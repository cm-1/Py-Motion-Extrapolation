# TODO:
# - Testing if ys have changed since last but y_errs have not.
# - Missing sums
# - Performance stuff, like boundscheck(False)
# - In set_y_errs, check that shape is correct!
# - Check annotated output
# - README on __cinit__

# from libc.stdio cimport printf

from sklearn.utils._typedefs cimport float64_t, int8_t, intp_t

from libc.string cimport memset

import numpy as np
cimport numpy as cnp
cnp.import_array()

from sklearn.tree._criterion cimport Criterion, ClassificationCriterion

cdef float64_t INFINITY = np.inf

cdef class WeightedErrorCriterion(ClassificationCriterion):
    r"""
    Calculating classification error when picking certain wrong classes would be
    worse than other wrong classes for a given sample.

    Thus, rather than assessing a node split by how many classes are assigned
    correctly, we can instead assess by the total error. 
    """

    cdef const float64_t[:, :, ::1] y_errs

    def __deepcopy__(self, memo):
        """Override deepcopy to manually handle copying y_errs."""
        new_obj = WeightedErrorCriterion(self.n_outputs, np.array(self.n_classes, dtype=np.intp))
        
        # Explicitly copy the y_errs memoryview
        new_obj.y_errs = self.y_errs  # Shallow copy reference
        return new_obj

    cpdef int set_y_errs(self, cnp.ndarray[float64_t, ndim=3] y_errs):
        self.y_errs = np.ascontiguousarray(y_errs, dtype=np.float64)
        return 0

    # The code for this override is nearly identical to the base class's, except
    # that sum_total's calculated from self.y_errs instead of self.y and an
    # error is raised if self.y_errs has not been set yet.
    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end
    ) except -1 nogil:
        """Initialize the criterion.

        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : ndarray, dtype=float64_t
            The target stored as a buffer for memory efficiency.
        sample_weight : ndarray, dtype=float64_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : float64_t
            The total weight of all samples
        sample_indices : ndarray, dtype=intp_t
            A mask on the samples. Indices of the samples in X and y we want to use,
            where sample_indices[start:end] correspond to the samples in this node.
        start : intp_t
            The first sample to use in the mask
        end : intp_t
            The last sample to use in the mask
        """
        #if not self.y_errs_supplied:
        #    raise Exception("Per-class errors not supplied to criterion!")
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef intp_t c
        cdef float64_t w = 1.0

        for k in range(self.n_outputs):
            memset(&self.sum_total[k, 0], 0, self.n_classes[k] * sizeof(float64_t))
        for p in range(start, end):
            i = sample_indices[p]

            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0.
            if sample_weight is not None:
                w = sample_weight[i]
            # Count weighted class frequency for each target
            for k in range(self.n_outputs):
                # The below two lines are pretty much the only way this function
                # differs from the version in the base class.
                for c in range(self.max_n_classes):
                    # We negate the errors because other code that we do not
                    # want to have to override chooses the class for a node
                    # based on an argmax, not argmin, of the sum_total.
                    self.sum_total[k, c] += -w * self.y_errs[i, k, c]

            self.weighted_n_node_samples += w
        # Reset to pos=start
        self.reset()
        return 0

    # The code for this override is nearly identical to the one in the base 
    # class, except that sums are calculated from self.y_errs instead of self.y.
    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left child.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        new_pos : intp_t
            The new ending position for which to move sample_indices from the right
            child to the left child.
        """
        cdef intp_t pos = self.pos
        # The missing samples are assumed to be in
        # self.sample_indices[-self.n_missing:] that is
        # self.sample_indices[end_non_missing:self.end].
        cdef intp_t end_non_missing = self.end - self.n_missing

        cdef const intp_t[:] sample_indices = self.sample_indices
        cdef const float64_t[:] sample_weight = self.sample_weight

        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef intp_t c
        cdef float64_t w = 1.0


        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.
        if (new_pos - pos) <= (end_non_missing - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    # The below two lines are pretty much the only way this 
                    # for loop differs from the version in the base class.
                    for c in range(self.max_n_classes):
                        # We negate the errors because other code that we do not
                        # want to have to override chooses the class for a node
                        # based on an argmax, not argmin, of the sum_total.
                        self.sum_left[k, c] += -w * self.y_errs[i, k, c]

                self.weighted_n_left += w

        else:
            self.reverse_reset()

            for p in range(end_non_missing - 1, new_pos - 1, -1):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    # The below two lines are pretty much the only way this 
                    # for loop differs from the version in the base class.
                    for c in range(self.max_n_classes):
                        # Here we do *NOT* negate the errors because the code in
                        # the base class has -= here instead of +=, meaning the
                        # negation needed in the prior for loop is canceled out.
                        self.sum_left[k, c] += w * self.y_errs[i, k, c]

                self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                self.sum_right[k, c] = self.sum_total[k, c] - self.sum_left[k, c]

        self.pos = new_pos
        return 0

    cdef float64_t node_impurity(self) noexcept nogil:
        """Evaluate the impurity of the current node.

        Impurity will be based on the average error if the single best class is
        chosen by the node.

        We want average error because the function for calculating proxy
        improvement will multiply left and right child impurities by the
        respective left and right sample counts.
        A possible TODO in the future could be to override that as well, so that
        the cancelled-out multiply and divide is omitted.

        Because higher impurity is considered worse, we will represent impurity
        with the total positive error. Whereas self.sum_total had to be negative
        error so that argmax in other code will work, here we un-negate things.
        """
        cdef float64_t max_neg_err
        cdef float64_t max_neg_err_sum = 0.0
        cdef float64_t neg_err
        cdef intp_t k
        cdef intp_t c

        for k in range(self.n_outputs):
            max_neg_err = -INFINITY
            for c in range(self.n_classes[k]):
                neg_err = self.sum_total[k, c]
                if neg_err > max_neg_err:
                    max_neg_err = neg_err
            max_neg_err_sum += max_neg_err

        # Negation happens here to make impurity positive.
        return -max_neg_err_sum / self.weighted_n_node_samples

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).

        Parameters
        ----------
        impurity_left : float64_t pointer
            The memory address to save the impurity of the left node
        impurity_right : float64_t pointer
            The memory address to save the impurity of the right node
        """
        cdef float64_t max_neg_err_left
        cdef float64_t max_neg_err_right
        cdef float64_t max_neg_err_sum_left = 0.0
        cdef float64_t max_neg_err_sum_right = 0.0
        cdef float64_t neg_err
        cdef intp_t k
        cdef intp_t c

        for k in range(self.n_outputs):
            max_neg_err_left = -INFINITY
            max_neg_err_right = -INFINITY
            for c in range(self.n_classes[k]):
                neg_err = self.sum_left[k, c]
                if neg_err > max_neg_err_left:
                    max_neg_err_left = neg_err
                neg_err = self.sum_right[k, c]
                if neg_err > max_neg_err_right:
                    max_neg_err_right = neg_err
         
            max_neg_err_sum_left += max_neg_err_left
            max_neg_err_sum_right += max_neg_err_right

        # Negation happens here to make impurity positive.
        impurity_left[0] = -max_neg_err_sum_left / self.weighted_n_left
        impurity_right[0] = -max_neg_err_sum_right / self.weighted_n_right
