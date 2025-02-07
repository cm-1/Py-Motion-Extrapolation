If you write other Cython code that needs to reference the current Cython code,
you will need to create a `weighted_impurity.pxd` file in the current directory
with the following content:


```
from sklearn.utils._typedefs cimport float64_t, int8_t, intp_t

from sklearn.tree._criterion cimport Criterion, ClassificationCriterion

import numpy as np
cimport numpy as cnp
cnp.import_array()

cdef class WeightedErrorCriterion(ClassificationCriterion):
    cdef float64_t[:, :, ::1] y_errs # The errors for each class for each sample.
    
    cpdef int set_y_errs(self, cnp.ndarray[float64_t, ndim=3] y_errs)
```
