
My custom decision tree criterion code extends a class in scikit-learn's
`criterion.pyx` file, and so some of the methods I overrode bear a strong
resemblence to the respective methods of the original, with only the relevant
lines changed. As such, I have included scikit-learn's license
(the `COPYING` file) in this folder and am noting here the connection between my
derivative work and the original.

# Building

Run `python .\setup.py build_ext --inplace`.

# Notes on using this code:

The most important notes are:

- I did not have missing values in my training data, and so I did not bother
  to override and adapt the code in `init_missing()` where `self.sum_missing` was 
  updated; for this criterion to correctly handle missing values, that must
  be added to this file.
- If you write other Cython code that needs to reference my Cython code,
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

Other possible considerations are mentioned in the comments at the top
(and, to a lesser extent, throughout) the `weighted_impurity.pyx` file.