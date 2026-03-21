import numpy as np

GT_PT_IND = 6
DYNAMIC_PT_IND = GT_PT_IND - 1
LAST_FIXED_PT_IND = GT_PT_IND - 2

ERR_NA_VAL = np.finfo(np.float32).max # A non-inf but inf-like value.

