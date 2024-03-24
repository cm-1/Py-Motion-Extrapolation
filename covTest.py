# This file is for allowing me to compare the results of my C++ covariance
# matrix implementation against Numpy as a quick sanity check.

import numpy as np

rng = np.random.RandomState(1234)

xVals = rng.random_sample((10, 2))

for x in xVals:
    print("testVals.push_back({{ {}, {} }});".format(x[0], x[1]))

print("cov:")
print(np.cov(xVals.T))
