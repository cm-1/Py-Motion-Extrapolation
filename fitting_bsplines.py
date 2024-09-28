import bspline
import numpy as np
import matplotlib.pyplot as plt


NUM_CTRL_PTS = 4
SPLINE_DEGREE = 2
ptsToFit = np.array([0.9, 0.8, 0.99, 1.2, 0.5, 0.1, 0.135])

knotList = np.arange(NUM_CTRL_PTS + SPLINE_DEGREE + 1)

mat = np.zeros((len(ptsToFit), NUM_CTRL_PTS))

iden = np.identity(NUM_CTRL_PTS)

# Interval over which basis functions sum to 1, assuming uniform knot seq.
uInterval = (knotList[SPLINE_DEGREE], knotList[-SPLINE_DEGREE - 1])
uVals = np.linspace(uInterval[0], uInterval[1], len(ptsToFit))
delta = SPLINE_DEGREE
for r in range(len(ptsToFit)):
    u = uVals[r]
    while (delta < len(ptsToFit) - 1 and u >= uVals[delta + 1]):
        delta = delta + 1 # muList[delta + 1]
    for c in range(NUM_CTRL_PTS):
        mat[r][c] = bspline.bSplineInner(
            u, SPLINE_DEGREE + 1, delta, iden[c], knotList
        )

ctrl_pts = np.linalg.lstsq(mat, ptsToFit)[0]
print(ctrl_pts)

ctrl_pts_4D = [bspline.MotionPoint((c,0,0,1)) for c in ctrl_pts]
spline = bspline.MotionBSpline(ctrl_pts_4D, SPLINE_DEGREE + 1, False)
spline_result = bspline.ptsFromNURBS(spline, 100, False)

plt.plot(uVals, ptsToFit, 'o')
plt.plot(spline_result.params, spline_result.points[:, 0])
plt.show()