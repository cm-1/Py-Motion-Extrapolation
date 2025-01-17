import numpy as np

from spline_approximation import BSplineFitCalculator, SplinePredictionMode
import bspline

SPLINE_DEGREE = 4
PTS_USED_TO_CALC_LAST = 10 # Must be at least spline's degree + 1.
DESIRED_CTRL_PTS = 9
PT_DIM = 6

# Set ctrl points, then generate values at select u vals, then fit b-spline to
# sample, then compare against original.

ctrl_pts = np.random.sample((DESIRED_CTRL_PTS, PT_DIM))
uInterval = (SPLINE_DEGREE, DESIRED_CTRL_PTS) 

m = DESIRED_CTRL_PTS - 1
k = SPLINE_DEGREE + 1
knots = np.arange(DESIRED_CTRL_PTS + SPLINE_DEGREE + 1)
muList = np.ones(len(knots), dtype=int)
num_params = PTS_USED_TO_CALC_LAST + 1
params = np.linspace(uInterval[0], uInterval[1], num_params)


weights = radii = np.ones(DESIRED_CTRL_PTS, dtype=int)
outputs = bspline.specifiedPtsFromNURBS(ctrl_pts, weights, m, k, knots, muList, params, radii)
out_pts = outputs.points

fit_modes = [SplinePredictionMode.EXTRAPOLATE]
fitter = BSplineFitCalculator(
    SPLINE_DEGREE, DESIRED_CTRL_PTS, PTS_USED_TO_CALC_LAST, fit_modes
)

res_extrap = fitter.fitAllData(out_pts)[-1]

res_ctrl = fitter.getCtrlPts(out_pts[:-1])

def maxDiff(a,b):
    return np.abs(a - b).max()
print("max extrap diff:", maxDiff(res_extrap, out_pts[-1]))
print("max ctrl diff:", maxDiff(res_ctrl, ctrl_pts))
