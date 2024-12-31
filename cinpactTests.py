#%%
import matplotlib.pyplot as plt
import numpy as np

from cinpact import CinpactCurve, CinpactLogic
from curvetools import tangentsFromPoints

#%% Derivative Tests
randPts = np.random.uniform(-35.0, 35.0, (35, 2))

randK = 10.0
randRad = 10.0
randOpen = True
randCurve = CinpactCurve(randPts, randOpen, randRad, randK, 3500)

rand_i = np.random.random_integers(0, 35, 1)[0]
rand_us = np.linspace(rand_i - 0.1, rand_i + 0.1, 101)
weight_vals = CinpactCurve.weightFunc(rand_us, rand_i, randRad, randK)

# delta_u = rand_us[1] - rand_us[0]
delta_u = randCurve.paramVals[1] - randCurve.paramVals[0]
approxDerivs = tangentsFromPoints(randCurve.curvePoints, False)
# approxDerivs = tangentsFromPoints(weight_vals, False)
approxDerivs /= delta_u
approx2ndDerivs = tangentsFromPoints(approxDerivs, False)
approx2ndDerivs /= delta_u
# theoreticalDerivInfo = CinpactCurve.weightAndDerivative(rand_us, rand_i, randRad, randK, True)
# theoreticalDerivs, theoreticalDerivs2 = theoreticalDerivInfo
sampleParams = randCurve.paramVals[::5]
theoreticalDerivs = np.empty((len(sampleParams), len(randPts[0])))
theoreticalDerivs2 = np.empty((len(sampleParams), len(randPts[0])))
theoreticals = (theoreticalDerivs, theoreticalDerivs2)
for i, u in enumerate(sampleParams):
    ptRange = randCurve.ctrlPtBoundsInclusive(u)
    dfilters = randCurve.weightDerivativeFilter(u, True)
    derivs = []
    selectCtrlPts = randPts[ptRange[0]:ptRange[1] + 1]
    for j, filter in enumerate(dfilters):
        if np.abs(filter.sum()) > 0.0001:
            print("filter:", filter, "sum=", filter.sum())
            raise Exception("Filter sum issue for order {}!".format(j))
        theoreticals[j][i] = np.dot(selectCtrlPts.transpose(), filter).transpose()

print("Deriv diff:", np.abs(approxDerivs[::5] - theoreticalDerivs).max())
print("Deriv2 diff:", np.abs(approx2ndDerivs[::5] - theoreticalDerivs2).max())

#%% Visual Test

julyPts = [
    (-0.4453846153846154, 0.16893899204244023),
    (-0.5400928381962864, 0.4914588859416445),
    (-0.1740583554376658, 0.8856498673740053),
    (-0.12030503978779838, 0.6245623342175066),
    (-0.8242175066312998, -0.0972679045092838),
    (-0.8421352785145888, 0.0870291777188329),
    (-0.2047745358090185, 0.3583554376657825),
    (-0.21245358090185684, 0.2559681697612731),
    (0.07423076923076921, 0.3993103448275862),
    (0.0435145888594165, 0.2687665782493368),
    (0.5452122015915121, 0.5477718832891247),
    (0.6552785145888594, 0.7781432360742706),
    (0.2150132625994695, 0.28668435013262594),
    (0.5324137931034482, 0.3865119363395225),
    (0.5144960212201591, 0.26620689655172414),
    (0.7883819628647214, 0.3890716180371353),
    (0.4351458885941645, 0.005119363395225451),
    (0.4121087533156499, 0.0972679045092838),
    (0.965, 0.2892440318302387)
]

julyPtsNP = np.array(julyPts)


july_curve = CinpactCurve(julyPtsNP, True, 10.0, 10.0, 1000)

fig, ax = plt.subplots()
ax.plot(*july_curve.curvePoints.transpose())#, marker='o')
ax.scatter(*julyPtsNP.transpose(), color='orange')
plt.show()

#%% 
# Graph that shows that CINPACT pts are not evenly spaced when ctrl pts are
# evenly spaced along a line.
pts_on_yeqx = np.linspace([0.0, 0.0], [8.0, 8.0], 8, axis=0)
cinpact_line = CinpactCurve(pts_on_yeqx, True, 3.0, 3.0, 35)
fig, ax = plt.subplots()
ax.plot(*cinpact_line.curvePoints.transpose(), marker='o')
# ax.scatter(*pts_on_yeqx.transpose(), color="orange")
plt.show()


#%% Graphs sum of unnormalized CINPACT basis functions.
fig2, ax2 = plt.subplots()

individ_weights = np.empty((len(randPts), len(randCurve.paramVals)))
for i in range(len(randPts)):
    individ_weights[i] = CinpactCurve.weightFunc(randCurve.paramVals, i, 3, 3)
    # ax2.plot(randCurve.paramVals, individ_weights[i])

ax2.plot(randCurve.paramVals, individ_weights.sum(axis=0))
plt.show()

#%%
# Graph that shows that a 1D CINPACT curve with ctrl pts that are linearly 
# ascending will be "wavy", whereas a 1D B-Spline curve will interpolate the
# inputs perfectly. This has implications for CINPACT's ability to model motion
# with constant velocity.
import bspline
fig2, ax2 = plt.subplots()

def bsplinePtsFromCtrlPts(ctrl_pts, k, num_samples):
    mpts = [bspline.MotionPoint(pt) for pt in ctrl_pts]
    ms = bspline.MotionBSpline(mpts, k, False)
    return bspline.ptsFromNURBS(ms, num_samples, False)

bspline_pts_1D = np.stack([
    np.arange(10, dtype=float), np.ones(10, dtype=float)
], axis=-1)
bspline_pts_10D = np.concat([np.eye(10), np.ones((10, 1))], axis = -1)


bpts = bsplinePtsFromCtrlPts(bspline_pts_10D, 3, 350)
bpts_1D = bsplinePtsFromCtrlPts(bspline_pts_1D, 3, 35)

ax2.scatter(bpts_1D.params, bpts_1D.points)

lin_mat = np.arange(10).reshape(1, 10)
lin_pts = lin_mat @ bpts.points.T
ax2.plot(bpts.params, lin_pts.flatten(), label="B-SPLINE for linear input")

c_weights_10 = np.empty((10, len(bpts.params)))

for i in range(10):
    ax2.plot(bpts.params, bpts.points[:, i])
    c_from_b_params = CinpactCurve.bsplineTransformU(bpts.params, i, 3)
    c_weights_10[i] = CinpactCurve.weightFunc(c_from_b_params, i, 3.898, 25.83)

c_sums_10 = c_weights_10.sum(axis=0, keepdims=True)
c_nweights_10 = c_weights_10 / c_sums_10
c_lin_pts = lin_mat @ c_nweights_10

ax2.scatter(
    bpts.params[::10], c_lin_pts.flatten()[::10], facecolors="none", 
    edgecolors="orange", label="B-CINPACT for linear input"
)

c_weights_10 = np.empty((len(bpts.params), i))

for i in range(10):
    ax2.plot(bpts.params, c_nweights_10[i], dashes=[1,1])
ax2.legend()
plt.show()