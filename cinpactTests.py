#%%
import matplotlib.pyplot as plt
import numpy as np

from cinpact import CinpactCurve, CinpactLogic
from bspline import tangentsFromPoints

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

curve = CinpactCurve(julyPtsNP, True, 10.0, 10.0, 1000)

fig, ax = plt.subplots()
ax.plot(*curve.curvePoints.transpose())
ax.scatter(*julyPtsNP.transpose())
plt.show()
