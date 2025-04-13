import numpy as np
import matplotlib.pyplot as plt

import cinpact

from gtCommon import PoseLoaderBCOT

bodIndex = 1
seqIndex = 11
cvFrameSkip = 2
calculator = PoseLoaderBCOT(bodIndex, seqIndex, cvFrameSkip)

def plotGTvsCalc(gt, calc, showOnlyGT = False):
    cvFrameSkip = calculator._cvFrameSkipForLoad
    val = gt[:, 0]
    indices = np.arange(len(gt))

    init_k = 10
    init_c = 10

    cinpactCurve = cinpact.CinpactCurve(calc, True, init_c, init_k, 1000)
    curvePts = cinpactCurve.curvePoints


    fig = plt.figure()
    ax = fig.add_subplot()
    scatterGT = ax.scatter(indices, val, marker='.')
    skippedIndices = indices[1::(cvFrameSkip + 1)]

    if not showOnlyGT and len(calc) > 0:
        scatterCalc = ax.scatter(skippedIndices, calc[:, 0], marker='+')

        line = ax.plot(cinpactCurve.paramVals, curvePts[:, 0])
        
    plt.show()

plotGTvsCalc(calculator.getRotationsGTNP(), calculator.getRotationsCalcNP(), False)

print("Issue frames for pose path", calculator.posePathGT, "are:", calculator.issueFrames)
