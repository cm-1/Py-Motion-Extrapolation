import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import cinpact

from gtCommon import BCOT_Data_Calculator

bodIndex = 1
seqIndex = 11
skipAmount = 2
calculator = BCOT_Data_Calculator(bodIndex, seqIndex, skipAmount)






def plotGTvsCalc(gt, calc, showOnlyGT = False):
    skipAmt = calculator.skipAmt
    x = gt[:, 0]
    y = gt[:, 1]
    z = gt[:, 2]
    w = np.arange(len(x))#translationsNP.shape[0])

    lastFrameNum = len(calc) * (calculator.skipAmt + 1)

    init_k = 10
    init_c = 10
    init_t = lastFrameNum
    init_endpt = -1

    curvePts = cinpact.CinpactCurve(calc, init_c, init_k, 1000).curvePoints

    scaled_w = (w - w.min()) / w.ptp()
    colors = plt.cm.coolwarm(scaled_w)

    fig = plt.figure()
    plt.subplots_adjust(bottom = 0.25)
    ax = fig.add_subplot(projection='3d')
    scatterGT = ax.scatter(x, y, z, marker='.', edgecolors=colors)
    if not showOnlyGT:
        colorsCalc = colors[1::(skipAmt + 1)]
        scatterCalc = ax.scatter(calc[:, 0], calc[:, 1], calc[:, 2], marker='+', edgecolors=colorsCalc)
        scatterCalc.set_edgecolor(colorsCalc) # Need to do this again to make it look right

        line = ax.plot(curvePts[:, 0], curvePts[:, 1], curvePts[:, 2])

        fig.subplots_adjust(left=0.25, bottom=0.25)
        kAx = fig.add_axes([0.25, 0.1, 0.20, 0.03])
        cAx = fig.add_axes([0.6, 0.1, 0.20, 0.03])
        tAx = fig.add_axes([0.25, 0.0, 0.20, 0.03])
        kSlider = Slider(
            ax=kAx,
            label='k',
            valmin=0.1,
            valmax=30,
            valinit=init_k,
        )
        cSlider = Slider(
            ax=cAx,
            label='c',
            valmin=1.1,
            valmax=30,
            valinit=init_c,
        )
        tSlider = Slider(
            ax = tAx,
            label ='t',
            valmin = 0,
            valmax = lastFrameNum,
            valstep = 1,
            valinit=init_t
        )

        def updateDisplay(_):
            end = int(tSlider.val) // (skipAmt + 1)
            endGT = int(tSlider.val) + 1
            scatterGT.set_offsets(gt[:endGT, :2])
            scatterGT.set_3d_properties(gt[:endGT, 2], 'z')
            scatterGT.set_edgecolor(colors[:endGT])
            scatterCalc.set_offsets(calc[:end, :2])
            scatterCalc.set_3d_properties(calc[:end, 2], 'z')
            scatterCalc.set_edgecolor(colors[1:(end * skipAmt)+1:(skipAmt + 1)])
            fig.canvas.draw_idle()

        def updatePts(_):
            global curvePts
            curvePts = cinpact.getSplinePts(calc[:init_endpt], cSlider.val, kSlider.val, 1000)
            end = int(tSlider.val)
            line[0].set_data_3d(
                curvePts[:end, 0], curvePts[:end, 1], curvePts[:end, 2]
            )
            fig.canvas.draw_idle()



        kSlider.on_changed(updatePts)
        cSlider.on_changed(updatePts)
        tSlider.on_changed(updateDisplay)
        
    plt.show()

plotGTvsCalc(calculator.rotationsGTNP, calculator.rotationsCalcNP, False)

print("Issue frames for pose path", calculator.posePathGT, "are:", calculator.issueFrames)
