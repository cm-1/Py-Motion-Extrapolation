import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import cinpact

from gtCommon import PoseLoaderBCOT

bodIndex = 1
seqIndex = 11
cvFrameSkip = 2
calculator = PoseLoaderBCOT(bodIndex, seqIndex, cvFrameSkip)

def plotGTvsCalc(gt, calc, showOnlyGT = False):
    cvFrameSkip = calculator._cvFrameSkipForLoad
    x = gt[:, 0]
    y = gt[:, 1]
    z = gt[:, 2]
    w = np.arange(len(x))#translationsNP.shape[0])

    lastFrameNum = len(gt) - (len(gt) % (cvFrameSkip + 1))
    # TODO: Remove the below sanity check after I get a chance to test with real data again.
    if len(calc) > 0 and lastFrameNum != len(calc) * (cvFrameSkip + 1):
        raise Exception("Logic error in frame num calc!")

    init_k = 10
    init_c = 10
    init_t = lastFrameNum
    init_endpt = -1

    curvePts = cinpact.CinpactCurve(calc, True, init_c, init_k, 1000).curvePoints

    scaled_w = (w - w.min()) / np.ptp(w)
    colors = plt.cm.coolwarm(scaled_w)

    fig = plt.figure()
    plt.subplots_adjust(bottom = 0.25)
    ax = fig.add_subplot(projection='3d')
    # The below gives a 1:1:1 aspect ratio.
    ax.set_box_aspect(np.ptp(gt, axis=0))
    scatterGT = ax.scatter(x, y, z, marker='.', edgecolors=colors)
    if not showOnlyGT:
        colorsCalc = colors[1::(cvFrameSkip + 1)]
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
            end = int(tSlider.val) // (cvFrameSkip + 1)
            endGT = int(tSlider.val) + 1
            scatterGT.set_offsets(gt[:endGT, :2])
            scatterGT.set_3d_properties(gt[:endGT, 2], 'z')
            scatterGT.set_edgecolor(colors[:endGT])
            scatterCalc.set_offsets(calc[:end, :2])
            scatterCalc.set_3d_properties(calc[:end, 2], 'z')
            scatterCalc.set_edgecolor(
                colors[1:(end * cvFrameSkip)+1:(cvFrameSkip + 1)]
            )
            fig.canvas.draw_idle()

        def updatePts(_):
            global curvePts
            curvePts = cinpact.CinpactCurve(calc[:init_endpt], True, cSlider.val, kSlider.val, 1000).curvePoints
            end = int(tSlider.val)
            line[0].set_data_3d(
                curvePts[:end, 0], curvePts[:end, 1], curvePts[:end, 2]
            )
            fig.canvas.draw_idle()



        kSlider.on_changed(updatePts)
        cSlider.on_changed(updatePts)
        tSlider.on_changed(updateDisplay)
        
    plt.show()

calculator.loadData()
# calculator.replaceDataWithHelix(False)
plotGTvsCalc(calculator.getTranslationsGTNP(), calculator.getTranslationsCalcNP(), False)

print("Issue frames for pose path", calculator.posePathGT, "are:", calculator.issueFrames)
