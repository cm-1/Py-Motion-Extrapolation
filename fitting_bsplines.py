# TODO: Look at https://matplotlib.org/stable/gallery/widgets/menu.html#sphx-glr-gallery-widgets-menu-py

from dataclasses import dataclass
import typing

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

import bspline
import gtCommon


# TODO: Allow custom caps, based on below constraint:
# Let n+1 be number of (known+unknown) pts "on" curve, g be the gap size.
# There will be m - k + 2 units of space in the domain of a uniform knot
# sequence for which the basis functions will sum to
# ng <= m - k + 2, m <= n - 1, because we have n+1 points on curve, n known points, and m+1 control points, and control point count can't be more than known points
# ng <= n - k + 1
# ng - n <= 1 - k
# g <= 1 - (k-1)/n

SPLINE_DEGREE = 2
PTS_USED_TO_CALC_LAST = 6 # Must be at least spline's degree + 1.
NUM_OUTPUT_PREDICTIONS = 1
DESIRED_CTRL_PTS = 5
if (PTS_USED_TO_CALC_LAST < SPLINE_DEGREE + 1):
    raise Exception("Need at least order=k=(degree + 1) input pts for calc!")
if DESIRED_CTRL_PTS > PTS_USED_TO_CALC_LAST:
    raise Exception("Need at least as many input points as control points!")
if DESIRED_CTRL_PTS < SPLINE_DEGREE + 1:
    raise Exception("Need at least order=k=(degree + 1) control points!")

BUTTON_OFF_COLOR = 'lightcoral'
BUTTON_ON_COLOR = 'lightgreen'


@dataclass
class PlotData:
    # componentKey: typing.Tuple[int, int]
    x_vals: typing.Any
    y_vals: typing.Any
    x_preds: typing.Any
    y_preds: typing.Any

@dataclass
class AxLineData:
    num_true_vals: int
    num_pred_vals: int
    true_line: typing.Any
    pred_line: typing.Any

axLinesDict = dict()

lastKnownPtIndex = 40 #41
lastPtIndex = lastKnownPtIndex + NUM_OUTPUT_PREDICTIONS




# nonRandomHeights = np.array([0.9, 0.8, 0.99, 1.2, 0.5, 0.1, 0.135])
# def randErr(numPts, low = -1, high = 1):
#     return np.cumsum(np.random.uniform(low, high, numPts))

# RANDS = 35
# ptsToFit0 = np.concatenate((randErr(RANDS), nonRandomHeights, randErr(RANDS)))
# ptsToFit1 = randErr(RANDS + len(nonRandomHeights) + RANDS)

bcotDataObjects = []
for seqIndex in range(len(gtCommon.BCOT_SEQ_NAMES)):
    bodList = []
    for bodIndex in range(len(gtCommon.BCOT_BODY_NAMES)):
        if gtCommon.isBodySeqPairValid(bodIndex, seqIndex):
            bodList.append(gtCommon.BCOT_Data_Calculator(bodIndex, seqIndex, 2))
        else:
            bodList.append(None)
    bcotDataObjects.append(bodList)



def fitBSpline(numCtrlPts, order, ptsToFit, uVals, knotVals, numUnknownPts = 0):
    if (len(ptsToFit) + numUnknownPts) != len(uVals):
        raise Exception(
            "Number of u values does not match number of known + unknown points!"
        )

    numInputPts = len(ptsToFit)

    mat = np.zeros((numInputPts, numCtrlPts))

    iden = np.identity(numCtrlPts)

    delta = order - 1
    for r in range(numInputPts):
        u = uVals[r]
        while (delta < numCtrlPts - 1 and u >= knotVals[delta + 1]):
            delta = delta + 1 # muList[delta + 1]
        for c in range(numCtrlPts):
            mat[r][c] = bspline.bSplineInner(u, order, delta, iden[c], knotVals)

    fitted_ctrl_pts = np.linalg.lstsq(mat, ptsToFit, rcond = None)[0]
    return fitted_ctrl_pts

startInd = max(0, lastKnownPtIndex - PTS_USED_TO_CALC_LAST + 1)
uInterval = (SPLINE_DEGREE, lastKnownPtIndex + 1 - startInd)
ctrlPtCount = min(lastKnownPtIndex + 1, DESIRED_CTRL_PTS)

# Interval over which basis functions sum to 1, assuming uniform knot seq.
knotList = np.arange(ctrlPtCount + SPLINE_DEGREE + 1)
uInterval = (knotList[SPLINE_DEGREE], knotList[-SPLINE_DEGREE - 1])
numTotal = lastKnownPtIndex + 1 - startInd + NUM_OUTPUT_PREDICTIONS
uVals = np.linspace(uInterval[0], uInterval[1], numTotal)
# uVals = uInterval[1] - gap*np.arange(numTotal - 1, -1, -1)



fig, ax = plt.subplots(figsize=(8, 6))
axisVals = [0, 5, -2, 2]
ax.axis(axisVals)

def on_press(event):
    x_shift = 0
    y_shift = 0
    if event.key == 'a':
        x_shift = -1
    elif event.key == 'd':
        x_shift = 1
    elif event.key == 'w':
        y_shift = 1
    elif event.key == 'x':
        y_shift = -1
    if x_shift!= 0 or y_shift != 0:
        axisVals[0] += x_shift
        axisVals[1] += x_shift
        axisVals[2] += y_shift
        axisVals[3] += y_shift
        ax.axis(axisVals)
        fig.canvas.draw()


fig.canvas.mpl_connect('key_press_event', on_press)


        

    
# We want options to show x, y, and z for each of the 3D points related to the
# model's pose. These include its origin (denoted as point "0"), the endpoints
# of the local axes (denoted as "1", "2", and "3"), and the axis-angle
# representation of the rotation (denoted "AA").
# These options will be available as a grid of buttons. 
row_labels = ['x', 'y', 'z']
column_labels = [str(i) for i in range(4)] + ["AA"]

def componentLabel(valIndex, axisIndex):
    return column_labels[axisIndex] + "," + row_labels[valIndex]

# ------------------------------------------------------------------------------
# Now comes all of the axes layout code.

plt.subplots_adjust(left=0.2, bottom=0.3, right=0.95, top=0.9)

coord_table_margin = 0.05
coord_cell_w = 0.03
coord_cell_h = 0.03
# Top of coord label = bottom margin + (a cell for each label + column headers):
coord_table_top = coord_table_margin + coord_cell_h * (len(row_labels) + 1)
def getButtonAxes(i, j):
    return plt.axes((
        coord_table_margin + coord_cell_w * j,
        coord_table_top - coord_cell_h * i, coord_cell_w, coord_cell_h
    ))

# Two sliders in the right half of the area underneath the plot.
# One controls the body index, the other the sequence index.
slider_ax1 = plt.axes([0.6, 0.2, 0.35, 0.03])#, facecolor='lightgoldenrodyellow')
slider_ax2 = plt.axes([0.6, 0.15, 0.35, 0.03])#, facecolor='lightgoldenrodyellow')

slider1 = Slider(
    slider_ax1, gtCommon.shortBodyNameBCOT(gtCommon.BCOT_BODY_NAMES[0]), 
    0, len(gtCommon.BCOT_BODY_NAMES) - 1, valstep=1
)
numSeqs = len(gtCommon.BCOT_SEQ_NAMES)
slider2 = Slider(
    slider_ax2, gtCommon.shortSeqNameBCOT(gtCommon.BCOT_SEQ_NAMES[numSeqs - 1]), 
    0, numSeqs - 1, valstep=1, valinit=numSeqs - 1
)




# Create a grid to store the state of each button
button_states = np.full((len(row_labels), len(column_labels)), False) # Initial state: all False (disabled)



def plotSelected():

    bInd = int(slider1.val + 0.001)
    sInd = int(slider2.val + 0.001)
    bcotDataObj = bcotDataObjects[sInd][bInd]
    retVal = dict()
    if bcotDataObj is None or not np.any(button_states):
        return retVal

    # Flatten needed here or else array looks like [[0], [2], ...]
    originComponents = np.argwhere(button_states[:, 0]).flatten()
    originData = bcotDataObj.getTranslationsGTNP(True)[:, originComponents]
    retIndices = [(0, i) for i in originComponents]

    # IGNORE MIDDLE BUTTON COLUMNS FOR NOW!

    # Flatten needed here for same reason as before.
    aaComponents = np.argwhere(button_states[:, 4]).flatten()
    aaData = bcotDataObj.getRotationsGTNP(True)[:, aaComponents]
    retIndices += [(4, i) for i in aaComponents]

    # Points stored such that each row is a timestamp and each column is a
    # component of either the origin, the axis-angle rotation, or another pt
    # on the object.
    ptsData = np.hstack((originData, aaData))


    for i, tupVal in enumerate(retIndices):
        retVal[tupVal] = PlotData(np.arange(len(ptsData)), ptsData[:, i], [], [])

    # Need an "empty" array with the right number of rows to hstack with.
    ctrlPts = np.empty((ctrlPtCount, 0)) 
    
    if lastKnownPtIndex >= SPLINE_DEGREE: # TODO: Replace with a clamp on index
        for i in range(ptsData.shape[1]):
            ptsToFit_i = ptsData[:, i]
            ptSubsetToFit = ptsToFit_i[startInd:lastKnownPtIndex + 1]

            ctrlPts1D = fitBSpline(
                ctrlPtCount, SPLINE_DEGREE + 1, ptSubsetToFit, uVals, knotList,
                NUM_OUTPUT_PREDICTIONS
            )
            # print(ctrl_pts)
            ctrlPts = np.hstack((ctrlPts, ctrlPts1D.reshape((ctrlPtCount, 1))))

        ctrlPts = np.hstack((ctrlPts, np.ones((len(ctrlPts), 1)))) # Weights

        ctrlPtDumbWrap = [bspline.MotionPoint(c) for c in ctrlPts]
        spline = bspline.MotionBSpline(ctrlPtDumbWrap, SPLINE_DEGREE + 1, False)
        spline_result = bspline.ptsFromNURBS(spline, 100, False)

        spline_result.params -= uInterval[0]
        spline_result.params *= ((lastPtIndex - startInd)/(uInterval[1] - uInterval[0]))
        spline_result.params += startInd

        for i, tupVal in enumerate(retIndices):
            retVal[tupVal].x_preds = spline_result.params
            retVal[tupVal].y_preds = spline_result.points[:, i]
    return retVal

    #         ptsToFitChoice = ptsToFit0 if dropdown.value == 0 else ptsToFit1
    #         line1.set_ydata(ptsToFitChoice)

def clearAndRedraw(componentDataDict):
    ax.clear()
    axLinesDict.clear()
    for k in componentDataDict:
        plotData = componentDataDict[k]
        line_true, = ax.plot(
            plotData.x_vals, plotData.y_vals, 'o',
            label=componentLabel(k[1], k[0])
        )
        line_pred, = ax.plot(plotData.x_preds, plotData.y_preds)
        axLinesDict[k] = AxLineData(
            len(plotData.x_vals), len(plotData.x_preds), line_true, line_pred
        )
    ax.axis(axisVals)
    ax.legend()
    fig.canvas.draw_idle()

def update(val):
    bInd = int(slider1.val + 0.001)
    sInd = int(slider2.val + 0.001)
    slider1.label.set_text(gtCommon.shortBodyNameBCOT(gtCommon.BCOT_BODY_NAMES[bInd]))
    slider2.label.set_text(gtCommon.shortSeqNameBCOT(gtCommon.BCOT_SEQ_NAMES[sInd]))

    componentDataDict = plotSelected()

    if len(componentDataDict.keys()) > 0:
        needsClearing = False
        for k in componentDataDict:
            firstComp = componentDataDict[k]
            firstAxLineData = axLinesDict[k]
            true_diff = len(firstComp.x_vals) != firstAxLineData.num_true_vals
            pred_diff = len(firstComp.x_preds) != firstAxLineData.num_pred_vals
            needsClearing = needsClearing or true_diff or pred_diff
        if needsClearing:
            clearAndRedraw(componentDataDict)
        else:
            for k in componentDataDict:
                plotData = componentDataDict[k]
                axLineDataVal = axLinesDict[k]
                axLineDataVal.true_line.set_ydata(plotData.y_vals)
                axLineDataVal.pred_line.set_ydata(plotData.y_preds)

            fig.canvas.draw_idle()


slider1.on_changed(update)
slider2.on_changed(update)

# Create a grid of buttons
buttons = []

# Function to handle button clicks (toggling state)
def on_button_clicked(event, row, col, button):
    
    if row > 0 and col > 0:
        # Toggle the state for the button at (row, col)
        button_states[row - 1, col - 1] = not button_states[row - 1, col - 1]
        # Update the button's color and text based on the new state
        if button_states[row - 1, col - 1]:
            button.color = BUTTON_ON_COLOR
        else:
            button.color = BUTTON_OFF_COLOR

    componentDataDict = plotSelected()
    
    clearAndRedraw(componentDataDict)



for i in range(len(row_labels) + 1):
    button_row = []
    for j in range(len(column_labels) + 1):
        # Compute position for each button (compact spacing)
        btn_ax = getButtonAxes(i, j)
        text = ""
        btn_color = "white"
        if i == 0 and j != 0:
            text = column_labels[j - 1]
        elif j == 0 and i != 0:
            text = row_labels[i - 1]
        if i > 0 and j > 0:
            btn_color = BUTTON_OFF_COLOR
        
        button = Button(btn_ax, text, color=btn_color)  # Default state is "Off"
        # Connect the click event with the button
        button.on_clicked(lambda event, row=i, col=j, btn=button: on_button_clicked(event, row, col, btn))
        button_row.append(button)
    buttons.append(button_row)

on_button_clicked(None, 0, 0, None)

# Show the plot
plt.show()
