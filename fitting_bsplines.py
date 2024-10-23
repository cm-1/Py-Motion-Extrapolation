# TODO: Look at https://matplotlib.org/stable/gallery/widgets/menu.html#sphx-glr-gallery-widgets-menu-py

from dataclasses import dataclass, field
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

FRAME_SKIP_AMT = 2
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

# TODO: Change "6" to something not hardcoded!
def getEmptyYData():
    return np.empty((0,6))

@dataclass
class PlotData:
    # componentKey: typing.Tuple[int, int]
    x_vals: typing.Any
    y_vals: typing.Any

@dataclass
class PredictionData:
    x_spline: typing.Any = field(default_factory=list)
    y_spline: typing.Any = field(default_factory=getEmptyYData)
    x_static: typing.Any = field(default_factory=list)
    y_static: typing.Any = field(default_factory=getEmptyYData)
    x_vel: typing.Any = field(default_factory=list)
    y_vel: typing.Any = field(default_factory=getEmptyYData)
    x_accel: typing.Any = field(default_factory=list)
    y_accel: typing.Any = field(default_factory=getEmptyYData)

@dataclass
class AxLines:
    true_line: typing.Any
    spline_line: typing.Any
    # static_line: typing.Any
    vel_line: typing.Any
    accel_line: typing.Any

class ObjSeqData:
    def __init__(self, bodID: int, seqID: int):
        self.bodID = bodID
        self.seqID = seqID
        self.hasData = False
        self.calculator = None
        if gtCommon.BCOT_Data_Calculator.isBodySeqPairValid(bodID, seqID):
            self.hasData = True
            self.calculator = gtCommon.BCOT_Data_Calculator(
                bodID, seqID, FRAME_SKIP_AMT
            )
        self.lastKnownPtIndex = 40# SPLINE_DEGREE
        self.plot_data = PlotData([], getEmptyYData())
        self.pred_data = PredictionData()
        self.y_window = (-2.0, 2.0)
        x_max = self.lastKnownPtIndex + NUM_OUTPUT_PREDICTIONS + 1
        x_min = self.lastKnownPtIndex - 3
        self.x_window = (x_min, x_max)

    # Returns [xmin, xmax, ymin, ymax]
    def getWindowVals(self):
        return [
            self.x_window[0], self.x_window[1],
            self.y_window[0], self.y_window[1]
        ]
    # data_col_keys: typing.Any = field(default_factory=list)
    # needs_reset: bool = True # TODO: Make "False" have an affect.


axLinesDict = dict()
buttonIndexToDataRow = {
    (0,0): 0, (1,0): 1, (2,0): 2, (0,4): 3, (1,4): 4, (2,4): 5
}

# nonRandomHeights = np.array([0.9, 0.8, 0.99, 1.2, 0.5, 0.1, 0.135])
# def randErr(numPts, low = -1, high = 1):
#     return np.cumsum(np.random.uniform(low, high, numPts))

# RANDS = 35
# ptsToFit0 = np.concatenate((randErr(RANDS), nonRandomHeights, randErr(RANDS)))
# ptsToFit1 = randErr(RANDS + len(nonRandomHeights) + RANDS)

objSeqDataGrid = []
for seqIndex in range(len(gtCommon.BCOT_SEQ_NAMES)):
    bodList = []
    for bodIndex in range(len(gtCommon.BCOT_BODY_NAMES)):
        bodList.append(ObjSeqData(bodIndex, seqIndex))
    objSeqDataGrid.append(bodList)



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




fig, ax = plt.subplots(figsize=(8, 6))
ax.axis([-1, 1, -1, 1])

def on_press(event):
    shiftIsDown = False
    x_shift = 0
    y_shift = 0
    bInd = int(slider1.val + 0.001)
    sInd = int(slider2.val + 0.001)
    selectedObjInfo = objSeqDataGrid[sInd][bInd]
    # Pressing "z" will fit the view height to the data visible on screen.
    if event.key == 'z' and selectedObjInfo.hasData:
        curr_window = selectedObjInfo.getWindowVals()
        rowsForStretch = [buttonIndexToDataRow[k] for k in axLinesDict.keys()]
        # Make sure the current window has data on-screen before attempting to
        # fit the view to the visible data.
        if curr_window[0] >= len(selectedObjInfo.plot_data.y_vals):
            xcess = len(selectedObjInfo.plot_data.y_vals) - 1 - curr_window[0]
            selectedObjInfo.x_window = (
                curr_window[0] + xcess, curr_window[1] + xcess
            )
            curr_window = selectedObjInfo.getWindowVals()

        dataSlice = selectedObjInfo.plot_data.y_vals[
            curr_window[0]:(curr_window[1] + 1), rowsForStretch
        ]
        minFromSlice = dataSlice.min()
        maxFromSlice = dataSlice.max()
        sliceHalfHeight = 0.55 *  (maxFromSlice - minFromSlice)
        midFromSlice = (maxFromSlice + minFromSlice)/2.0
        selectedObjInfo.y_window = (
            midFromSlice - sliceHalfHeight, midFromSlice + sliceHalfHeight)
        ax.axis(selectedObjInfo.getWindowVals())
        fig.canvas.draw_idle()
        return
    elif event.key == '1':
        slider1.set_val(max(slider1.val - 1, 0))
    elif event.key == '2':
        slider1.set_val(min(slider1.val + 1, len(gtCommon.BCOT_BODY_NAMES) - 1))
    elif event.key == '4':
        slider2.set_val(max(slider2.val - 1, 0))
    elif event.key == '5':
        slider2.set_val(min(slider2.val + 1, len(gtCommon.BCOT_SEQ_NAMES) - 1))
    elif event.key == 'a':
        x_shift = -1
    elif event.key == 'A':
        x_shift = -1
        shiftIsDown = True
    elif event.key == 'd':
        x_shift = 1
    elif event.key == 'D':
        x_shift = 1
        shiftIsDown = True
    elif event.key == 'w':
        y_shift = 1
    elif event.key == 'x' and ax.get_navigate_mode() is None:
        y_shift = -1
    if x_shift != 0 or y_shift != 0 and selectedObjInfo.hasData:
        prevYRange = selectedObjInfo.y_window
        newYRange = (prevYRange[0] + y_shift, prevYRange[1] + y_shift)
        selectedObjInfo.y_window = newYRange
        if x_shift != 0 and shiftIsDown:
            selectedObjInfo.lastKnownPtIndex += x_shift
            update(None)
        else:
            prevXRange = selectedObjInfo.x_window
            newXRange = (prevXRange[0] + x_shift, prevXRange[1] + x_shift)
            selectedObjInfo.x_window = newXRange
            ax.axis(selectedObjInfo.getWindowVals())
            fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_press)


        

    
# We want options to show x, y, and z for each of the 3D points related to the
# model's pose. These include its origin (denoted as point "0"), the endpoints
# of the local axes (denoted as "1", "2", and "3"), and the axis-angle
# representation of the rotation (denoted "AA").
# These options will be available as a grid of buttons. 
row_labels = ['x', 'y', 'z']
column_labels = [str(i) for i in range(4)] + ["AA"]

# vectorRowIndex is either x, y, or z.
# axisIndex refers to the origin, or the angle-axis vector, etc.
def componentLabel(vectorRowIndex, axisIndex):
    return column_labels[axisIndex] + "," + row_labels[vectorRowIndex]

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



def updatePredictionData(objSeqDataInfo):

    if (not objSeqDataInfo.hasData) or (not np.any(button_states)):
        return
    

    selectedCalc = objSeqDataInfo.calculator

    # Flatten needed here or else array looks like [[0], [2], ...]
    originData = selectedCalc.getTranslationsGTNP(True)

    # IGNORE MIDDLE BUTTON COLUMNS FOR NOW!

    # Flatten needed here for same reason as before.
    aaData = selectedCalc.getRotationsGTNP(True)

    # Points stored such that each row is a timestamp and each column is a
    # component of either the origin, the axis-angle rotation, or another pt
    # on the object.
    ptsData = np.hstack((originData, aaData))


    objSeqDataInfo.plot_data = PlotData(np.arange(len(ptsData)), ptsData)

    lkpi_ceiling = len(ptsData) - NUM_OUTPUT_PREDICTIONS - 1
    lastKnownSplineIndex = min(objSeqDataInfo.lastKnownPtIndex, lkpi_ceiling)
    lastKnownSplineIndex = max(lastKnownSplineIndex, SPLINE_DEGREE)

    lastKnownVelIndex = min(objSeqDataInfo.lastKnownPtIndex, len(ptsData) - 2)
    lastKnownVelIndex = max(lastKnownVelIndex, 2) # TODO: Make diff for accel.
    # TODO: Might need to change this if using different limits for accel index:
    objSeqDataInfo.lastKnownPtIndex = lastKnownVelIndex 

    startInd = max(0, lastKnownSplineIndex - PTS_USED_TO_CALC_LAST + 1)
    uInterval = (SPLINE_DEGREE, lastKnownSplineIndex + 1 - startInd)
    ctrlPtCount = min(lastKnownSplineIndex + 1, DESIRED_CTRL_PTS)


    # Interval over which basis functions sum to 1, assuming uniform knot seq.
    knotList = np.arange(ctrlPtCount + SPLINE_DEGREE + 1)
    uInterval = (knotList[SPLINE_DEGREE], knotList[-SPLINE_DEGREE - 1])
    numTotal = lastKnownSplineIndex + 1 - startInd + NUM_OUTPUT_PREDICTIONS
    uVals = np.linspace(uInterval[0], uInterval[1], numTotal)
    # uVals = uInterval[1] - gap*np.arange(numTotal - 1, -1, -1)

    # Need an "empty" array with the right number of rows to hstack with.
    ctrlPts = np.empty((ctrlPtCount, 0)) 

    # We'll keep track of velocities at two timesteps; the velocity at index 1
    # will be the "current" velocity, and the velocity at index 0 will be the
    # previous velocity (needed to approximate acceleration).
    # TODO: Replace hardcoded "6" with something automatic.
    velPts = np.empty((2, 6)) 
    accelPt = np.empty((1, 6)) 
    
    for i in range(ptsData.shape[1]):
        ptsToFit_i = ptsData[:, i]
        ptSubsetToFit = ptsToFit_i[startInd:lastKnownSplineIndex + 1]

        ctrlPts1D = fitBSpline(
            ctrlPtCount, SPLINE_DEGREE + 1, ptSubsetToFit, uVals, knotList,
            NUM_OUTPUT_PREDICTIONS
        )
        # print(ctrl_pts)
        ctrlPts = np.hstack((ctrlPts, ctrlPts1D.reshape((ctrlPtCount, 1))))

        velPts[1, i] = ptsToFit_i[lastKnownVelIndex] - ptsToFit_i[lastKnownVelIndex - 1]
        velPts[0, i] = ptsToFit_i[lastKnownVelIndex - 1] - ptsToFit_i[lastKnownVelIndex - 2]
    accelPt[0] = velPts[1] - velPts[0]
    constAccelPt = ptsData[lastKnownVelIndex] + velPts[1] + 0.5 * accelPt[0]

    ctrlPts = np.hstack((ctrlPts, np.ones((len(ctrlPts), 1)))) # Weights

    ctrlPtDumbWrap = [bspline.MotionPoint(c) for c in ctrlPts]
    spline = bspline.MotionBSpline(ctrlPtDumbWrap, SPLINE_DEGREE + 1, False)
    spline_result = bspline.ptsFromNURBS(spline, 100, False)

    splineInputLen = lastKnownSplineIndex + NUM_OUTPUT_PREDICTIONS - startInd
    spline_result.params -= uInterval[0]
    spline_result.params *= (splineInputLen/(uInterval[1] - uInterval[0]))
    spline_result.params += startInd


    pred_data = PredictionData()
    pred_data.x_spline = spline_result.params
    pred_data.y_spline = spline_result.points

    velXRange = (lastKnownVelIndex - 1, lastKnownVelIndex + 1)
    velInputYPrev = ptsData[velXRange[0]]
    newPointsYFromVel = velInputYPrev + 2 * velPts[1]

    pred_data.x_vel = np.array(velXRange)
    pred_data.y_vel = np.stack((velInputYPrev, newPointsYFromVel))
    pred_data.x_accel = pred_data.x_vel[1:]
    pred_data.y_accel = np.array([constAccelPt])

    objSeqDataInfo.pred_data = pred_data
    return

    #         ptsToFitChoice = ptsToFit0 if dropdown.value == 0 else ptsToFit1
    #         line1.set_ydata(ptsToFitChoice)

def clearAndRedraw(objSeqDataInfo):
    ax.clear()
    axLinesDict.clear()
    onButtonIndices = [tuple(btn_ind) for btn_ind in np.argwhere(button_states)]
    plot_data = objSeqDataInfo.plot_data
    pred_data = objSeqDataInfo.pred_data
    for k in onButtonIndices:
        i = buttonIndexToDataRow[k]
        line_true, = ax.plot(
            plot_data.x_vals, plot_data.y_vals[:, i], 'o',
            label=componentLabel(k[0], k[1])
        )
        line_spline, = ax.plot(pred_data.x_spline, pred_data.y_spline[:, i])
        line_vel = ax.plot(pred_data.x_vel, pred_data.y_vel[:, i], "-.")[0]
        line_accel = ax.plot(pred_data.x_accel, pred_data.y_accel[:, i], "s")[0]
        axLinesDict[k] = AxLines(line_true, line_spline, line_vel, line_accel)
    ax.axis(objSeqDataInfo.getWindowVals())
    ax.legend()
    fig.canvas.draw_idle()

def update(val):
    bInd = int(slider1.val + 0.001)
    sInd = int(slider2.val + 0.001)
    slider1.label.set_text(gtCommon.shortBodyNameBCOT(gtCommon.BCOT_BODY_NAMES[bInd]))
    slider2.label.set_text(gtCommon.shortSeqNameBCOT(gtCommon.BCOT_SEQ_NAMES[sInd]))

    objSeqDataInfo = objSeqDataGrid[sInd][bInd]
    updatePredictionData(objSeqDataInfo)

    if len(axLinesDict.keys()) > 0:
        objPts = objSeqDataInfo.plot_data
        objPreds = objSeqDataInfo.pred_data
        for k in axLinesDict.keys():
            i = buttonIndexToDataRow[k]
            axLineK = axLinesDict[k]
            true_diff = len(objPts.x_vals) != len(axLineK.true_line.get_xdata())
            spline_diff = len(objPreds.x_spline) != len(axLineK.spline_line.get_xdata())

            # I haven't profiled how fast set_data is compared to set_y_data,
            # but there's no real point in doing unnecessary setting.
            if true_diff:
                axLineK.true_line.set_data([objPts.x_vals, objPts.y_vals[:, i]])
            else:
                axLineK.true_line.set_ydata(objPts.y_vals[:, i])

            # TODO: Is it worth checking if an x_shift occurred, and if not,
            # then only shifting the y_data?
            axLineK.spline_line.set_data([
                objPreds.x_spline, objPreds.y_spline[:, i]
            ])
            

            axLineK.vel_line.set_data([objPreds.x_vel, objPreds.y_vel[:, i]])
            axLineK.accel_line.set_data([
                objPreds.x_accel, objPreds.y_accel[:, i]
            ])
        ax.axis(objSeqDataInfo.getWindowVals())

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

    bInd = int(slider1.val + 0.001)
    sInd = int(slider2.val + 0.001)
    bcotDataObj = objSeqDataGrid[sInd][bInd]
    updatePredictionData(bcotDataObj)
    
    clearAndRedraw(bcotDataObj)



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
