# TODO: Look at https://matplotlib.org/stable/gallery/widgets/menu.html#sphx-glr-gallery-widgets-menu-py

from dataclasses import dataclass, field
import typing

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
# from matplotlib.widgets import Slider

import gtCommon as gtc
import posemath as pm

FRAME_SKIP_AMT = 0
MAX_BCOT_FRAMES = 359 # Frames in longest BCOT vid.
print("Hey, maybe it's worth comparing skip amt graphs.")

class ObjSeqData:
    def __init__(self, bodID: int, seqID: int):
        self._bodID = bodID
        self._seqID = seqID
        self._hasData = False
        self._calculator = None
        if gtc.PoseLoaderBCOT.isBodySeqPairValid(bodID, seqID):
            self._hasData = True
            self._calculator = gtc.PoseLoaderBCOT(bodID, seqID)
        self.numDataToShow = -0
        self._initialized = False
        self._accelData = None
        self._accelDataSegments = None
        self._originSegments = None
        self._dataBounds = None
        self._signedAccelNorms = None

    def initialize(self):
        if self._hasData and not self._initialized:
            self._initialized = True
            step = (FRAME_SKIP_AMT + 1)
            translations = self._calculator.getTranslationsGTNP()[::step]
            self._accelData = np.diff(translations, 2, axis = 0)
            reps = np.full(len(self._accelData), 2)
            reps[0] = 1
            reps[-1] = 1
            repeated = np.repeat(self._accelData, reps, axis = 0)
            self._accelDataSegments = repeated.reshape((-1, 2, 3))
            self.numDataToShow = len(self._accelData)
            
            self._dataBounds = np.stack(
                (self._accelData.min(axis=0), self._accelData.max(axis=0))
            )

            # Find acceleration norms, and negate norms to match whether 
            # consecutive acceleration *vectors* point in the same direction.
            self._signedAccelNorms = np.linalg.norm(self._accelData, axis = -1)
            accelDots = pm.einsumDot(self._accelData[1:], self._accelData[:-1]) 
            flip = np.logical_xor.accumulate(accelDots < 0, axis = -1)
            self._signedAccelNorms[1:][flip] = -self._signedAccelNorms[1:][flip]

            self._originSegments = np.zeros((3, 2, 3))
            for i in range(3):
                self._originSegments[i, :, i] = self._dataBounds[:, i]
        return
    
    def hasData(self):
        return self._hasData
    
    def getData(self):
        self.initialize()
        return self._accelData
    
    def getSegments(self):
        self.initialize()
        return self._accelDataSegments

    def getBounds(self):
        return self._dataBounds
    
    def getOriginSegments(self):
        return self._originSegments
    
    def getSignedNorms(self):
        return self._signedAccelNorms

objSeqDataGrid = []
combos = []
for seqIndex in range(len(gtc.BCOT_SEQ_NAMES)):
    bodList = []
    for bodIndex in range(len(gtc.BCOT_BODY_NAMES)):
        bodList.append(ObjSeqData(bodIndex, seqIndex))
        if bodList[-1].hasData():
            combos.append((seqIndex, bodIndex))
    objSeqDataGrid.append(bodList)

maxFramesWithSkip = 1 + (MAX_BCOT_FRAMES // (FRAME_SKIP_AMT + 1)) 
timestampsForLongest = np.arange(maxFramesWithSkip)





fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(1, 2, 1, projection="3d")
scatterPlot = ax.scatter([], [], [], marker='.', edgecolors=[])
linesPlot = Line3DCollection([], colors=[])
ax.add_collection(linesPlot)
originLines = Line3DCollection([], color="black")
ax.add_collection(originLines)

ax2D = fig.add_subplot(1, 2, 2)

colors = plt.cm.coolwarm(timestampsForLongest/maxFramesWithSkip)

# ------------------------------------------------------------------------------
# Now comes all of the axes layout code.

plt.subplots_adjust(left=0.2, bottom=0.3, right=0.95, top=0.9)

noDataTextAxis = plt.axes((0.2, 0.5, 0.1, 0.1), frameon=False)
noDataTextAxis.text(1.0, 1.0, "No data for obj/seq pair!")
noDataTextAxis.get_xaxis().set_visible(False)
noDataTextAxis.get_yaxis().set_visible(False)

vidNameTextAxis = plt.axes([0.6, 0.15, 0.35, 0.03], frameon=False)
vidNameTextAxis.text(0.0, 0.0, "no text yet!")
vidNameTextAxis.get_xaxis().set_visible(False)
vidNameTextAxis.get_yaxis().set_visible(False)


def switchObjects():
    sInd = combos[selectedComboInd][0]
    bInd = combos[selectedComboInd][1]
    objSeqDataInfo = objSeqDataGrid[sInd][bInd]

    bName = gtc.truncateName(gtc.BCOT_BODY_NAMES[bInd])
    sName = gtc.shortSeqNameBCOT(gtc.BCOT_SEQ_NAMES[sInd])

    noDataTextAxis.set_visible(not objSeqDataInfo.hasData)

    vidNameTextAxis.clear()
    vidNameTextAxis.text(0, 0, bName + ", " + sName)

    accData = objSeqDataInfo.getData()
    dataLen = len(accData)
    colorMapStep = maxFramesWithSkip // dataLen
    dataEndInd = objSeqDataInfo.numDataToShow
    colorMapEnd = colorMapStep * (dataEndInd)
    colorsSlice = colors[:colorMapEnd:colorMapStep]

    accDataSlice = accData[:dataEndInd]
    scatterPlot.set_offsets(accDataSlice[:, :2])
    scatterPlot.set_3d_properties(accDataSlice[:, 2], 'z')
    scatterPlot.set_edgecolor(colorsSlice)

    linesPlot.set(
        segments = objSeqDataInfo.getSegments()[:dataEndInd - 1],
        colors = colorsSlice[:-1]
    )
    originLines.set(segments = objSeqDataInfo.getOriginSegments())
    ax.draw_artist(linesPlot)
    ax.draw_artist(originLines)
    bounds = objSeqDataInfo.getBounds()

    ax.axes.set_xlim3d(left = bounds[0, 0], right = bounds[1, 0])
    ax.axes.set_ylim3d(bottom = bounds[0, 1], top = bounds[1, 1])
    ax.axes.set_zlim3d(bottom = bounds[0, 2], top = bounds[1, 2])

    ax2D.clear()

    ax2D.bar(
        timestampsForLongest[:dataEndInd],
        objSeqDataInfo.getSignedNorms()[:dataEndInd],
        align="center"
    )

    fig.canvas.draw_idle()


selectedComboInd = 0

def on_press(event):
    global selectedComboInd
    prevComboInd = selectedComboInd
    if event.key == '1':
        selectedComboInd = max(0, selectedComboInd - 1)
    elif event.key == '2':
        selectedComboInd = min(len(combos) - 1, selectedComboInd + 1)
    if selectedComboInd != prevComboInd:
        switchObjects()

fig.canvas.mpl_connect('key_press_event', on_press)


switchObjects()

# Show the plot
plt.show()
