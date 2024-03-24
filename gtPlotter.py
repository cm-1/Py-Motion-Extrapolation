import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import math
import pathlib
import re

import cinpact

def axisAngleFromMat(matrix):
    angle = math.acos((matrix.trace() - 1)/2.0)
    axisDir = np.array([
        matrix[2,1] - matrix[1,2],
        matrix[0,2] - matrix[2,0],
        matrix[1,0] - matrix[0,1]
    ])
    return (angle / (2.0 * math.sin(angle))) * axisDir

bcotDir = pathlib.Path("D:\\DatasetsAndResults\\BCOT")
datasetDir = bcotDir / "BCOT_dataset"
poseExportDir = bcotDir / "srt3d_results_bcot"

body_names = [
    "3D Touch",  "Ape", "Auto GPS", "Bracket", "Cat", "Deadpool", "Driller",
    "FlashLight",  "Jack",        "Lamp Clamp",                 "RJ45 Clip",
    "Squirrel",     "Standtube",          "Stitch",         "Teapot",
    "Vampire Queen" , "RTI Arm",      "Wall Shelf" , "Lego", "Tube", 
]

sequence_names = [
    "complex_movable_handheld",
    "complex_movable_suspension",
    "complex_static_handheld",
    "complex_static_suspension",
    "complex_static_trans",
    "easy_static_handheld",
    "easy_static_suspension",
    "easy_static_trans",
    "light_movable_handheld",
    "light_movable_suspension",
    "light_static_handheld",
    "light_static_suspension",
    "light_static_trans",
    "occlusion_movable_suspension",
    "outdoor_scene1_movable_handheld_cam1",
    "outdoor_scene1_movable_handheld_cam2",
    "outdoor_scene1_movable_suspension_cam1",
    "outdoor_scene1_movable_suspension_cam2",
    "outdoor_scene2_movable_handheld_cam1",
    "outdoor_scene2_movable_handheld_cam2",
    "outdoor_scene2_movable_suspension_cam1",
    "outdoor_scene2_movable_suspension_cam2",
]


posePathGT = datasetDir / sequence_names[11] / body_names[1] / "pose.txt"
print("Pose path:", posePathGT)

posePathCalc = poseExportDir / ("cvOnlySkip0_poses_" + sequence_names[11] + "_" + body_names[1] +".txt")

patternNum = r"(-?\d+\.?\d*)"
patternTrans = re.compile((r"\s+" + patternNum) * 3 + r"\s*$")
patternRot = re.compile(r"^\s*" + (patternNum + r"\s+") * 9)

translationsGT = []
translationsCalc = []
rotationsGT = []
rotationsCalc = []
with open(posePathGT, "r") as file:
    for line in file.readlines():
        transMatch = patternTrans.search(line)
        translationsGT.append(np.array([float(g) for g in transMatch.groups()]))
        rotMatch = patternRot.search(line)
        rotRead = np.array([float(g) for g in rotMatch.groups()])
        rotationsGT.append(np.array(rotRead).reshape((3,3)))
        
with open(posePathCalc, "r") as file:
    for line in file.readlines():
        transMatch = patternTrans.search(line)
        translationsCalc.append(np.array([float(g) for g in transMatch.groups()]))
        rotMatch = patternRot.search(line)
        rotRead = np.array([float(g) for g in rotMatch.groups()])
        rotationsCalc.append(np.array(rotRead).reshape((3,3)))

translationsGTNP = np.array(translationsGT)
translationsCalcNP = np.array(translationsCalc)

rotationsGTNP = np.array([axisAngleFromMat(m) for m in rotationsGT])
rotationsCalcNP = np.array([axisAngleFromMat(m) for m in rotationsCalc])
for rotArr in [rotationsGTNP, rotationsCalcNP]:
    for i in range(1, rotArr.shape[0]):
        if np.dot(rotArr[i-1], rotArr[i]) < 0:
            rotArr[i] *= -1

def plotGTvsCalc(gt, calc, showOnlyGT = False):
    x = gt[:, 0]
    y = gt[:, 1]
    z = gt[:, 2]
    w = np.arange(len(x))#translationsNP.shape[0])

    init_k = 10
    init_c = 10
    init_t = len(calc)
    init_endpt = -1

    curvePts = cinpact.getSplinePts(calc, init_c, init_k, 1000)

    scaled_w = (w - w.min()) / w.ptp()
    colors = plt.cm.coolwarm(scaled_w)

    fig = plt.figure()
    plt.subplots_adjust(bottom = 0.25)
    ax = fig.add_subplot(projection='3d')
    scatterGT = ax.scatter(x, y, z, marker='.', edgecolors=colors)
    if not showOnlyGT:
        scatterCalc = ax.scatter(calc[:, 0], calc[:, 1], calc[:, 2], marker='+', edgecolors=colors)
        scatterCalc.set_edgecolor(colors[:len(calc)]) # Need to do this again to make it look right

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
            valmax = len(calc),
            valstep = 1,
            valinit=init_t
        )

        def updateDisplay(_):
            end = int(tSlider.val)
            endGT = end + 1
            scatterGT.set_offsets(gt[:endGT, :2])
            scatterGT.set_3d_properties(gt[:endGT, 2], 'z')
            scatterGT.set_edgecolor(colors[:endGT])
            scatterCalc.set_offsets(calc[:end, :2])
            scatterCalc.set_3d_properties(calc[:end, 2], 'z')
            scatterCalc.set_edgecolor(colors[:end])
            fig.canvas.draw_idle()

        def updatePts(_):
            global curvePts
            curvePts = cinpact.getSplinePts(calc[:init_endpt:3], cSlider.val, kSlider.val, 1000)
            end = int(tSlider.val)
            line[0].set_data_3d(
                curvePts[:end, 0], curvePts[:end, 1], curvePts[:end, 2]
            )
            fig.canvas.draw_idle()



        kSlider.on_changed(updatePts)
        cSlider.on_changed(updatePts)
        tSlider.on_changed(updateDisplay)

    plt.show()

plotGTvsCalc(rotationsGTNP, rotationsCalcNP, False)

