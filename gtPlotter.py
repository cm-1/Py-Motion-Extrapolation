import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import math
import pathlib
import re

import cinpact

# Makes rotation matrix from an axis (with angle being encoded in axis length).
# Uses common formula that you can google if need-be.
def matFromAxisAngle(scaledAxis):
    angle = np.linalg.norm(scaledAxis)
    unitAxis = scaledAxis / angle
    x, y, z = unitAxis.flatten()
    skewed = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])
    return np.identity(3) + np.sin(angle) * skewed + (1.0 - np.cos(angle)) * (skewed @ skewed)

# 1st return value: rotation angle
# 2nd return value: axis pre-scaled by rotation angle (so NOT unit axis!)
def axisAngleFromMat(matrix, lastAngle, lastAxis):
    # Vec3 representing the direction of our rotation axis. Not yet unit length.
    nonUnitAxisDir = np.array([
        matrix[2,1] - matrix[1,2],
        matrix[0,2] - matrix[2,0],
        matrix[1,0] - matrix[0,1]
    ])
    
    # The angle of rotation about the above axis direction.
    # Outputs of acos are constrained to [0, pi], which will impact later code. 
    angle = math.acos((matrix.trace() - 1)/2.0)

    # TL;DR: Detect axis flips and increment/decrement angle to prevent jumps.
    # Can skip longer explanation below if you want or need to.
    # ------------------------------------------------------------------------
    # Technically, we can now just return `angle/(2*sin(angle)) * axis``, and
    # then we have an **accurate** angle-axis result stored as a vec3.
    # **However**, because acos only returns values in [0, pi], rotations above
    # pi about an axis will be represented by rotations under pi about the
    # flipped version of the axis. That means that, on frame "n", you have a
    # rotation of (pi - epsilon) about axis "v", and then on frame "n+1",
    # instead of a rotation of (pi + epsilon) about axis "v", you get back a
    # rotation of (pi - epsilon) about axis "-v".
    # Technically correct, but very jumpy....
    # So, we perform the following steps:
    #  1. Detect if axis flipped, via dot product.
    #  2. If axis was flipped, negate angle to maintain accuracy.
    #     We also need to flip the axis, but we'll do it later, at the same time
    #     that we scale it, to slightly save on operations.
    #  3. Add necessary multiples of 2pi to angle to prevent large angle jump.
    #     (May be necessary even if axis did not flip on this frame!)
    #     (E.g., corrections to previous frames could lead to lastAngle > 2pi)
    # ------------------------------------------------------------------------
    # TODO: Calculating (via floor/ceil/mult/div) number of 2pi incs needed
    # may be faster than while loop? A worst vs. avg case trade-off, I guess.
    # Is there a way where, e.g., an object on a carousel won't just have the
    # angle accumulate to be higher and higher? I guess you could keep the angle
    # uncorrected/unnegated and within [0,pi], just flip the axis if need-be,
    # and store flip sign in some 4th component scaled by sin(angle) so as to be
    # continuous... which may be just a worse version of quaternions.
    if np.dot(lastAxis, nonUnitAxisDir) < 0.0:
        angle = -angle
    while lastAngle - angle > math.pi:
        angle += 2.0 * math.pi
    while angle - lastAngle > math.pi:
        angle -= 2.0 * math.pi
        
    # (!!!) PLEASE READ THIS COMMENT BEFORE MOVING/EDITING THIS LINE!
    # Our final vec3 output will be `angle * unitAxis`,
    # where `unitAxis = nonUnitAxis/2sin(angle)`.
    # Without flip correction, angle is in [0, pi] and so sin(angle) >= 0; thus,
    # if we say:
    # `originalAxis = nonUnitAxis/2sin(originalAngle)`
    # and if we do a flip correction, our new axis will have to be:
    # `newAxis = -originalAxis`
    #        ` = nonUnitAxis/2sin(-originalAngle) = nonUnitAxis/2sin(newAngle)`
    # What this means is that, if we calculate the sin() scaling after modifying
    # the angle AND we use this scaling to flip the axis dir if need-be, then
    # we save on a few extra calculations.
    nonUnitAxisLen = 2.0 * math.sin(angle)
    
    return angle, (angle / nonUnitAxisLen) * nonUnitAxisDir



def axisAngleListFromMats(matList):
    # differences = []
    lastAngle = 0.0
    lastDir = np.array([0.0, 0.0, 0.0])
    retList = []
    for m in matList:
        lastAngle, val = axisAngleFromMat(m, lastAngle, lastDir)
        lastDir = val
        retList.append(val)
        # recovered = matFromAxisAngle(val)
        # differences.append(np.linalg.norm(recovered - m))
    # print("Max difference:", np.max(np.array(differences)))
    
    return np.array(retList)

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

skipAmt = 2
posePathCalc = poseExportDir / ("cvOnlySkip" + str(skipAmt) + "_poses_" + sequence_names[11] + "_" + body_names[1] +".txt")

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

rotationsGTNP = axisAngleListFromMats(rotationsGT)
rotationsCalcNP = axisAngleListFromMats(rotationsCalc)

issueFrames = []
for rotArr in [rotationsGTNP, rotationsCalcNP]:
    for i in range(1, rotArr.shape[0]):
        if np.dot(rotArr[i-1], rotArr[i]) < 0:
            issueFrames.append((i, rotArr[i-1], rotArr[i])) #rotArr[i] *= -1

def plotGTvsCalc(gt, calc, showOnlyGT = False):
    x = gt[:, 0]
    y = gt[:, 1]
    z = gt[:, 2]
    w = np.arange(len(x))#translationsNP.shape[0])

    lastFrameNum = len(calc) * (skipAmt + 1)

    init_k = 10
    init_c = 10
    init_t = lastFrameNum
    init_endpt = -1

    curvePts = cinpact.getSplinePts(calc, init_c, init_k, 1000)

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

plotGTvsCalc(rotationsGTNP, rotationsCalcNP, False)

print("Issue frames for pose path", posePathGT, "are:", issueFrames)
