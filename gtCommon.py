import numpy as np

import math
import pathlib
import os # TODO: replace fully with pathlib.


BCOT_DIR = pathlib.Path("C:\\Users\\U01\\Documents\\Datasets\\BCOT")
BCOT_DATASET_DIR = BCOT_DIR / "BCOT_Dataset"
BCOT_POSE_EXPORT_DIR = BCOT_DIR / "srt3d_results_bcot"


BCOT_BODY_NAMES = [
    "3D Touch",  "Ape", "Auto GPS", "Bracket", "Cat", "Deadpool", "Driller",
    "FlashLight",  "Jack",        "Lamp Clamp",                 "RJ45 Clip",
    "Squirrel",     "Standtube",          "Stitch",         "Teapot",
    "Vampire Queen" , "RTI Arm",      "Wall Shelf" , "Lego", "Tube", 
]

BCOT_SEQ_NAMES = [
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

def shortBodyNameBCOT(longName):
    maxLen = 11
    if len(longName) < maxLen:
        return longName
    return longName[:maxLen - 3] + "..."

def shortSeqNameBCOT(longName):
    words = longName.split("_")
    retVal = ""
    for w in words:
        retVal += w[0]
        if w[-1] in "12":
            retVal += w[-1]
        retVal += "_"
    return retVal[:-1]

# Makes rotation matrix from an axis (with angle being encoded in axis length).
# Uses common formula that you can google if need-be.
def matFromAxisAngle(scaledAxis):
    angle = np.linalg.norm(scaledAxis)
    if angle == 0.0:
        return np.identity(3)
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
    # Input needs to be clamped to [-1, 1] in case fp precision causes it to
    # exit that interval and, thus, the domain for acos. 
    acosInput = max(-1.0, min((matrix.trace() - 1)/2.0, 1.0))
    angle = math.acos(acosInput)

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
    if nonUnitAxisLen == 0.0: # Separate return statement here to avoid div-by-0
        return 0, np.zeros(3)

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

def isBodySeqPairValid(bodyIndex, seqIndex):
    seq = BCOT_SEQ_NAMES[seqIndex]
    bod = BCOT_BODY_NAMES[bodyIndex]
    
    posePathGT = BCOT_DATASET_DIR / seq / bod
    return os.path.isdir(posePathGT)



class BCOT_Data_Calculator:
    def __init__(self, bodyIndex, seqIndex, skipAmt):

        
        self.skipAmt = skipAmt
        self.issueFrames = []

        self._seq = BCOT_SEQ_NAMES[seqIndex]
        self._bod = BCOT_BODY_NAMES[bodyIndex]
        
        self.posePathGT = BCOT_DATASET_DIR / self._seq / self._bod / "pose.txt"

        self._translationsGTNP = np.zeros((0,3), dtype=np.float64)
        self._translationsCalcNP = np.zeros((0,3), dtype=np.float64)

        self._rotationsGTNP = np.zeros((0,3), dtype=np.float64)
        self._rotationsCalcNP = np.zeros((0,3), dtype=np.float64)
        self._dataLoaded = False
    
    def loadData(self):
        if self._dataLoaded:
            return

        print("Pose path:", self.posePathGT)
        posePathCalc = BCOT_POSE_EXPORT_DIR / ("cvOnly_skip" + str(self.skipAmt) + "_poses_" + self._seq + "_" + self._bod +".txt")

        #patternNum = r"(-?\d+\.?\d*e?-?\d*)" # E.g., should match "-0.11e-07"
        #patternTrans = re.compile((r"\s+" + patternNum) * 3 + r"\s*$")
        #patternRot = re.compile(r"^\s*" + (patternNum + r"\s+") * 9)

        gtMatData = BCOT_Data_Calculator.matrixDataFromFile(self.posePathGT)
        self._translationsGTNP = gtMatData[1]

        self._rotationsGTNP = axisAngleListFromMats(gtMatData[0])

        # Check if file for CV-calculated pose data exists; if so, load it too.
        if os.path.exists(posePathCalc):
            calcMatData = BCOT_Data_Calculator.matrixDataFromFile(posePathCalc)
            self._translationsCalcNP = calcMatData[1]
            self._rotationsCalcNP = axisAngleListFromMats(calcMatData[0])
        

        # Our earlier attempts to make sure the axis-angle calculations do not
        # result in an axis flip between consecutive frames should have worked.
        # But if not, we'll keep track of the frames that are an issue for the
        # sake of debugging.
        # TODO: Should probably remove this code at some point, because I don't
        # think any issues have been detected in any of my runs on the whole
        # dataset at this point.
        for rotArr in [self._rotationsGTNP, self._rotationsCalcNP]:
            for i in range(1, rotArr.shape[0]):
                if np.dot(rotArr[i-1], rotArr[i]) < 0:
                    self.issueFrames.append((i, rotArr[i-1], rotArr[i])) #rotArr[i] *= -1

        self._dataLoaded = True

    # Returns (rotation mat data, translation data) tuple, where each element is
    # a numpy array, the former with shape (n,3,3), the latter with shape (n,3).
    def matrixDataFromFile(filepath):
        data = np.loadtxt(filepath)
    
        # Assumes that each line has 12 floats, where the first 9 are the 3x3
        # rotation matrix entries and the last 3 are the translation.
        rotations = data[:, :9].reshape((-1, 3, 3)) 
        translations = data[:, 9:12]
        return (rotations, translations)

    def getTranslationsGTNP(self, useSkipAmt):
        self.loadData()
        step = (1 + self.skipAmt) if useSkipAmt else 1
        return self._translationsGTNP[::step]
    def getTranslationsCalcNP(self):
        self.loadData()
        return self._translationsCalcNP
    def getRotationsGTNP(self, useSkipAmt):
        self.loadData()
        step = (1 + self.skipAmt) if useSkipAmt else 1
        return self._rotationsGTNP[::step]
    def getRotationsCalcNP(self):
        self.loadData()
        return self._rotationsCalcNP