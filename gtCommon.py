import numpy as np

import pathlib
import os # TODO: replace fully with pathlib.


BCOT_DIR = pathlib.Path("C:\\Users\\U01\\Documents\\Datasets\\BCOT")
BCOT_DATASET_DIR = BCOT_DIR / "BCOT_Dataset"
BCOT_CV_POSE_EXPORT_DIR = BCOT_DIR / "srt3d_results_bcot"


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

# Input is assumed to be a numpy array with shape (n,3,3) for some n > 0.
# Return value thus has shape (n,3).
def axisAngleFromMatArray(matrix):
    # Vec3 representing the direction of our rotation axis. Not yet unit length.
    nonUnitAxisDirs = np.stack([
        matrix[...,2,1] - matrix[...,1,2],
        matrix[...,0,2] - matrix[...,2,0],
        matrix[...,1,0] - matrix[...,0,1]
    ], axis=-1) # Axis specification needed for input being a list of matrices.
    
    # The angle of rotation about the above axis direction.
    # Outputs of acos are constrained to [0, pi], which will impact later code.
    # Input needs to be clamped to [-1, 1] in case fp precision causes it to
    # exit that interval and, thus, the domain for acos.
    # We'll specify the trace axes so that we can get the trace of each rotation
    # matrix from an array of them and still get correct output.
    matrixTraces = matrix.trace(axis1 = -2, axis2 = -1)
    acosInput = np.clip((matrixTraces - 1.0)/2.0, -1.0, 1.0)
    angles = np.arccos(acosInput)

    

    # TL;DR: Angles, as output of acos, start out in interval [0, pi]. Thus,
    # similar rotations [pi - epsilon, 0, 0] and [pi + epsilon, 0, 0] will be
    # represented by distant vectors (pi - epsilon)[+1, 0, 0] and 
    # (pi - epsilon)[-1, 0 0]. We detect when this happens by observing the axes
    # and then we correct angles (e.g., adding 2pi multiples, i.e. taus) to fix.
    # --------------------------------------------------------------------------
    # If we now just return `angle/(2*sin(angle)) * unnormalizedAxis`, we would
    # have an *accurate* angle-axis result stored as a vec3. However, because
    # acos only returns values in [0, pi], you can get axis-angle vec3 "jumps"
    # like above over small rotation changes. So, we look at the unnormalized
    # axes for flips like that, and we start our corrections by negating
    # corresponding angles. Now, you might be thinking this either (a) generates
    # an incorrect result, or (b) does nothing. You might think (a) because
    # obviously rotations by +alpha and -alpha about the same axis differ. But
    # in a later step, we get our final axis-angle vec3 by multiplying
    # `(angle/(2*sin(angle))) * nonUnitAxisDirs`. Because
    # `angle/sin(angle) == (-angle)/sin(-angle)`, there would be zero effect on
    # our output if we took no further steps. Which may lead to one thinking
    # that (b) applies. HOWEVER, now angles are set up for corrections by adding
    # taus. E.g., in our "pi + epsilon" example, after negation, our consecutive
    # angles will become "pi - epsilon" and "-(pi - epsilon)"; we can correct
    # the latter by adding 2pi to get "pi + epsilon", which is the "best" way to
    # represent those consecutive rotations!
    # We'll only add taus s.t. angles become within pi distance of each other.
    # E.g., consecutive angles +epsilon and -epsilon would not be affected.
    # --------------------------------------------------------------------------
    # If you still doubt any of the above, please at least be very careful in
    # making any "corrections". I've thought about this quite thoroughly, but I 
    # don't want to take up too much space justifying it further.
    # --------------------------------------------------------------------------
    # In summary, we perform the following steps:
    #  1. Detect if axis flipped, via dot product.
    #  2. If axis was flipped, negate angle. 
    #  3. Add necessary multiples of 2pi to angles to prevent large angle jump.
    #     (May be necessary even if axis did not flip on this frame!)
    #     (E.g., corrections to previous frames could lead to lastAngle > 2pi)
    # --------------------------------------------------------------------------

    # Numpy-styled axis-flip-detection:
    # Einsum is used to perform a dot product between consecutive axes. For
    # understanding einsum, this ref might be handy:
    # https://ajcr.net/Basic-guide-to-einsum/
    # Here's a StackOverflow post suggesting that einsum might be faster than
    # doing `(arr[1:] * arr[:-1]).sum(axis=1)`:
    # https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
    # (see answer with plots further down page)
    # For the first negative dot prod, we will need to flip the corresponding
    # axis. If the next index's *original* dot product was positive, we'll need
    # to flip the next axis also, to keep it aligned with the "new" previous, 
    # and so on until we reach another *original* dot product that was negative.
    # I.e., we "accumulate" the number of flips needed, and even numbers cancel
    # out. This is why we use an accumulation of xors; it sort of functions like
    # mod 2 addition, but I'm hoping it's cheaper.
    angle_dots = np.einsum(
        'ij,ij->i', nonUnitAxisDirs[1:], nonUnitAxisDirs[:-1]
    ) # Dot products along 'j' axis; preserve existence of the 'i' axis.
    needs_flip = np.logical_xor.accumulate(angle_dots < 0)
    angles[1:][needs_flip] = -angles[1:][needs_flip]

    # Now we add 2pi multiples to make angles within pi of each other.
    # We want the following difference-to-correction mapping:
    # ..., (-3pi, -pi) -> tau, (-pi, pi) -> 0, (pi, 3pi) -> -tau,
    # (3pi, 5pi) -> -2tau, ... (note: interval endpoints don't matter)
    # The following code achieves this:
    np_tau = 2.0 * np.pi
    tau_facs = np.round(np.diff(angles) / np_tau)
    # We need to accumulate the correction sum because, if two angles are within
    # pi of each other, and the former gets incremented by n*tau, the latter 
    # must also be incremented by n*tau so that they stay within pi distance.
    angle_corrections = np_tau * np.cumsum(tau_facs)
    angles[1:] -= angle_corrections

    # TODO: Remove this after sufficient testing!
    if np.any(np.greater(np.abs(np.diff(angles)), np.pi + 0.00001)):
        raise Exception("Numpification of AA code resulted in angle diff > pi!")
        
    
    # (!!!) PLEASE READ THIS COMMENT BEFORE MOVING/EDITING THIS LINE!
    # Our final vec3 output will be `angle * unitAxis`,
    # where `unitAxis = nonUnitAxis/2sin(angle)`.
    # In above comments, we noted that we're "allowed" to replace `angle` with
    # `-1*angle` in the first step of our angle corrections, because
    # `angle/sin(angle) == -angle/sin(-angle)` (and then of course adding taus
    # also preserves accuracy). IF WE MOVE THE DIVISION BY 2SIN EARLIER, WE LOSE
    # THIS ABILITY that comes  from the negation of the angle and the sin(angle)
    # cancelling out; we'd have to take a separate step to flip required axes!
    nonUnitAxisLen = 2.0 * np.sin(angles)

    # zeroDivIndices = (nonUnitAxisLen != 0.0)

    angle_scale_amt = angles / nonUnitAxisLen
    return np.einsum('i,ij->ij', angle_scale_amt, nonUnitAxisDirs)



# def axisAngleListFromMats(matList):
#     # differences = []
#     lastAngle = 0.0
#     lastDir = np.array([0.0, 0.0, 0.0])
#     retList = []
#     for m in matList:
#         lastAngle, val = axisAngleFromMat(m, lastAngle, lastDir)
#         lastDir = val
#         retList.append(val)
#         # recovered = matFromAxisAngle(val)
#         # differences.append(np.linalg.norm(recovered - m))
#     # print("Max difference:", np.max(np.array(differences)))
    
#     return np.array(retList)

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
        posePathCalc = BCOT_CV_POSE_EXPORT_DIR / ("cvOnly_skip" + str(self.skipAmt) + "_poses_" + self._seq + "_" + self._bod +".txt")

        #patternNum = r"(-?\d+\.?\d*e?-?\d*)" # E.g., should match "-0.11e-07"
        #patternTrans = re.compile((r"\s+" + patternNum) * 3 + r"\s*$")
        #patternRot = re.compile(r"^\s*" + (patternNum + r"\s+") * 9)

        gtMatData = BCOT_Data_Calculator.matrixDataFromFile(self.posePathGT)
        self._translationsGTNP = gtMatData[1]
        self._rotationMatsGTNP = gtMatData[0]

        self._rotationsGTNP = axisAngleFromMatArray(gtMatData[0])

        # Check if file for CV-calculated pose data exists; if so, load it too.
        if os.path.exists(posePathCalc):
            calcMatData = BCOT_Data_Calculator.matrixDataFromFile(posePathCalc)
            self._translationsCalcNP = calcMatData[1]
            self._rotationsCalcNP = axisAngleFromMatArray(calcMatData[0])

        # TODO: In the past, I had a test to make sure the axis-angle rotations
        # did not have sharp flips between consecutive axes. This test was
        # a bit misguided, because consecutive axes [+epsilon, 0, 0] and 
        # [-epsilon, 0, 0] would be totally fine. But I'm keeping it this way
        # for consistency's sake for now to make sure other code areas didn't
        # break. But SOON, I should either remove this or look at euclidean
        # distance between vectors instead!
        for rotArr in [self._rotationsGTNP, self._rotationsCalcNP]:
            flipPlaces = np.einsum('ij,ij->i', rotArr[1:], rotArr[:-1]) <= 0
            if np.any(flipPlaces):
                flipInds = 1 + np.argwhere(flipPlaces)
                self.issueFrames = np.hstack((
                    flipInds, rotArr[flipInds], rotArr[flipInds - 1]
                ))
                print("Issue frames:", self.issueFrames)
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