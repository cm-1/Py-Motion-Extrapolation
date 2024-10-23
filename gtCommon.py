import numpy as np

import pathlib
import json



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

def quatsFromAxisAngles(axisAngleVals):
    angles = np.linalg.norm(axisAngleVals, axis=1, keepdims=True)

    normed = axisAngleVals / angles

    quaternions = np.hstack((np.cos(angles / 2), np.sin(angles / 2) * normed))
    return quaternions


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
def axisAngleFromMatArray(matrixArray, zeroAngleThresh = 0.0001):
    # TODO: I think the only parts of the code below that do not yet support
    # more than 3 dimensions are the handling of axes for angles of zero.
    # There may not yet be a *benefit* to full support, but noting just in case.
    if matrixArray.ndim > 3:
        raise Exception("Input array of matrices must have (n,3,3) shape!")

    # TODO: Replace instances of np.stack(...), np.concat(...), and similar with
    # np.empty(...) followed by assigning to slices. I'm guessing it'd be more
    # efficient? Less allocating/freeing of memory, right?
    # TODO: Last I checked (2024-10-19), it's fine, but if changed since, see if
    # supporting a arrays with more dims than shape (n,3,3) leads to any 
    # inefficiency; if so, remove, or use an "if" to switch to better "flat"
    # version, because I don't know of a practical purpose off-hand for
    # supporting more dims than that. In fact, it'd possibly hinder multicore 
    # processing, which may be the better and/or faster approach for more dims.

    # --------------------------------------------------------------------------
    # We'll start by using Shepperd's algorithm to obtain initial results:
    #   Shepperd, Stanley W. "Quaternion from rotation matrix."
    #   Journal of guidance and control 1.3 (1978): 223-224.
    #   doi:10.2514/3.55767b
    # Then we'll modify the initial results to remove discontinuous jumps
    # between consecutive axis-angle vectors, as described later.
    # --------------------------------------------------------------------------
    # In Shepperd's algorithm, we take different steps for each rotation matrix
    # depending on whether the trace or a diagonal entry is larger. We'll 
    # use numpy slices to accomplish this.

    # --------------------------------------------------------------------------
    # Shepperd's Algorithm: Initial Setup
    # --------------------------------------------------------------------------


    # We'll specify the trace axes so that we can get the trace of each rotation
    # matrix from an array of them and still get correct output.
    matrixTraceVals = matrixArray.trace(axis1 = -2, axis2 = -1)
    # np.diagonal(...) makes no copy, so this should be reasonably efficient.
    matrixDiags = np.diagonal(matrixArray, axis1 = -2, axis2 = -1)
    
    angles = np.empty(matrixArray.shape[:-2]) # Storage for resulting angles.
    nonUnitAxes = np.empty(matrixDiags.shape) # Storage for resulting axes.

    # Get largest diagonal entries' locations. We'll reuse this later on too.
    whereMaxDiag = np.argmax(matrixDiags, axis = -1, keepdims=True)
    # Extract the value of the largest diagonal entry.
    # Earlier numpy versions don't have `take_along_axis`; in that case, you'd
    # use something like arr[np.arange(len(...)), colIndices].
    diagMaxes = np.take_along_axis(matrixDiags, whereMaxDiag, axis = -1)[..., 0]

    # Slice indices for applying different steps to different rotation matrices.
    useTraceBool = np.greater(matrixTraceVals, diagMaxes)
    useDiagBool = np.invert(useTraceBool)
    # I think indices are oft faster than bool indexing. But needs confirming.
    useTrace = np.nonzero(useTraceBool)
    useDiag = np.nonzero(useDiagBool)
    

    # --------------------------------------------------------------------------
    # Shepperd's Algorithm: Case Where Trace Was Greater.
    # --------------------------------------------------------------------------

    # The angle of rotation about the above axis direction.
    # Outputs of acos are constrained to [0, pi], which will impact later code.
    # Input needs to be clamped to [-1, 1] in case fp precision causes it to
    # exit that interval and, thus, the domain for acos.
    acosInput = np.clip((matrixTraceVals[useTrace] - 1.0)/2.0, -1.0, 1.0)
    angles[useTrace] = np.arccos(acosInput)

    matrixOffDiags = matrixArray[useTrace]


    # Vec3 representing the direction of our rotation axis. Not yet unit length.
    nonUnitAxes[useTrace] = np.stack([
        matrixOffDiags[...,2,1] - matrixOffDiags[...,1,2],
        matrixOffDiags[...,0,2] - matrixOffDiags[...,2,0],
        matrixOffDiags[...,1,0] - matrixOffDiags[...,0,1]
    ], axis=-1) # Axis specification needed for input being a list of matrices.


    # --------------------------------------------------------------------------
    # Shepperd's Algorithm: Case Where a Diagonal Entry Was Greater.
    # --------------------------------------------------------------------------

    i_s = whereMaxDiag[useDiag][:, 0]
    j_s = (i_s + 1) % 3
    k_s = (j_s + 1) % 3

    matsWhereDiagUsed = matrixArray[useDiag]
    # The only way I know of to slice with variable last-axis-indices is to
    # pass in an arange for the first axis; simply using `[:, i_s, j_s]` FAILS!
    # Maybe there's a better way I'm unaware of, though.
    arangeUseDiag = np.arange(len(i_s))
    Aij = matsWhereDiagUsed[arangeUseDiag, i_s, j_s]
    Aji = matsWhereDiagUsed[arangeUseDiag, j_s, i_s]
    Aik = matsWhereDiagUsed[arangeUseDiag, i_s, k_s]
    Aki = matsWhereDiagUsed[arangeUseDiag, k_s, i_s]
    Ajk = matsWhereDiagUsed[arangeUseDiag, j_s, k_s]
    Akj = matsWhereDiagUsed[arangeUseDiag, k_s, j_s]

    diagMaxSubset = diagMaxes[useDiag]
    
    # The below is `2sin(angle/2) * axis`
    sqrtInput = 1 + diagMaxSubset + diagMaxSubset - matrixTraceVals[useDiag]
    ax_i = np.sqrt(np.fmax(0.0, sqrtInput)) # fmax to protect from fp issues.
    nonUnitAxes[useDiag, i_s] = ax_i
    nonUnitAxes[useDiag, j_s] = (Aij + Aji)/ax_i
    nonUnitAxes[useDiag, k_s] = (Aik + Aki)/ax_i

    # Again, we need to clamp/clip in case of fp precision causing problems.
    acosInput = np.clip((Akj - Ajk)/(ax_i + ax_i), -1.0, 1.0)
    halfAngles = np.arccos(acosInput)
    angles[useDiag] = halfAngles + halfAngles # Will be between 0 and 2pi.

    # TODO: Remove this test:
    if np.any(angles < 0):
        raise Exception("I was wrong about all pos angles at this step!")

    # --------------------------------------------------------------------------
    # "Corrections" Proceeding Shepperd's Algorithm
    # --------------------------------------------------------------------------

    # For reasons described shortly, we may want to use an axis other than the
    # default [0, 0, 0] to represent a rotation by `2*n*pi` radians.
    # To fix this, we'll choose to propagate the last nonzero axis.
    # Note: if the FIRST axes are all [0, 0, 0], that's okay; we don't need to
    # worry about propagating BACKWARDS, just FORWARDS. We'll  use the technique
    # proposed in a 2015-05-27 StackOverflow answer by user "jme" (1231929/jme)
    # to a 2015-05-27 question, "Fill zero values of 1d numpy array with last
    # non-zero values" (https://stackoverflow.com/q/30488961) by user "mgab"
    # (3406913/mgab). A 2016-12-16 edit to "Most efficient way to forward-fill 
    # NaN values in numpy array" by user Xukrao (7306999/xukrao) shows this to
    # be more efficient than similar for-loop, numba, pandas, etc. solutions.
    # --------------------------------------------------------------------------
    # The below only works for a flat list of angles! *Might* be neat to
    # consider more dimensions (but efficiency?). I think this could work by:
    #  * `inds = np.tile(np.arange(angles.shape[-1]), angles.shape[:-1] + (1,))`
    #    * An array with same shape as angles, but with aranges on last axis.
    #  * Do the zero-setting and max accumulation similar to before.
    #  * For copying, could use something like `put_along_axis`, but that's
    #    copying way more items than need-be. Could instead maybe do:
    #      * `where = np.argwhere(angles < zeroThresh).transpose()`
    #      * `where[-1] = accumulated_inds[angles < zeroThresh]`
    #      * `axes[angles < zeroThresh] = axes[tuple(where)]`
    #  * Or could flatten earlier and accept weirdness if first angles are 0.
    angleInds = np.arange(len(angles))
    # At this step, all angles SHOULD be positive.
    zeroAngleInds = np.nonzero(angles < zeroAngleThresh)
    angleInds[zeroAngleInds] = 0
    angleInds = np.maximum.accumulate(angleInds, axis = -1)
    nonUnitAxes[zeroAngleInds] = nonUnitAxes[angleInds[zeroAngleInds]]

    # TL;DR: Angles for the 1st case of Shepperd's algorithm, as output of acos,
    # start out in interval [0, pi]. Thus, the similar rotations
    # [pi - epsilon, 0, 0] and [pi + epsilon, 0, 0] will be represented by
    # distant vectors (pi - epsilon)[+1, 0, 0] and (pi - epsilon)[-1, 0, 0].
    # Similar *could* happen for the 2nd case too, though is "less guaranteed".
    # Anyway, we detect when this happens by observing the axes and then we
    # correct the angles (e.g., adding 2pi multiples, i.e. taus) to fix.
    # --------------------------------------------------------------------------
    # Let `norm = angle/2sin(angle)` for the 1st case of Shepperd's algorithm,
    # and let `norm = angle/2sin(angle/2)` for the 2nd. If we now just return
    # `norm * nonUnitAxes`, we would have *accurate* angle-axis results stored 
    # as vec3s. However, you can get axis-angle vec3 "jumps", like we described
    # above, over small rotation changes. So, we look at the unnormalized axes
    # for such flips, and we start our corrections by negating corresponding
    # angles. Now, you might think this either (a) generates incorrect results,
    # or (b) does nothing. You might think (a) because obviously rotations by
    # +alpha and -alpha about the same axis differ. But in a later step, because
    # we get our final axis-angle vec3s by multiplying `norm * nonUnitAxes`, and
    # because `angle/sin(angle) == (-angle)/sin(-angle)`, this would have no
    # effect on our output if we took no further steps. Which may lead one to
    # think that (b) applies. BUT, now angles are set up for correction by 
    # adding taus: in our "pi + epsilon" example, our consecutive angles become
    # "pi - epsilon" and "-(pi - epsilon)" after negation; we can correct the
    # latter by adding 2pi to get "pi + epsilon", which is the "best" way to
    # represent those consecutive rotations!
    # We'll only add taus s.t. angles become within pi distance of each other.
    # E.g., consecutive angles +epsilon and -epsilon would not be affected.
    # Unfortunately, because `sin((angle + tau)/2) = -sin(angle/2)`, an extra
    # negation gets introduced into the later normalization of the axes for the
    # 2nd case of Shepperd's algorithm. SO, we should normalize the axes AFTER
    # negating angles but BEFORE adding taus!!!
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
        '...ij,...ij->...i', nonUnitAxes[..., 1:, :], nonUnitAxes[..., :-1, :]
    ) # Dot products along 'j' axis; preserve existence of the 'i' axis.
    needs_flip = np.logical_xor.accumulate(angle_dots < 0, axis = -1)
    angles[..., 1:][needs_flip] = -angles[..., 1:][needs_flip]

    # (!!!) PLEASE DO NOT MOVE THIS LINE WITHOUT READING EARLIER COMMENTS
    #       EXPLAINING WHY IT SHOULD BE *EXACTLY* HERE!
    # Now comes the axis normalization/flipping. We have 2 cases to consider:
    #   1. Axes that need dividing by 2sin(angle) or 2sin(angle/2). For these,
    #      as shown before, no further action is required in terms of sign flips
    #      and whatnot if we normalize NOW, before adding taus.
    #   2. Axes for angles 2pi*n, which copied the last non-zero-angle axis.
    #      For these, we'll just copy over the normalized axes.
    unitAxes = np.empty(matrixDiags.shape) # Storage for resulting axes.
    
    # First, we'll handle Shepperd's Algorithm case 1:
    sinVals = np.sin(angles[useTrace])
    sinVals += sinVals
    unitAxes[useTrace] = \
        nonUnitAxes[useTrace] / sinVals[..., np.newaxis]
    # Then Shepperd's Algorithm case 2:
    sinVals = np.sin(angles[useDiag]/2.0)
    unitAxes[useDiag] = nonUnitAxes[useDiag] / (sinVals + sinVals)[..., np.newaxis]
    # Then zero-angle:
    unitAxes[zeroAngleInds] = unitAxes[angleInds[zeroAngleInds]]
    # If the first rotations are zero-angle, then they have no previous axis to
    # copy. So they'll still be NaNs or whatever, so we should fix those.
    numIssueAxesAtFront = 0
    while numIssueAxesAtFront < len(angles):
        if angles[numIssueAxesAtFront] >= zeroAngleThresh:
            break
        numIssueAxesAtFront += 1
    unitAxes[:numIssueAxesAtFront] = 0.0

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
    angle_corrections = np_tau * np.cumsum(tau_facs, axis = -1)
    angles[..., 1:] -= angle_corrections

    # TODO: Remove this after sufficient testing!
    if np.any(np.greater(np.abs(np.diff(angles)), np.pi + 0.00001)):
        raise Exception("Numpification of AA code resulted in angle diff > pi!")
        
    print("Reminder to remove Exception checks and look @ other TODOs.")
        
    # Now we combine the angles and unit axes into a final array of vec3s.
    return np.einsum('...i,...ij->...ij', angles, unitAxes)



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





class BCOT_Data_Calculator:

    _DATASET_DIR = None
    _CV_POSE_EXPORT_DIR = None
    _dir_paths_initialized = False

    def __init__(self, bodyIndex, seqIndex, skipAmt):
        BCOT_Data_Calculator._setupPosePaths()
        
        self.skipAmt = skipAmt
        self.issueFrames = []

        self._seq = BCOT_SEQ_NAMES[seqIndex]
        self._bod = BCOT_BODY_NAMES[bodyIndex]
        
        self.posePathGT = BCOT_Data_Calculator._DATASET_DIR / self._seq \
            / self._bod / "pose.txt"

        self._translationsGTNP = np.zeros((0,3), dtype=np.float64)
        self._translationsCalcNP = np.zeros((0,3), dtype=np.float64)

        self._rotationsGTNP = np.zeros((0,3), dtype=np.float64)
        self._rotationsCalcNP = np.zeros((0,3), dtype=np.float64)
        self._dataLoaded = False

    def _setupPosePaths():
        if BCOT_Data_Calculator._dir_paths_initialized:
            return
        settingsDir = pathlib.Path(__file__).parent.parent.resolve()
        jsonPath = settingsDir / "local.config.json"
        with open(jsonPath) as f:
            d = json.load(f)
            BCOT_Data_Calculator._DATASET_DIR = pathlib.Path(
                d["dataset_directory"]
            )
            BCOT_Data_Calculator._CV_POSE_EXPORT_DIR = pathlib.Path(
                d["result_directory"]
            )
            
    def isBodySeqPairValid(bodyIndex, seqIndex):
        BCOT_Data_Calculator._setupPosePaths()

        seq = BCOT_SEQ_NAMES[seqIndex]
        bod = BCOT_BODY_NAMES[bodyIndex]
        
        posePathGT = BCOT_Data_Calculator._DATASET_DIR / seq / bod
        return posePathGT.is_dir()

    def loadData(self):
        if self._dataLoaded:
            return

        calcFName = "cvOnly_skip" + str(self.skipAmt) + "_poses_" \
            + self._seq + "_" + self._bod +".txt"

        print("Pose path:", self.posePathGT)
        posePathCalc = BCOT_Data_Calculator._CV_POSE_EXPORT_DIR / calcFName
        #patternNum = r"(-?\d+\.?\d*e?-?\d*)" # E.g., should match "-0.11e-07"
        #patternTrans = re.compile((r"\s+" + patternNum) * 3 + r"\s*$")
        #patternRot = re.compile(r"^\s*" + (patternNum + r"\s+") * 9)

        gtMatData = BCOT_Data_Calculator.matrixDataFromFile(self.posePathGT)
        self._translationsGTNP = gtMatData[1]
        self._rotationMatsGTNP = gtMatData[0]

        self._rotationsGTNP = axisAngleFromMatArray(gtMatData[0])

        # Check if file for CV-calculated pose data exists; if so, load it too.
        if posePathCalc.is_file():
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
                self.issueFrames = 1 + np.argwhere(flipPlaces)
                # self.issueFrames = np.hstack((
                #     flipInds, rotArr[flipInds], rotArr[flipInds - 1]
                # ))
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