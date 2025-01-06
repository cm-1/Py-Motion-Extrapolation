import numpy as np

import pathlib
import json

import posemath as pm

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

def shortBodyNameBCOT(longName, maxLen = 11):
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

        self._rotationsGTNP = pm.axisAngleFromMatArray(gtMatData[0])

        # Check if file for CV-calculated pose data exists; if so, load it too.
        if posePathCalc.is_file():
            calcMatData = BCOT_Data_Calculator.matrixDataFromFile(posePathCalc)
            self._translationsCalcNP = calcMatData[1]
            self._rotationsCalcNP = pm.axisAngleFromMatArray(calcMatData[0])

        # TODO: In the past, I had a test to make sure the axis-angle rotations
        # did not have sharp flips between consecutive axes. This test was
        # a bit misguided, because consecutive axes [+epsilon, 0, 0] and 
        # [-epsilon, 0, 0] would be totally fine. But I'm keeping it this way
        # for consistency's sake for now to make sure other code areas didn't
        # break. But SOON, I should either remove this or look at euclidean
        # distance between vectors instead!
        for rotArr in [self._rotationsGTNP, self._rotationsCalcNP]:
            flipPlaces = pm.einsumDot(rotArr[1:], rotArr[:-1]) <= 0
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
    def getRotationMatsGTNP(self, useSkipAmt):
        self.loadData()
        step = (1 + self.skipAmt) if useSkipAmt else 1
        return self._rotationMatsGTNP[::step]
    def getRotationsCalcNP(self):
        self.loadData()
        return self._rotationsCalcNP

    # Replace the file-loaded rotation data with a const-angular-accel sim.
    def replaceRotationData(self):
        if not self._dataLoaded:
            raise NotImplementedError("Data must be loaded before replacing!")
        num_const_a_vals = len(self.getTranslationsGTNP(False))
        start_ang_vel = np.random.uniform(-0.08, 0.08, 3)
        start_ang_vel_angle = np.linalg.norm(start_ang_vel)
        const_a_ax = start_ang_vel / start_ang_vel_angle
        # The values of 0.1 a few lines above and 0.0015 on the next line were
        # chosen so that the maximum angular displacement between frames with a
        # skip amount of 2 would still be under pi radians. In this case with a
        # step of 3, the max start vel becomes 3*sqrt(0.24), the acceleration is
        # "increased" by 3^2 = 9 (since 1/s^2 is in the unit), and there'd be
        # a max of 120 "3-steps" in this dataset (max 360 frames per vid),
        # so the max velocity becomes 3*sqrt(0.24) + 120*9*0.0015.
        # Hope there's no mistake in the above math. Didn't double-check because
        # this code worked "good enough" in tests.
        const_a = np.random.sample(1)[0] * 0.0015 * const_a_ax
        # print("const_a:", np.linalg.norm(const_a))
        const_a_angle = np.linalg.norm(const_a)
        const_a_times = np.arange(num_const_a_vals).reshape(-1, 1)

        const_a_v_deltas = start_ang_vel_angle * const_a_times
        const_a_a_deltas = 0.5 * const_a_angle * const_a_times**2
        const_a_disp_angles = start_ang_vel_angle + const_a_v_deltas + const_a_a_deltas 

        start_quat_2D = pm.normalizeAll(np.random.uniform(-1, 1, (1, 4)))
        start_mat = pm.matFromAxisAngle(pm.axisAnglesFromQuats(start_quat_2D)[0])
        repeat_axes = np.repeat(
            const_a_ax.reshape(1, -1), num_const_a_vals, axis=0
        )
        delta_mats = pm.matsFromAxisAngleArrays(
            const_a_disp_angles.flatten(), repeat_axes 
        )
        final_mats = np.einsum(
            'ij,bjk->bik', start_mat, delta_mats
        )
        self._rotationMatsGTNP = final_mats
        self._rotationsGTNP = pm.axisAngleFromMatArray(final_mats)
        return
