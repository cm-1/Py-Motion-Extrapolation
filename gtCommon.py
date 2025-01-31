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
            
    def isBodySeqPairValid(bodyIndex, seqIndex, exclude_cam2 = False):
        BCOT_Data_Calculator._setupPosePaths()

        seq = BCOT_SEQ_NAMES[seqIndex]
        bod = BCOT_BODY_NAMES[bodyIndex]

        if exclude_cam2 and "cam2" in seq:
            return False
        
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

    def _getNumFrames(self):
        if not self._dataLoaded:
            self.loadData()
        return len(self._translationsGTNP)

    @staticmethod
    def _randFloat(upper: float, lower: float = None):
        if lower is None:
            lower = -upper
        # Return single float extracted from array of length 1.
        return np.random.uniform(lower, upper, 1)[0]
    
    # Updates self and then returns the random rotation.
    def _applyRandRotToAll(self, rot_mats):
        rr = pm.randomRotationMat()
        # Front-multiply each other matrix by our random one.
        new_mats = np.einsum('ij,bjk->bik', rr, rot_mats)
        self._rotationMatsGTNP = new_mats
        self._rotationsGTNP = pm.axisAngleFromMatArray(new_mats)
        return rr

    # The times parameter may either 1D array or a "keepdims=True" result.
    # The axis can be a vec3 or an array of vec3s.
    # Returns a (matrices, angles) tuple.
    @staticmethod
    def _constAngAccelRotMats(times: np.ndarray, axis: np.ndarray, v: float,
                           a: float, start_angle: float = 0.0):
        
        num_frames = len(times)

        # We need a "keepdims=True" version of the times for multiplications.
        times_rs = times.reshape(-1, 1) if times.ndim < 2 else times

        # If "axis" is a single vec3 instead of many of them, we need to
        # make a copy of the axis for each frame.
        all_axes_given = (axis.ndim == 2 and len(axis) == num_frames)
        axes = axis if all_axes_given else np.repeat([axis], num_frames, axis=0)
            
        const_a_v_deltas = v * times_rs
        const_a_a_deltas = 0.5 * a * times_rs**2 if a != 0.0 else 0.0
        const_a_disp_angles = start_angle + const_a_v_deltas + const_a_a_deltas 

        delta_mats = pm.matsFromAxisAngleArrays(
            const_a_disp_angles.flatten(), axes 
        )

        return (delta_mats, const_a_disp_angles)

    # Replace the file-loaded rotation data with a const-angular-accel sim.
    def replaceDataWithConstAngAccel(self):
        num_const_a_vals = self._getNumFrames()
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
        const_a = np.random.sample(1)[0] * 0.0015 #* const_a_ax
        # print("const_a:", np.linalg.norm(const_a))
        # const_a_angle = np.linalg.norm(const_a)
        const_a_times = np.arange(num_const_a_vals).reshape(-1, 1)

        delta_mats, _ = self._constAngAccelRotMats(
            const_a_times, const_a_ax, start_ang_vel_angle, const_a,
            start_ang_vel_angle
        )

        _ = self._applyRandRotToAll(delta_mats)
        
        return
    

    def replaceDataWithHelix(self, useAccel: bool):
        if useAccel:
            self._spiralHelixHelper(self._spiralHelixWithAccXY)
        else:
            self._spiralHelixHelper(self._helixXY)

    def _spiralHelixHelper(self, custom_func):
        num_frames = self._getNumFrames()

        rot_rate = self._randFloat(np.pi / 4)
        v_mag = self._randFloat(35)

        times = np.arange(num_frames)
        z_axes = np.zeros((num_frames, 3))
        z_axes[:, -1] = 1.0

        rot_mats, thetas = self._constAngAccelRotMats(
            times, z_axes, rot_rate, 0.0
        )
        thetas = thetas.flatten()

        c = np.cos(thetas)
        s = np.sin(thetas)

        # Call the custom function to get xs and ys
        xs, ys, zs = custom_func(v_mag, rot_rate, times, c, s)

        
        centred_spiral = np.stack((xs, ys, zs), axis=-1)

        spiral_shift = np.random.uniform(-50, 50, 3)
        vertical_spiral = centred_spiral + spiral_shift

        spiral_tilt = self._applyRandRotToAll(rot_mats)
        self._translationsGTNP = vertical_spiral @ spiral_tilt.transpose()

        return

    def _spiralHelixWithAccXY(self, v_mag: float, rot_rate: float,
                              times: np.ndarray,
                              cos_vals: np.ndarray, sin_vals: np.ndarray):
        a_in_vdir = 0#self._randFloat(3.35)
        a_ortho = 0# self._randFloat(3.35)

        # Let R_k be our 2D object-to-world rotation matrix at frame k.
        # We want our velocity at time k to be R_k * (v_0 + a_0 * t). If we let
        # v_0 = [v, 0] for some v and a_0 = [a_v, a_p], then we want our
        # velocity to be (v + a_v*t)[cos, sin] + a_p*t[-sin, cos]. 
        # If we then take the integral of this, we get the following:
        ratio_v_no_t = (v_mag - a_ortho / rot_rate) / rot_rate
        ratio_v_t = (a_in_vdir / rot_rate) * times
        ratio_v = ratio_v_no_t + ratio_v_t

        ratio_no_v_no_t = (a_in_vdir) / (rot_rate ** 2)
        ratio_no_v_t = (a_ortho / rot_rate) * times
        ratio_no_v = ratio_no_v_no_t + ratio_no_v_t

        xs = ratio_v * sin_vals + ratio_no_v * cos_vals
        ys = ratio_no_v * sin_vals - ratio_v * cos_vals

        v_z = self._randFloat(3.35)
        a_z = self._randFloat(0.35)
        zs = v_z * times + (a_z/2.0) * (times**2)


        return xs, ys, zs

    def _helixXY(self, v_mag: float, rot_rate: float, times: np.ndarray, 
                 cos_vals: np.ndarray, sin_vals: np.ndarray):

        v_z = self._randFloat(3.35)
        zs = v_z * times

        # Let R_k be our 2D object-to-world rotation matrix at frame k.
        # We want our velocity at time k to be R_k * v_0. If we let v_0 = [v, 0]
        # for some v, then we want our velocity to be
        # v[cos(theta_k), sin(theta_k)]. If we then take the integral of this,
        # we get the position we have below.
        ratio = v_mag/rot_rate
        return ratio * sin_vals, -ratio * cos_vals, zs
