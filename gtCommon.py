from abc import ABC, abstractmethod
import pathlib
import json
import re
import typing

import numpy as np
from numpy.typing import NDArray

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

def truncateName(longName, maxLen = 11):
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


class VidBCOT(typing.NamedTuple):
    body_ind: int
    seq_ind: int

class PoseLoader(ABC):

    def __init__(self):
        self._setupPosePaths()
        
        self.issueFrames = []

        self._translationsGTNP = np.zeros((0,3), dtype=np.float64)
        self._translationsCalcNP = np.zeros((0,3), dtype=np.float64)

        self._rotationsGTNP = np.zeros((0,3), dtype=np.float64)
        self._rotationsCalcNP = np.zeros((0,3), dtype=np.float64)
        self._dataLoaded = False

    @classmethod
    @abstractmethod
    def getAllIDs(cls) -> typing.List:
        return []

    # The below code is meant to replace sklearn's train_test_split because
    # I want this class to be importable without having to install sklearn.
    # However, to ensure my other code that relied on train_test_split still
    # gives the same output, I made sure behaviour matched the sklearn source:
    #   sklearn/model_selection/_split.py
    # My code is a lot simpler (e.g., assumes random seed's type is never an 
    # existing rng instead of an int) but should hopefully be "good enough".
    @staticmethod
    def trainValidationTestSplit(data_to_split: NDArray,
                                 validation_ratio: float = 0.15,
                                 test_ratio: float = 0.2,
                                 random_seed: int = 0):
        n_total = len(data_to_split)

        # sklearn's train_test_split uses ceil for test and floor for train.
        n_test = int(np.ceil(test_ratio * n_total))
        n_validation = int(np.ceil(validation_ratio * n_total))
        n_test_and_valid = (n_test + n_validation)
        n_train = n_total - n_test_and_valid
        if n_train <= 0:
            raise ValueError(
                "Validation ({}) and test ({}) ratios too high; no training data!".format(
                    validation_ratio, test_ratio
                )
            )
        
        # This part also matches sklearn source code functionality, though 
        # significantly simplified (doesn't handle all the "edge cases").
        rng = np.random.RandomState(random_seed)
        rng_inds = rng.permutation(n_total)
        test_inds = rng_inds[:n_test]
        validation_inds = rng_inds[n_test: n_test_and_valid]
        train_inds = rng_inds[n_test_and_valid:]

        return (
            data_to_split[train_inds], data_to_split[validation_inds],
            data_to_split[test_inds]
        )
    
    @classmethod
    def trainValidationTestSplitIDs(cls, validation_ratio = 0.15, test_ratio = 0.2, random_seed = 0) -> typing.Tuple[typing.List, typing.List, typing.List]:
        all_ids = cls.getAllIDs()

        return PoseLoader.trainValidationTestSplit(
            all_ids, validation_ratio, test_ratio, random_seed
        )
        
    @classmethod
    def trainTestIDs(cls, test_ratio = 0.2, random_seed = 0) -> typing.Tuple[typing.List, typing.List]:
        train_valid_test = cls.trainValidationTestSplit(
            0.0, test_ratio, random_seed
        )
        # Extract the (train, test) from the (train, validation, test) tuple,
        # where validation == [].
        return (train_valid_test[0], train_valid_test[2])

    @classmethod
    @abstractmethod
    def _setPosePathsFromJSON(cls, json_read_result):
        pass

    @abstractmethod
    def getVidID(self):
        pass

    # The error-handling here was added by ChatGPT, but all remaining code is
    # human-written.
    @classmethod
    def _setupPosePaths(cls):
        if cls._dir_paths_initialized:
            return
            
        settingsDir = pathlib.Path(__file__).parent.resolve()
        jsonPath = settingsDir / "config" / "local.config.json"
        d = None
        try:
            with open(jsonPath, "r") as f:
                d = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {jsonPath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON in: {jsonPath}") from e
        except OSError as e:
            raise RuntimeError(f"Error reading config file: {jsonPath}") from e

        cls._setPosePathsFromJSON(d)
    
    @abstractmethod
    def _getPosesFromDisk(self):
        pass
            
    def loadData(self):
        if self._dataLoaded:
            return

        gtMatData, calcMatData = self._getPosesFromDisk()

        self._translationsGTNP = gtMatData[1]
        self._rotationMatsGTNP = gtMatData[0]

        self._rotationsGTNP = pm.axisAngleFromMatArray(gtMatData[0])

        # Check if file for CV-calculated pose data exists; if so, load it too.
        if calcMatData is not None:
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
    def posesFromMatsFile(filepath):
        data = np.loadtxt(filepath)
    
        # Assumes that each line has 12 floats, where the first 9 are the 3x3
        # rotation matrix entries and the last 3 are the translation.
        rotations = data[:, :9].reshape((-1, 3, 3)) 
        translations = data[:, 9:12]
        return (rotations, translations)

    def getTranslationsGTNP(self):
        self.loadData()
        return self._translationsGTNP
    def getTranslationsCalcNP(self):
        self.loadData()
        return self._translationsCalcNP
    def getRotationsGTNP(self):
        self.loadData()
        return self._rotationsGTNP
    def getRotationMatsGTNP(self):
        self.loadData()
        return self._rotationMatsGTNP
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

class PoseLoaderBCOT(PoseLoader):
    # Video motion categories.
    # TODO: Make enum.
    motion_kinds = [
        "movable_handheld", "movable_suspension", "static_handheld",
        "static_suspension", "static_trans"
    ]

    _DATASET_DIR = None
    _CV_POSE_EXPORT_DIR = None
    _dir_paths_initialized = False

    def __init__(self, bodyIndex: int, seqIndex: int, cvFrameSkipForLoad = -1):
        '''If cvFrameSkipForLoad < 0, we do not load poses calculated with computer vision.'''
        super(PoseLoaderBCOT, self).__init__()

        self._bod_index = bodyIndex
        self._seq_index = seqIndex
        self._seq = BCOT_SEQ_NAMES[self._seq_index]
        self._bod = BCOT_BODY_NAMES[self._bod_index]
        self._cvFrameSkipForLoad = cvFrameSkipForLoad
        
        self.posePathGT = PoseLoaderBCOT._DATASET_DIR / self._seq \
            / self._bod / "pose.txt"
        
        calcFName = "cvOnly_skip" + str(self._cvFrameSkipForLoad) + "_poses_" \
            + self._seq + "_" + self._bod +".txt"
        self.posePathCalc = PoseLoaderBCOT._CV_POSE_EXPORT_DIR / calcFName


    def getVidID(self):
        return (self._bod_index, self._seq_index)

    @classmethod
    def getAllIDs(cls):
        '''
        Generate the following tuples that represent each video:
            
            (sequence_name, body_name, motion_kind)
        
        The first two tuple elements uniquely identify a video, while the third
        is redundant (it's part of each sequence name) but might be used for 
        more convenient filtering of videos.

        ---
        In the BCOT dataset, videos are categorized first by the "sequence" type 
        (which is motion/lighting/background), and then by the object ("body") 
        featured in the video. Each "combo" of a sequence and body thus represents
        a distinct video.
        '''

        combos = []
        for s, s_val in enumerate(BCOT_SEQ_NAMES):
            k = ""
            # For now, using a for loop, not regex, to get motion kind from seq name.
            for k_opt in PoseLoaderBCOT.motion_kinds:
                if k_opt in s_val:
                    k = k_opt
                    break
            for b in range(len(BCOT_BODY_NAMES)):
                # Some sequence-body pairs do not have videos, and some have two videos
                # with identical motion but a different camera. So we first check that 
                # a video exists and has unique motion.
                if PoseLoaderBCOT.isBodySeqPairValid(b, s, True):
                    combos.append((b, s, k))
        return combos

    @staticmethod
    def combosByBodyIDs(bods):
        '''Filter out combos based on the 3D object ("body") subset chosen.''' 
        combos = PoseLoaderBCOT.getAllIDs()
        return [c for c in combos if c[0] in bods]

    @classmethod
    def trainValidationTestByBody(cls, validation_ratio = 0.15, test_ratio = 0.2, random_seed = 0) -> typing.Tuple[typing.List, typing.List, typing.List]:
        '''
        We'll split our data into train/validation/test sets where the vids for
        a single body will either all be train vids or all be test vids. This 
        way (a) we are guaranteed to have every motion "class" in our train and 
        test sets, and (b) we'll know how well the models generalize to new 3D 
        objects not trained on.
        '''
        all_bodies = np.arange(len(BCOT_BODY_NAMES))

        body_split = PoseLoader.trainValidationTestSplit(
            all_bodies, validation_ratio, test_ratio, random_seed
        )

        return tuple(PoseLoaderBCOT.combosByBodyIDs(b_ids) for b_ids in body_split)

    @classmethod
    def trainTestByBody(cls, test_ratio = 0.2, random_seed = 0) -> typing.Tuple[typing.List, typing.List, typing.List]:
        '''See the documentation for trainValidationTestByBody().'''
        train_valid_test = cls.trainValidationTestByBody(
            0.0, test_ratio, random_seed
        )
        # Extract the (train, test) from the (train, validation, test) tuple,
        # where validation == [].
        return (train_valid_test[0], train_valid_test[2])


    @classmethod
    def _setPosePathsFromJSON(cls, json_read_result):
        PoseLoaderBCOT._DATASET_DIR = pathlib.Path(
            json_read_result["bcot_dataset_directory"]
        )
        PoseLoaderBCOT._CV_POSE_EXPORT_DIR = pathlib.Path(
            json_read_result["bcot_result_directory"]
        )

    def _getPosesFromDisk(self):

        print("Pose path:", self.posePathGT)
        #patternNum = r"(-?\d+\.?\d*e?-?\d*)" # E.g., should match "-0.11e-07"
        #patternTrans = re.compile((r"\s+" + patternNum) * 3 + r"\s*$")
        #patternRot = re.compile(r"^\s*" + (patternNum + r"\s+") * 9)

        gtMatData = PoseLoader.posesFromMatsFile(self.posePathGT)
        calcMatData: typing.Optional[typing.Tuple[NDArray, NDArray]] = None
        if self._cvFrameSkipForLoad >= 0 and self.posePathCalc.is_file():
            calcMatData = PoseLoader.posesFromMatsFile(self.posePathCalc)

        return (gtMatData, calcMatData)

    @staticmethod
    def isBodySeqPairValid(bodyIndex, seqIndex, exclude_cam2 = False):
        PoseLoaderBCOT._setupPosePaths()

        seq = BCOT_SEQ_NAMES[seqIndex]
        bod = BCOT_BODY_NAMES[bodyIndex]

        if exclude_cam2 and "cam2" in seq:
            return False
        
        posePathGT = PoseLoaderBCOT._DATASET_DIR / seq / bod
        return posePathGT.is_dir()

class PoseLoaderBOP(PoseLoader, ABC):
    def __init__(self):
        super(PoseLoaderBOP, self).__init__() 

    @staticmethod
    def _getPosesFromFileBOP(filename):
        with open(filename, 'r') as f:
            content = f.read()

        matches = []
        pattern = r'"(\d+)"\s*:\s*\[\s*{[^}]*?"cam_R_m2c"\s*:\s*\[([^\]]+)\],\s*"cam_t_m2c"\s*:\s*\[([^\]]+)\]'
        for line in content.split("\n"):
            matches += re.findall(pattern, content)

        keys = []
        rotations = []
        translations = []

        for key, r_str, t_str in matches:
            keys.append(int(key))
            r = np.fromstring(r_str, sep=',')
            t = np.fromstring(t_str, sep=',')
            rotations.append(r.reshape(3, 3))
            translations.append(t)

        return (
            np.stack(rotations),
            np.stack(translations),
            np.diff(np.array(keys, dtype=int), prepend=keys[0])
        )
    
    @staticmethod
    def _posePathFromSeq(parent_path: pathlib.Path, seq_num: int):
        num_str = "{n:0{w}}".format(n=seq_num, w=6)
        return parent_path / num_str / "scene_gt.json"

class PoseLoaderTUDL(PoseLoaderBOP):
    _NUM_SEQS = 3
    _DATASET_DIR = None
    _dir_paths_initialized = False


    def __init__(self, seq_num: int):
        super(PoseLoaderTUDL, self).__init__()

        assert seq_num > 0, "Sequence # must be > 0."
        assert seq_num <= PoseLoaderTUDL._NUM_SEQS, "Sequence # must be <= 3."
        self.seq_num = seq_num


        self.posePathGT = self._posePathFromSeq(
            PoseLoaderTUDL._DATASET_DIR / "test", seq_num
        )
    
    @classmethod
    def getAllIDs(cls):
        return np.arange(PoseLoaderTUDL._NUM_SEQS)
    

    @classmethod
    def _setPosePathsFromJSON(cls, json_read_result):
        PoseLoaderTUDL._DATASET_DIR = pathlib.Path(
            json_read_result["tudl_dataset_directory"]
        )

    def _getPosesFromDisk(self):
        print("Pose path:", self.posePathGT)
       
        gtMatData = self._getPosesFromFileBOP(self.posePathGT)
        
        return (gtMatData, None)
    
