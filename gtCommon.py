from abc import ABC, abstractmethod
from enum import Enum
import pathlib
import json
import re
import typing

import numpy as np
from numpy.typing import NDArray
from motiontools.shared_constants import *
from datatools.data_splitting import DataSubsetKind

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

class _LoadedPoses:
    def __init__(self, translations: NDArray,
                 mat_rotations: typing.Optional[NDArray] = None,
                 aa_rotations: typing.Optional[NDArray] = None):
        if aa_rotations is None and mat_rotations is None:
            raise ValueError("Must supply either AA or matrix rotations!")
        
        self.translations = translations
        if mat_rotations is None:
            mat_rotations = pm.matsFromScaledAxisAngleArray(aa_rotations)
        if aa_rotations is None:
            aa_rotations = pm.axisAngleFromMatArray(mat_rotations)
        self.mat_rotations: NDArray = mat_rotations
        self.aa_rotations: NDArray = aa_rotations

class _LoadedData(typing.NamedTuple):
    gt_data: _LoadedPoses
    cv_data: typing.Optional[_LoadedPoses] 
    
class PoseLoader(ABC):

    def __init__(self, are_timestamps_const: bool):
        self._setupPosePaths()
        
        self._translationsGTNP = np.zeros((0,3), dtype=np.float64)
        self._translationsCalcNP = np.zeros((0,3), dtype=np.float64)

        self._rotationsGTNP = np.zeros((0,3), dtype=np.float64)
        self._rotationsCalcNP = np.zeros((0,3), dtype=np.float64)

        self._rotationMatsGTNP = np.zeros((0,3,3), dtype=np.float64)
        self._rotationMatsCalcNP = np.zeros((0,3,3), dtype=np.float64)


        self._dataLoaded = False

        self._are_timestamps_const = are_timestamps_const
        self._timestamps: typing.Optional[NDArray] = None

    @staticmethod
    @abstractmethod
    def datasetName() -> str:
        return "UNDEFINED"

    
    def areTimestampsConst(self):
        return self._are_timestamps_const

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
    def trainValidationTestSplit(data_to_split: typing.Union[NDArray, typing.List],
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

        train_data = [data_to_split[i] for i in train_inds]
        val_data = [data_to_split[i] for i in validation_inds]
        test_data = [data_to_split[i] for i in test_inds]

        return (train_data, val_data, test_data)
    
    @classmethod
    def trainValidationTestSplitIDs(cls, validation_ratio = 0.15,
                                    test_ratio = 0.2, random_seed = 0,
                                    *args, **kwargs) -> typing.Tuple[typing.List, typing.List, typing.List]:
        all_ids = cls.getAllIDs(*args, **kwargs)

        return PoseLoader.trainValidationTestSplit(
            all_ids, validation_ratio, test_ratio, random_seed
        )
        
    @classmethod
    def trainTestIDs(cls, test_ratio = 0.2, random_seed = 0) -> typing.Tuple[typing.List, typing.List]:
        train_valid_test = cls.trainValidationTestSplitIDs(
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
        return ()

    # The error-handling here was added by ChatGPT, but all remaining code is
    # human-written.
    @classmethod
    def _setupPosePaths(cls):
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
    def _getPosesFromDisk(self) -> _LoadedData:
        pass
            
    def loadData(self):
        if self._dataLoaded:
            return

        gtMatData, calcMatData = self._getPosesFromDisk()

        self._translationsGTNP = gtMatData.translations
        self._rotationMatsGTNP = gtMatData.mat_rotations
        self._rotationsGTNP = gtMatData.aa_rotations

        # Check if file for CV-calculated pose data exists; if so, load it too.
        if calcMatData is not None:
            self._translationsCalcNP = calcMatData.translations
            self._rotationMatsCalcNP = calcMatData.mat_rotations
            self._rotationsCalcNP = calcMatData.aa_rotations

        # This was a test for "bad" flips in the axis angle creation from
        # matrix arrays. I say "bad" flips because "small" flips from, say,
        # [+epsilon, 0, 0] to [-epsilon, 0, 0] would be totally fine. So one
        # also needs to test for the magnitude of the difference vector.
        # for rotArr in [self._rotationsGTNP, self._rotationsCalcNP]:
        #     flipPlaces = pm.einsumDot(rotArr[1:], rotArr[:-1]) <= 0
        #     if np.any(flipPlaces):
        #         eucDists = np.linalg.norm(rotArr[1:][flipPlaces], axis=-1)
        #         eucDists += np.linalg.norm(rotArr[:-1][flipPlaces], axis=-1)
        #         if np.any(eucDists > np.pi):
        #     # Also, testing for big jumps in non-flipping cases.
        #     jumps = np.linalg.norm(np.diff(rotArr, 1, axis=0), axis=-1)
        #     big_jumps = jumps > np.pi
        #     if np.any(big_jumps):
        #         tau = 2 * np.pi
        #         where_jump,  = np.nonzero(big_jumps)
        #         unexplained_jumps = []
        #         for wj in where_jump:
        #             jump_vec0 = rotArr[wj]
        #             jump_vec1 = rotArr[wj + 1]
        #             jv1_dir = jump_vec1 / np.linalg.norm(jump_vec1)
        #             wj_unexplained = False
        #             for fix_dir in [-1, 1]:
        #                 jump_vec1_new = jump_vec1 + (tau * fix_dir) * jv1_dir
        #                 new_dist = np.linalg.norm(jump_vec1_new - jump_vec0)
        #                 if new_dist < jumps[wj]:
        #                     wj_unexplained = True
        #             if wj_unexplained:
        #                 unexplained_jumps.append(wj)
        #         if len(unexplained_jumps) > 0:
            self._dataLoaded = True
        

    # Returns (rotation mat data, translation data) tuple, where each element is
    # a numpy array, the former with shape (n,3,3), the latter with shape (n,3).
    @staticmethod
    def posesFromMatsTXT(filepath):
        data = np.loadtxt(filepath)
    
        # Assumes that each line has 12 floats, where the first 9 are the 3x3
        # rotation matrix entries and the last 3 are the translation.
        rotations = data[:, :9].reshape((-1, 3, 3)) 
        translations = data[:, 9:12]
        return _LoadedPoses(translations, rotations)

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
    
    def getTimestamps(self):
        self.loadData()
        if self._are_timestamps_const:
            return np.arange(len(self._translationsGTNP))
        else:
            return self._timestamps

    def getNoisyTranslation(self, std_dev):
        self.loadData()
        h = hash(self.getVidID())
        rng = np.random.default_rng(seed=abs(h)) # TODO: Replace abs with something better.
        noise = rng.normal(0.0, std_dev, self._translationsGTNP.shape)
        return self._translationsGTNP + noise

    def getNoisyRotation(self, std_dev_deg):
        return pm.applyRandomAANoise(self._rotationsGTNP, std_dev_deg, abs(hash(self.getVidID())))
    

    def getTimestamps(self):
        self.loadData()
        if self._are_timestamps_const:
            return np.arange(len(self._translationsGTNP))
        else:
            return self._timestamps

    def getNoisyTranslation(self, std_dev):
        self.loadData()
        h = hash(self.getVidID())
        rng = np.random.default_rng(seed=abs(h)) # TODO: Replace abs with something better.
        noise = rng.normal(0.0, std_dev, self._translationsGTNP.shape)
        return self._translationsGTNP + noise

    def getNoisyRotation(self, std_dev_deg):
        return pm.applyRandomAANoise(self._rotationsGTNP, std_dev_deg, abs(hash(self.getVidID())))
    

    def getTimestamps(self):
        self.loadData()
        if self._are_timestamps_const:
            return np.arange(len(self._translationsGTNP))
        else:
            return self._timestamps

    def _getNumFrames(self):
        if not self._dataLoaded:
            self.loadData()
        return len(self._translationsGTNP)

    @staticmethod
    def _randHelper(shape, upper: float, lower: typing.Optional[float] = None):
        if lower is None:
            lower = -upper
        return np.random.uniform(lower, upper, shape)

    @staticmethod
    def _randFloat(upper: float, lower: typing.Optional[float] = None):
        # Return single float extracted from array of length 1.
        return PoseLoader._randHelper(1, upper, lower)[0]
    
    @classmethod
    def prepIDsForConstructor(cls, ids: typing.List) -> typing.List:
        return ids
    
    @classmethod
    def getAllMinimalIDsAndLoaders(cls, *args, **kwargs):
        ids = cls.getAllIDs(*args, **kwargs)
        clean_ids = cls.prepIDsForConstructor(ids)
        loaders = [cls(*ci) for ci in clean_ids]
        return clean_ids, loaders

    @classmethod
    def getSamplesByCriteria(cls, vid_ids, realign: bool, allow_overlap: bool,
                             criteria: str, step: int, num_to_find: int = -1,
                             static_thresh = 4.0, planar_thresh: float = 1e-2,
                             verbose: bool = False):
        '''
        Get a set of sample sequences (enough needed to calculate crackle) from
        a set of specified videos, using one of several criteria
        ("static", "planar", or "none").

        Parameters:
            vid_ids (list): List of video identifier tuples to search through.
            realign (bool): Whether to realign the coordinate frame so the last
                two points are on the x axis and the last three in the XY plane.
            allow_overlap (bool): Whether returned sequences can share points.
            step (int): The step between sampled frames to use.
            num_to_find (int): How many such sequences to find (if possible).
                Defaults to -1, meaning to find all possible ones.
            criteria (str): Which selection: 'static', 'planar', or 'none'.
            static_thresh (float, optional): Threshold for "static" criterion.
            planar_thresh (float, optional): Tolerance for planarity.
            verbose (bool, optional): Print successfulness. Defaults to False.

        Returns:
            list: A list of 2-tuples of arrays (pts_subset, aas_subset) where:
                - pts_subset (ndarray): Array of point positions, potentially
                realigned and translated if realign=True
                - aas_subset (ndarray): Array of rotations (as axis-angles),
                potentially realigned if realign=True
        '''
        # Create a loader instance from each ID.
        loaders = [cls(*i) for i in cls.prepIDsForConstructor(vid_ids)]
        rets: typing.List[typing.Tuple[NDArray, NDArray]] = [] # Returned list.
        num_found = 0
        find_all = num_to_find < 0
        for tl in loaders:
            all_pts = tl.getTranslationsGTNP()[::step]
            tl_diffs = np.diff(all_pts, 1, axis=0)
            
            i = 0
            seq_len = (GT_PT_IND + 1)
            i_lim = len(all_pts) - seq_len
            # Continue until we run out of frames or found enough examples.
            while i < i_lim and (num_found < num_to_find or find_all):
                accept = False
                end_ind = i + seq_len
                end_diff_ind = end_ind - 1
                if criteria == "static":
                    # See if first few points are mostly static.
                    tl_diff_subset = tl_diffs[i:end_diff_ind]
                    early_speeds = np.linalg.norm(
                        tl_diff_subset[:LAST_FIXED_PT_IND], axis=-1
                    )
                    if np.sum(early_speeds) < static_thresh:
                        accept = True
                elif criteria == "planar":
                    # Fit a plane to pts_subset (using SVD)
                    pts_subset = all_pts[i:end_ind]
                    mean = np.mean(pts_subset, axis=0)
                    U_S_Vt = np.linalg.svd(pts_subset - mean)
                    # Smallest singular values ratio = distance from planarity
                    if U_S_Vt[1][-1]/U_S_Vt[1][-2] < planar_thresh:
                        accept = True
                elif criteria == "none":
                    accept = True
                else:
                    raise ValueError(f"Unknown criteria: {criteria}")

                if accept:
                    # If so, we collect the poses of them and the next frames.
                    aas_subset = tl.getRotationsGTNP()[::step][i:end_ind]
                    pts_subset = all_pts[i:end_ind]
                    # May move points so that the one before LAST_FIXED_PT_IND
                    # is at origin, the point at LAST_FIXED_PT_IND is on x-axis,
                    # and the one two prior is in XY plane.
                    if realign:
                        tl_diff_subset = tl_diffs[i:end_diff_ind]
                        # Get rotation matrix to move the mentioned points into
                        # XY plane, with velocity along x-axis.
                        diff_mat = pm.getOrthonormalFrames(
                            False,
                            tl_diff_subset[LAST_FIXED_PT_IND - 1:LAST_FIXED_PT_IND],
                            tl_diff_subset[LAST_FIXED_PT_IND:DYNAMIC_PT_IND]
                        )[1][0]
                        aas_subset = aas_subset @ diff_mat
                        new_frame_pts = pts_subset @ diff_mat
                        # Move points closer to origin.
                        pts_subset = new_frame_pts - new_frame_pts[LAST_FIXED_PT_IND - 1]
                    rets.append((pts_subset, aas_subset))
                    num_found += 1
                    i = ((i + 1) if allow_overlap else end_ind)
                else:
                    i += 1
            if (num_found >= num_to_find and not find_all):
                break
        if verbose:
            find_str = str(num_to_find) if num_to_find > 0 else "all"
            print(f"Found {num_found}/{find_str} {criteria} sequences.")
        return rets

    @classmethod
    def getGroupedSamplesByCriteria(cls, train_ids, val_ids, test_ids,
                                    realign: bool, allow_overlap: bool,
                                    criteria: str, step: int,
                                    num_to_find: int = -1, static_thresh = 4.0,
                                    planar_thresh: float = 1e-2,
                                    verbose: bool = False):
        '''
        Same as getSamplesByCriteria, but grouped by train/val/test.

        Parameters:
            train_ids, val_ids, test_ids: Lists of video identifier tuples.
            Remaining parameters: See getSamplesByCriteria.

        Returns:
            dict: A dict where the keys are DataSubsetKind values and the items
                are NDArrays structured like lists of 2-tuples of arrays
                (pts_subset, aas_subset) where:
                    - pts_subset (ndarray): Array of point positions, potentially
                      realigned and translated if realign=True
                    - aas_subset (ndarray): Array of rotations (as axis-angles),
                      potentially realigned if realign=True
                
        '''
        # Dictionary to store sequences from each set
        sequence_sets = {
            DataSubsetKind.TRAIN: train_ids,
            DataSubsetKind.VALIDATION: val_ids,
            DataSubsetKind.TEST: test_ids
        }
        # Get sequences for each set
        sequences_by_set: typing.Dict[DataSubsetKind, NDArray] = dict()
        for set_name, vid_ids in sequence_sets.items():
            tup_list = cls.getSamplesByCriteria(
                vid_ids, realign, allow_overlap, criteria, step, num_to_find,
                static_thresh, planar_thresh, verbose
            )
            e = (0, 2, GT_PT_IND + 1, 3)
            pose_info = np.stack(tup_list, axis=0) if tup_list else np.empty(e)
            pose_info.setflags(write=False)
            sequences_by_set[set_name] = pose_info
        return sequences_by_set

class SyntheticPoseLoader(PoseLoader):
    def __init__(self, num_frames: int, const_rot_accel: bool, helix: bool,
                 const_deriv_lim: int = -1):
        super(SyntheticPoseLoader, self).__init__(True)

        self.num_frames = num_frames
        self.const_rot_accel = const_rot_accel
        self.helix = helix
        self.const_deriv_lim = const_deriv_lim
        # Reusability-TODO: Change the constructor so that seeds can be
        # specified. Then, here, save the seeds if they're provided, and if not,
        # create random seeds. Could use a lambda for that part.

    @staticmethod
    def datasetName():
        return "Synthetic"

    @classmethod
    def getAllIDs(cls, max_id: int = 1) -> typing.List:
        raise AttributeError("Synthetic pose loader lacks pre-set IDs.")

    @classmethod
    def _setPosePathsFromJSON(cls, json_read_result):
        pass

    def getVidID(self):
        # Reusability-TODO: Also return the random seeds used to generate the
        # data. 
        return (
            self.num_frames, self.const_rot_accel, self.helix,
            self.const_deriv_lim
        )

    def _getPosesFromDisk(self):
        rotations = np.empty((0,3))
        translations = np.empty((0,3))

        if self.helix:
            rotations, translations = self.getHelixRotationMatsAndPositions(
                self.helix
            )
        elif self.const_rot_accel:
            rotations = self.getConstAngAccelMats()

        if self.const_deriv_lim > 0:
            init_vals = [
                self._randHelper(3, 33) for _ in range(self.const_deriv_lim + 1)
            ]
            diffs = np.broadcast_to(init_vals[-1], (self.num_frames, 3))
            for iv in init_vals[-2::-1]:
                diffs = iv + np.cumsum(diffs, axis=0)
            translations = diffs          

        # If translations or rotations were not created above, set to defaults.
        if len(translations) == 0:
            translations = np.zeros((self.num_frames, 3))
        if len(rotations) == 0:
            iden = np.eye(3).reshape(1, 3, 3)
            rotations = np.repeat(iden, self.num_frames, axis=0)

        gtMatData = _LoadedPoses(translations, rotations)
        calcMatData = None # May use this in the future somehow?
        return _LoadedData(gtMatData, calcMatData)

    # Updates self and then returns the random rotation.
    @staticmethod
    def _applyRandRotToAll(rot_mats):
        rr = pm.randomRotationMat()
        # Front-multiply each other matrix by our random one.
        new_mats = np.einsum('ij,bjk->bik', rr, rot_mats)
        return rr, new_mats

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
    def getConstAngAccelMats(self):
        num_const_a_vals = self.num_frames
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

        return self._applyRandRotToAll(delta_mats)[1]
            

    def getHelixRotationMatsAndPositions(self, useAccel: bool):
        retVal: typing.Tuple[NDArray, NDArray]
        if useAccel:
            retVal = self._spiralHelixHelper(self._spiralHelixWithAccXY)
        else:
            retVal = self._spiralHelixHelper(self._helixXY)
        return retVal
    
    def _spiralHelixHelper(self, custom_func):
        num_frames = self.num_frames

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

        spiral_tilt, final_rot_mats = self._applyRandRotToAll(rot_mats)
        final_translations = vertical_spiral @ spiral_tilt.transpose()

        return (final_rot_mats, final_translations)

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

    _DATASET_DIR: pathlib.Path
    _CV_POSE_EXPORT_DIR: pathlib.Path
    _dir_paths_initialized: bool = False

    def __init__(self, bodyIndex: int, seqIndex: int, cvFrameSkipForLoad = -1):
        '''If cvFrameSkipForLoad < 0, we do not load poses calculated with computer vision.'''
        super(PoseLoaderBCOT, self).__init__(True)

        self._bod_index = bodyIndex
        self._seq_index = seqIndex
        self._seq = BCOT_SEQ_NAMES[self._seq_index]
        self._bod = BCOT_BODY_NAMES[self._bod_index]
        self._cvFrameSkipForLoad = cvFrameSkipForLoad
        
        self.poseDirGT = PoseLoaderBCOT._DATASET_DIR / self._seq / self._bod
        self.posePathGT = self.poseDirGT / "pose.txt"
        
        calcFName = "cvOnly_skip" + str(self._cvFrameSkipForLoad) + "_poses_" \
            + self._seq + "_" + self._bod +".txt"
        self.posePathCalc = PoseLoaderBCOT._CV_POSE_EXPORT_DIR / calcFName

    def datasetName():
        return "BCOT"

    def getVidID(self):
        return (self._bod_index, self._seq_index)

    @staticmethod
    def getMotionKind(seq_index: int):
        # TODO: Precalculate and store this so that it's O(1) instead of O(n)
        k = ""
        for k_opt in PoseLoaderBCOT.motion_kinds:
            if k_opt in BCOT_SEQ_NAMES[seq_index]:
                k = k_opt
                break
        return k

    @classmethod
    def getAllIDs(cls, exclude_cam2: bool = True):
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
        for s in range(len(BCOT_SEQ_NAMES)):
            k = cls.getMotionKind(s)
            for b in range(len(BCOT_BODY_NAMES)):
                # Some sequence-body pairs do not have videos, and some have two videos
                # with identical motion but a different camera. So we first check that 
                # a video exists and has unique motion.
                if PoseLoaderBCOT.isBodySeqPairValid(b, s, exclude_cam2):
                    combos.append((b, s, k))
        return combos

    @staticmethod
    def combosByBodyIDs(bods, exclude_cam2: bool):
        '''Filter out combos based on the 3D object ("body") subset chosen.''' 
        combos = PoseLoaderBCOT.getAllIDs(exclude_cam2)
        return [c for c in combos if c[0] in bods]

    @classmethod
    def trainValidationTestByBody(cls, validation_ratio = 0.15,
                                  test_ratio = 0.2, random_seed = 0,
                                  exclude_cam2: bool = True) -> typing.Tuple[typing.List, typing.List, typing.List]:
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

        return typing.cast(
            typing.Tuple[typing.List, typing.List, typing.List],
            tuple(
                PoseLoaderBCOT.combosByBodyIDs(b_ids, exclude_cam2)
                for b_ids in body_split
            )
        )

    @classmethod
    def trainTestByBody(cls, test_ratio = 0.2, random_seed = 0) -> typing.Tuple[typing.List, typing.List]:
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
        PoseLoaderBCOT._dir_paths_initialized = True

    def _getPosesFromDisk(self):

        print("Pose path:", self.posePathGT)
        #patternNum = r"(-?\d+\.?\d*e?-?\d*)" # E.g., should match "-0.11e-07"
        #patternTrans = re.compile((r"\s+" + patternNum) * 3 + r"\s*$")
        #patternRot = re.compile(r"^\s*" + (patternNum + r"\s+") * 9)

        gtMatData = PoseLoader.posesFromMatsTXT(self.posePathGT)
        calcMatData = None
        if self._cvFrameSkipForLoad >= 0 and self.posePathCalc.is_file():
            calcMatData = PoseLoader.posesFromMatsTXT(self.posePathCalc)

        return _LoadedData(gtMatData, calcMatData)

    @staticmethod
    def isBodySeqPairValid(bodyIndex: int, seqIndex: int, exclude_cam2: bool = False):
        if not PoseLoaderBCOT._dir_paths_initialized:
            PoseLoaderBCOT._setupPosePaths()

        seq = BCOT_SEQ_NAMES[seqIndex]
        bod = BCOT_BODY_NAMES[bodyIndex]

        if exclude_cam2 and "cam2" in seq:
            return False
        
        posePathGT = PoseLoaderBCOT._DATASET_DIR / seq / bod
        return posePathGT.is_dir()
    
    @classmethod
    def prepIDsForConstructor(cls, ids):
        return [i[:2] for i in ids]

class PoseLoaderBOP(PoseLoader, ABC):
    def __init__(self, are_timestamps_const: bool):
        super(PoseLoaderBOP, self).__init__(are_timestamps_const)
        raise NotImplementedError("Need to figure out how to handle this better!")

    @staticmethod
    def _getPosesFromFileBOP(filename: str,
                             line_range: \
                                typing.Optional[typing.Tuple[int, int]] = None
                             ):
        '''NOTE: Line range has inclusive lower, exclusive upper, like range().'''
        with open(filename, 'r') as f:
            content = f.readlines()
            
        if content[0].strip() == "{":
            content = content[1:] # Exclude first line, "{"

        matches = []
        pattern = r'"(\d+)"\s*:\s*\[\s*{[^}]*?"cam_R_m2c"\s*:\s*\[([^\]]+)\],\s*"cam_t_m2c"\s*:\s*\[([^\]]+)\]'
        
        selected_content = content
        if line_range is not None:
            selected_content = content[line_range[0]:line_range[1]]
        
        for line in selected_content:
            curr_matches = re.findall(pattern, line)
            if len(curr_matches) > 1:
                raise NotImplementedError("More than one object per frame!")
            elif len(curr_matches) == 0 and line.strip() != '}':
                raise Exception("No pose found on line!")
            matches += curr_matches

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
    _NUM_SUBSEQS = 8
    _SUBSEQ_SPLITS = {
        (True, 1): (1085, 2030, 2982, 4074, 5320, 6440, 7630),
        (True, 2): (987, 1948, 2907, 4013, 5035, 5994, 7184),
        (True, 3): (966, 1903, 2981, 4010, 4974, 6066, 7347),
        (False, 1): (1421, 3059, 4445, 6027, 7994, 9646, 11865),
        (False, 2): (1274, 2765, 3899, 5180, 6622, 7749, 9793),
        (False, 3): (1806, 3794, 5117, 6706, 8246, 9772, 11851)
    }
    _DATASET_DIR = None


    def __init__(self, is_test: bool, seq_num: int, subseq_num: int):
        super(PoseLoaderTUDL, self).__init__(False)
        raise NotImplementedError("Need to check if TUDL timestamps are constant!")

        assert seq_num > 0, "Sequence # must be > 0."
        assert seq_num <= PoseLoaderTUDL._NUM_SEQS, "Sequence # must be <= 3."

        # -1 means we use whole folder, including discontinuities.
        assert subseq_num >= -1, "Subsequence # must be >= -1."
        assert subseq_num < PoseLoaderTUDL._NUM_SUBSEQS, "Subsequence # must be < 8."
        
        self.seq_num = seq_num
        self.subseq_num = subseq_num
        self.is_test = is_test

        tt_folder = "test" if is_test else "train_real"


        self.posePathGT = self._posePathFromSeq(
            PoseLoaderTUDL._DATASET_DIR / tt_folder, seq_num
        )

        low = 0
        if subseq_num > 0:
            low = PoseLoaderTUDL._SUBSEQ_SPLITS[is_test, seq_num][subseq_num - 1]

        high = None
        # -1 means we use whole folder, including discontinuities.
        if subseq_num >= 0 and (subseq_num + 1) < PoseLoaderTUDL._NUM_SUBSEQS:
            high = PoseLoaderTUDL._SUBSEQ_SPLITS[is_test, seq_num][subseq_num]
        

        self.subseq_interval = (low, high)
    
    @staticmethod
    def datasetName():
        return "TUDL"
    
    @classmethod
    def getAllIDs(cls):
        return [
            (is_test, seq_num, subseq_num)
            for is_test in (True, False) 
            for seq_num in range(1, PoseLoaderTUDL._NUM_SEQS + 1)
            for subseq_num in range(PoseLoaderTUDL._NUM_SUBSEQS)
        ]
    
    def getVidID(self):
        return (self.is_test, self.seq_num, self.subseq_num)
    

    @classmethod
    def _setPosePathsFromJSON(cls, json_read_result):
        PoseLoaderTUDL._DATASET_DIR = pathlib.Path(
            json_read_result["tudl_dataset_directory"]
        )

    def _getPosesFromDisk(self):
        print("Pose path:", self.posePathGT)
       
        gtMatData = self._getPosesFromFileBOP(
            self.posePathGT, self.subseq_interval
        )
        
        return (gtMatData, None)
    
class PoseLoaderPauwels(PoseLoader):
    _DATASET_DIR = None

    def __init__(self, cvFrameSkipForLoad = -1):
        '''If cvFrameSkipForLoad < 0, we do not load poses calculated with computer vision.'''
        super(PoseLoaderPauwels, self).__init__(False)
        raise NotImplementedError("Need to check if this dataset has timestamps!")
        self._cvFrameSkipForLoad = cvFrameSkipForLoad
        
        self.posePathGT = PoseLoaderPauwels._DATASET_DIR / "ground_truth.txt"
        
        # calcFName = "cvOnly_skip" + str(self._cvFrameSkipForLoad) + "_poses_" \
        #     + self._seq + "_" + self._bod +".txt"
        # self.posePathCalc = PoseLoaderPauwels._CV_POSE_EXPORT_DIR / calcFName

    @staticmethod
    def datasetName():
        return "Pauwels"

    def getVidID(self):
        return () # empty tuple

    @classmethod
    def getAllIDs(cls):
        return [()]

    @classmethod
    def _setPosePathsFromJSON(cls, json_read_result):
        PoseLoaderPauwels._DATASET_DIR = pathlib.Path(
            json_read_result["pauwels_dataset_directory"]
        )
        # PoseLoaderPauwels._CV_POSE_EXPORT_DIR = pathlib.Path(
        #     json_read_result["bcot_result_directory"]
        # )

    def _getPosesFromDisk(self):
        data = np.loadtxt(self.posePathGT)
    
        v3_rotations = data[:, 3:]
        translations = data[:, :3]
        gtMatData = _LoadedPoses(translations, None, v3_rotations)

        calcMatData: typing.Optional[typing.Tuple[NDArray, NDArray]] = None
        # if self._cvFrameSkipForLoad >= 0 and self.posePathCalc.is_file():
        #     calcMatData = PoseLoader.posesFromMatsTXT(self.posePathCalc)

        return _LoadedData(gtMatData, calcMatData)

    def getRotationsGTNP(self):
        raise NotImplementedError("Could make this > efficient?")
        return super().getRotationsGTNP()

class SepPoseComponents(typing.NamedTuple):
    translations: NDArray
    rot_mats: NDArray
    rot_aas: NDArray # Axis-angles

class PoseLoaderClipsHOT3D(PoseLoader):

    # _FRAMES_PER_VID = 150
    # _ZEROS_WIDTH = 6

    # _ALL_CAM_KEYS = {
    #     CAM_TYPE.ARIA: ("1201-1", "1201-2", "214-1"),
    #     CAM_TYPE.QUEST: ("1201-1", "1201-2")
    # }

    # # The value for the Quest3 does not actually matter, since both of its
    # # cameras always output the exact same timestamp.
    # _ALL_REF_CAM_IDXS = {CAM_TYPE.ARIA: 2, CAM_TYPE.QUEST: 0}

    class CAM_TYPE(Enum):
        ARIA = 1
        QUEST = 2

    _CAM_TYPE_CLIP_RANGES = {
        CAM_TYPE.ARIA: (1849, 3364), CAM_TYPE.QUEST: (0, 1287)
    }

    _POSE_FNAMES = {
        CAM_TYPE.ARIA: "hot3d_all_aria_poses.npz",
        CAM_TYPE.QUEST: "hot3d_all_quest_poses.npz"
    }

    _TIMESTAMP_FNAMES = {
        CAM_TYPE.ARIA: "hot3d_aria_times.npz",
        CAM_TYPE.QUEST: "hot3d_quest_times.npz"
    }

    _DATASET_DIR: pathlib.Path
    # _CV_POSE_EXPORT_DIR = None
    _dir_paths_initialized: bool = False

    _ALL_CLIP_OBJS: typing.Dict[CAM_TYPE, NDArray] = dict()
    _ALL_CLIP_BASE_INDS: typing.Dict[CAM_TYPE, NDArray] = dict()
    _ALL_INV_CAM_POSES: typing.Dict[CAM_TYPE, SepPoseComponents] = dict()
    _ALL_OBJ_CAM_POSES: typing.Dict[CAM_TYPE, SepPoseComponents] = dict()
    _ALL_OBJ_WORLD_POSES: typing.Dict[CAM_TYPE, SepPoseComponents] = dict()
    _ALL_TIMESTAMPS: typing.Dict[int, NDArray] = dict()


    def __init__(self, clip_num: int, obj_num: int, ignore_cam: bool = False):
        '''Use an `obj_num` of 0 to indicate "stationary" objects, which means
        you basically just get the camera motion.'''
        super(PoseLoaderClipsHOT3D, self).__init__(  False  )  
        if obj_num < 0:
            raise ValueError("Object number must be positive!")
        elif obj_num == 0 and ignore_cam:
            raise ValueError("Cannot ignore camera for static object poses!")
        
        self._clip_num = clip_num
        self._obj_num = obj_num
        self._ignore_cam = ignore_cam
        cam_type_found = False

        self.clip_type: PoseLoaderClipsHOT3D.CAM_TYPE
        lower_clip_ind = 0
        # Loop over camera/device types to see which one this clip belongs to.
        for key, rnge in PoseLoaderClipsHOT3D._CAM_TYPE_CLIP_RANGES.items():
            if self._clip_num >= rnge[0] and self._clip_num <= rnge[1]:
                self.clip_type = key
                cam_type_found = True
                lower_clip_ind = rnge[0]
                break
        self._local_clip_ind = self._clip_num - lower_clip_ind

        if not cam_type_found:
            raise Exception("Clip #" + str(clip_num) + " not recognized!")
        
        times_fname = PoseLoaderClipsHOT3D._TIMESTAMP_FNAMES[self.clip_type]
        poses_fname = PoseLoaderClipsHOT3D._POSE_FNAMES[self.clip_type]

        self.poseDirGT = PoseLoaderClipsHOT3D._DATASET_DIR 
        self.posePathGT = self.poseDirGT / poses_fname
        self.timesPathGT = self.poseDirGT / times_fname

        self._ind_in_loaded = -1
        
    @staticmethod
    def datasetName():
        return "HOT3D"

    def getVidID(self):
        return (self._clip_num, self._obj_num, self._ignore_cam)

    @classmethod
    def getAllIDs(cls, include_o2c = False, include_w2c = False,
                  include_o2w = False):
        if not (include_o2c or include_o2w or include_w2c):
            raise ValueError("Must specify which pose kinds to include!")
        
        # We need to load info on which objects are moving in each clip.
        if not PoseLoaderClipsHOT3D._dir_paths_initialized:
            PoseLoaderClipsHOT3D._setupPosePaths()

        # (Clip #, ObjectID (0 for stationary), )
        # TODO: Need to include model numbers here too!
        ret = []
        for cam, rnge in PoseLoaderClipsHOT3D._CAM_TYPE_CLIP_RANGES.items():
            # First, all of the cam-only 
            if include_w2c:
                ret += [(x, 0, False) for x in range(rnge[0], rnge[1] + 1)]
            if include_o2c or include_o2w:
                bool_opts = []
                if include_o2c:
                    bool_opts.append(False)
                if include_o2w:
                    bool_opts.append(True)
                obj_enumeration = enumerate(
                    PoseLoaderClipsHOT3D._ALL_CLIP_OBJS[cam]
                )
                for clip_offset, objs in obj_enumeration:
                    clip = clip_offset + rnge[0]
                    for b in bool_opts:
                        ret += [(clip, obj, b) for obj in objs if obj > 0]
        return ret
    
    @staticmethod
    def _np_load_single(arr: NDArray, multiplier: float = 1.0):
        ac = arr.copy()
        if multiplier != 1.0:
            ac *= multiplier
        ac.setflags(write=False)
        return ac
    
    @staticmethod
    def _np_load_sep(arr_load, base_str: str):
        ts = PoseLoaderClipsHOT3D._np_load_single(
            arr_load[base_str + "translations"], 1000.0
        )
        mats = PoseLoaderClipsHOT3D._np_load_single(
            arr_load[base_str + "rmats"]
        )
        aas = PoseLoaderClipsHOT3D._np_load_single(
            arr_load[base_str + "axisangles"]
        )
        return SepPoseComponents(ts, mats, aas)
    
    @classmethod
    def _setPosePathsFromJSON(cls, json_read_result):
        if PoseLoaderClipsHOT3D._dir_paths_initialized:
            return
        
        data_dir = pathlib.Path(
            json_read_result["hot3d_dataset_directory"]
        )
        PoseLoaderClipsHOT3D._DATASET_DIR = data_dir

        # PoseLoaderClipsHOT3D._CV_POSE_EXPORT_DIR = pathlib.Path(
        #     json_read_result["hot3d_result_directory"]
        # )

        # Because all of the dataset pose info is stored in just a few *sorta*
        # small files, it makes more sense for efficiency to store those
        # entire files in memory than to keep opening/closing the same one over
        # and over again. 
        for ct in PoseLoaderClipsHOT3D.CAM_TYPE:
            pose_path = data_dir / PoseLoaderClipsHOT3D._POSE_FNAMES[ct]
            with np.load(pose_path, allow_pickle=False) as f:
                PoseLoaderClipsHOT3D._ALL_CLIP_OBJS[ct] = \
                    PoseLoaderClipsHOT3D._np_load_single(f['obj_ids'])
                PoseLoaderClipsHOT3D._ALL_CLIP_BASE_INDS[ct] = \
                    PoseLoaderClipsHOT3D._np_load_single(f[
                        'moving_data_base_ind_per_clip'
                    ])     
                PoseLoaderClipsHOT3D._ALL_INV_CAM_POSES[ct] = \
                    PoseLoaderClipsHOT3D._np_load_sep(f, 'w2c_')
                PoseLoaderClipsHOT3D._ALL_OBJ_CAM_POSES[ct] = \
                    PoseLoaderClipsHOT3D._np_load_sep(f, 'o2c_')
                PoseLoaderClipsHOT3D._ALL_OBJ_WORLD_POSES[ct] = \
                    PoseLoaderClipsHOT3D._np_load_sep(f, 'o2w_')     
            f.close()

            times_path = data_dir / PoseLoaderClipsHOT3D._TIMESTAMP_FNAMES[ct]
            clip_range = PoseLoaderClipsHOT3D._CAM_TYPE_CLIP_RANGES[ct]
            clip_nums_copy = None
            all_ts_copy = None
            with np.load(times_path, allow_pickle=False) as np_load:
                clip_nums_copy = PoseLoaderClipsHOT3D._np_load_single(
                    np_load['clip_nums']
                )
                all_ts_copy = PoseLoaderClipsHOT3D._np_load_single(
                    np_load['timestamps']
                )
            np_load.close()
            for row, clipn in enumerate(clip_nums_copy):
                if clipn >= clip_range[0] and clipn <= clip_range[1]:
                    PoseLoaderClipsHOT3D._ALL_TIMESTAMPS[clipn] = all_ts_copy[row]
                        
        PoseLoaderClipsHOT3D._dir_paths_initialized = True

    def _getPosesFromDisk(self):
        # print("Key:", self._clip_num, ". Pose path:", self.posePathGT)

        self._timestamps = \
            PoseLoaderClipsHOT3D._ALL_TIMESTAMPS[self._clip_num]

        poses_to_sel_from: typing.Optional[SepPoseComponents] = None
        self._ind_in_loaded = self._local_clip_ind
        if self._obj_num == 0:
            poses_to_sel_from = \
                PoseLoaderClipsHOT3D._ALL_INV_CAM_POSES[self.clip_type]
        else:
            pose_ind = \
                PoseLoaderClipsHOT3D._ALL_CLIP_BASE_INDS[self.clip_type][self._local_clip_ind]
            objs_for_clip = \
                PoseLoaderClipsHOT3D._ALL_CLIP_OBJS[self.clip_type][self._local_clip_ind]
            sub_ind = 0
            for obj_num in objs_for_clip:
                if obj_num > 0 and obj_num != self._obj_num:
                    sub_ind += 1
                elif obj_num == self._obj_num:
                    break
            pose_ind += sub_ind
            self._ind_in_loaded = pose_ind

            if self._ignore_cam:
                poses_to_sel_from = \
                    PoseLoaderClipsHOT3D._ALL_OBJ_CAM_POSES[self.clip_type]
            else:
                poses_to_sel_from = \
                    PoseLoaderClipsHOT3D._ALL_OBJ_WORLD_POSES[self.clip_type]

        translations = poses_to_sel_from.translations[self._ind_in_loaded]
        mats = poses_to_sel_from.rot_mats[self._ind_in_loaded]
        aas = poses_to_sel_from.rot_aas[self._ind_in_loaded]
        gtMatData = _LoadedPoses(translations, mats, aas)
        calcMatData = None

        return _LoadedData(gtMatData, calcMatData)
    