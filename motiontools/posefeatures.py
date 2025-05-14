from enum import Enum
import typing
import copy

from dataclasses import dataclass
from math import floor

import numpy as np
from numpy.typing import NDArray

from joblib import Parallel, delayed
import joblib

import posemath as pm
import poseextrapolation as pex
import minjerk as mj

import gtCommon as gtc

DEFAULT_OBJ_STATIC_THRESH_MM = 10.0 # 10 millimeters; semi-arbitrary
DEFAULT_STRAIGHT_ANG_THRESH_DEG = 30.0
DEFAULT_MIN_JERK_OPT_ITER_LIM = 33
DEFAULT_SPLIT_MIN_JERK_OPT_ITER_LIM = 33
DEFAULT_ERR_RADIUS_RATIO_THRESH = 0.10
FLOAT_32_MAX = np.finfo(np.float32).max


class MOTION_MODEL(Enum):
    STATIC = 1
    VEL_DEG1 = 2
    VEL_DEG2 = 3
    ACC_DEG2 = 4
    JERK = 5
    CIRC_VEL_DEG1 = 6
    CIRC_VEL_DEG2 = 7
    CIRC_ACC = 8
    MIN_JERK = 9
    MIN_JERK_SPLIT = 10


class MOTION_DATA(Enum):
    SPEED_DEG1 = 1
    SPEED_DEG2 = 2
    ACC_VEC3 = 3
    LAST_BEST_LABEL = 4
    TIMESTEP = 5

    JERK_VEC3 = 6
    CIRC_RAD = 7
    CIRC_SPEED = 8
    CIRC_ACC = 9
    CIRC_VEL_DEG1_ERR_VEC3 = 10
    CIRC_VEL_DEG2_ERR_VEC3 = 11
    CIRC_ACC_ERR_VEC3 = 12
    JERK_ERR_VEC3 = 13
    AX3_SQ_DIFF = 14
    ROTATION_VEC3 = 15
    CIRC_ANG_SPEED = 16
    CIRC_ANG_ACC = 17

    DISP_MAG_DIFF = 18
    DISP_MAG_DIFF_TIMESCALED = 19
    DISP_MAG_RATIO = 20
    BOUNCE_ANGLE = 21

    UNIT_ROT_AX_DIFF = 22
    UNIT_ROT_AX_DIFF_TIMESCALED = 23

    RAD_DIFF = 24
    TIMESCALED_RAD_DIFF = 25

    # Norms of vec6s composed of circle centres and radii-scaled normals.
    CIRC_VEC6_DIFF = 26

    TIME_SINCE_STATIONARY = 27
    TIME_SINCE_DIR_CHANGE = 28
    DIST_SINCE_DIR_CHANGE = 29

    TIME_CIRC_MOTION = 30
    ANG_SUM_CIRC_MOTION = 31
    DIST_SUM_CIRC_MOTION = 32

    CIRC_CENTRE_DIFF = 33

    ROT_ACC_VEC3 = 34
    FRAME_NUM = 35
    TIMESTAMP = 36

    PREV_FA_ANG_ACC = 37
    NEXT_FA_ANG_ACC = 38

    SPEED_ACC_RATIO = 39
    VEL_BCS_RATIOS = 40

    ORTHO_ACC_MAG = 41

    VEL_DOT = 42 # Units match work per kg (J/kg)

    SPEED_JERK_RATIO = 43
    ACC_JERK_RATIO = 44
    SPEED_ORTHO_ACC_RATIO = 45
    CIRC_ACC_CIRC_SPEED_RATIO = 46
    CIRC_ANG_ACC_CIRC_ANG_SPEED_RATIO = 47
    CIRC_ANG_RATIO = 48

    BOUNCE_ANGLE_2_SUM = 49

    ACC_VEL_DEG2_ERR_RATIO = 50

    PLANE_NORMAL_DOT = 51

    DIST_FROM_CIRCLE = 52
    RATIO_FROM_CIRCLE = 53

    ACC_VEL_DEG1_DOT = 54
    ACC_VEL_DEG2_DOT = 55
    VEL_DEG2_MAG_DIFF = 56
    VEL_DEG2_MAG_DIFF_TIMESCALED = 57

    INV_DISP_MAG_RATIO = 58
    INV_VEL_BCS_RATIOS = 59
    INV_CIRC_ANG_RATIO = 60

    # GT0 = 61
    # GT1 = 62
    # GT2 = 63
    # GT3 = 64
    # GT4 = 65
    # GT5 = 66

    JERK_VEL_DEG1_DOT = 67
    JERK_VEL_DEG2_DOT = 68
    JERK_ACC_DOT = 69

class RELATIVE_AXIS(Enum):
    VEL_DEG1 = 1
    VEL_DEG2 = 2
    ACC_FULL = 3
    ACC_ORTHO_DEG1 = 4
    PLANE_ORTHO = 5
    ROTATION = 6
    ITSELF = 7

class ANG_OR_MAG(Enum):
    ANG = 1
    MAG = 2

class SpecifiedMotionData(typing.NamedTuple):
    base_cat: MOTION_DATA
    axis: RELATIVE_AXIS
    ang_or_mag: ANG_OR_MAG
    bidirectional: bool
    is_timestep_shifted: bool

    @property
    def name(self):
        ret_name = ""
        bn = self.base_cat.name 
        last_underscore_ind = bn.rfind("_")
        if bn[last_underscore_ind:] != "_VEC3":
            raise ValueError("No \"VEC3\" found in base type {}!".format(bn))
        
        ret_name = bn[:last_underscore_ind] 

        if self.axis != RELATIVE_AXIS.ITSELF:
            ret_name += "_" + self.axis.name

        ret_name += "_" + self.ang_or_mag.name

        if self.bidirectional:
            ret_name += "_BIDIR"
        if self.is_timestep_shifted:
            ret_name += "_SHIFT"
        return ret_name

class Vec3Data:
    def __init__(self, vecs = None, unit_vecs = None, norms = None):
        if vecs is None:
            if unit_vecs is None or norms is None:
                raise ValueError("Not enough info to reconstruct scaled vec3s!")
            self.norms = norms.flatten()
            self.unit_vecs = unit_vecs
            self.vecs = pm.scalarsVecsMul(self.norms, unit_vecs)
        else:
            self.vecs = vecs
            self.unit_vecs = unit_vecs
            if norms is None:
                self.norms = np.linalg.norm(vecs, axis=-1)
            else:
                self.norms = norms.flatten()
            
            if unit_vecs is None:                
                self.unit_vecs = pm.safelyNormalizeArray(
                    vecs, self.norms.reshape(-1,1)
                )
            # Note: Do NOT want to call normalization function with param
            # that back-propagates axis dir if first axes are 0, since that 
            # would not be a real option at runtime (requires knowing future)!
            # TODO: Should maybe make this NaN later (as well as some of the
            # things like "time since stationary" at a video start) and then
            # make separate trees based on whether or not those attributes
            # are "available"?
            if self.norms[0] == 0.0:
                if np.any(self.vecs[0] != 0):
                    raise Exception("Norm and scaled vec 0-len inconsisitency!")
                self.unit_vecs[0] = 0.0
                # MAYBE setting unit dir to 0 is a workaround if this happens?

MOTION_DATA_KEY_TYPE = typing.Union[MOTION_DATA, SpecifiedMotionData]
PoseLoaderList = typing.List[gtc.PoseLoader]

# Finds closest points on hyperplanes with the given normals and offsets.
# The below function is the result of me feeding my original getClosestPoint()
# function through Copilot/Claude to accomodate multiple points per hyperplane.
# TODO: Need to manually verify logic and clean things up a bit.
def getClosestPoint(normals: NDArray, scaled_plane_offsets: NDArray, points: NDArray, return_dists: bool = False):
    """
    Parameters:
        normals: shape (m, k) where m is number of hyperplanes, k is dimensionality
        scaled_plane_offsets: shape (m,) offset for each hyperplane
        points: shape (n, k) points to find closest hyperplane points for
        return_dists: whether to return distances along with closest points
    """
    # Reshape for broadcasting:
    # normals: (m, k, 1)
    # scaled_plane_offsets: (m, 1)
    # points: (1, k, n)
    normals_exp = normals.reshape(normals.shape[0], normals.shape[1], 1)
    offsets_exp = scaled_plane_offsets.reshape(-1, 1)
    points_exp = points.T.reshape(1, points.shape[1], points.shape[0])
    
    # Calculate dot products: (m, 1, n)
    norm_sq = pm.einsumDot(normals, normals).reshape(-1, 1, 1)
    pn = np.sum(normals_exp * points_exp, axis=1, keepdims=True)  # (m, 1, n)
    
    # Calculate scalars: (m, 1, n)
    scalar = (offsets_exp.reshape(-1, 1, 1) - pn) / norm_sq
    
    # Calculate displacements: (m, k, n)
    disps = scalar * normals_exp
    
    # Calculate closest points: (m, k, n)
    closest = points_exp + disps
    
    if return_dists:
        # Calculate distances: (m, n)
        distances = np.sqrt(np.sum(disps * disps, axis=1))
        return (closest.transpose(2, 1, 0),  # (n, k, m)
                distances.T)                  # (n, m)
    return closest.transpose(2, 1, 0)        # (n, k, m)

# The below function is the result of me feeding my original gtMultipliers6()
# function through Copilot/Claude to accomodate multiple baseline_m6 values.
# TODO: Need to manually verify logic and clean things up a bit.
def gtMultipliers6(y_true: NDArray, baseline_m6: NDArray):
    """
    Parameters:
        y_true: shape (m, 15) where m is number of data points
        baseline_m6: shape (n, 6) where n is number of baseline points to consider
    """
    # Calculate j_o directly as it doesn't depend on baseline_m6
    j_o = y_true[:, 8] / y_true[:, 5]
    
    # First hyperplane (a and j components)
    points_aj = baseline_m6[:, [2, 4]]  # Shape: (n, 2)
    closest_points_aj, dists_aj = getClosestPoint(
        y_true[:, [2, 4]], y_true[:, 7], points_aj, return_dists=True
    )
    # For each hyperplane (each row in y_true), find the closest among n points
    min_indices_aj = np.argmin(dists_aj, axis=0)  # Shape: (m,)
    # Get the closest points using the indices
    # closest_points_aj shape is (n, 2, m), we want to select best n for each m
    a_a = closest_points_aj[min_indices_aj, 0, range(len(y_true))]
    j_a = closest_points_aj[min_indices_aj, 1, range(len(y_true))]
    
    # Second hyperplane (v, a, and j components)
    points_vaj = baseline_m6[:, [0, 1, 3]]  # Shape: (n, 3)
    closest_points_vaj, dists_vaj = getClosestPoint(
        y_true[:, [0, 1, 3]], y_true[:, 6], points_vaj, return_dists=True
    )
    # For each hyperplane, find the closest among n points
    min_indices_vaj = np.argmin(dists_vaj, axis=0)  # Shape: (m,)
    # Get the closest points using the indices
    # closest_points_vaj shape is (n, 3, m), we want to select best n for each m
    v_v = closest_points_vaj[min_indices_vaj, 0, range(len(y_true))]
    a_v = closest_points_vaj[min_indices_vaj, 1, range(len(y_true))]
    j_v = closest_points_vaj[min_indices_vaj, 2, range(len(y_true))]
    
    return np.stack([v_v, a_v, a_a, j_v, j_a, j_o], axis=-1)


def getMissingMotionDataKeys(keys: typing.List[MOTION_DATA_KEY_TYPE]):
        missing_keys: typing.List[MOTION_DATA] = []
        for motion_data_kind in MOTION_DATA:
            md_kind_present = False
            for k in keys:
                if isinstance(k, MOTION_DATA):
                    md_kind_present = md_kind_present or k == motion_data_kind
                elif isinstance(k, SpecifiedMotionData):
                    base_cat_eq = k.base_cat == motion_data_kind
                    md_kind_present = md_kind_present or base_cat_eq
                else:
                    raise ValueError("Bad key type! {}".format(k))
                
                if md_kind_present:
                    break

            if not md_kind_present:
                missing_keys.append(motion_data_kind)

        return missing_keys

@dataclass
class FeaturesAndResultsForVid:
    motion_data: typing.List[typing.Dict[MOTION_DATA_KEY_TYPE, NDArray]]
    err_norms: typing.List[typing.Dict[MOTION_MODEL, NDArray]]
    min_norm_labels: typing.List[NDArray]
    min_norm_vecs: typing.List[NDArray]
    # err3D_lists[skip_amt][c2] = curr_errs_3D

class CalcsForVideo:
    def __init__(self, 
                 obj_static_thresh_mm: float = DEFAULT_OBJ_STATIC_THRESH_MM,
                 straight_angle_thresh_deg: float = DEFAULT_STRAIGHT_ANG_THRESH_DEG,
                 err_na_val: float = FLOAT_32_MAX,
                 min_jerk_opt_iter_lim: int = DEFAULT_MIN_JERK_OPT_ITER_LIM,
                 split_min_jerk_opt_iter_lim: int = DEFAULT_SPLIT_MIN_JERK_OPT_ITER_LIM,
                 err_radius_ratio_thresh: float = DEFAULT_ERR_RADIUS_RATIO_THRESH
                 ):
        self.obj_static_thresh_mm = obj_static_thresh_mm
        self.straight_angle_thresh_rad = np.deg2rad(straight_angle_thresh_deg)
        self.min_jerk_opt_iter_lim = min_jerk_opt_iter_lim
        self.split_min_jerk_opt_iter_lim = split_min_jerk_opt_iter_lim
        self.err_radius_ratio_thresh = err_radius_ratio_thresh
        self.err_na_val = err_na_val

        self.motion_mod_keys = [
            MOTION_MODEL(i) for i in range(1, len(MOTION_MODEL) + 1)
        ]

        # I had to accomodate a venv where I have Python 3.7 for running
        # tensorflow-gpu on Windows. Unfortunately, newer joblib versions 
        # require Python 3.8. So that meant downgrading joblib to an older 
        # version (1.2), but said version did not have `parallel_config`, which
        # I think is probably a good idea to use when possible. So I wrote some 
        # code to use it if the joblib version is new enough, but still allow 
        # the old joblib for that one venv.
        joblib_vers = tuple((int(v) for v in joblib.__version__.split(".")))
        joblib_1_3_supported = False
        joblib_between_1_3__2_0 = joblib_vers[0] == 1 and joblib_vers[1] >= 3
        # Test if joblib version >= 1.3
        if joblib_vers[0] > 1 or joblib_between_1_3__2_0:
            joblib_1_3_supported = True
        self._joblib_1_3_supported = joblib_1_3_supported
        # End of constructor.
        
    def getAll(self, pose_loaders: PoseLoaderList, num_procs: int = -1,
               max_threads_per_proc = 2):
        
        results = None

        # For the 1st combo, check to make sure all keys are there.
        # This way we can stop a lot sooner if we are missing a key.
        result_for_1st = self.getInputFeatures(
            pose_loaders[0], True
        )[1]

        remaining_loaders = pose_loaders[1:]

        if num_procs == 1:
            results = dict(map(self.getInputFeatures, remaining_loaders))
        else: 
            cpu_count = joblib.cpu_count()
            # If caller did not specify how many children processes...
            if num_procs <= 0:
                num_procs = 1
                # Excepting the (rare!) case where our device only has 1 CPU...
                if cpu_count > 1:
                    # ... we want at least 2 children, and ideally we'd have
                    # as many children as possible to fill up our cores while
                    # still letting each child use multiple CPUs, as the joblib
                    # docs specify that the "inner_max_num_threads" specifies
                    # the number of threads that calls to numpy/BLAS or other
                    # similar libraries can use within each child process.
                    num_procs = max(2, cpu_count // max_threads_per_proc)
            # Ensure we do not exceed our CPU count in total.
            max_available_per = int(floor(cpu_count / num_procs))
            per = min(max_threads_per_proc, max_available_per)
            # We want the loky backend to get true parallelism without worrying
            # about the headaches the multiparallelism backend creates on
            # Windows, where we need __name__ == "__main__" checks and whatnot.
            if self._joblib_1_3_supported:
                print("Running {} processes each with {} threads.".format(
                    num_procs, per
                ))
                with joblib.parallel_config(backend="loky", inner_max_num_threads=per):
                    results_list = Parallel(n_jobs=num_procs)(
                        delayed(self.getInputFeatures)(c)
                        for c in remaining_loaders
                    )
                    results = dict(results_list)
            else:
                print("Running {} processes.".format(num_procs))
                results_list = Parallel(n_jobs=num_procs)(
                    delayed(self.getInputFeatures)(c) for c in remaining_loaders
                )
                results = dict(results_list)
        results[pose_loaders[0].getVidID()] = result_for_1st

        self.all_motion_data = [dict() for _ in range(3)]
        self.min_norm_labels = [dict() for _ in range(3)]
        self.err_norm_lists = [dict() for _ in range(3)]
        self.min_norm_vecs = [dict() for _ in range(3)]
        combos = [pl.getVidID() for pl in pose_loaders]
        for i in range(3):
            for combo in combos:
                res = results[combo]
                self.all_motion_data[i][combo] = res.motion_data[i]
                self.err_norm_lists[i][combo] = res.err_norms[i]
                self.min_norm_labels[i][combo] = res.min_norm_labels[i]
                self.min_norm_vecs[i][combo] = res.min_norm_vecs[i]

    @staticmethod
    def _addDotsAndAngs(dict_to_update: typing.Dict[MOTION_DATA_KEY_TYPE, NDArray],
               motion_data_base_key: MOTION_DATA, relative_axis: RELATIVE_AXIS,
               shift: bool, dots_with_unit_axis: NDArray, vec_norms: NDArray):
        SMD = SpecifiedMotionData
        AM = ANG_OR_MAG
        km      = SMD(motion_data_base_key, relative_axis, AM.MAG, False, shift)
        kmbidir = SMD(motion_data_base_key, relative_axis, AM.MAG, True, shift)
        ka      = SMD(motion_data_base_key, relative_axis, AM.ANG, False, shift)
        kabidir = SMD(motion_data_base_key, relative_axis, AM.ANG, True, shift)

        dict_to_update[km] = dots_with_unit_axis
        dict_to_update[kmbidir] = np.abs(dots_with_unit_axis)
        
        curr_angs = np.arccos(np.clip(dots_with_unit_axis / vec_norms, -1, 1))
        if vec_norms[0] == 0.0:
            curr_angs[0] = 0.0
        dict_to_update[ka] = curr_angs
        dict_to_update[kabidir] = pm.getAcuteAngles(curr_angs)


    def getInputFeatures(self, pose_loader: gtc.PoseLoader,
                         check_key_completeness: bool = False):
        # The below code will use MOTION_DATA_KEY_TYPE classes so often that some
        # shorter aliases might be helpful.
        MD = MOTION_DATA
        SMD = SpecifiedMotionData
        RA = RELATIVE_AXIS
        AM = ANG_OR_MAG
        V3D = Vec3Data

        
        all_translations = pose_loader.getTranslationsGTNP()
        aa_rotations = pose_loader.getRotationsGTNP()
        all_quats = pm.quatsFromAxisAngleVec3s(aa_rotations)
        all_rotation_mats = pose_loader.getRotationMatsGTNP()

        motion_datas = []
        all_err_norms = []
        min_err_labels = []
        min_err_vecs = []

        for step in range(1, 4):
            step_sq = step*step
            translations = all_translations[::step]

            quats = all_quats[::step][:-1]
            inv_quats = pm.conjugateQuats(quats)
            quat_diffs = pm.multiplyQuatLists(quats[1:], inv_quats[:-1])

            vel_axes, vel_angs_unflat = pm.axisAnglesFromQuats(quat_diffs)
            vel_angs = vel_angs_unflat.flatten()
            vel_angs_timescaled = vel_angs / step
            timescaled_vel_axes = pm.scalarsVecsMul(vel_angs_timescaled, vel_axes)


            translation_diffs = np.diff(translations, 1, axis=0)
            prev_translations = translations[:-1]
            n_input_frames = len(prev_translations)
            deg1_vels = translation_diffs[:-1]
            deg1_vel_diffs = np.diff(deg1_vels, 1, axis=0)
            half_deg1_vel_diffs = deg1_vel_diffs / 2.0
            
            deg2_vels = deg1_vels[1:] + half_deg1_vel_diffs

            t_jerk_preds = 4 * prev_translations[3:] - 6 * prev_translations[2:-1] \
                + 4 * prev_translations[1:-2] - prev_translations[:-3]

            t_jerk_amt = prev_translations[3:] - 3*deg1_vels[1:-1] - prev_translations[:-3]

            cma = pex.CircularMotionAnalysis(
                translations, translation_diffs, None
            )

            temp_preds = dict()
            temp_preds[MOTION_MODEL.STATIC] = prev_translations
            temp_preds[MOTION_MODEL.VEL_DEG1] = prev_translations[1:] + deg1_vels
            temp_preds[MOTION_MODEL.VEL_DEG2] = prev_translations[2:] + deg2_vels
            temp_preds[MOTION_MODEL.ACC_DEG2] = \
                temp_preds[MOTION_MODEL.VEL_DEG2] + half_deg1_vel_diffs
            temp_preds[MOTION_MODEL.JERK] = t_jerk_preds
            temp_preds[MOTION_MODEL.CIRC_VEL_DEG1] = cma.vel_deg1_preds_3D
            temp_preds[MOTION_MODEL.CIRC_VEL_DEG2] = cma.vel_deg2_preds_3D
            temp_preds[MOTION_MODEL.CIRC_ACC] = cma.acc_preds_3D

            n_jerk_preds = len(t_jerk_preds)

            motion_data = dict()
            motion_data[MOTION_DATA.TIMESTEP] = np.full(n_jerk_preds, step)

            unit_rot_ax_diffs = np.diff(vel_axes, 1, axis=0)[-n_jerk_preds:]
            unit_rot_ax_diff_mags = np.linalg.norm(unit_rot_ax_diffs, axis=-1)

            motion_data[MOTION_DATA.UNIT_ROT_AX_DIFF] = unit_rot_ax_diff_mags
            motion_data[MOTION_DATA.UNIT_ROT_AX_DIFF_TIMESCALED] = \
                                                        unit_rot_ax_diff_mags / step

            deg1_speeds_full = np.linalg.norm(deg1_vels, axis=-1, keepdims=True)
            deg1_speeds = deg1_speeds_full[-n_jerk_preds:].flatten()
            deg2_speeds_full = np.linalg.norm(deg2_vels, axis=-1, keepdims=True)
            deg2_speeds = deg2_speeds_full[-n_jerk_preds:]
            timescaled_speeds = deg2_speeds.flatten() / step
            motion_data[MOTION_DATA.SPEED_DEG1] = deg1_speeds / step
            motion_data[MOTION_DATA.SPEED_DEG2] = timescaled_speeds


            deg2_accs = deg1_vel_diffs / step_sq
            acc_mags_full = np.linalg.norm(deg2_accs, axis=-1, keepdims=True)
            acc_mags = acc_mags_full[-n_jerk_preds:].flatten()

            motion_data[MOTION_DATA.SPEED_ACC_RATIO] = timescaled_speeds / acc_mags
            vel_dots = pm.einsumDot(
                deg1_vels[-n_jerk_preds:], deg1_vels[-(n_jerk_preds + 1):-1]
            )
            vel_bcs_ratios = vel_dots / (deg1_speeds[-n_jerk_preds:]**2)
            motion_data[MOTION_DATA.VEL_BCS_RATIOS] = vel_bcs_ratios
            motion_data[MOTION_DATA.INV_VEL_BCS_RATIOS] = 1.0 / vel_bcs_ratios

            disp_mag_diffs = np.diff(deg1_speeds_full, 1, axis=0)[-n_jerk_preds:].flatten()
            motion_data[MOTION_DATA.DISP_MAG_DIFF] = disp_mag_diffs
            motion_data[MOTION_DATA.DISP_MAG_DIFF_TIMESCALED] = disp_mag_diffs / step
            speed_deg2_diffs = np.diff(deg2_speeds_full, 1, axis=0)[-n_jerk_preds:].flatten()
            motion_data[MOTION_DATA.VEL_DEG2_MAG_DIFF] = speed_deg2_diffs
            motion_data[MOTION_DATA.VEL_DEG2_MAG_DIFF_TIMESCALED] = speed_deg2_diffs / step
            disp_mag_div = deg1_speeds_full[1:] / deg1_speeds_full[:-1]
            motion_data[MOTION_DATA.DISP_MAG_RATIO] = disp_mag_div[-n_jerk_preds:].flatten()
            motion_data[MOTION_DATA.INV_DISP_MAG_RATIO] = 1.0 / disp_mag_div[-n_jerk_preds:].flatten()

            unit_vels_deg1 = pm.safelyNormalizeArray(deg1_vels, deg1_speeds_full)
            unit_vels_deg2 = pm.safelyNormalizeArray(deg2_vels, deg2_speeds_full)
            unit_accs = pm.safelyNormalizeArray(deg2_accs, acc_mags_full)

            radii = cma.getRadii()
            motion_data[MOTION_DATA.CIRC_RAD] = radii[-n_jerk_preds:]

            circ_ang_speeds = cma.second_angles / step
            prev_circ_ang_speeds = cma.first_angles / step
            radii_subset = radii[-n_jerk_preds:]
            circ_ang_speeds_subset = circ_ang_speeds[-n_jerk_preds:]
            circ_speeds = radii_subset * circ_ang_speeds_subset
            motion_data[MOTION_DATA.CIRC_SPEED] = circ_speeds
            motion_data[MOTION_DATA.CIRC_ANG_SPEED] = circ_ang_speeds_subset
            prev_circ_ang_speeds_subset = prev_circ_ang_speeds[-n_jerk_preds:]
            circ_accs = (circ_speeds - (radii_subset * prev_circ_ang_speeds_subset)) / step
            circ_ang_accs = circ_ang_speeds - prev_circ_ang_speeds
            motion_data[MOTION_DATA.CIRC_ACC] = circ_accs
            motion_data[MOTION_DATA.CIRC_ANG_ACC] = circ_ang_accs[-n_jerk_preds:]

            motion_data[MOTION_DATA.CIRC_ACC_CIRC_SPEED_RATIO] = circ_accs / circ_speeds
            motion_data[MOTION_DATA.CIRC_ANG_ACC_CIRC_ANG_SPEED_RATIO] = \
                circ_ang_accs[-n_jerk_preds:] / circ_ang_speeds_subset
            c_ang_ratio = cma.second_angles / cma.first_angles
            motion_data[MOTION_DATA.CIRC_ANG_RATIO] = c_ang_ratio[-n_jerk_preds:]
            motion_data[MOTION_DATA.INV_CIRC_ANG_RATIO] = 1.0 / c_ang_ratio[-n_jerk_preds:]
            

            motion_data[MOTION_DATA.CIRC_VEC6_DIFF] = cma.vec6CircleDists()

            radii_diffs = np.diff(radii, 1, axis=0)[-n_jerk_preds:]
            motion_data[MOTION_DATA.RAD_DIFF] = radii_diffs
            motion_data[MOTION_DATA.TIMESCALED_RAD_DIFF] = radii_diffs / step

            rotation_mats = all_rotation_mats[::step]
            rot_accs = np.diff(timescaled_vel_axes, 1, axis=0) / step
            prev_fixed_ax_angs = pm.closestAnglesAboutAxis(
                rotation_mats[1:-2], rotation_mats[2:-1], vel_axes[:-1]
            )
            next_fixed_ax_angs = pm.closestAnglesAboutAxis(
                rotation_mats[:-3], rotation_mats[1:-2], vel_axes[1:]
            )

            prev_fa_ang_accs = (vel_angs[1:] - prev_fixed_ax_angs) / step_sq
            next_fa_ang_accs = (next_fixed_ax_angs - vel_angs[:-1]) / step_sq
            motion_data[MOTION_DATA.PREV_FA_ANG_ACC] = prev_fa_ang_accs[-n_jerk_preds:]
            motion_data[MOTION_DATA.NEXT_FA_ANG_ACC] = next_fa_ang_accs[-n_jerk_preds:]

            xyz_axes = prev_translations.reshape(-1,3,1) + rotation_mats[:-1]
            n_xyz_axes = prev_translations.reshape(-1,3,1) + rotation_mats[:-1]

            mat_diffs = xyz_axes[1:] - xyz_axes[:-1]
            n_mat_diffs = n_xyz_axes[1:] - n_xyz_axes[:-1]
            vec9s = mat_diffs.reshape(-1,9)[-n_jerk_preds:]
            n_vec9s = n_mat_diffs.reshape(-1,9)[-n_jerk_preds:]
            vec18s = np.concatenate((vec9s, n_vec9s), axis=1)
            motion_data[MOTION_DATA.AX3_SQ_DIFF] = pm.einsumDot(vec18s, vec18s)

            c_centre_diff_vecs = np.diff(cma.getCentres3D(), axis=0)
            c_centre_diff_norms = np.linalg.norm(c_centre_diff_vecs, axis=-1)
            motion_data[MOTION_DATA.CIRC_CENTRE_DIFF] = c_centre_diff_norms

            
            t_diff_angs = pm.anglesBetweenVecs(
                unit_vels_deg1[:-1], unit_vels_deg1[1:], False
            )
            t_diff_dots = pm.einsumDot(deg1_speeds_full[:-1], deg1_speeds_full[1:])
            motion_data[MOTION_DATA.BOUNCE_ANGLE] = t_diff_angs[-n_jerk_preds:]
            bounce_ang_pair_sums = t_diff_angs[1:] + t_diff_angs[:-1]
            motion_data[MOTION_DATA.BOUNCE_ANGLE_2_SUM] = \
                bounce_ang_pair_sums[-n_jerk_preds:]
            motion_data[MOTION_DATA.VEL_DOT] = t_diff_dots[-n_jerk_preds:]

            avd1m = pm.einsumDot(deg2_accs, deg1_vels[1:])
            motion_data[MOTION_DATA.ACC_VEL_DEG1_DOT] = avd1m[-n_jerk_preds:]
            avd2m = pm.einsumDot(deg2_accs, deg2_vels)
            motion_data[MOTION_DATA.ACC_VEL_DEG2_DOT] = avd2m[-n_jerk_preds:] 

            motion_data[MOTION_DATA.JERK_VEL_DEG1_DOT] = pm.einsumDot(
                t_jerk_amt, deg1_vels[2:]
            )
            motion_data[MOTION_DATA.JERK_VEL_DEG2_DOT] = pm.einsumDot(
                t_jerk_amt, deg2_vels[1:]
            )
            motion_data[MOTION_DATA.JERK_ACC_DOT] = pm.einsumDot(
                t_jerk_amt, deg2_accs[1:]
            )


            d_under_thresh = deg1_speeds_full < self.obj_static_thresh_mm
            a_over_thresh = t_diff_angs > self.straight_angle_thresh_rad
        
            mj_preds = None
            acc_preds = temp_preds[MOTION_MODEL.ACC_DEG2]
            if self.min_jerk_opt_iter_lim > 0:
                mj_preds = mj.min_jerk_lsq(
                    prev_translations, d_under_thresh.flatten(), 
                    a_over_thresh.flatten(),
                    max_opt_iters=self.min_jerk_opt_iter_lim,
                    vels = deg1_vels, accs = deg2_accs, jerks = t_jerk_amt
                )
                mj_preds = mj_preds[-len(acc_preds):]
                mj_na = np.isnan(mj_preds)[:, 0]
                mj_preds[mj_na] = acc_preds[mj_na]

            temp_preds[MOTION_MODEL.MIN_JERK] = mj_preds


            mj_split_preds = None
            if self.split_min_jerk_opt_iter_lim > 0:
                _ds_under_thresh = deg1_vels < self.obj_static_thresh_mm
                _vel_deg1_signs = np.sign(deg1_vels)
                _as_over_thresh = _vel_deg1_signs[1:] != _vel_deg1_signs[:-1]

                split_mj_pred_list = []
                for mjsi in range(3):
                    split_mj_preds_i = mj.min_jerk_lsq(
                        prev_translations[:, mjsi:(mjsi + 1)], 
                        _ds_under_thresh[:, mjsi], _as_over_thresh[:, mjsi],
                        max_opt_iters = self.split_min_jerk_opt_iter_lim
                    )
                    split_mj_pred_list.append(split_mj_preds_i)
                mj_split_preds = np.concatenate(split_mj_pred_list, axis=-1)
                mj_split_preds = mj_split_preds[-len(acc_preds):]

                for mjsi in range(3):
                    mj_split_na = np.isnan(mj_split_preds[:, mjsi])
                    mj_split_preds[mj_split_na, mjsi] = acc_preds[mj_split_na, mjsi]

            temp_preds[MOTION_MODEL.MIN_JERK_SPLIT] = mj_split_preds

            _, time_since_static, _ = pm.since_calc(
                d_under_thresh, n_input_frames, [], 0
            )
            # We want the total distance traveled since there was last a big angle. 
            _, time_since_big_ang, (dist_since_big_ang,) = \
                pm.since_calc(a_over_thresh, n_input_frames, [deg1_speeds_full], 1)



            motion_data[MOTION_DATA.TIME_SINCE_STATIONARY] = time_since_static[-n_jerk_preds:] * step
            motion_data[MOTION_DATA.TIME_SINCE_DIR_CHANGE] = time_since_big_ang[-n_jerk_preds:] * step
            motion_data[MOTION_DATA.DIST_SINCE_DIR_CHANGE] = dist_since_big_ang[-n_jerk_preds:]

            
            
            is_circ_res = cma.isMotionStillCircular(
                prev_translations[3:], self.err_radius_ratio_thresh, FLOAT_32_MAX
            )

            _, time_circ, (ang_sum_circ, dist_sum_circ) = pm.since_calc(
                is_circ_res.non_circ_bool_inds, n_input_frames, 
                [cma.second_angles, radii * cma.second_angles], 3
            )

            time_circ = (time_circ + 3) * step

            motion_data[MOTION_DATA.TIME_CIRC_MOTION] = time_circ[-n_jerk_preds:]
            motion_data[MOTION_DATA.ANG_SUM_CIRC_MOTION] = ang_sum_circ[-n_jerk_preds:]
            motion_data[MOTION_DATA.DIST_SUM_CIRC_MOTION] = dist_sum_circ[-n_jerk_preds:]

            frame_nums = np.arange(n_input_frames - n_jerk_preds, n_input_frames)
            motion_data[MOTION_DATA.FRAME_NUM] = frame_nums
            motion_data[MOTION_DATA.TIMESTAMP] = frame_nums * step


            motion_data[MOTION_DATA.DIST_FROM_CIRCLE] = is_circ_res.dists
            motion_data[MOTION_DATA.RATIO_FROM_CIRCLE] = is_circ_res.dist_radius_ratios


            # ======================================================================
            # Calculating prediction errors starts here!

            curr_err_norms = np.empty((len(MOTION_MODEL), n_jerk_preds + 1))

            t_subset = translations[-(n_jerk_preds + 1):] # To calc errors against.

            curr_err_norms_dict = dict()
            curr_errs_3D = dict()
            for i, motion_mod in enumerate(self.motion_mod_keys):
                pred_subset = None
                if motion_mod != MOTION_MODEL.JERK:
                    if temp_preds[motion_mod] is None:
                        tdim = translations.shape[1]
                        pred_subset = np.full(
                            (n_jerk_preds + 1, tdim), FLOAT_32_MAX
                        )
                    else:
                        pred_subset = temp_preds[motion_mod][-(n_jerk_preds + 1):]
                else:
                    pred_subset = np.empty((n_jerk_preds + 1, 3))
                    pred_subset[1:] = temp_preds[motion_mod]
                    pred_subset[0] = np.inf

                errs = t_subset - pred_subset
                curr_err_norms[i] = np.linalg.norm(errs, axis=-1)

                curr_errs_3D[motion_mod] = errs

                curr_err_norms_dict[motion_mod] = curr_err_norms[i, 1:]
            curr_min_norm_labels = np.argmin(curr_err_norms, axis=0).flatten()
            motion_data[MOTION_DATA.LAST_BEST_LABEL] = curr_min_norm_labels[:-1]

            curr_min_keys = [
                self.motion_mod_keys[i] for i in curr_min_norm_labels[1:]
            ]

            curr_min_norm_vecs = np.array([
                curr_errs_3D[k][i + 1] for i, k in enumerate(curr_min_keys)
            ])

            # curr_min_norm_vecs = np.take_along_axis(
            #     curr_errs_3D[1:], curr_min_norm_labels[1:].reshape(-1, 1, 1),
            #     axis=1
            # )
            
            circ_vd1_ind = MOTION_MODEL.CIRC_VEL_DEG1.value - 1
            circ_vd2_ind = MOTION_MODEL.CIRC_VEL_DEG2.value - 1
            circ_acc_ind = MOTION_MODEL.CIRC_ACC.value - 1
            jerk_ind = MOTION_MODEL.JERK.value - 1
            prev_circ_errs_vd1 = curr_errs_3D[MOTION_MODEL.CIRC_VEL_DEG1][:-1]
            prev_circ_errs_vd2 = curr_errs_3D[MOTION_MODEL.CIRC_VEL_DEG2][:-1]
            prev_circ_errs_acc = curr_errs_3D[MOTION_MODEL.CIRC_ACC][:-1]
            prev_circ_err_mags_vd1 = curr_err_norms[circ_vd1_ind][:-1]
            prev_circ_err_mags_vd2 = curr_err_norms[circ_vd2_ind][:-1]
            prev_circ_err_mags_acc = curr_err_norms[circ_acc_ind][:-1]
        

            step_4_pow = step**4
            prev_jerk_errs = curr_errs_3D[MOTION_MODEL.JERK][:-1] / step_4_pow
            prev_jerk_errs[0] = 0 # Overwrite with 0 so it = the "4th derivative" 
            prev_jerk_err_mags = curr_err_norms[jerk_ind][:-1]
            prev_jerk_err_mags[0] = 0
            
            acc_vel_deg1_mag = pm.einsumDot(deg2_accs, unit_vels_deg1[1:])
            acc_vel_deg1_parallel = pm.scalarsVecsMul(acc_vel_deg1_mag, unit_vels_deg1[1:])
            
            acc_ortho_deg1_vecs = deg2_accs - acc_vel_deg1_parallel
            acc_ortho_deg1_mags = np.linalg.norm(
                acc_ortho_deg1_vecs, axis=-1, keepdims=True
            )
            
            unit_acc_ortho_deg1_vecs = pm.safelyNormalizeArray(
                acc_ortho_deg1_vecs, acc_ortho_deg1_mags
            )
            motion_data[MOTION_DATA.ORTHO_ACC_MAG] = acc_ortho_deg1_mags[-n_jerk_preds:].flatten()

            vel_deg2_acc_cross = np.cross(unit_vels_deg2, unit_accs)
            sin_vel_deg2_acc_angs = np.linalg.norm(
                vel_deg2_acc_cross, axis=-1, keepdims=True
            )


            precalced_mags = [
                deg1_speeds_full, deg2_speeds_full, acc_mags_full,
                sin_vel_deg2_acc_angs
            ]
            for mags in precalced_mags:
                if mags[0] == 0.0:
                    raise ValueError("Relative axes cannot have a norm of 0.0!")

            acc_vel_deg2_mag = avd2m / deg2_speeds_full.flatten()
            self._addDotsAndAngs(
                motion_data, MD.ACC_VEC3, RA.VEL_DEG2, False,
                acc_vel_deg2_mag[-n_jerk_preds:], acc_mags.flatten()
            )

            
            ortho_dirs = pm.safelyNormalizeArray(vel_deg2_acc_cross, sin_vel_deg2_acc_angs)

            plane_dots = pm.einsumDot(
                ortho_dirs[-n_jerk_preds:], ortho_dirs[-(n_jerk_preds + 1):-1]
            )
            motion_data[MOTION_DATA.PLANE_NORMAL_DOT] = plane_dots
            

            rot_v3d = Vec3Data(timescaled_vel_axes, vel_axes, vel_angs_timescaled[-n_jerk_preds:])
            vec3s_dict: typing.Dict[MOTION_DATA, Vec3Data] = {
                MD.ACC_VEC3: V3D(deg2_accs, unit_accs, acc_mags),
                MD.JERK_VEC3: V3D(t_jerk_amt / step**3), MD.ROTATION_VEC3: rot_v3d,
                MD.CIRC_VEL_DEG1_ERR_VEC3: V3D(
                    prev_circ_errs_vd1, None, prev_circ_err_mags_vd1
                ),
                MD.CIRC_VEL_DEG2_ERR_VEC3: V3D(
                    prev_circ_errs_vd2, None, prev_circ_err_mags_vd2
                ),
                MD.CIRC_ACC_ERR_VEC3: V3D(
                    prev_circ_errs_acc, None, prev_circ_err_mags_acc
                ),
                MD.JERK_ERR_VEC3: V3D(prev_jerk_errs, None, prev_jerk_err_mags),
                MD.ROT_ACC_VEC3: V3D(rot_accs[-n_jerk_preds:], None, None)
            }
            for md_k, v3s in vec3s_dict.items():
                motion_data[SMD(md_k, RA.ITSELF, AM.MAG, False, False)] = v3s.norms


            rel_axes_dict: typing.Dict[RELATIVE_AXIS, NDArray] = {
                RA.VEL_DEG1: unit_vels_deg1, RA.VEL_DEG2: unit_vels_deg2,
                RA.ACC_FULL: unit_accs, RA.ACC_ORTHO_DEG1: unit_acc_ortho_deg1_vecs,
                RA.PLANE_ORTHO: ortho_dirs, RA.ROTATION: vel_axes
            }

            ra_keys = list(rel_axes_dict.keys())
            vel_ra_keys = [RELATIVE_AXIS.VEL_DEG1, RELATIVE_AXIS.VEL_DEG2]


            ra_mats = np.stack([
                rel_axes_dict[k][-(n_jerk_preds + 1):] for k in ra_keys
            ], axis=1)

            vel_ra_mats = ra_mats[:, [ra_keys.index(k) for k in vel_ra_keys]]

            for shiftRel in [True, False]:
                ra_start = -(n_jerk_preds + int(shiftRel))
                ra_end = -1 if shiftRel else None

                ra_mats_sub = ra_mats[ra_start:ra_end]
                vel_ra_mats_sub = vel_ra_mats[ra_start:ra_end]

                for md_k, v3s in vec3s_dict.items():
                    curr_ra_mats = ra_mats_sub
                    curr_ra_keys = ra_keys
                    if md_k == MOTION_DATA.ACC_VEC3:
                        curr_ra_mats = vel_ra_mats_sub
                        curr_ra_keys = vel_ra_keys


                    ra_dots = pm.einsumMatVecMul(
                        curr_ra_mats, v3s.vecs[-n_jerk_preds:]
                    )

                    for ra_row, ra_k in enumerate(curr_ra_keys):

                        # Skipping values that were already derived+stored earlier.
                        if md_k == MD.ACC_VEC3 and ra_k == RA.VEL_DEG2 and not shiftRel:
                            continue 

                        self._addDotsAndAngs(
                            motion_data, md_k, ra_k, shiftRel,
                            ra_dots[-n_jerk_preds:, ra_row], v3s.norms[-n_jerk_preds:]
                        )
                        
            
            jerk_mags = vec3s_dict[MD.JERK_VEC3].norms
            acc_jerk_ratio = acc_mags / jerk_mags
            motion_data[MOTION_DATA.SPEED_JERK_RATIO] = timescaled_speeds / jerk_mags
            motion_data[MOTION_DATA.ACC_JERK_RATIO] = acc_jerk_ratio
            motion_data[MOTION_DATA.SPEED_ORTHO_ACC_RATIO] = \
                timescaled_speeds / acc_ortho_deg1_mags.flatten()[-n_jerk_preds:]

            motion_data[MOTION_DATA.ACC_VEL_DEG2_ERR_RATIO] = acc_jerk_ratio / step

            keys_to_check_for_completeness = motion_data.keys()
            if check_key_completeness:
                missing_keys = getMissingMotionDataKeys(
                    keys_to_check_for_completeness
                )
                if len(missing_keys) > 0:
                    raise Exception("Keys {} missing!".format([
                        mk.name for mk in missing_keys
                    ]))
            motion_datas.append(motion_data)
            min_err_labels.append(curr_min_norm_labels[1:])
            all_err_norms.append(curr_err_norms_dict)
            min_err_vecs.append(curr_min_norm_vecs)

        return (pose_loader.getVidID(), FeaturesAndResultsForVid(
            motion_datas, all_err_norms, min_err_labels, min_err_vecs
        ))
        # all_motion_data[skip_amt][c2] = motion_data
        # err_norm_lists[skip_amt][c2] = curr_err_norms_dict
        # # err3D_lists[skip_amt][c2] = curr_errs_3D
        # min_norm_labels[skip_amt][c2] = curr_min_norm_labels[1:]
        
class JAV(Enum):
    VELOCITY = 1
    ACCELERATION = 2
    JERK = 3

NumpyForSkipAndID = typing.List[typing.Dict[typing.Any, NDArray]]
OrderForJAV = typing.Tuple[JAV, JAV, JAV]


def dataForCombosJAV(pose_loaders: PoseLoaderList, vec_order: OrderForJAV,
                     return_world2locals: bool = False, 
                     return_translations: bool = False):
    '''
    For each frame of video, we consider a coordinate frame where one axis is
    aligned with the object's velocity and another is aligned with the
    acceleration (or, at least, the part of it orthogonal to velocity).
    We then calculate and return the speed, acceleration, and jerk for the current
    time and the position at the next time in this frame.
    Returns a List[Dict[Combo, NDArray]] that again separates things
    by frame skip amount and by combo.
    '''

    # Empty dict for each skip amount.
    all_data: NumpyForSkipAndID = [dict() for _ in range(3)]
    all_world2local_mats: NumpyForSkipAndID = [dict() for _ in range(3)]
    all_translations: NumpyForSkipAndID = [dict() for _ in range(3)]
    
    if vec_order is None:
        vec_order = (JAV.VELOCITY, JAV.ACCELERATION, JAV.JERK)
    if len(vec_order) != 3 or {v.value for v in vec_order} != {1, 2, 3}:
        raise ValueError("Vector order must be a permutation of (velocity, acceleration, jerk)!")
    
    skip_end = 3#1 if onlySkip0 else 3
    for calc_obj in pose_loaders:
        c = calc_obj.getVidID()
        curr_translations = calc_obj.getTranslationsGTNP()
        for skip in range(skip_end):
            step = skip + 1
            translations = curr_translations[::step]
            vels = np.diff(translations, axis=0)
            # We need a velocity for the last timestep, but not an acceleration,
            # because we need the vectors that take each current position to
            # the next when calculating the "ground truth" for displacement
            # predictions. This is the velocity vector; acceleration vectors
            # are not needed for this; we only need "current" acceleration.
            accs = np.diff(vels[:-1], axis=0)
            jerks = np.diff(accs, axis=0)

            # Here we specify which order in which we orthonormalize our
            # velocity, acceleration, and jerk vectors into orthonormal frames.
            # The first-chosen of these gets aligned exactly with an axis, while
            # the others only get orthogonal components aligned with an axis.
            
            # To only calculate as much as we need, we clip the arrays' fronts
            # off when we can.
            default_ordered = (vels[2:-1], accs[1:], jerks)
            ordered = copy.copy(default_ordered)
            if vec_order is not None:
                ordered = tuple(default_ordered[v.value - 1] for v in vec_order)

            mags0 = np.linalg.norm(ordered[0], axis=-1)
            unit_vecs0 = pm.safelyNormalizeArray(
                ordered[0], mags0[:, np.newaxis]
            )
            # Find the magnitude of the second vector that is parallel to and
            # orthogonal to the first.
            mags_p1 = pm.einsumDot(ordered[1], unit_vecs0) # Parallel magnitude
            vecs_p1 = pm.scalarsVecsMul(mags_p1, unit_vecs0) # Parallel vec3
            vecs_o1 = ordered[1] - vecs_p1 # Orthogonal vec3
            mags_o1 = np.linalg.norm(vecs_o1, axis=-1) # Orthogonal magnitude

            unit_vecs1 = pm.safelyNormalizeArray(
                vecs_o1, mags_o1[:, np.newaxis]
            )

            mags_p20 = pm.einsumDot(ordered[2], unit_vecs0) # Parallel magnitude
            vecs_p20 = pm.scalarsVecsMul(mags_p20, unit_vecs0)
            mags_p21 = pm.einsumDot(ordered[2], unit_vecs1)
            vecs_p21 = pm.scalarsVecsMul(mags_p21, unit_vecs1)
            vecs_o2 = ordered[2] - (vecs_p20 + vecs_p21)
            mags_o2 = np.linalg.norm(vecs_o2, axis=-1)

            unit_vecs2 = pm.safelyNormalizeArray(
                vecs_o2, mags_o2[:, np.newaxis]
            )

            # We now have matrices to convert vectors in world space into
            # these local vector-aligned frames.
            mats = np.stack([unit_vecs0, unit_vecs1, unit_vecs2], axis=1)

            # Transform each third vector and to-next-frame displacement into
            # this frame via matmul.
            # local_vecs2 = pm.einsumMatVecMul(mats, ordered[2])
            local_diffs = pm.einsumMatVecMul(mats, vels[3:])

            # We'll now return all of the data needed to convert velocity,
            # acceleration, and jerk multipliers into local vectors in these
            # new frames. To do this, we don't need to return the coordinate
            # frames themselves: we just need to know the velocity in this
            # frame (a vector [speed, 0, 0]), the acceleration in this frame
            # (i.e. [a_p, a_o, 0]), etc. And since we don't need to return 0s,
            # we can just return the following:
            c_res = (
                mags0, mags_p1, mags_o1, mags_p20, mags_p21, mags_o2,
                *(local_diffs.T)
            )

            all_data[skip][c] = np.stack(c_res, axis=-1)
            if return_world2locals:
                all_world2local_mats[skip][c] = mats
            if return_translations:
                all_translations[skip][c] = translations
    if return_world2locals or return_translations:
        res = (all_data, )
        if return_world2locals:
            res += (all_world2local_mats, )
        if return_translations:
            res += (all_translations, )
        return res
    return all_data

