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
    LAST_BEST_LABEL_ONEHOT = 1
    TIMESTEP = 2

    VEL_DEG1_VEC3 = 3
    VEL_DEG2_VEC3 = 4
    ACC_VEC3 = 5
    JERK_VEC3 = 6
    JERK_ERR_VEC3 = 7
    CRACKLE_VEC3 = 8
    ROTATION_VEC3 = 9
    ROT_ACC_VEC3 = 10
    CIRC_VEL_DEG1_ERR_VEC3 = 11
    CIRC_VEL_DEG2_ERR_VEC3 = 12
    CIRC_ACC_ERR_VEC3 = 13

    CIRC_RAD = 14
    CIRC_SPEED = 15
    CIRC_ACC = 16
    CIRC_ANG_SPEED = 17
    CIRC_ANG_ACC = 18

    AX3_SQ_DIFF = 19

    DISP_MAG_DIFF = 20
    DISP_MAG_DIFF_TIMESCALED = 21
    DISP_MAG_RATIO = 22
    # BOUNCE_ANGLE = 21 # Redundant now with VEL_DEG1_VEC3 stuff.

    UNIT_ROT_AX_DIFF = 23
    UNIT_ROT_AX_DIFF_TIMESCALED = 24

    RAD_DIFF = 25
    TIMESCALED_RAD_DIFF = 26

    # Norms of vec6s composed of circle centres and radii-scaled normals.
    CIRC_VEC6_DIFF = 27

    TIME_SINCE_STATIONARY = 28
    TIME_SINCE_DIR_CHANGE = 29
    DIST_SINCE_DIR_CHANGE = 30

    TIME_CIRC_MOTION = 31
    ANG_SUM_CIRC_MOTION = 32
    DIST_SUM_CIRC_MOTION = 33

    CIRC_CENTRE_DIFF = 34

    FRAME_NUM = 35
    TIMESTAMP = 36

    PREV_FA_ANG_ACC = 37
    NEXT_FA_ANG_ACC = 38

    SPEED_ACC_RATIO = 39
    VEL_BCS_RATIOS = 40

    CURVATURE = 41
    LAST_CURVATURE = 42


    SPEED_JERK_RATIO = 43
    ACC_JERK_RATIO = 44
    SPEED_ORTHO_ACC_RATIO = 45
    CIRC_ACC_CIRC_SPEED_RATIO = 46
    CIRC_ANG_ACC_CIRC_ANG_SPEED_RATIO = 47
    CIRC_ANG_RATIO = 48

    BOUNCE_ANGLE_2_SUM = 49

    PLANE_NORMAL_DOT = 50

    DIST_FROM_CIRCLE = 51
    RATIO_FROM_CIRCLE = 52

    VEL_DEG2_MAG_DIFF = 53
    VEL_DEG2_MAG_DIFF_TIMESCALED = 54

    INV_DISP_MAG_RATIO = 55
    INV_VEL_BCS_RATIOS = 56
    INV_CIRC_ANG_RATIO = 57

    GT0 = 58
    GT1 = 59
    GT2 = 60
    GT3 = 61
    GT4 = 62
    GT5 = 63

    ROT_JERK_VEC3 = 64


    # VEL_DOT = 64                             # Units match work per kg (J/kg)


    # CURVATURE_V = 65
    # CURVATURE_A = 66
    # CURVATURE_J = 67
    # LAST_CURVATURE_V = 68
    # LAST_CURVATURE_A = 69
    # LAST_CURVATURE_J = 70


class OTHER_DIRECTION(Enum):
    ACC_ORTHO_DEG1 = 1
    PLANE_ORTHO = 2

RELATIVE_VECTOR = typing.Union[MOTION_DATA, OTHER_DIRECTION]
ALL_RELATIVE_VECTORS = (
    MOTION_DATA.VEL_DEG1_VEC3, MOTION_DATA.VEL_DEG2_VEC3, MOTION_DATA.ACC_VEC3,
    MOTION_DATA.JERK_VEC3, MOTION_DATA.JERK_ERR_VEC3, MOTION_DATA.ROTATION_VEC3,
    MOTION_DATA.ROT_ACC_VEC3, MOTION_DATA.ROT_JERK_VEC3,
    OTHER_DIRECTION.ACC_ORTHO_DEG1, OTHER_DIRECTION.PLANE_ORTHO
)

class OrthoVecDirPair(typing.NamedTuple):
    vec3: MOTION_DATA
    axis: OTHER_DIRECTION

ORTHO_VEC3_AX_PAIRS = [
    OrthoVecDirPair(MOTION_DATA.VEL_DEG1_VEC3, OTHER_DIRECTION.ACC_ORTHO_DEG1),
    OrthoVecDirPair(MOTION_DATA.VEL_DEG1_VEC3, OTHER_DIRECTION.PLANE_ORTHO),
    OrthoVecDirPair(MOTION_DATA.VEL_DEG2_VEC3, OTHER_DIRECTION.PLANE_ORTHO),
    OrthoVecDirPair(MOTION_DATA.ACC_VEC3, OTHER_DIRECTION.PLANE_ORTHO)
]

class ANG_OR_MAG(Enum):
    ANG = 1
    MAG_PROJ = 2
    MAG_DOT = 3

class SpecifiedMotionData(typing.NamedTuple):
    base_cat: MOTION_DATA
    axis: RELATIVE_VECTOR
    ang_or_mag: ANG_OR_MAG
    bidirectional: bool
    is_timestep_shifted: bool

    @property
    def name(self):
        ret_name = ""
        bn = self.base_cat.name 
        rn = self.axis.name
        last_underscore_ind = bn.rfind("_")
        if bn[last_underscore_ind:] != "_VEC3":
            raise ValueError("No \"VEC3\" found in base type {}!".format(bn))
        
        ret_name = bn[:last_underscore_ind] 

        dirs_eq = False
        if isinstance(self.axis, MOTION_DATA):
            if self.base_cat != self.axis:
                r_last_underscore_ind = rn.rfind("_")
                if rn[r_last_underscore_ind:] != "_VEC3":
                    raise ValueError(
                        "No \"VEC3\" found in relative key {}!".format(rn)
                    )
                ret_name += "_" + rn[:r_last_underscore_ind]
            else:
                dirs_eq= True
        else:
            ret_name += "_" + rn
        am = self.ang_or_mag.name
        if dirs_eq and not self.is_timestep_shifted:
            if self.ang_or_mag == ANG_OR_MAG.MAG_PROJ:
                am = "MAG"
            else:
                raise Exception("Redundant self-axis value!")
        ret_name += "_" + am

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
            for i in range(2):
                if self.norms[i] == 0.0:
                    if np.any(self.vecs[i] != 0.0):
                        raise Exception(
                            "Norm and scaled vec 0-len inconsistency!"
                        )
                    self.unit_vecs[i] = 0.0
                    # MAYBE setting unit dir to 0 is a workaround in this case?

class OneHotMotionData(typing.NamedTuple):
    base_cat: MOTION_DATA
    cat_num: int

    @property
    def name(self):
        bn = self.base_cat.name 
        last_underscore_ind = bn.rfind("_")
        if bn[last_underscore_ind:] != "_ONEHOT":
            raise ValueError("No \"ONEHOT\" found in base type {}!".format(bn))
        
        return bn[:(last_underscore_ind + 1)] + "CAT" + str(self.cat_num) 

def _getUnflatTimestamps(num_displacements: int,
                         timestamps: typing.Optional[NDArray]):
    unflat_timestamps: typing.Optional[NDArray] = None
    if timestamps is not None:
        if len(timestamps) != (num_displacements + 1):
            raise ValueError(
                "Must have n+1 timestamps for n displacements!"
            )
        unflat_timestamps = timestamps.reshape(-1, 1)
    return unflat_timestamps

class PositionDerivativeCollection:
    def __init__(self, displacements: NDArray,
                 timestamps: typing.Optional[NDArray] = None):
        self.velocities = displacements.copy()
        unflat_timestamps = _getUnflatTimestamps(len(displacements), timestamps)
        if timestamps is not None:
            time_deltas = np.diff(unflat_timestamps, 1, axis=0)
            self.velocities = displacements / time_deltas

        self.accelerations = self._recursiveDeriv(
            self.velocities, 2, unflat_timestamps
        )
        self.jerks = self._recursiveDeriv(
            self.accelerations, 3, unflat_timestamps
        )
        self.snaps = self._recursiveDeriv(
            self.jerks, 4, unflat_timestamps
        )
        self.crackles = self._recursiveDeriv(
            self.snaps, 5, unflat_timestamps
        )

    @staticmethod   
    def _recursiveDeriv(prev_vals: NDArray, deriv_power: int,
                        full_timestamps: typing.Optional[NDArray]):
        '''E.g., prev_vals are the last accelerations, deriv_power is 3 
        (for jerk), and full_timestamps are the timestamps for all positions
        (minus perhaps the last one for which no prediction follows) reshaped
        to (n, 1).'''
        ret_val = np.diff(prev_vals, 1, axis=0)
        if full_timestamps is not None:
            time_deltas = \
                full_timestamps[deriv_power:] - full_timestamps[:-deriv_power]
            scalars = deriv_power / time_deltas
            ret_val = scalars * ret_val
        return ret_val

class RotationDerivativeCollection:
    def __init__(self, known_rotation_diffs_aa: NDArray, 
                 timestamps: typing.Optional[NDArray] = None):
        '''The param known_rotation_diffs_aa means taking the axis-angle forms
        of the rotations between frames WITHOUT considering timestamps yet!'''
        timestamps_given = (timestamps is not None)
        unflat_timestamps = _getUnflatTimestamps(
            len(known_rotation_diffs_aa), timestamps
        )
        
        self.velocities = known_rotation_diffs_aa
        if timestamps_given:
            time_deltas = np.diff(unflat_timestamps, 1, axis=0)
            self.velocities = self.velocities / time_deltas
        
        self.accelerations = np.diff(self.velocities, 1, axis=0)
        if timestamps_given:
            # TODO: Try the "easier" version of this instead and compare error.
            t2_m_t0 = unflat_timestamps[2:] - unflat_timestamps[-2:]
            self.accelerations *= (2 / t2_m_t0)


MOTION_DATA_KEY_TYPE = typing.Union[
    MOTION_DATA, SpecifiedMotionData, OneHotMotionData
]
PoseLoaderList = typing.List[gtc.PoseLoader]

# Finds closest points on hyperplanes with the given normals and offsets.
# The below function is the result of me feeding my original getClosestPoint()
# function through Copilot/Claude to accomodate multiple points per hyperplane.
# TODO: Need to manually verify logic and clean things up a bit.
def getClosestPoint(normals: NDArray, scaled_plane_offsets: NDArray, points: NDArray, return_sq_dists: bool = False):
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
    points_exp = points.T.reshape(1, points.shape[1], points.shape[0])
    
    # Calculate dot products: (m, 1, n)
    norm_sq = pm.einsumDot(normals, normals).reshape(-1, 1, 1)
    pn = np.sum(normals_exp * points_exp, axis=1, keepdims=True)  # (m, 1, n)
    
    # Calculate scalars: (m, 1, n)
    scalar = (scaled_plane_offsets.reshape(-1, 1, 1) - pn) / norm_sq
    
    # Calculate displacements: (m, k, n)
    disps = scalar * normals_exp
    
    # Calculate closest points: (m, k, n)
    closest = points_exp + disps
    
    if return_sq_dists:
        # Calculate distances: (m, n)
        sq_distances = np.sum(disps * disps, axis=1)
        return (closest,
                sq_distances)
    return closest

def getBaselineJAV6(acc_multiplier_options, jerk_multiplier_options):
    base = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    for a1_ind, a1 in enumerate(acc_multiplier_options):
        j_end_ind = a1_ind if a1_ind < 2 else 3
        for a0_ind, a0 in enumerate(acc_multiplier_options[:(a1_ind + 1)]):
            for j2_ind, j2 in enumerate(jerk_multiplier_options[:(j_end_ind + 1)]):
                for j1_ind, j1 in enumerate(jerk_multiplier_options[:(j2_ind + 1)]):
                    j_end_from_a0 = a0_ind if a0_ind < 2 else 3
                    jv_end_ind = min(j2_ind, j_end_from_a0)
                    for j0 in jerk_multiplier_options[:(jv_end_ind + 1)]:
                        base.append([1.0, a0, a1, j0, j1, j2])
    return np.asarray(base)

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
    pts_aj = baseline_m6[:, [2, 4]]  # Shape: (n, 2)
    u_pts_aj, u_pts_aj_inds = np.unique(pts_aj, return_inverse=True, axis=0)
    closest_points_aj, sq_dists_aj = getClosestPoint(
        y_true[:, [2, 4]], y_true[:, 7], u_pts_aj, return_sq_dists=True
    )


    
    # Second hyperplane (v, a, and j components)
    pts_vaj = baseline_m6[:, [0, 1, 3]]  # Shape: (n, 3)
    u_pts_vaj, u_pts_vaj_inds = np.unique(pts_vaj, return_inverse=True, axis=0)
    closest_points_vaj, sq_dists_vaj = getClosestPoint(
        y_true[:, [0, 1, 3]], y_true[:, 6], u_pts_vaj, return_sq_dists=True
    )

    all_sq_dists_aj = sq_dists_aj[:, u_pts_aj_inds]
    all_sq_dists_vaj = sq_dists_vaj[:, u_pts_vaj_inds]
    all_sq_dists = all_sq_dists_aj + all_sq_dists_vaj

    min_indices = np.argmin(all_sq_dists, axis=1)
    min_indices_aj = u_pts_aj_inds[min_indices]
    min_indices_vaj = u_pts_vaj_inds[min_indices]

    # Get the closest points using the indices
    # closest_points_aj shape is (n, 2, m), we want to select best n for each m
    a_a = closest_points_aj[range(len(y_true)), 0, min_indices_aj]
    j_a = closest_points_aj[range(len(y_true)), 1, min_indices_aj]
    # Get the closest points using the indices
    # closest_points_vaj shape is (n, 3, m), we want to select best n for each m
    v_v = closest_points_vaj[range(len(y_true)), 0, min_indices_vaj]
    a_v = closest_points_vaj[range(len(y_true)), 1, min_indices_vaj]
    j_v = closest_points_vaj[range(len(y_true)), 2, min_indices_vaj]
    
    return np.stack([v_v, a_v, a_a, j_v, j_a, j_o], axis=-1)


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
                 err_radius_ratio_thresh: float = DEFAULT_ERR_RADIUS_RATIO_THRESH,
                 exclude_onehots: bool = True, exclude_past_muls: bool = True,
                 exclude_bidir: bool = True, exclude_axis_angs: bool = True,
                 exclude_circ_data: bool = True, exclude_vel_deg2: bool = True,
                 exclude_timescaled: bool = True,
                 other_exclusions: typing.Optional[typing.List[MOTION_DATA_KEY_TYPE]] = None
                 ):
        self.obj_static_thresh_mm = obj_static_thresh_mm
        self.straight_angle_thresh_rad = np.deg2rad(straight_angle_thresh_deg)
        self.min_jerk_opt_iter_lim = min_jerk_opt_iter_lim
        self.split_min_jerk_opt_iter_lim = split_min_jerk_opt_iter_lim
        self.err_radius_ratio_thresh = err_radius_ratio_thresh
        self.err_na_val = err_na_val

        self.exclude_onehots = exclude_onehots
        self.exclude_past_muls = exclude_past_muls
        self.exclude_bidir = exclude_bidir
        self.exclude_axis_angs = exclude_axis_angs
        self.exclude_circ_data = exclude_circ_data
        self.exclude_vel_deg2 = exclude_vel_deg2
        self.exclude_timescaled = exclude_timescaled
        self.other_exclusions = other_exclusions
        if self.other_exclusions is None:
            self.other_exclusions = set()

        self.base_key_exclusions: typing.Set[MOTION_DATA] = set()

        if self.exclude_onehots:
            self.base_key_exclusions.update(
                k for k in MOTION_DATA if k.name.upper().endswith("ONEHOT")
            )
        self.base_JAV6 = np.empty((0, 6))
        if self.exclude_past_muls:
            self.base_key_exclusions.update(
                k for k in MOTION_DATA
                if len(k.name) == 3 and k.name.upper().startswith("GT")
            )
        else:
            base_a_opts = [0.0, 0.5, 1.0]
            # See other commenting on how these possible multiplier totals are
            # found when looking at the lagrange polynomial derivatives.
            base_j_opts = [0, 1/6, 2/3, 1.0]

            self.base_JAV6 = getBaselineJAV6(base_a_opts, base_j_opts)

        if self.exclude_circ_data:
            self.base_key_exclusions.update(
                k for k in MOTION_DATA if self._isDataCircleRelated(k)
            )
        if self.exclude_vel_deg2:
            self.base_key_exclusions.update(
                k for k in MOTION_DATA if "VEL_DEG2" in k.name.upper()
            )

        if self.exclude_timescaled:
            self.base_key_exclusions.update(
                k for k in MOTION_DATA if "TIMESCALE" in k.name.upper()
            )

        self.base_key_exclusions.update(self.other_exclusions)

        self.motion_mod_keys = [
            MOTION_MODEL(i) for i in range(1, len(MOTION_MODEL) + 1)
        ]

        self._initializeFutureVars()

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

    def _initializeFutureVars(self):
        # Initialize to-be-filled variables with empty lists.        
        self.all_motion_data: typing.List[
            typing.Dict[typing.Any, typing.Dict[MOTION_DATA_KEY_TYPE, NDArray]]
        ] = []
        self.err_norm_lists: typing.List[
            typing.Dict[typing.Any, typing.Dict[MOTION_MODEL, NDArray]]
        ] = []

        self.min_norm_labels: typing.List[typing.Dict[typing.Any, NDArray]] = []

        self.min_norm_vecs: typing.List[typing.Dict[typing.Any, NDArray]] = []

    def freeUpMemory(self):
        del self.all_motion_data
        del self.err_norm_lists
        del self.min_norm_labels
        del self.min_norm_vecs
        self._initializeFutureVars()

    @staticmethod
    def _isDataCircleRelated(data_key: MOTION_DATA):
        n = data_key.name.upper()
        circ_name = "_CIRC_" in n or "CIRCLE" in n
        circ_name |= n.startswith("CIRC_") or n.endswith("_CIRC")
        circ_name |= "RAD_DIFF" in n
        return circ_name
    
    def validateKeys(self, keys: typing.List[MOTION_DATA_KEY_TYPE]):
        all_possible_keys = set(MOTION_DATA)

        present_keys = {k for k in keys if isinstance(k, MOTION_DATA)}
        for k in keys:
            if isinstance(k, SpecifiedMotionData) or isinstance(k, OneHotMotionData):
                present_keys.add(k.base_cat)
        
        invalid_keys = present_keys.difference(all_possible_keys)
        if len(invalid_keys) > 0:
            raise ValueError("Bad types for keys: {}".format(invalid_keys))

        missing_keys = all_possible_keys.difference(present_keys)

        wrong_missing_keys = missing_keys.difference(self.base_key_exclusions)
        wrong_present_keys = self.base_key_exclusions.difference(missing_keys)
        
        if len(wrong_missing_keys) > 0:
            raise Exception("Keys {} missing when they should not!".format(
                [mk.name for mk in wrong_missing_keys]
            ))
        if len(wrong_present_keys) > 0:
            raise Exception("Keys {} present despite contrary !settings".format(
                [mk.name for mk in wrong_present_keys]
            ))
        
        return
    
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

    def _addDotsAndAngs(self,
               dict_to_update: typing.Dict[MOTION_DATA_KEY_TYPE, NDArray],
               motion_data_base_key: MOTION_DATA, relative_axis: RELATIVE_VECTOR,
               shift: bool, dot_products: NDArray, vec_norms: NDArray, *,
               other_norms: typing.Optional[NDArray] = None,
               dots_with_unit_axis: typing.Optional[NDArray] = None):
        
        SMD = SpecifiedMotionData
        AM = ANG_OR_MAG

        # Key ceation.
        kmp = SMD(
            motion_data_base_key, relative_axis, AM.MAG_PROJ, False, shift
        )
        kmpbidir = SMD(
            motion_data_base_key, relative_axis, AM.MAG_PROJ, True, shift
        )
        kmd = SMD(
            motion_data_base_key, relative_axis, AM.MAG_DOT, False, shift
        )
        kmdbidir = SMD(
            motion_data_base_key, relative_axis, AM.MAG_DOT, True, shift
        )
        ka      = SMD(motion_data_base_key, relative_axis, AM.ANG, False, shift)
        kabidir = SMD(motion_data_base_key, relative_axis, AM.ANG, True, shift)

        # Storing/calculating the data.
        if other_norms is not None or dots_with_unit_axis is not None:
            if dots_with_unit_axis is None:
                dots_with_unit_axis = dot_products / other_norms
                # TODO: Handle jerk, snap, and crackle better here.
                # Probably best to have a param where if these columns are included,
                # then the beginning "zero" rows are excluded?
                for i in range(2):
                    if other_norms[i] == 0.0:
                        dots_with_unit_axis[i] = 0.0
            dict_to_update[kmp] = dots_with_unit_axis
            if not self.exclude_bidir:
                dict_to_update[kmpbidir] = np.abs(dots_with_unit_axis)
        else:
            kmd = kmp
            kmdbidir = kmpbidir
            dots_with_unit_axis = dot_products

        duplicate_dot = (
            isinstance(relative_axis, MOTION_DATA) and (not shift)
            and motion_data_base_key.value < relative_axis.value
        )
        if not duplicate_dot:
            dict_to_update[kmd] = dot_products
            if not self.exclude_bidir:
                dict_to_update[kmdbidir] = np.abs(dot_products)
        
        if not self.exclude_axis_angs:
            curr_angs = np.arccos(
                np.clip(dots_with_unit_axis / vec_norms, -1, 1)
            )
            # TODO: Handle jerk, snap, and crackle better here.
            # Probably best to have a param where if these columns are included,
            # then the beginning "zero" rows are excluded?
            for i in range(2):
                if vec_norms[i] == 0.0:
                    curr_angs[i] = 0.0
            dict_to_update[ka] = curr_angs
            if not self.exclude_bidir:
                dict_to_update[kabidir] = pm.getAcuteAngles(curr_angs)


    def getInputFeatures(self, pose_loader: gtc.PoseLoader,
                         check_key_completeness: bool = False):
        # The below code will use MOTION_DATA_KEY_TYPE classes so often that some
        # shorter aliases might be helpful.
        MD = MOTION_DATA
        SMD = SpecifiedMotionData
        OD = OTHER_DIRECTION
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

            rderivs = RotationDerivativeCollection(timescaled_vel_axes)
            rot_accs = rderivs.accelerations / step

            rot_jerks = np.diff(rot_accs, 1, axis=0) / step
            rot_jerks = np.pad(rot_jerks, ((1, 0), (0, 0)))


            pderivs = PositionDerivativeCollection(translation_diffs[:-1])
            deg1_vels = pderivs.velocities
            deg2_accs = pderivs.accelerations / step_sq
            t_jerk_amt = pderivs.jerks

            # Pad with 0 so it equals the "4th derivative".
            # TODO: handle better!
            scaled_snaps = pderivs.snaps / (step**4)
            snap_mags = np.linalg.norm(scaled_snaps, axis=-1)
            # Why pad the snaps and crackles both by 2? The snaps need to be a
            # bit longer because they're used as a relative axis but the
            # crackles do not.
            prev_jerk_errs = np.pad(scaled_snaps, ((2, 0), (0, 0)))
            prev_jerk_err_mags = np.pad(snap_mags, ((2, 0)))

            crackles = np.pad(pderivs.crackles / (step**5), ((2, 0), (0, 0)))

            half_deg1_vel_diffs = 0.5 * pderivs.accelerations
            deg2_vels = deg1_vels[1:] + half_deg1_vel_diffs


            t_jerk_preds = 4 * prev_translations[3:] - 6 * prev_translations[2:-1] \
                + 4 * prev_translations[1:-2] - prev_translations[:-3]


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
            if not self.exclude_timescaled:
                motion_data[MOTION_DATA.UNIT_ROT_AX_DIFF_TIMESCALED] \
                                                = unit_rot_ax_diff_mags / step

            deg1_speeds_full = np.linalg.norm(deg1_vels, axis=-1, keepdims=True)
            deg1_speeds = deg1_speeds_full[-n_jerk_preds:].flatten()
            deg2_speeds_full = np.linalg.norm(deg2_vels, axis=-1, keepdims=True)
            timescaled_speeds_deg2_full = deg2_speeds_full / step
            timescaled_speeds_deg1_full = deg1_speeds_full / step
            timescaled_speeds_deg1 = \
                timescaled_speeds_deg1_full.flatten()[-n_jerk_preds:]


            acc_mags_full = np.linalg.norm(deg2_accs, axis=-1, keepdims=True)
            acc_mags = acc_mags_full[-n_jerk_preds:].flatten()

            motion_data[MOTION_DATA.SPEED_ACC_RATIO] = timescaled_speeds_deg1 / acc_mags
            vel_dots = pm.einsumDot(
                deg1_vels[-n_jerk_preds:], deg1_vels[-(n_jerk_preds + 1):-1]
            )
            vel_bcs_ratios = vel_dots / (deg1_speeds[-n_jerk_preds:]**2)
            motion_data[MOTION_DATA.VEL_BCS_RATIOS] = vel_bcs_ratios
            motion_data[MOTION_DATA.INV_VEL_BCS_RATIOS] = 1.0 / vel_bcs_ratios

            disp_mag_diffs = np.diff(deg1_speeds_full, 1, axis=0)[-n_jerk_preds:].flatten()
            motion_data[MOTION_DATA.DISP_MAG_DIFF] = disp_mag_diffs
            if not self.exclude_timescaled:
                motion_data[MOTION_DATA.DISP_MAG_DIFF_TIMESCALED] \
                    = disp_mag_diffs / step
            disp_mag_div = deg1_speeds_full[1:] / deg1_speeds_full[:-1]
            motion_data[MOTION_DATA.DISP_MAG_RATIO] = disp_mag_div[-n_jerk_preds:].flatten()
            motion_data[MOTION_DATA.INV_DISP_MAG_RATIO] = 1.0 / disp_mag_div[-n_jerk_preds:].flatten()

            unit_vels_deg1 = pm.safelyNormalizeArray(deg1_vels, deg1_speeds_full)
            unit_vels_deg2 = pm.safelyNormalizeArray(deg2_vels, deg2_speeds_full)
            unit_accs = pm.safelyNormalizeArray(deg2_accs, acc_mags_full)

            if not self.exclude_circ_data:
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
                if not self.exclude_timescaled:
                    motion_data[MOTION_DATA.TIMESCALED_RAD_DIFF] = radii_diffs / step

                c_centre_diff_vecs = np.diff(cma.getCentres3D(), axis=0)
                c_centre_diff_norms = np.linalg.norm(c_centre_diff_vecs, axis=-1)
                motion_data[MOTION_DATA.CIRC_CENTRE_DIFF] = c_centre_diff_norms


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

                motion_data[MOTION_DATA.DIST_FROM_CIRCLE] = is_circ_res.dists
                motion_data[MOTION_DATA.RATIO_FROM_CIRCLE] = is_circ_res.dist_radius_ratios

            rotation_mats = all_rotation_mats[::step]
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

            
            t_diff_angs = pm.anglesBetweenVecs(
                unit_vels_deg1[:-1], unit_vels_deg1[1:], False
            )
            # motion_data[MOTION_DATA.BOUNCE_ANGLE] = t_diff_angs[-n_jerk_preds:]
            bounce_ang_pair_sums = t_diff_angs[1:] + t_diff_angs[:-1]
            motion_data[MOTION_DATA.BOUNCE_ANGLE_2_SUM] = \
                bounce_ang_pair_sums[-n_jerk_preds:]


            scaled_jerks = np.empty_like(deg2_accs)
            scaled_jerks[0] = 0.0 # TODO: handle this better!
            scaled_jerks[1:] = t_jerk_amt / (step ** 3)

            if not self.exclude_vel_deg2:
                speed_deg2_diffs = np.diff(deg2_speeds_full, 1, axis=0)[-n_jerk_preds:].flatten()
                motion_data[MOTION_DATA.VEL_DEG2_MAG_DIFF] = speed_deg2_diffs
                if not self.exclude_timescaled:
                    motion_data[MOTION_DATA.VEL_DEG2_MAG_DIFF_TIMESCALED] \
                        = speed_deg2_diffs / step


            d_under_thresh = deg1_speeds_full < self.obj_static_thresh_mm
            a_over_thresh = t_diff_angs > self.straight_angle_thresh_rad
        
            mj_preds = None
            acc_preds = temp_preds[MOTION_MODEL.ACC_DEG2]
            if self.min_jerk_opt_iter_lim > 0:
                mj_preds = mj.min_jerk_lsq(
                    prev_translations, d_under_thresh.flatten(), 
                    a_over_thresh.flatten(),
                    max_opt_iters=self.min_jerk_opt_iter_lim,
                    vels = deg1_vels, accs = deg2_accs, jerks = scaled_jerks[1:]
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

            
            

            frame_nums = np.arange(n_input_frames - n_jerk_preds, n_input_frames)
            motion_data[MOTION_DATA.FRAME_NUM] = frame_nums
            motion_data[MOTION_DATA.TIMESTAMP] = frame_nums * step



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
            if not self.exclude_onehots:
                for mn in range(len(self.motion_mod_keys)):
                    motion_data[
                        OneHotMotionData(MOTION_DATA.LAST_BEST_LABEL_ONEHOT, mn)
                    ] = (curr_min_norm_labels[:-1] == mn)

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
            prev_circ_errs_vd1 = curr_errs_3D[MOTION_MODEL.CIRC_VEL_DEG1][:-1]
            prev_circ_errs_vd2 = curr_errs_3D[MOTION_MODEL.CIRC_VEL_DEG2][:-1]
            prev_circ_errs_acc = curr_errs_3D[MOTION_MODEL.CIRC_ACC][:-1]
            prev_circ_err_mags_vd1 = curr_err_norms[circ_vd1_ind][:-1]
            prev_circ_err_mags_vd2 = curr_err_norms[circ_vd2_ind][:-1]
            prev_circ_err_mags_acc = curr_err_norms[circ_acc_ind][:-1]
        


            avd1m = pm.einsumDot(deg2_accs, deg1_vels[1:])
            acc_vel_deg1_mag = avd1m / deg1_speeds_full[1:].flatten()
            acc_vel_deg1_parallel = pm.scalarsVecsMul(acc_vel_deg1_mag, unit_vels_deg1[1:])
            
            acc_ortho_deg1_vecs = deg2_accs - acc_vel_deg1_parallel
            acc_ortho_deg1_mags = np.linalg.norm(
                acc_ortho_deg1_vecs, axis=-1, keepdims=True
            )
            
            unit_acc_ortho_deg1_vecs = pm.safelyNormalizeArray(
                acc_ortho_deg1_vecs, acc_ortho_deg1_mags
            )

            vel_deg2_acc_cross = np.cross(unit_vels_deg2, unit_accs)
            sin_vel_deg2_acc_angs = np.linalg.norm(
                vel_deg2_acc_cross, axis=-1, keepdims=True
            )

            

            jvd1m = pm.einsumDot(scaled_jerks[1:], deg1_vels[2:])
            jerk_vel_deg1_mags = jvd1m / deg1_speeds
            
            jerk_acc_ortho_mags = pm.einsumDot(
                scaled_jerks[1:], unit_acc_ortho_deg1_vecs[-n_jerk_preds:]
            )
            
            jerk_vel_vecs = pm.scalarsVecsMul(
                jerk_vel_deg1_mags, unit_vels_deg1[-n_jerk_preds:]
            )
            jerk_acc_vecs = pm.scalarsVecsMul(
                jerk_acc_ortho_mags, unit_acc_ortho_deg1_vecs[-n_jerk_preds:]
            )
            jerk_ortho_vecs = scaled_jerks[1:] - (jerk_vel_vecs + jerk_acc_vecs)
            jerk_ortho_norms = np.linalg.norm(jerk_ortho_vecs, axis=-1, keepdims=True)

            
            vec3_precalc_keys = [
                (MD.ACC_VEC3, MD.VEL_DEG1_VEC3),
                (MD.ACC_VEC3, OD.ACC_ORTHO_DEG1),
                (MD.JERK_VEC3, MD.VEL_DEG1_VEC3),
                (MD.JERK_VEC3, OD.ACC_ORTHO_DEG1),
                (MD.JERK_VEC3, OD.PLANE_ORTHO)
            ]

            precalced_mags = [
                deg1_speeds_full, deg2_speeds_full, acc_mags_full,
                sin_vel_deg2_acc_angs, jerk_ortho_norms
            ]
            for mags in precalced_mags:
                mags_0 = mags[0]
                mags_0_v = mags_0 if np.isscalar(mags_0) else mags_0[0]
                if mags_0_v == 0.0:
                    raise ValueError("Relative axes cannot have a norm of 0.0!")
            
            # NOTE: We *could* use the orthogonal component of jerk to set the
            # sign/direction of the plane normals instead of just using the sign
            # of the cross product. However, for whatever reason, using the
            # cross product gives a better neural network score.
            # So I'm going to keep using the cross product unless I can make a
            # big improvement somewhere else so that this stops mattering. 
            ortho_dirs = pm.safelyNormalizeArray(vel_deg2_acc_cross, sin_vel_deg2_acc_angs)

            plane_dots = pm.einsumDot(
                ortho_dirs[-n_jerk_preds:], ortho_dirs[-(n_jerk_preds + 1):-1]
            )
            motion_data[MOTION_DATA.PLANE_NORMAL_DOT] = plane_dots
            
            unit_jerks = np.empty((n_jerk_preds + 1, 3))
            jerk_norms = np.linalg.norm(scaled_jerks, axis=-1)
            unit_jerks = pm.safelyNormalizeArray(scaled_jerks, jerk_norms[..., np.newaxis])

            unit_snaps = pm.safelyNormalizeArray(prev_jerk_errs, prev_jerk_err_mags[..., np.newaxis])

            rot_v3d = Vec3Data(timescaled_vel_axes, vel_axes, vel_angs_timescaled)
            vec3s_dict: typing.Dict[MOTION_DATA, Vec3Data] = {
                MD.ACC_VEC3: V3D(deg2_accs, unit_accs, acc_mags_full),
                MD.JERK_VEC3: V3D(scaled_jerks, unit_jerks, jerk_norms),
                MD.ROTATION_VEC3: rot_v3d,
                MD.JERK_ERR_VEC3: V3D(prev_jerk_errs, unit_snaps, prev_jerk_err_mags),
                MD.CRACKLE_VEC3: V3D(crackles, None, None),
                MD.ROT_ACC_VEC3: V3D(rot_accs, None, None),
                MD.ROT_JERK_VEC3: V3D(rot_jerks, None, None),
                MD.VEL_DEG1_VEC3: V3D(
                    deg1_vels / step, unit_vels_deg1, timescaled_speeds_deg1_full
                )
            }
            if not self.exclude_vel_deg2:
                vec3s_dict[MD.VEL_DEG2_VEC3] = V3D(
                    deg2_vels / step, unit_vels_deg2, timescaled_speeds_deg2_full
                )
            if not self.exclude_circ_data:
                vec3s_dict[MD.CIRC_VEL_DEG1_ERR_VEC3] = V3D(
                    prev_circ_errs_vd1, None, prev_circ_err_mags_vd1
                )
                if not self.exclude_vel_deg2:
                    vec3s_dict[MD.CIRC_VEL_DEG2_ERR_VEC3] = V3D(
                        prev_circ_errs_vd2, None, prev_circ_err_mags_vd2
                    )
                vec3s_dict[MD.CIRC_ACC_ERR_VEC3] = V3D(
                    prev_circ_errs_acc, None, prev_circ_err_mags_acc
                )



            for md_k, v3s in vec3s_dict.items():
                motion_data[SMD(md_k, md_k, AM.MAG_PROJ, False, False)] = \
                    v3s.norms[-n_jerk_preds:]

            other_dirs_dict: typing.Dict[OTHER_DIRECTION, NDArray] = {
                OD.ACC_ORTHO_DEG1: unit_acc_ortho_deg1_vecs,
                OD.PLANE_ORTHO: ortho_dirs
            }

            vec3_stack_for_mat = []
            selected_rel_vecs = []
            for k in ALL_RELATIVE_VECTORS:
                if isinstance(k, MOTION_DATA):
                    if k == MOTION_DATA.VEL_DEG2_VEC3 and self.exclude_vel_deg2:
                        continue
                    vec3_stack_for_mat.append(vec3s_dict[k].vecs[-(n_jerk_preds + 1):])
                elif isinstance(k, OTHER_DIRECTION):
                    vec3_stack_for_mat.append(other_dirs_dict[k][-(n_jerk_preds + 1):])
                selected_rel_vecs.append(k)

            ra_mats = np.stack(vec3_stack_for_mat, axis=1)

            for shiftRel in [True, False]:
                ra_start = -(n_jerk_preds + int(shiftRel))
                ra_end = -1 if shiftRel else None

                ra_mats_sub = ra_mats[ra_start:ra_end]

                for md_k, v3s in vec3s_dict.items():

                    ra_dots = pm.einsumMatVecMul(
                        ra_mats_sub, v3s.vecs[-n_jerk_preds:]
                    )
                    for ra_row, ra_k in enumerate(selected_rel_vecs):

                        # Skipping values that were already derived+stored earlier.
                        if not shiftRel:
                            key_precalced = False
                            for md_kp, ra_kp in vec3_precalc_keys:
                                if md_kp == md_k and ra_kp == ra_k:
                                    key_precalced = True
                                    break
                            if key_precalced:
                                continue
                            # Also skip if the axis matches or is ortho to the
                            # vec3, since we have a specialized "itself" axis 
                            # for the first case and the second yields no useful
                            # data.
                            if isinstance(ra_k, MOTION_DATA):
                                if md_k == ra_k:
                                    continue
                            elif (md_k, ra_k) in ORTHO_VEC3_AX_PAIRS:
                                continue
                        else:
                            # TODO: Fix shift for jerk or snap as prev vec so 
                            # that values generated are not NaN.
                            if isinstance(ra_k, MOTION_DATA) and (
                                ra_k == MOTION_DATA.JERK_VEC3 or ra_k == MOTION_DATA.JERK_ERR_VEC3
                            ):
                                ... # continue

                        _ra_norms = None
                        if isinstance(ra_k, MOTION_DATA):
                            _ra_norms = vec3s_dict[ra_k].norms[ra_start:ra_end]

                        self._addDotsAndAngs(
                            motion_data, md_k, ra_k, shiftRel,
                            ra_dots[-n_jerk_preds:, ra_row],
                            v3s.norms[-n_jerk_preds:], other_norms=_ra_norms
                        )

            flat_accs = acc_mags.flatten()
            self._addDotsAndAngs(
                motion_data, MD.ACC_VEC3, OD.ACC_ORTHO_DEG1, False,
                acc_ortho_deg1_mags[-n_jerk_preds:].flatten(), flat_accs
            )

            self._addDotsAndAngs(
                motion_data, MD.ACC_VEC3, MD.VEL_DEG1_VEC3, False,
                avd1m[-n_jerk_preds:], flat_accs,
                dots_with_unit_axis=acc_vel_deg1_mag[-n_jerk_preds:]
            )

            self._addDotsAndAngs(
                motion_data, MD.JERK_VEC3, MD.VEL_DEG1_VEC3, False,
                jvd1m[-n_jerk_preds:], jerk_norms[-n_jerk_preds:],
                dots_with_unit_axis=jerk_vel_deg1_mags[-n_jerk_preds:]
            )

            self._addDotsAndAngs(
                motion_data, MD.JERK_VEC3, OD.ACC_ORTHO_DEG1, False,
                jerk_acc_ortho_mags[-n_jerk_preds:], jerk_norms[-n_jerk_preds:]
            )
            self._addDotsAndAngs(
                motion_data, MD.JERK_VEC3, OD.PLANE_ORTHO, False,
                jerk_ortho_norms.flatten(), jerk_norms[1:].flatten()
            )

            jerk_mags = vec3s_dict[MD.JERK_VEC3].norms
            acc_jerk_ratio = acc_mags / jerk_mags[-n_jerk_preds:]
            motion_data[MOTION_DATA.SPEED_JERK_RATIO] = timescaled_speeds_deg1 / jerk_mags[-n_jerk_preds:]
            motion_data[MOTION_DATA.ACC_JERK_RATIO] = acc_jerk_ratio
            motion_data[MOTION_DATA.SPEED_ORTHO_ACC_RATIO] = \
                timescaled_speeds_deg1 / acc_ortho_deg1_mags.flatten()[-n_jerk_preds:]


            if not self.exclude_past_muls:
                gt_calc_jav_input = np.empty((n_jerk_preds, 9))

                gt_calc_jav_input[:, 0] = deg1_speeds
                gt_calc_jav_input[:, 1] = acc_vel_deg1_mag[-n_jerk_preds:] * step_sq
                gt_calc_jav_input[:, 2] = acc_ortho_deg1_mags[-n_jerk_preds:, 0] * step_sq
                gt_calc_jav_input[:, 3] = jerk_vel_deg1_mags * (step**3)
                gt_calc_jav_input[:, 4] = jerk_acc_ortho_mags * (step**3)
                gt_calc_jav_input[:, 5] = jerk_ortho_norms.flatten() * (step**3)
                
                jav_mats = np.stack([
                    unit_vels_deg1[-n_jerk_preds:],
                    unit_acc_ortho_deg1_vecs[-n_jerk_preds:],
                    pm.safelyNormalizeArray(jerk_ortho_vecs, jerk_ortho_norms)
                ], axis=1)

                prev_start = -(n_jerk_preds + 1)
                gt_calc_jav_input[:, 6:9] = pm.einsumMatVecMul(
                    jav_mats, translation_diffs[prev_start:-1]
                )

                gt_jav6 = gtMultipliers6(gt_calc_jav_input, self.base_JAV6)

                motion_data[MOTION_DATA.GT0] = gt_jav6[:, 0]
                motion_data[MOTION_DATA.GT1] = gt_jav6[:, 1]
                motion_data[MOTION_DATA.GT2] = gt_jav6[:, 2]
                motion_data[MOTION_DATA.GT3] = gt_jav6[:, 3]
                motion_data[MOTION_DATA.GT4] = gt_jav6[:, 4]
                motion_data[MOTION_DATA.GT5] = gt_jav6[:, 5]

            ck_denom = deg1_speeds_full.flatten()[1:]**(3/2)
            ck_num_terms = []
            vel_vecs = deg1_vels[1:]
            for exc_i in range(3):
                exc_ip = (exc_i + 1) % 3
                ck_num_term = \
                    pderivs.accelerations[:, exc_i] * vel_vecs[:, exc_ip]
                ck_num_term -= \
                    pderivs.accelerations[:, exc_ip] * vel_vecs[:, exc_i]
                ck_num_terms.append(ck_num_term**2)
            ck_num = np.sqrt(np.sum(ck_num_terms, axis=0))
            curvatures = ck_num / ck_denom
            motion_data[MOTION_DATA.CURVATURE] = curvatures[1:]
            motion_data[MOTION_DATA.LAST_CURVATURE] = curvatures[:-1]

            # Remove any specified keys.
            for k in self.other_exclusions:
                motion_data.pop(k, None)

            '''
            Slower curvature thing that didn't improve accuracy
            _, jav_frames = pm.getOrthonormalFrames(
                True, deg1_vels[2:], deg1_vel_diffs[1:], t_jerk_amt
            )

            curve_keys = (
                MOTION_DATA.CURVATURE_V, MOTION_DATA.CURVATURE_A,
                MOTION_DATA.CURVATURE_J
            )
            last_curve_keys = (
                MOTION_DATA.LAST_CURVATURE_V, MOTION_DATA.LAST_CURVATURE_A,
                MOTION_DATA.LAST_CURVATURE_J
            )

            for fn, frame in enumerate(jav_frames):
                t_jav_vecs = frame @ translations[fn:(fn + 3)]
                v_jav_vecs = np.diff(t_jav_vecs, 1, axis=0)
                a_jav_vecs = np.diff(v_jav_vecs, 1, axis=0)
                curvatures = np.abs(a_jav_vecs) / ((1 + v_jav_vecs[-2:]**2)**(3/2))
                for c, (ck, lck) in enumerate(zip(curve_keys, last_curve_keys)):
                    if fn == 0:
                        motion_data[ck] = np.empty(n_jerk_preds)
                        motion_data[lck] = np.empty(n_jerk_preds)
                    motion_data[ck][fn] = curvatures[-1, c]
                    motion_data[lck][fn] = curvatures[0, c]
            '''


            if check_key_completeness:
                self.validateKeys(motion_data.keys())
                
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

def _listOfEmptyDicts(size: int):
    return [dict() for _ in range(size)]

def dataForCombosJAV(pose_loaders: PoseLoaderList, vec_order: OrderForJAV,
                     return_world2locals: bool = False, 
                     return_translations: bool = False,
                     return_rotation_mats: bool = False,
                     return_rotation_vels: bool = False
                     ):
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
    all_data: NumpyForSkipAndID = _listOfEmptyDicts(3)
    all_world2local_mats: NumpyForSkipAndID = _listOfEmptyDicts(3)
    all_translations: NumpyForSkipAndID = _listOfEmptyDicts(3)

    all_rotation_mats: NumpyForSkipAndID = _listOfEmptyDicts(3)
    all_rotation_vels: NumpyForSkipAndID = _listOfEmptyDicts(3)
    
    if vec_order is None:
        vec_order = (JAV.VELOCITY, JAV.ACCELERATION, JAV.JERK)
    if len(vec_order) != 3 or {v.value for v in vec_order} != {1, 2, 3}:
        raise ValueError("Vector order must be a permutation of (velocity, acceleration, jerk)!")
    
    skip_end = 3#1 if onlySkip0 else 3
    for calc_obj in pose_loaders:
        c = calc_obj.getVidID()
        curr_translations = calc_obj.getTranslationsGTNP()

        # Only applicable if we want to return associated rotations.
        curr_rotation_mats: typing.Optional[NDArray] = None
        if return_rotation_mats or return_rotation_vels:
            curr_rotation_mats = calc_obj.getRotationMatsGTNP()
        
        for skip in range(skip_end):
            step = skip + 1
            translations = curr_translations[::step]
            displacements = np.diff(translations, axis=0)
            # We need a velocity for the last timestep, but not an acceleration,
            # because we need the vectors that take each current position to
            # the next when calculating the "ground truth" for displacement
            # predictions. This is the velocity vector; acceleration vectors
            # are not needed for this; we only need "current" acceleration.
            times: typing.Optional[NDArray] = None
            if not calc_obj.areTimestampsConst():
                times = calc_obj.getTimestamps()[::step][:-1]
            mds = PositionDerivativeCollection(displacements[:-1], times)
            vels = mds.velocities
            accs = mds.accelerations
            jerks = mds.jerks
            snaps = np.insert(mds.snaps, 0, np.zeros(3), axis=0)
            crackles = np.concatenate((np.zeros((2, 3)), mds.crackles), axis=0)

            # Here we specify which order in which we orthonormalize our
            # velocity, acceleration, and jerk vectors into orthonormal frames.
            # The first-chosen of these gets aligned exactly with an axis, while
            # the others only get orthogonal components aligned with an axis.
            
            # To only calculate as much as we need, we clip the arrays' fronts
            # off when we can.
            default_ordered = (vels[2:], accs[1:], jerks)
            ordered = copy.copy(default_ordered)
            if vec_order is not None:
                ordered = tuple(default_ordered[v.value - 1] for v in vec_order)


            all_mags, mats = pm.getOrthonormalFrames(True, *ordered, False)
                        
            # Transform each third vector and to-next-frame displacement into
            # this frame via matmul.
            # local_vecs2 = pm.einsumMatVecMul(mats, ordered[2])
            local_diffs = pm.einsumMatVecMul(mats, displacements[3:])
            local_snaps = pm.einsumMatVecMul(mats, snaps)
            local_crackles = pm.einsumMatVecMul(mats, crackles)

            # We'll now return all of the data needed to convert velocity,
            # acceleration, and jerk multipliers into local vectors in these
            # new frames. To do this, we don't need to return the coordinate
            # frames themselves: we just need to know the velocity in this
            # frame (a vector [speed, 0, 0]), the acceleration in this frame
            # (i.e. [a_p, a_o, 0]), etc. And since we don't need to return 0s,
            # we can just return the following:
            c_res = (
                *all_mags, *(local_snaps.T), *(local_crackles.T),
                *(local_diffs.T)
            )

            all_data[skip][c] = np.stack(c_res, axis=-1)
            if return_world2locals:
                all_world2local_mats[skip][c] = mats
            if return_translations:
                all_translations[skip][c] = translations
            if return_rotation_mats or return_rotation_vels:
                rotation_mats = curr_rotation_mats[::step]
                if return_rotation_mats:
                    all_rotation_mats[skip][c] = rotation_mats
                if return_rotation_vels:
                    rev_rotation_mats = np.swapaxes(rotation_mats[:-1], -2, -1)
                    rotation_vel_mats = pm.einsumMatMatMul(
                        rotation_mats[1:], rev_rotation_mats
                    )
                    all_rotation_vels[skip][c] = pm.axisAngleFromMatArray(
                        rotation_vel_mats
                    )

    if return_world2locals or return_translations:
        res = (all_data, )
        if return_world2locals:
            res += (all_world2local_mats, )
        if return_translations:
            res += (all_translations, )
        if return_rotation_mats:
            res += (all_rotation_mats, )
        if return_rotation_vels:
            res += (all_rotation_vels, )
        return res
    return all_data

