#%% Imports
import typing
from enum import Enum
import copy
import time

import numpy as np
from numpy.typing import NDArray

# Decision tree imports ========================================================
from sklearn import tree as sk_tree
from sklearn.tree._tree import TREE_LEAF
from sklearn.model_selection import train_test_split

from custom_tree.weighted_impurity import WeightedErrorCriterion

# Local code imports ===========================================================
import posemath as pm
import poseextrapolation as pex
from gtCommon import BCOT_Data_Calculator
import gtCommon as gtc

OBJ_IS_STATIC_THRESH_MM = 10.0 # 20 millimeters; semi-arbitrary
STRAIGHT_LINE_ANG_THRESH = np.deg2rad(5)
CIRC_ERR_RADIUS_RATIO_THRESH = 0.10

motion_kinds = [
    "movable_handheld", "movable_suspension", "static_handheld",
    "static_suspension", "static_trans"
]

combos = []
for s, s_val in enumerate(gtc.BCOT_SEQ_NAMES):
    k = ""
    for k_opt in motion_kinds:
        if k_opt in s_val:
            k = k_opt
            break
    for b in range(len(gtc.BCOT_BODY_NAMES)):
        if BCOT_Data_Calculator.isBodySeqPairValid(b, s, True):
            combos.append((b, s, k))

all_translations: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
all_rotations: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
all_rotation_mats: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
for combo in combos:
    calculator = BCOT_Data_Calculator(combo[0], combo[1], 0)
    all_translations[combo[:2]] = calculator.getTranslationsGTNP(False)
    aa_rotations = calculator.getRotationsGTNP(False)
    quats = pm.quatsFromAxisAngleVec3s(aa_rotations)
    all_rotations[combo[:2]] = quats
    all_rotation_mats[combo[:2]] = calculator.getRotationMatsGTNP(False)
#%%
class MOTION_MODEL(Enum):
    STATIC = 1
    VEL_DEG1 = 2
    VEL_DEG2 = 3
    ACC_DEG2 = 4
    JERK = 5
    CIRC = 6


# Minimum Jerk params? E.g., time/dist to motion start/end?
# 
# Then have to debug and make sure the attributes were calculated correctly...
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
    CIRC_ERR_VEC3 = 10
    JERK_ERR_VEC3 = 11
    AX3_SQ_DIFF = 12
    ROTATION_VEC3 = 13
    CIRC_ANG_SPEED = 14
    CIRC_ANG_ACC = 15

    DISP_MAG_DIFF = 16
    DISP_MAG_DIFF_TIMESCALED = 17
    DISP_MAG_RATIO = 18
    BOUNCE_ANGLE = 19

    UNIT_ROT_AX_DIFF = 20
    UNIT_ROT_AX_DIFF_TIMESCALED = 21

    RAD_DIFF = 22
    TIMESCALED_RAD_DIFF = 23

    # Norms of vec6s composed of circle centres and radii-scaled normals.
    CIRC_VEC6_DIFF = 24

    TIME_SINCE_STATIONARY = 25
    TIME_SINCE_DIR_CHANGE = 26
    DIST_SINCE_DIR_CHANGE = 27

    TIME_CIRC_MOTION = 28
    ANG_SUM_CIRC_MOTION = 29
    DIST_SUM_CIRC_MOTION = 30

    CIRC_CENTRE_DIFF = 31

    ROT_ACC_VEC3 = 32
    FRAME_NUM = 33
    TIMESTAMP = 34

    PREV_FA_ANG_ACC = 35
    NEXT_FA_ANG_ACC = 36

    SPEED_ACC_RATIO = 37
    VEL_BCS_RATIOS = 38

    ORTHO_ACC_MAG = 39

class RELATIVE_AXIS(Enum):
    VEL_DEG1 = 1
    VEL_DEG2 = 2
    ACC_FULL = 3
    ACC_ORTHO_DEG2 = 4
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

# The below code will use MOTION_DATA_KEY_TYPE classes so often that some
# shorter aliases might be helpful.
MD = MOTION_DATA
SMD = SpecifiedMotionData
RA = RELATIVE_AXIS
AM = ANG_OR_MAG
V3D = Vec3Data

def since_calc(bool_inds: np.ndarray, num_total_input_frames: int, 
               features_to_sum: typing.List[np.ndarray],
               prev_inds_included_in_sum: int):
    # If we have n+1 total frames and n "input" frames, then we have n-1
    # velocities, n-2 accelerations, etc. so the number of bools for our events
    # is n-k for some k. But to make things a bit easier to USE (even if 
    # CREATION is harder), I think all arrays should be length n. 
    # We'll also init using 0s, not empty, in order to propagate last indices.
    last_inds = np.zeros(num_total_input_frames, dtype=np.int32)
    k = num_total_input_frames - len(bool_inds)
    # We need to shift all bool-to-int inds by k to compensate.
    int_inds = np.where(bool_inds)[0] + k
    # We'll assume the event happened for the first k frames, because otherwise
    # we must make an arbitrary nonzero choice of time-since-event for them. 
    for i in range(1, k): # last_inds[0] == 0 already.
        last_inds[i] = i
    last_inds[int_inds] = int_inds
    
    # As described better (e.g., efficiency) in other comments in my xode where
    # I do ths,we'll use the technique proposed in a 2015-05-27 StackOverflow 
    # answer by user "jme" (1231929/jme) to a 2015-05-27 question, "Fill zero 
    # values of 1d numpy array with last non-zero values" 
    # (https://stackoverflow.com/q/30488961) by user "mgab" (3406913/mgab). 
    last_inds = np.maximum.accumulate(last_inds)
    
    time_since = np.arange(num_total_input_frames) - last_inds
    
    ret_features: typing.List[np.ndarray] = []
    for feature in features_to_sum:
        feature_len_diff = num_total_input_frames - len(feature)
        # We want to take the sum since frame 0 to the "present" and subtract 
        # the sum from frame 0 until the last event.
        feature_sums = np.empty(num_total_input_frames)
        feature_sums[:feature_len_diff] = 0.0
        feature_sums[feature_len_diff:] = np.cumsum(feature)
        start_inds = last_inds - prev_inds_included_in_sum
        inds_to_subtract = np.maximum(start_inds, 0)
        feature_since = feature_sums - feature_sums[inds_to_subtract]
        ret_features.append(feature_since)
        
    
    return (time_since, ret_features)

#%%

motion_mod_keys = [MOTION_MODEL(i) for i in range(1, len(MOTION_MODEL) + 1)] 

MOTION_DATA_DICT_TYPE = typing.List[typing.Dict[
    typing.Tuple[int, int], typing.Dict[MOTION_DATA_KEY_TYPE, np.ndarray]
]]
def get_per_combo_datastruct() -> MOTION_DATA_DICT_TYPE:
    return [{ck[:2]: None for ck in combos} for _ in range(3)]

err3D_lists = get_per_combo_datastruct()
err_norm_lists = get_per_combo_datastruct()
min_norm_labels = get_per_combo_datastruct()

all_motion_data = get_per_combo_datastruct()

for skip_amt in range(3):
    step = 1 + skip_amt
    step_sq = step**2

    for combo in combos:
        c2 = combo[:2]
        translations = all_translations[c2][::step]
        n_next_translations = len(translations) - 1

        quats = all_rotations[c2][::step][:-1]
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
        temp_preds[MOTION_MODEL.CIRC] = cma.c_trans_preds

        n_jerk_preds = len(t_jerk_preds)

        motion_data = dict()
        motion_data[MOTION_DATA.TIMESTEP] = np.full(n_jerk_preds, step)

        unit_rot_ax_diffs = np.diff(vel_axes, 1, axis=0)[-n_jerk_preds:]
        unit_rot_ax_diff_mags = np.linalg.norm(unit_rot_ax_diffs, axis=-1)

        motion_data[MOTION_DATA.UNIT_ROT_AX_DIFF] = unit_rot_ax_diff_mags
        motion_data[MOTION_DATA.UNIT_ROT_AX_DIFF_TIMESCALED] = \
                                                    unit_rot_ax_diff_mags / step

        deg1_vel_subset = deg1_vels[-n_jerk_preds:]
        deg2_vel_subset = deg1_vels[-n_jerk_preds:]

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

        disp_mag_diffs = np.diff(deg1_speeds_full, 1, axis=0)[-n_jerk_preds:].flatten()
        motion_data[MOTION_DATA.DISP_MAG_DIFF] = disp_mag_diffs
        motion_data[MOTION_DATA.DISP_MAG_DIFF_TIMESCALED] = disp_mag_diffs / step
        disp_mag_div = deg1_speeds_full[1:] / deg1_speeds_full[:-1]
        motion_data[MOTION_DATA.DISP_MAG_RATIO] = disp_mag_div[-n_jerk_preds:].flatten()

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

        
        motion_data[MOTION_DATA.CIRC_VEC6_DIFF] = cma.vec6CircleDists()

        radii_diffs = np.diff(radii, 1, axis=0)[-n_jerk_preds:]
        motion_data[MOTION_DATA.RAD_DIFF] = radii_diffs
        motion_data[MOTION_DATA.TIMESCALED_RAD_DIFF] = radii_diffs / step

        rotation_mats = all_rotation_mats[c2][::step]
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

        mat_diffs = xyz_axes[1:] - xyz_axes[:-1]
        vec9s = mat_diffs.reshape(-1,9)[-n_jerk_preds:]
        motion_data[MOTION_DATA.AX3_SQ_DIFF] = pm.einsumDot(vec9s, vec9s)

        c_centre_diff_vecs = np.diff(cma.getCentres3D(), axis=0)
        c_centre_diff_norms = np.linalg.norm(c_centre_diff_vecs, axis=-1)
        motion_data[MOTION_DATA.CIRC_CENTRE_DIFF] = c_centre_diff_norms

        
        t_diff_angs = pm.anglesBetweenVecs(
            unit_vels_deg1[:-1], unit_vels_deg1[1:], False
        )
        motion_data[MOTION_DATA.BOUNCE_ANGLE] = t_diff_angs[-n_jerk_preds:]

        

        d_under_thresh = deg1_speeds_full < OBJ_IS_STATIC_THRESH_MM
        a_over_thresh = t_diff_angs > STRAIGHT_LINE_ANG_THRESH
     


        time_since_static, _ = since_calc(d_under_thresh, n_input_frames, [], 0)
        # We want the total distance traveled since there was last a big angle. 
        time_since_big_ang, (dist_since_big_ang,) = since_calc(
            a_over_thresh, n_input_frames, [deg1_speeds_full], 1
        )


        motion_data[MOTION_DATA.TIME_SINCE_STATIONARY] = time_since_static[-n_jerk_preds:] * step
        motion_data[MOTION_DATA.TIME_SINCE_DIR_CHANGE] = time_since_big_ang[-n_jerk_preds:] * step
        motion_data[MOTION_DATA.DIST_SINCE_DIR_CHANGE] = dist_since_big_ang[-n_jerk_preds:]

        
        
        non_circ_bool_inds = cma.isMotionStillCircular(
            prev_translations[3:], CIRC_ERR_RADIUS_RATIO_THRESH
        )

        time_circ, (ang_sum_circ, dist_sum_circ) = since_calc(
            non_circ_bool_inds, n_input_frames, 
            [cma.second_angles, radii * cma.second_angles], 3
        )

        time_circ = (time_circ + 3) * step

        motion_data[MOTION_DATA.TIME_CIRC_MOTION] = time_circ[-n_jerk_preds:]
        motion_data[MOTION_DATA.ANG_SUM_CIRC_MOTION] = ang_sum_circ[-n_jerk_preds:]
        motion_data[MOTION_DATA.DIST_SUM_CIRC_MOTION] = dist_sum_circ[-n_jerk_preds:]

        frame_nums = np.arange(n_input_frames - n_jerk_preds, n_input_frames)
        motion_data[MOTION_DATA.FRAME_NUM] = frame_nums
        motion_data[MOTION_DATA.TIMESTAMP] = frame_nums * step

        # ======================================================================
        # Calculating prediction errors starts here!

        curr_err_norms = np.empty((len(MOTION_MODEL), n_jerk_preds + 1))

        t_subset = translations[-(n_jerk_preds + 1):] # To calc errors against.

        curr_err_norms_dict = dict()
        curr_errs_3D = dict()
        for i, motion_mod in enumerate(motion_mod_keys):
            pred_subset = None
            if motion_mod != MOTION_MODEL.JERK:
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
        min_norm_labels[skip_amt][c2] = curr_min_norm_labels[1:]
        motion_data[MOTION_DATA.LAST_BEST_LABEL] = curr_min_norm_labels[:-1]

        
        circ_ind = MOTION_MODEL.CIRC.value - 1
        jerk_ind = MOTION_MODEL.JERK.value - 1
        prev_circ_errs = curr_errs_3D[MOTION_MODEL.CIRC][:-1]
        prev_circ_err_mags = curr_err_norms[circ_ind][:-1]
    

        step_4_pow = step**4
        prev_jerk_errs = curr_errs_3D[MOTION_MODEL.JERK][:-1] / step_4_pow
        prev_jerk_errs[0] = 0 # Overwrite with 0 so it = the "4th derivative" 
        prev_jerk_err_mags = curr_err_norms[jerk_ind][:-1]
        prev_jerk_err_mags[0] = 0
        
        acc_vel_deg2_mag = pm.einsumDot(deg2_accs, unit_vels_deg2)
        acc_vel_deg2_parallel = pm.scalarsVecsMul(acc_vel_deg2_mag, unit_vels_deg2)
        
        acc_ortho_deg2_vecs = deg2_accs - acc_vel_deg2_parallel
        acc_ortho_deg2_mags = np.linalg.norm(
            acc_ortho_deg2_vecs, axis=-1, keepdims=True
        )
        
        unit_acc_ortho_deg2_vecs = pm.safelyNormalizeArray(
            acc_ortho_deg2_vecs, acc_ortho_deg2_mags
        )
        motion_data[MOTION_DATA.ORTHO_ACC_MAG] = acc_ortho_deg2_mags[-n_jerk_preds:].flatten()

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

        a_v2_mag_k       = SMD(MD.ACC_VEC3, RA.VEL_DEG2, AM.MAG, False, False)
        a_v2_mag_k_bidir = SMD(MD.ACC_VEC3, RA.VEL_DEG2, AM.MAG, True,  False)
        a_v2_ang_k       = SMD(MD.ACC_VEC3, RA.VEL_DEG2, AM.ANG, False, False)
        a_v2_ang_k_bidir = SMD(MD.ACC_VEC3, RA.VEL_DEG2, AM.ANG, True,  False)

        motion_data[a_v2_mag_k] = acc_vel_deg2_mag[-n_jerk_preds:]
        motion_data[a_v2_mag_k_bidir] = np.abs(acc_vel_deg2_mag[-n_jerk_preds:])
        acc_vel_deg2_angs = np.arcsin(np.clip(sin_vel_deg2_acc_angs, -1.0, 1.0))
        acc_vel_deg2_ang_subset = acc_vel_deg2_angs[-n_jerk_preds:].flatten()
        motion_data[a_v2_ang_k] = acc_vel_deg2_ang_subset
        motion_data[a_v2_ang_k_bidir] = pm.getAcuteAngles(acc_vel_deg2_ang_subset)

        ortho_dirs = pm.safelyNormalizeArray(vel_deg2_acc_cross, sin_vel_deg2_acc_angs)

        rot_v3d = V3D(timescaled_vel_axes, vel_axes, vel_angs_timescaled[-n_jerk_preds:])
        vec3s_dict: typing.Dict[MOTION_DATA, Vec3Data] = {
            MD.ACC_VEC3: V3D(deg2_accs, unit_accs, acc_mags),
            MD.JERK_VEC3: V3D(t_jerk_amt / step**3), MD.ROTATION_VEC3: rot_v3d,
            MD.CIRC_ERR_VEC3: V3D(prev_circ_errs, None, prev_circ_err_mags),
            MD.JERK_ERR_VEC3: V3D(prev_jerk_errs, None, prev_jerk_err_mags),
            MD.ROT_ACC_VEC3: V3D(rot_accs[-n_jerk_preds:], None, None)
        }
        for md_k, v3s in vec3s_dict.items():
            motion_data[SMD(md_k, RA.ITSELF, AM.MAG, False, False)] = v3s.norms


        rel_axes_dict: typing.Dict[RELATIVE_AXIS, np.ndarray] = {
            RA.VEL_DEG1: unit_vels_deg1, RA.VEL_DEG2: unit_vels_deg2,
            RA.ACC_FULL: unit_accs, RA.ACC_ORTHO_DEG2: unit_acc_ortho_deg2_vecs,
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
            
                    k_m       = SMD(md_k, ra_k, AM.MAG, False, shiftRel)
                    k_m_bidir = SMD(md_k, ra_k, AM.MAG, True,  shiftRel)
                    k_a       = SMD(md_k, ra_k, AM.ANG, False, shiftRel)
                    k_a_bidir = SMD(md_k, ra_k, AM.ANG, True,  shiftRel)

                    motion_data[k_m] = ra_dots[:, ra_row]
                    motion_data[k_m_bidir] = np.abs(ra_dots[:, ra_row])
                    
                    curr_angs = pm.anglesBetweenVecs(
                        rel_axes_dict[ra_k][ra_start:ra_end],
                        v3s.unit_vecs[-n_jerk_preds:], False
                    )
                    if v3s.norms[0] == 0.0:
                        curr_angs[0] = 0.0
                    motion_data[k_a] = curr_angs
                    motion_data[k_a_bidir] = pm.getAcuteAngles(curr_angs)

        all_motion_data[skip_amt][c2] = motion_data
        err_norm_lists[skip_amt][c2] = curr_err_norms_dict
        err3D_lists[skip_amt][c2] = curr_errs_3D

# Make sure all keys present.
missing_keys: typing.List[MOTION_DATA] = []
for motion_data_kind in MOTION_DATA:
    md_kind_present = False
    for k in all_motion_data[0][combos[0][:2]].keys():
        if isinstance(k, MOTION_DATA):
            md_kind_present = md_kind_present or k == motion_data_kind
        elif isinstance(k, SpecifiedMotionData):
            md_kind_present = md_kind_present or k.base_cat == motion_data_kind
        else:
            raise ValueError("Bad key type! {}".format(k))
        
        if md_kind_present:
            break

    if not md_kind_present:
        missing_keys.append(motion_data_kind)

if len(missing_keys) > 0:
    raise Exception("Keys {} missing!".format([mk.name for mk in missing_keys]))

motion_kinds_plus = motion_kinds + ["all"]
#%%

def concatForComboSubset(data, combo_subset, front_trim: int = 0):
    ret_val = []
    for els_for_skip in data:
        subset_via_combos = [els_for_skip[ck[:2]] for ck in combo_subset]
        concated = None
        if isinstance(subset_via_combos[0], dict):
            concated = dict()
            for k in subset_via_combos[0].keys():
                concated[k] = np.concatenate([
                    svc[k][front_trim:] for svc in subset_via_combos
                ])
        else:
            concated = np.concatenate([
                svc[front_trim:] for svc in subset_via_combos
            ]) 
        ret_val.append(concated)
    return ret_val

def combosByBod(bods):
    return [c for c in combos if c[0] in bods]

def get2DArrayFromDataStruct(data: MOTION_DATA_DICT_TYPE, 
                             ks: typing.List[MOTION_DATA_KEY_TYPE] = None,
                             stack_axis: int = 0):
    if ks is None:
        ks = list(data[0].keys())
    concated = {k: np.concatenate([d[k] for d in data]) for k in ks}
    stacked = np.stack([concated[k] for k in ks], stack_axis)
    return ks, stacked

# def concatForKeys(data, keys):
#     return np.stack((data[k] for k in keys))

#%%

bod_arange = np.arange(len(gtc.BCOT_BODY_NAMES), dtype=int)
train_bodies, test_bodies = train_test_split(bod_arange, test_size = 0.2, random_state=0)

train_combos = combosByBod(train_bodies)
test_combos = combosByBod(test_bodies)


train_data = concatForComboSubset(all_motion_data, train_combos)
train_labels = concatForComboSubset(min_norm_labels, train_combos)
train_errs = concatForComboSubset(err_norm_lists, train_combos)
test_labels = concatForComboSubset(min_norm_labels, test_combos)
test_data = concatForComboSubset(all_motion_data, test_combos)
test_errs = concatForComboSubset(err_norm_lists, test_combos)


concat_train_labels = np.concatenate(train_labels)
concat_test_labels = np.concatenate(test_labels)
motion_data_keys, concat_train_data = get2DArrayFromDataStruct(train_data)
concat_test_data = get2DArrayFromDataStruct(test_data, motion_data_keys)[1]

concat_train_errs = get2DArrayFromDataStruct(train_errs, motion_mod_keys, -1)[1]
concat_test_errs = get2DArrayFromDataStruct(test_errs, motion_mod_keys, -1)[1]

#%%
best_seq_means = []
test_best_seq_means = []
for skip in range(3):
    best_seq_scores = []
    test_best_seq_scores = []
    for seq in range(len(gtc.BCOT_SEQ_NAMES)):
        seq_data = []
        test_seq_data = []
        for combo in combos:
            if combo[1] == seq:
                seq_combo_scores_stacked = np.stack(
                    list(err_norm_lists[skip][combo[:2]].values()), axis=-1
                )
                seq_data.append(seq_combo_scores_stacked)
                if combo[0] in test_bodies:
                    test_seq_data.append(seq_combo_scores_stacked)
        if len(seq_data) > 0:
            concat_seq_data = np.concatenate(seq_data, axis=0)
            concat_test_seq_data = np.concatenate(test_seq_data, axis=0)
            seq_sums = np.sum(concat_seq_data, axis=0)
            assert seq_sums.shape == (len(MOTION_MODEL), )
            assert concat_seq_data.shape[1] == len(MOTION_MODEL)
            
            seq_best_mode = np.argmin(seq_sums)
            best_seq_scores.append(concat_seq_data[:, seq_best_mode])
            test_best_seq_scores.append(concat_test_seq_data[:, seq_best_mode])
    seq_best_mean = np.mean(np.concatenate(best_seq_scores))
    seq_best_test_mean = np.mean(np.concatenate(test_best_seq_scores))
    
    best_seq_means.append(seq_best_mean)
    test_best_seq_means.append(seq_best_test_mean)
print("Best case one-model-per sequence results for skips 0, 1, 2:")
print("All data:", best_seq_means)
print("Test data:", test_best_seq_means)


#%%
def assessPredError(concat_errs, pred_labels, inds_dict = None):
    if inds_dict is None:
        inds_dict = {"all:": None}
    ret_dict = dict()
    pred_labels_rs = pred_labels.reshape(-1,1)
    taken_errs = np.take_along_axis(concat_errs, pred_labels_rs, axis=1)
    for k, inds in inds_dict.items():
        ret_dict[k] = taken_errs[inds].mean()
    return ret_dict
#%%

s_inds = []
s_train_inds = []
ts_ind = motion_data_keys.index(MOTION_DATA.TIMESTEP)
for i in range(1,4): # We have data for frame steps of 1, 2, and 3.
    curr_s_inds = concat_test_data[ts_ind] == i
    s_inds.append(curr_s_inds)
    s_train_inds.append(concat_train_data[0] == i)

s_ind_dict = {"skip" + str(i): s_inds[i] for i in range(3)}
s_ind_dict["all"] = None

s_train_ind_dict = {"skip" + str(i): s_train_inds[i] for i in range(3)}
s_train_ind_dict["all"] = None

mc = WeightedErrorCriterion(1, np.array([len(MOTION_MODEL)], dtype=np.intp))
y_errs_reshape = concat_train_errs.reshape((
    concat_train_errs.shape[0], 1, concat_train_errs.shape[1]
))
mc.set_y_errs(y_errs_reshape)

error_lim = assessPredError(concat_test_errs, concat_test_labels, s_ind_dict)
print("Error limit:", error_lim)

# A tree depth of 8 is already way beyond "human-readable", and I think the
# graphs don't show miraculous improvements past 8, so 8 seems like a good max.
max_depth = 8


#%% Training decision tree at max depth.
# ---
# We'll train a tree at max depth and then trim it to smaller depths to evaulate
# the performance at lower depths. This yields the exact same trees as if we
# were to train individual ones with lower max depths (confirmed via tests), but
# eliminates duplicated training time. I might leave the old code for training
# individual trees below as a comment, for reference.

big_tree = sk_tree.DecisionTreeClassifier(max_depth=max_depth, criterion=mc)
print("Starting decision tree training!")
start_time = time.time()
big_tree = big_tree.fit(concat_train_data.T, concat_train_labels)
print("Done!")
print("Time spent:", time.time() - start_time)
#%%

# A recursive function to help trim_to_depth(). The params should be a tree's
# children_left and children_right, then a specified node's index and its
# depth, and a target tree depth. 
def _trim_to_depth_helper(left_children_inds: NDArray[np.signedinteger],
                          right_children_inds: NDArray[np.signedinteger],
                          current_node_index: int, current_node_depth: int,
                          target_depth: int):
    if current_node_depth >= target_depth:
        left_children_inds[current_node_index] = TREE_LEAF
        right_children_inds[current_node_index] = TREE_LEAF
    else:
        # Recurse left and right subtrees to set required nodes to leaves.
        _trim_to_depth_helper(
            left_children_inds, right_children_inds, 
            left_children_inds[current_node_index], current_node_depth + 1,
            target_depth
        )
        _trim_to_depth_helper(
            left_children_inds, right_children_inds, 
            right_children_inds[current_node_index], current_node_depth + 1,
            target_depth
        )
    return
        
def trim_to_depth(tree: sk_tree._classes.DecisionTreeClassifier, depth):
    assert depth >= 1, "Depth to trim tree to must be >= 1!"
    depth = min(depth, tree.get_depth())
    tree_copy = copy.deepcopy(tree)  # Make a copy to avoid modifying original.
    lefts = tree_copy.tree_.children_left
    rights = tree_copy.tree_.children_right

    _trim_to_depth_helper(lefts, rights, 0, 0, depth)
    return tree_copy

#%%
scores = {k: np.empty(max_depth) for k in s_ind_dict.keys()}
depths = np.arange(1, max_depth + 1)
for d in depths:
    # Preiously, we trained new trees from scratch using the below code, but as
    # described above, we'll instead trim the "main" tree to get new ones.
    #         clf = sk_tree.DecisionTreeClassifier(max_depth=d, criterion=mc)
    #         clf = clf.fit(concat_train_data.T, concat_train_labels)
    trimmed_tree = trim_to_depth(big_tree, d)
    graph_pred = trimmed_tree.predict(concat_test_data.T)
    scores_for_depth = assessPredError(concat_test_errs, graph_pred, s_ind_dict)
    for k, score_for_depth in scores_for_depth.items():
        scores[k][d-1] = score_for_depth

print("Scores for trimmed trees:")
print(scores)
#%%
import matplotlib.pyplot as plt

for k, score_sub in scores.items():
    #score_normed = (score_sub - score_sub.min()) / np.ptp(score_sub)
    curr_plt_ln, = plt.plot(depths, score_sub, label=k)
    err_lim_k = error_lim[k]
    curr_plt_col = curr_plt_ln.get_color()
    plt.plot(
        [1, max_depth], [err_lim_k, err_lim_k], color=curr_plt_col, dashes=[1,1]
    )
plt.legend()
plt.ylabel("Test Set Error")# (normed to [0,1])")
plt.xlabel("Max decision tree depth")
plt.show()
print("TODO: Add single 'limit' legend entry!")

#%%
# Again, we'll replace old code with a trimming of our main tree.
#         mclf = sk_tree.DecisionTreeClassifier(max_depth=4, criterion=mc)
#         mclf = mclf.fit(concat_train_data.T, concat_train_labels)
mclf = trim_to_depth(big_tree, 4)
mclfps = mclf.predict(concat_test_data.T).copy()

#%%

mc_errs = assessPredError(concat_test_errs, mclfps, s_ind_dict)
print("Error for my decision tree=", mc_errs)
print("Done!", mclfps)
#%%
from sklearn.tree import export_graphviz
import pathlib

tree_path = pathlib.Path(__file__).parent.resolve() / "results" / "tree.dot"
feature_names = [e.name for e in motion_data_keys]
class_names = [str(i) for i in range(1, len(MOTION_MODEL) + 1)]

export_graphviz(
    mclf, out_file=str(tree_path), 
    feature_names=feature_names, 
    class_names=class_names,
    filled=True, rounded=True, special_characters=True,
    
)
# Convert to .pdf with:
# Graphviz\bin\dot.exe -Tpdf tree.dot -o tree.pdf

#%%
import tensorflow as tf
import keras
#%%
def customPoseLoss(y_true, y_pred):
    probs = tf.nn.softmax(y_pred, axis=1)
    return tf.reduce_sum(y_true * probs, axis=1)
# Example Usage
tf_concat_train_errs = tf.convert_to_tensor(concat_train_errs, dtype=tf.float32)
tf_loss_fn = customPoseLoss # CustomLossWithErrors(concat_train_errs)

input_dim = concat_train_data.shape[0]
num_classes = len(MOTION_MODEL)

onehot_train_labels = tf.one_hot(concat_train_labels, num_classes)

tfmodel = keras.Sequential([
    keras.layers.Input((input_dim,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    # keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    # keras.layers.Dropout(0.2),
    # keras.layers.Dense(1, activation='sigmoid'),
    keras.layers.Dense(num_classes, activation='sigmoid')])

tfmodel.summary()

tf_loss_fn2 = keras.losses.CategoricalCrossentropy(from_logits=True)
tfmodel.compile(optimizer='adam', loss=tf_loss_fn)
tfmodel.fit(concat_train_data.T, concat_train_errs, epochs=5, shuffle=True)
#%%
bgtrain = big_tree.predict(concat_train_data.T)
bgtrain_score = assessPredError(concat_train_errs, bgtrain, s_train_ind_dict)
print("Big tree train errs=", bgtrain_score)

mclf_train = mclf.predict(concat_train_data.T)
mclf_train_score = assessPredError(concat_train_errs, mclf_train, s_train_ind_dict)
print("Smaller tree train errs=", mclf_train_score)

tf_train_preds = np.argmax(tfmodel(concat_train_data.T).numpy(), axis=1)
tf_train_errs = assessPredError(concat_train_errs, tf_train_preds, s_train_ind_dict)
print("TF train errs=", tf_train_errs)
print()

tf_test_preds = np.argmax(tfmodel(concat_test_data.T).numpy(), axis=1)
tf_test_errs = assessPredError(concat_test_errs, tf_test_preds, s_ind_dict)
print("TF test errs=", tf_test_errs)
