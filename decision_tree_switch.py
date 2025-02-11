#%%
import typing
from collections import namedtuple
from enum import Enum

import numpy as np
from sklearn import tree as sk_tree
from sklearn.model_selection import train_test_split

import posemath as pm
import poseextrapolation as pex
from gtCommon import BCOT_Data_Calculator
import gtCommon as gtc

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
for combo in combos:
    calculator = BCOT_Data_Calculator(combo[0], combo[1], 0)
    all_translations[combo[:2]] = calculator.getTranslationsGTNP(False)
    aa_rotations = calculator.getRotationsGTNP(False)
    quats = pm.quatsFromAxisAngleVec3s(aa_rotations)
    all_rotations[combo[:2]] = quats
#%%
class MOTION_MODEL(Enum):
    STATIC = 1
    VEL_DEG1 = 2
    VEL_DEG2 = 3
    ACC_DEG2 = 4
    JERK = 5
    CIRC = 6

# Jerk and Circle error mags.
#   Maybe parallel and ortho error mags?
#   Also err angles.
# Add "ang with ortho" to each ang option.
# Bounce angle
# 3-ax diff?
# Vel/acc ratios?
# Vel len diffs?
# circ angles?
class MOTION_DATA(Enum):
    SPEED_DEG1 = 1
    SPEED_DEG2 = 2
    ACC_MAG = 3
    JERK_MAG = 4
    TIMESTEP = 5
    ROT_SPEED = 6
    LAST_BEST_LABEL = 7
    ACC_ANG_WITH_VEL_DEG1 = 8
    ACC_ANG_WITH_VEL_DEG2 = 9
    ACC_MAG_ORTHO_DEG2 = 10
    ACC_MAG_PARALLEL_DEG2 = 11
    JERK_MAG_VEL_DEG2 = 12
    JERK_MAG_ACC = 13
    JERK_MAG_ORTHO = 14
    JERK_ANG_WITH_VEL_DEG1 = 15
    JERK_ANG_WITH_VEL_DEG2 = 16
    JERK_ANG_WITH_ACC = 17
    CIRC_RAD = 18
    CIRC_SPEED = 19
    CIRC_ACC = 20
    # CIRC_ERR_MAG = 21
    # CIRC_ERR_MAG_VEL_DEG2 = 22
    # CIRC_ERR_MAG_ACC = 23
    # CIRC_ERR_MAG_ORTHO = 24
    # CIRC_ERR_ANG_WITH_VEL_DEG1 = 25
    # CIRC_ERR_ANG_WITH_VEL_DEG2 = 26
    # CIRC_ERR_ANG_WITH_ACC = 27



def get_per_combo_datastruct():
    return [{ck[:2]: None for ck in combos} for _ in range(3)]

err3D_lists = get_per_combo_datastruct()
err_norm_lists = get_per_combo_datastruct()
min_norm_labels = get_per_combo_datastruct()

all_motion_data = get_per_combo_datastruct()

for skip_amt in range(3):
    step = 1 + skip_amt

    for combo in combos:
        c2 = combo[:2]
        translations = all_translations[c2][::step]
        n_next_translations = len(translations) - 1

        quats = all_rotations[c2][::step][:-1]
        inv_quats = pm.conjugateQuats(quats)
        quat_diffs = pm.multiplyQuatLists(quats[1:], inv_quats[:-1])

        vel_axes, vel_angs_unflat = pm.axisAnglesFromQuats(quat_diffs)
        vel_angs = vel_angs_unflat.flatten()

        translation_diffs = np.diff(translations, 1, axis=0)
        prev_translations = translations[:-1]
        deg1_vels = translation_diffs[:-1]
        deg2_acc = np.diff(deg1_vels, 1, axis=0)
        half_deg2_acc = deg2_acc / 2
        
        deg2_vels = deg1_vels[1:] + half_deg2_acc

        t_jerk_preds = 4 * translations[3:-1] - 6 * translations[2:-2] + 4 * translations[1:-3] - translations[:-4]

        t_jerk_amt = translations[3:-1] - 3*deg1_vels[1:-1] - translations[:-4]

        c_info = pex.circle_preds(translations, translation_diffs, None)

        temp_preds = dict()
        temp_preds[MOTION_MODEL.STATIC] = prev_translations
        temp_preds[MOTION_MODEL.VEL_DEG1] = translations[1:-1] + deg1_vels
        temp_preds[MOTION_MODEL.VEL_DEG2] = translations[2:-1] + deg2_vels
        temp_preds[MOTION_MODEL.ACC_DEG2] = \
            temp_preds[MOTION_MODEL.VEL_DEG2] + half_deg2_acc
        temp_preds[MOTION_MODEL.JERK] = t_jerk_preds
        temp_preds[MOTION_MODEL.CIRC] = c_info.predictions

        n_jerk_preds = len(t_jerk_preds)

        motion_data = dict()
        motion_data[MOTION_DATA.TIMESTEP] = np.full(n_jerk_preds, step)

        deg1_vel_subset = deg1_vels[-n_jerk_preds:]
        deg2_vel_subset = deg1_vels[-n_jerk_preds:]
        acc_subset = deg2_acc[-n_jerk_preds:]

        deg1_speeds = np.linalg.norm(deg1_vel_subset, axis=-1, keepdims=True)
        deg2_speeds = np.linalg.norm(deg2_vel_subset, axis=-1, keepdims=True)
        acc_mags = np.linalg.norm(acc_subset, axis=-1, keepdims=True)
        jerk_mags = np.linalg.norm(t_jerk_amt, axis=-1, keepdims=True) 
        motion_data[MOTION_DATA.SPEED_DEG1] = deg1_speeds.flatten() / step
        motion_data[MOTION_DATA.SPEED_DEG2] = deg2_speeds.flatten() / step
        motion_data[MOTION_DATA.ACC_MAG] = acc_mags.flatten() / step**2
        motion_data[MOTION_DATA.JERK_MAG] = jerk_mags.flatten() / step**3

        motion_data[MOTION_DATA.ROT_SPEED] = vel_angs[-n_jerk_preds:] / step

        unit_vels_deg1 = pm.safelyNormalizeArray(deg1_vel_subset, deg1_speeds)
        unit_vels_deg2 = pm.safelyNormalizeArray(deg2_vel_subset, deg2_speeds)
        unit_accs = pm.safelyNormalizeArray(acc_subset, acc_mags)
        motion_data[MOTION_DATA.ACC_ANG_WITH_VEL_DEG1] = pm.anglesBetweenVecs(
            unit_vels_deg1, unit_accs, False
        )
        motion_data[MOTION_DATA.ACC_ANG_WITH_VEL_DEG2] = pm.anglesBetweenVecs(
            unit_vels_deg2, unit_accs, False
        )
        acc_paral, acc_ortho = pm.parallelAndOrthoParts(
            acc_subset, unit_vels_deg2, True
        )
        motion_data[MOTION_DATA.ACC_MAG_PARALLEL_DEG2] = np.linalg.norm(
            acc_paral, axis=-1
        )
        motion_data[MOTION_DATA.ACC_MAG_ORTHO_DEG2] = np.linalg.norm(
            acc_ortho, axis=-1
        )

        deg2_vel_frames = pm.getOrthonormalFrames(deg2_vel_subset, acc_subset)
        deg2_vel_frames = np.swapaxes(deg2_vel_frames, -1, -2) # Transposes

        jerk_reframed = pm.einsumMatVecMul(deg2_vel_frames, t_jerk_amt)
        motion_data[MOTION_DATA.JERK_MAG_VEL_DEG2] = jerk_reframed[:, 0]
        motion_data[MOTION_DATA.JERK_MAG_ACC] = jerk_reframed[:, 1]
        motion_data[MOTION_DATA.JERK_MAG_ORTHO] = jerk_reframed[:, 2]

        unit_jerks = pm.safelyNormalizeArray(t_jerk_amt, jerk_mags)
        motion_data[MOTION_DATA.JERK_ANG_WITH_VEL_DEG1] = pm.anglesBetweenVecs(
            unit_vels_deg1, unit_jerks, False
        )
        motion_data[MOTION_DATA.JERK_ANG_WITH_VEL_DEG2] = pm.anglesBetweenVecs(
            unit_vels_deg2, unit_jerks, False
        )
        motion_data[MOTION_DATA.JERK_ANG_WITH_ACC] = pm.anglesBetweenVecs(
            unit_accs, unit_jerks, False
        )

        radii = np.sqrt(c_info.sq_radii)
        motion_data[MOTION_DATA.CIRC_RAD] = radii[-n_jerk_preds:]

        circ_speeds = radii * c_info.angle_pairs[1] / step
        motion_data[MOTION_DATA.CIRC_SPEED] = circ_speeds[-n_jerk_preds:]
        prev_circ_speeds = radii * c_info.angle_pairs[0] / step
        circ_accs = (circ_speeds - prev_circ_speeds) / step
        motion_data[MOTION_DATA.CIRC_ACC]  = circ_accs[-n_jerk_preds:]

        curr_err_norms = np.empty((len(MOTION_MODEL), n_jerk_preds))

        t_subset = translations[-n_jerk_preds:]

        pre_jerk_preds = np.empty((len(MOTION_MODEL), 3))
        ind_before_jerk = -(n_jerk_preds + 1)
        curr_err_norms_dict = dict()
        curr_errs_3D = dict()
        for i, motion_mod in enumerate(MOTION_MODEL):
            if motion_mod != MOTION_MODEL.JERK:
                el_before_jerk = temp_preds[motion_mod][ind_before_jerk]
                pre_jerk_preds[i] = el_before_jerk
            else:
                pre_jerk_preds[i] = np.inf
            pred_subset = temp_preds[motion_mod][-n_jerk_preds:]

            errs = t_subset - pred_subset
            curr_errs_3D[motion_mod] = errs

            curr_err_norms[i] = np.linalg.norm(errs, axis=-1)
            curr_err_norms_dict[motion_mod] = curr_err_norms[i]
        pre_jerk_errs = translations[ind_before_jerk] - pre_jerk_preds
        pre_jerk_norms = np.linalg.norm(pre_jerk_errs, axis=-1)
        min_norm_labels[skip_amt][c2] = np.argmin(curr_err_norms, axis=0)
        min_norms = np.min(curr_err_norms, axis=0)
        last_labels = np.roll(min_norm_labels[skip_amt][c2], 1)
        last_labels[0] = np.argmin(pre_jerk_norms)
        last_min_norms = np.roll(min_norms, 1)
        last_min_norms[0] = pre_jerk_norms[last_labels[0]]
        motion_data[MOTION_DATA.LAST_BEST_LABEL] = last_labels

        all_motion_data[skip_amt][c2] = motion_data
        err_norm_lists[skip_amt][c2] = curr_err_norms_dict
        err3D_lists[skip_amt][c2] = curr_errs_3D

motion_kinds_plus = motion_kinds + ["all"]
#%%

def concatForComboSubset(data, combo_subset, keys = None):
    ret_val = []
    for els_for_skip in data:
        subset_via_combos = [els_for_skip[ck[:2]] for ck in combo_subset]
        concated = None
        if isinstance(subset_via_combos[0], dict):
            concated = dict()
            for k in subset_via_combos[0].keys():
                concated[k] = np.concatenate([svc[k] for svc in subset_via_combos])
        else:
            concated = np.concatenate(subset_via_combos) 
        ret_val.append(concated)
    return ret_val

def combosByBod(bods):
    return [c for c in combos if c[0] in bods]

def get2DArrayFromDataStruct(data, ks = None, stack_axis = 0):
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
data_headers_enums, concat_train_data = get2DArrayFromDataStruct(train_data)
concat_test_data = get2DArrayFromDataStruct(test_data, data_headers_enums)[1]

motion_mod_keys = [MOTION_MODEL(i) for i in range(1, len(MOTION_MODEL) + 1)] 
concat_train_errs = get2DArrayFromDataStruct(train_errs, motion_mod_keys, -1)[1]
concat_test_errs = get2DArrayFromDataStruct(test_errs, motion_mod_keys, -1)[1]

#%%
def assessPredError(concat_errs, pred_labels):
    pred_labels_rs = pred_labels.reshape(-1,1)
    taken_errs = np.take_along_axis(concat_errs, pred_labels_rs, axis=1)
    return taken_errs.mean()

#%%
from custom_tree.weighted_impurity import WeightedErrorCriterion

from time import time

start_time = time()

s_inds = []
s_train_inds = []
mc_errs = []
for i in range(1,4): # We have data for frame steps of 1, 2, and 3.
    curr_s_inds = concat_test_data[0] == i
    s_inds.append(curr_s_inds)
    s_train_inds.append(concat_train_data[0] == i)

mc = WeightedErrorCriterion(1, np.array([len(MOTION_MODEL)], dtype=np.intp))
y_errs_reshape = concat_train_errs.reshape((
    concat_train_errs.shape[0], 1, concat_train_errs.shape[1]
))
mc.set_y_errs(y_errs_reshape)

error_lim = assessPredError(concat_test_errs, concat_test_labels)
print("Error limit:", error_lim)

# In tests up until now, best score was for depth 7, regardless of skip amt.
max_depth = 8
scores = np.empty((4, max_depth))
depths = np.arange(1, max_depth + 1)
print("Done depth:", end='', flush=True)
for d in depths:
    clf = sk_tree.DecisionTreeClassifier(max_depth=d, criterion=mc)
    clf = clf.fit(concat_train_data.T, concat_train_labels)
    graph_pred = clf.predict(concat_test_data.T)
    for i, s_ind_sub in enumerate(s_inds):
        scores[i, d-1] = assessPredError(
            concat_test_errs[s_ind_sub], graph_pred[s_ind_sub]
        )
    scores[3, d-1] = assessPredError(concat_test_errs, graph_pred)
    print(d, end=",", flush=True)
print("done!")
print("Time spent:", time() - start_time)
#%%
import matplotlib.pyplot as plt

for i, score_sub in enumerate(scores):
    i_label = "skip={}".format(i) if i < 3 else "all"
    #score_normed = (score_sub - score_sub.min()) / np.ptp(score_sub)
    plt.plot(depths, score_sub, label=i_label)
plt.plot([1, max_depth], [error_lim, error_lim], label="all (limit)")
plt.legend()
plt.ylabel("Test Set Error")# (normed to [0,1])")
plt.xlabel("Max decision tree depth")
plt.show()

#%%
mclf = sk_tree.DecisionTreeClassifier(max_depth=4, criterion=mc)
mclf = mclf.fit(concat_train_data.T, concat_train_labels)
mclfps = mclf.predict(concat_test_data.T).copy()

#%%

for sis in s_inds:
    curr_mc_err = assessPredError(concat_test_errs[sis, :], mclfps[sis])
    mc_errs.append(curr_mc_err)
mc_err_all = assessPredError(concat_test_errs, mclfps)
mc_errs.append(mc_err_all)
print("Error for my decision tree on skip=0,1,2,all:", *mc_errs)
print("Done!", mclfps)
#%%
from sklearn.tree import export_graphviz
import pathlib

tree_path = pathlib.Path(__file__).parent.resolve() / "results" / "tree.dot"
feature_names = [e.name for e in data_headers_enums]
class_names = [str(i) for i in range(1, len(MOTION_MODEL) + 1)]

export_graphviz(
    mclf, out_file=str(tree_path), 
    feature_names=feature_names, 
    class_names=class_names,
    filled=True, rounded=True, special_characters=True,
    
)
# Convert to .pdf with:
# Graphviz\bin\dot.exe -Tpdf tree.dot -o tree.pdf
