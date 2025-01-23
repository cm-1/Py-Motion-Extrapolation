#%%
import typing
from collections import namedtuple
from enum import Enum

import numpy as np
from sklearn import tree as sk_tree
from sklearn.model_selection import train_test_split

import posemath as pm
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
    all_rotations[combo:2] = quats
#%%
class MOTION_MODEL(Enum):
    STATIC = 1
    VEL_DEG1 = 2
    VEL_DEG2 = 3
    ACC_DEG2 = 4
    JERK = 5

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

def get_per_combo_and_motion_mod_datastruct():
    return {mm: [
        {ck[:2]: None for ck in combos} for _ in range(3)
    ] for mm in MOTION_MODEL}

def get_per_combo_datastruct():
    return [{ck[:2]: None for ck in combos} for _ in range(3)]

err3D_lists = get_per_combo_and_motion_mod_datastruct()
err_norm_lists = get_per_combo_and_motion_mod_datastruct()
min_norm_labels = get_per_combo_datastruct()

motion_data = {md: {ck[:2]: [] for ck in combos} for md in MOTION_DATA}

for skip_amt in range(3):
    step = 1 + skip_amt

    for combo in combos:
        c2 = combo[:2]
        translations = all_translations[c2][::step]
        n_next_translations = len(translations) - 1

        quats = all_rotations[c2][::step][:-1]
        inv_quats = pm.conjugateQuats(quats)
        quat_diffs = pm.multiplyQuatLists(quats[1:], inv_quats[:-1])

        vel_axes, vel_angs = pm.axisAnglesFromQuats(quat_diffs)

        prev_translations = translations[:-1]
        deg1_vels = np.diff(prev_translations, 1, axis=0)
        deg2_acc = np.diff(deg1_vels, 1, axis=0)
        half_deg2_acc = deg2_acc / 2
        
        deg2_vels = deg1_vels[1:] + half_deg2_acc

        t_jerk_preds = 4 * translations[3:-1] - 6 * translations[2:-2] + 4 * translations[1:-3] - translations[:-4]

        t_jerk_amt = translations[3:-1] - 3*deg1_vels[1:-1] + translations[:-4]

        temp_preds = dict()
        temp_preds[MOTION_MODEL.STATIC] = prev_translations
        temp_preds[MOTION_MODEL.VEL_DEG1] = translations[1:-1] + deg1_vels
        temp_preds[MOTION_MODEL.VEL_DEG2] = translations[2:-1] + deg2_vels
        temp_preds[MOTION_MODEL.ACC_DEG2] = \
            temp_preds[MOTION_MODEL.VEL_DEG2] + half_deg2_acc
        temp_preds[MOTION_MODEL.JERK] = t_jerk_preds

        n_jerk_preds = len(t_jerk_preds)

        motion_data[MOTION_DATA.TIMESTEP][c2].append(np.full(n_jerk_preds, step))

        deg1_vel_subset = deg1_vels[-n_jerk_preds:]
        deg2_vel_subset = deg1_vels[-n_jerk_preds:]
        acc_subset = deg2_acc[-n_jerk_preds:]

        deg1_speeds = np.linalg.norm(deg1_vel_subset, axis=-1, keepdims=True)
        deg2_speeds = np.linalg.norm(deg2_vel_subset, axis=-1, keepdims=True)
        acc_mags = np.linalg.norm(acc_subset, axis=-1, keepdims=True)
        motion_data[MOTION_DATA.SPEED_DEG1][c2].append(deg1_speeds.flatten() / step)
        motion_data[MOTION_DATA.SPEED_DEG2][c2].append(deg2_speeds.flatten() / step)
        motion_data[MOTION_DATA.ACC_MAG][c2].append(acc_mags.flatten() / step**2)
        motion_data[MOTION_DATA.JERK_MAG][c2].append(np.linalg.norm(
            t_jerk_amt, axis=-1
        ) / step**3)

        motion_data[MOTION_DATA.ROT_SPEED][c2].append(
            vel_angs[-n_jerk_preds:] / step
        )

        unit_vels_deg1 = deg1_vel_subset / deg1_speeds
        unit_vels_deg2 = deg2_vel_subset / deg2_speeds
        unit_accs = acc_subset / acc_mags
        vd1a_dots = np.clip(pm.einsumDot(unit_vels_deg1, unit_accs), -1.0, 1.0)
        vd2a_dots = np.clip(pm.einsumDot(unit_vels_deg2, unit_accs), -1.0, 1.0)
        motion_data[MOTION_DATA.ACC_ANG_WITH_VEL_DEG1][c2].append(
            np.arccos(vd1a_dots)
        )
        motion_data[MOTION_DATA.ACC_ANG_WITH_VEL_DEG2][c2].append(
            np.arccos(vd2a_dots)
        )

        curr_err_norms = np.empty((len(MOTION_MODEL), n_jerk_preds))

        t_subset = translations[-n_jerk_preds:]

        pre_jerk_preds = np.empty(4)
        for i, motion_mod in enumerate(MOTION_MODEL):
            pred_subset = temp_preds[motion_mod][-n_jerk_preds:]

            errs = t_subset - pred_subset
            err3D_lists[motion_mod][skip_amt][c2] = errs

            curr_err_norms[i] = np.linalg.norm(errs, axis=-1)
            err_norm_lists[motion_mod][skip_amt][c2] = curr_err_norms[i]
        
        min_norm_labels[skip_amt][c2] = np.argmin(curr_err_norms, axis=0)
        last_labels = np.roll(min_norm_labels[skip_amt][c2], 1)
        last

motion_kinds_plus = motion_kinds + ["all"]
#%%
def concatForComboSubset(data, combo_subset):
    ret_val = []
    for els_for_skip in data:
        subset_via_combos = [els_for_skip[ck[:2]] for ck in combo_subset]
        concated = np.concatenate(subset_via_combos) 
        ret_val.append(concated)
    return ret_val

def combosByBod(bods):
    return [c for c in combos if c[0] in bods]

# def concatForKeys(data, keys):
#     return np.stack((data[k] for k in keys))

#%%

bod_arange = np.arange(len(gtc.BCOT_BODY_NAMES), dtype=int)
train_bodies, test_bodies = train_test_split(bod_arange, test_size = 0.2, random_state=0)

train_combos = combosByBod(train_bodies)
test_combos = combosByBod(test_bodies)

train_speeds = concatForComboSubset(speeds_deg1, train_combos)
train_labels = concatForComboSubset(min_norm_labels, train_combos)
test_labels = concatForComboSubset(min_norm_labels, test_combos)
test_speeds = concatForComboSubset(speeds_deg1, test_combos)

#%%
import matplotlib.pyplot as plt
TRAIN_SKIP = 2

max_depth = 14
scores = np.empty(max_depth)
depths = np.arange(1, max_depth + 1)
for d in depths:
    clf = sk_tree.DecisionTreeClassifier(max_depth=d)
    clf = clf.fit(train_speeds[TRAIN_SKIP].reshape(-1, 1), train_labels[TRAIN_SKIP])

    scores[d-1] = clf.score(test_speeds[TRAIN_SKIP].reshape(-1,1), test_labels[TRAIN_SKIP])

plt.plot(depths, scores)
plt.show()

#%%
clf = sk_tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(train_speeds[TRAIN_SKIP].reshape(-1, 1), train_labels[TRAIN_SKIP])



sk_tree.plot_tree(
    clf, feature_names=["v"], class_names=['1', '2', '3', '4', '5'], fontsize=5
)
plt.show()
