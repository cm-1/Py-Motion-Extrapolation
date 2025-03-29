#%% Imports
import typing
import copy
import time

import numpy as np
from numpy.typing import NDArray

# Decision tree imports ========================================================
from sklearn import tree as sk_tree
from sklearn.tree._tree import TREE_LEAF
from sklearn.model_selection import train_test_split

from posefeatures import MOTION_DATA, MOTION_MODEL
from posefeatures import MOTION_DATA_KEY_TYPE, CalcsForCombo


from custom_tree.weighted_impurity import WeightedErrorCriterion

# Local code imports ===========================================================
from gtCommon import BCOT_Data_Calculator
import gtCommon as gtc

OBJ_IS_STATIC_THRESH_MM = 10.0 # 10 millimeters; semi-arbitrary
STRAIGHT_LINE_ANG_THRESH_DEG = 30.0
CIRC_ERR_RADIUS_RATIO_THRESH = 0.10
MAX_MIN_JERK_OPT_ITERS = 33
MAX_SPLIT_MIN_JERK_OPT_ITERS = 33
ERR_NA_VAL = np.finfo(np.float32).max


motion_kinds = [
    "movable_handheld", "movable_suspension", "static_handheld",
    "static_suspension", "static_trans"
]
# motion_kinds_plus = motion_kinds + ["all"]

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


#%%



#%%

# NamedTuple combos.
nametup_combos = [gtc.Combo(*c[:2]) for c in combos]
cfc = CalcsForCombo(
    nametup_combos, obj_static_thresh_mm=OBJ_IS_STATIC_THRESH_MM, 
    straight_angle_thresh_deg=STRAIGHT_LINE_ANG_THRESH_DEG,
    err_na_val=ERR_NA_VAL, min_jerk_opt_iter_lim=MAX_MIN_JERK_OPT_ITERS,
    split_min_jerk_opt_iter_lim = MAX_SPLIT_MIN_JERK_OPT_ITERS,
    err_radius_ratio_thresh=CIRC_ERR_RADIUS_RATIO_THRESH
)
results = cfc.getAll()
all_motion_data = cfc.all_motion_data
min_norm_labels = cfc.min_norm_labels
err_norm_lists = cfc.err_norm_lists

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

def get2DArrayFromDataStruct(data: typing.List[typing.Dict[typing.Any, NDArray]], 
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

motion_mod_keys = [MOTION_MODEL(i) for i in range(1, len(MOTION_MODEL) + 1)] 
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
def getErrorPerSkip(concat_errs, pred_labels, inds_dict):
    pred_labels_rs = pred_labels.reshape(-1,1)
    taken_errs = np.take_along_axis(concat_errs, pred_labels_rs, axis=1)
    ret_dict = {k: taken_errs[inds] for k, inds in inds_dict.items()}
    return ret_dict

def assessPredError(concat_errs, pred_labels, inds_dict = None):
    if inds_dict is None:
        inds_dict = {"all:": None}
    all_errs_dict = getErrorPerSkip(concat_errs, pred_labels, inds_dict)
    mean_dict = {k: v.mean() for k, v in all_errs_dict.items()}
    return mean_dict
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
import posemath as pm
from sklearn.preprocessing import StandardScaler

ComboList = typing.List[gtc.Combo]
def dataForCombosJAV(combos: ComboList, onlySkip0: bool = False):
    all_data = [dict() for _ in range(3)]

    skip_end = 1 if onlySkip0 else 3
    for c in combos:
        calc_obj = BCOT_Data_Calculator(c.body_ind, c.seq_ind, 0)
        all_translations = calc_obj.getTranslationsGTNP(False)
        for skip in range(skip_end):
            step = skip + 1
            translations = all_translations[::step]
            vels = np.diff(translations, axis=0)
            accs = np.diff(vels[:-1], axis=0)
            jerks = np.diff(accs, axis=0)

            # We only calculate what we need, so we clip the arrays' fronts off.
            speeds = np.linalg.norm(vels[2:-1], axis=-1)
            unit_vels = pm.safelyNormalizeArray(vels[2:-1], speeds[:, np.newaxis])
            acc_p = pm.einsumDot(accs[1:], unit_vels)
            acc_p_vecs = pm.scalarsVecsMul(acc_p, unit_vels)
            acc_o_vecs = accs[1:] - acc_p_vecs
            acc_o = np.linalg.norm(acc_o_vecs, axis=-1)

            unit_acc_o = pm.safelyNormalizeArray(acc_o_vecs, acc_o[:, np.newaxis])

            plane_orthos = np.cross(unit_vels, unit_acc_o)
            mats = np.stack([unit_vels, unit_acc_o, plane_orthos], axis=1)

            local_jerks = pm.einsumMatVecMul(mats, jerks)

            local_diffs = pm.einsumMatVecMul(mats, vels[3:])

            c_res = {
                'v': speeds, 'a_p': acc_p, 'a_o': acc_o, 
                'j_v': local_jerks[:, 0], 'j_a': local_jerks[:, 1],
                'j_cross': local_jerks[:, 2], 'd_v': local_diffs[:, 0],
                'd_a': local_diffs[:, 1], 'd_cross': local_diffs[:, 2]
            }

            all_data[skip][c] = c_res
    return all_data

def dataForComboSplitJAV(combos: ComboList, train_combos: ComboList, test_combos: ComboList):    
    all_data = dataForCombosJAV(combos)
    
    key_ord = [
        'v', 'a_p', 'a_o', 'j_v', 'j_a', 'j_cross', 'd_v', 'd_a', 'd_cross'
    ]
    _, train_res = get2DArrayFromDataStruct(
        concatForComboSubset(all_data, train_combos), key_ord, -1
    )
    _, test_res = get2DArrayFromDataStruct(
        concatForComboSubset(all_data, test_combos), key_ord, -1
    )
    return train_res, test_res

def poseLossJAV(y_true, y_pred):
    '''
    Assumes y_true contains the following columns, in order:
     - speed
     - accel parallel to velocity
     - accel ortho to velocity
     - jerk parallel to speed, ortho to speed but in acc plane, ortho to plane
     - correct pose displacement in the same "coordinate frame" as the jerk.

    In other words, we are working in an orthonormal coordinate frame where one 
    axis is aligned with velocity, another with acceleration, and then a third
    orthogonal to both.
    
    Then, we assume y_pred contains the multipliers for velocity, acceleration,
    and jerk. The predicted "local" displacement is thus:
    [[speed, acc_p, jerk_v],       [vel_multiplier,
     [0,     acc_o, jerk_a],     x  acc_multiplier,
     [0,     0,     jerk_cross]]    jerk_multiplier]

    Then after this matrix multiplication, we find the distance between it and
    the correct pose displacement, both vec3s.
    '''
    pred_disp_0 = y_true[:, 0] * y_pred[:, 0] + y_true[:, 1] * y_pred[:, 1] \
        + y_true[:, 3] * y_pred[:, 2]
    pred_disp_1 = y_true[:, 2] * y_pred[:, 1] + y_true[:, 4] * y_pred[:, 2]
    pred_disp_2 = y_true[:, 5] * y_pred[:, 2]

    pred_disp = tf.stack([pred_disp_0, pred_disp_1, pred_disp_2], axis=-1)

    true_disp = y_true[:, 6:]

    err_vec3 = true_disp - pred_disp
    return tf.norm(err_vec3, axis=-1)
#%%
colin_thresh = 0.7
nonco_cols, co_mat = pm.non_collinear_features(concat_train_data.T, colin_thresh)

vel_bcs_ind = motion_data_keys.index(MOTION_DATA.VEL_BCS_RATIOS)

last_best_ind = motion_data_keys.index(MOTION_DATA.LAST_BEST_LABEL)
timestamp_ind = motion_data_keys.index(MOTION_DATA.TIMESTAMP)

max_bcs_co = max(np.max(co_mat[vel_bcs_ind]), np.max(co_mat[:, vel_bcs_ind]))

if not nonco_cols[vel_bcs_ind] or max_bcs_co > colin_thresh:
    raise Exception("Need to decide how to handle this case!")

nonco_cols[vel_bcs_ind] = False # It's the variable we want to predict.
nonco_cols[last_best_ind] = False # Needs one-hot encoding or similar.
nonco_cols[timestamp_ind] = False

nonco_train_data = concat_train_data[nonco_cols]
nonco_test_data = concat_test_data[nonco_cols]

# %%
bcs_model = keras.Sequential([
    keras.layers.Input((nonco_train_data.shape[0],)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3)
])

bcs_model.summary()
bcs_model.compile(loss=poseLossJAV, optimizer='adam')
# %%
bcs_train, bcs_test = dataForComboSplitJAV(
    nametup_combos, train_combos, test_combos
)

bcs_train = tf.convert_to_tensor(bcs_train, dtype=tf.float32)
bcs_test = tf.convert_to_tensor(bcs_test, dtype=tf.float32)

bcs_scalar = StandardScaler()
z_nonco_train_data = bcs_scalar.fit_transform(nonco_train_data.T)
bcs_model.fit(z_nonco_train_data, bcs_train, epochs=32, shuffle=True)

#%%
z_nonco_test_data = bcs_scalar.transform(nonco_test_data.T)
bcs_pred = bcs_model.predict(z_nonco_test_data)
bcs_test_errs = poseLossJAV(bcs_test, bcs_pred)
#%%
bcs_test_scores = {k: np.mean(bcs_test_errs[v]) for k, v in s_ind_dict.items()}
print(bcs_test_scores)
#%%
nonco_featnames = [
    k.name for i, k in enumerate(motion_data_keys) if nonco_cols[i]
]

def errsForColScramble(model, data, col_ind, y_true):
    data_scramble = data.copy()
    data_scramble[:, col_ind] = np.random.default_rng().choice(
        data_scramble[:, col_ind], len(data_scramble), False
    )
    preds = model.predict(data_scramble)
    errs = model.loss(y_true, preds)
    return errs

keys_s_ind = s_ind_dict.keys()
scramble_scores = np.empty((z_nonco_test_data.shape[1], len(s_ind_dict.keys())))
for col_ind in range(z_nonco_test_data.shape[1]):
    errs = errsForColScramble(bcs_model, z_nonco_test_data, col_ind, bcs_test)
    for i, k in enumerate(keys_s_ind):
        scramble_scores[col_ind, i] = np.mean(errs[s_ind_dict[k]])
#%%
scramble_rank = np.argsort(scramble_scores, axis=0)[::-1]
print("Most important feature inds:", scramble_rank[:10], sep='\n')
    

#%%
nonco_col_nums = np.where(nonco_cols)[0]


#%%
import shap
default_rng = np.random.default_rng()
rng_choice = default_rng.choice(z_nonco_test_data, 100, False, axis=0)
shap_ex = shap.DeepExplainer(bcs_model, rng_choice)

#%%
rng_single = default_rng.choice(rng_choice, 1, axis=0)
shap_values = shap_ex.shap_values(rng_single)#, samples=500)
# shap.initjs()

# shap.summary_plot(shap_values=shap_values, features=rng_choice[35:36])

shap.force_plot(
    shap_ex.expected_value[1].numpy(), shap_values[:, :, 1] #, feature_names=X_train.columns
)
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


#%%
import matplotlib.pyplot as plt

big_tree_preds = big_tree.predict(concat_test_data.T)
tree_errs_for_plt = getErrorPerSkip(
    concat_test_errs, big_tree_preds, s_ind_dict
)
acc_only_preds = np.full_like(
    big_tree_preds, motion_mod_keys.index(MOTION_MODEL.ACC_DEG2)
)
acc_errs_for_plt = getErrorPerSkip(concat_test_errs, acc_only_preds, s_ind_dict)


bar_skip_key = 'skip2'
bar_data = [
    tree_errs_for_plt[bar_skip_key], acc_errs_for_plt[bar_skip_key],
    bcs_test_errs[s_ind_dict[bar_skip_key]]
]
bar_labels = ['Tree', 'Acc Only', 'JAV NN']
for i, arr in enumerate(bar_data):
    if arr.ndim > 1 and arr.shape[1] == 1:
        bar_data[i] = arr.flatten()
    elif arr.ndim > 1:
        raise Exception("Unexpected shape!")

bar_percentiles = [np.percentile(b, 90) for b in bar_data]
bar_pmax = np.max(bar_percentiles)

fig, ax = plt.subplots()
ax.hist(
    bar_data, histtype='step', stacked=False, fill=False, label=bar_labels,
    bins=35, density=True, range=(0, bar_pmax)
)
ax.legend()
ax.set_ylabel("Proportion")
ax.set_xlabel("Translation Error (mm)")
plt.show()

