#%% Imports
import typing
import copy
import time

import concurrent.futures

import numpy as np
from numpy.typing import NDArray

# Decision tree imports ========================================================
from sklearn import tree as sk_tree
from sklearn.tree._tree import TREE_LEAF
from sklearn.model_selection import train_test_split

from posefeatures import MOTION_DATA, MOTION_MODEL, SpecifiedMotionData
from posefeatures import MOTION_DATA_KEY_TYPE, CalcsForCombo


from custom_tree.weighted_impurity import WeightedErrorCriterion

# Local code imports ===========================================================
from gtCommon import BCOT_Data_Calculator
import gtCommon as gtc

OBJ_IS_STATIC_THRESH_MM = 10.0 # 10 millimeters; semi-arbitrary
STRAIGHT_LINE_ANG_THRESH_DEG = 30.0
CIRC_ERR_RADIUS_RATIO_THRESH = 0.10
MAX_MIN_JERK_OPT_ITERS = 33
ERR_NA_VAL = np.finfo(np.float32).max


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


#%%



#%%

motion_mod_keys = [MOTION_MODEL(i) for i in range(1, len(MOTION_MODEL) + 1)] 

# MOTION_DATA_DICT_TYPE = typing.List[typing.Dict[
#     typing.Tuple[int, int], typing.Dict[MOTION_DATA_KEY_TYPE, np.ndarray]
# ]]
# def get_per_combo_datastruct() -> MOTION_DATA_DICT_TYPE:
#     return [{ck[:2]: None for ck in combos} for _ in range(3)]

# err3D_lists = get_per_combo_datastruct()

all_motion_data = [dict() for _ in range(3)]
min_norm_labels = [dict() for _ in range(3)]
err_norm_lists = [dict() for _ in range(3)]

# NamedTuple combos.
nt_combos = [gtc.Combo(*c[:2]) for c in combos]
cfc = CalcsForCombo(
    nt_combos, obj_static_thresh_mm=OBJ_IS_STATIC_THRESH_MM, 
    straight_angle_thresh_deg=STRAIGHT_LINE_ANG_THRESH_DEG,
    err_na_val=ERR_NA_VAL, min_jerk_opt_iter_lim=MAX_MIN_JERK_OPT_ITERS,
    err_radius_ratio_thresh=CIRC_ERR_RADIUS_RATIO_THRESH
    )
results = cfc.getAll()
all_motion_data = cfc.all_motion_data
min_norm_labels = cfc.min_norm_labels
err_norm_lists = cfc.err_norm_lists



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
