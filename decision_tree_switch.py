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

from custom_tree.weighted_impurity import WeightedErrorCriterion

# Local code imports ===========================================================
# For reading the dataset into numpy arrays:
from gtCommon import BCOT_Data_Calculator
import gtCommon as gtc

# Stuff needed for calculating the input features for the non-RNN models.
# MOTION_DATA is an enum representing input feature column "names", while
# MOTION_MODEL is an enum that represents some physical non-ML motion prediction
# schemes like constant-velocity, constant-acceleration, etc.
from posefeatures import MOTION_DATA, MOTION_MODEL # Enums
from posefeatures import MOTION_DATA_KEY_TYPE, CalcsForCombo

# Some consts used in calculating the input features.
OBJ_IS_STATIC_THRESH_MM = 10.0 # 10 millimeters; semi-arbitrary
STRAIGHT_LINE_ANG_THRESH_DEG = 30.0 # 30deg as arbitrary max "straight" angle.
CIRC_ERR_RADIUS_RATIO_THRESH = 0.10 # Threshold for if motion's circular.
MAX_MIN_JERK_OPT_ITERS = 33 # Max iters for min jerk optimization calcs.
MAX_SPLIT_MIN_JERK_OPT_ITERS = 33
ERR_NA_VAL = np.finfo(np.float32).max # A non-inf but inf-like value.

# Video categories; currently not *really* used in this file, but that might 
# change in the near future if I want to analyze the categories separately.
motion_kinds = [
    "movable_handheld", "movable_suspension", "static_handheld",
    "static_suspension", "static_trans"
]
# motion_kinds_plus = motion_kinds + ["all"]

# Generate the following tuples that represent each video:
#     (sequence_name, body_name, motion_kind)
# The first two tuple elements uniquely identify a video, while the third is
# redundant (it's part of each sequence name) but might be used for more
# convenient filtering of videos.
# ---
# In the BCOT dataset, videos are categorized first by the "sequence" type 
# (which is motion/lighting/background), and then by the object ("body") 
# featured in the video. Each "combo" of a sequence and body thus represents
# a distinct video.
combos = []
for s, s_val in enumerate(gtc.BCOT_SEQ_NAMES):
    k = ""
    # For now, using a for loop, not regex, to get motion kind from seq name.
    for k_opt in motion_kinds:
        if k_opt in s_val:
            k = k_opt
            break
    for b in range(len(gtc.BCOT_BODY_NAMES)):
        # Some sequence-body pairs do not have videos, and some have two videos
        # with identical motion but a different camera. So we first check that 
        # a video exists and has unique motion.
        if BCOT_Data_Calculator.isBodySeqPairValid(b, s, True):
            combos.append((b, s, k))


#%%



#%%

# From the combo 3-tuples, construct nametuple versions containing only the
# uniquely-identifying parts. Some functions expect this instead of the 3-tuple. 
nametup_combos = [gtc.Combo(*c[:2]) for c in combos]

cfc = CalcsForCombo(
    nametup_combos, obj_static_thresh_mm=OBJ_IS_STATIC_THRESH_MM, 
    straight_angle_thresh_deg=STRAIGHT_LINE_ANG_THRESH_DEG,
    err_na_val=ERR_NA_VAL, min_jerk_opt_iter_lim=MAX_MIN_JERK_OPT_ITERS,
    split_min_jerk_opt_iter_lim = MAX_SPLIT_MIN_JERK_OPT_ITERS,
    err_radius_ratio_thresh=CIRC_ERR_RADIUS_RATIO_THRESH
)
results = cfc.getAll()

# Input features like velocity, acceleration, jerk, rotation speed, etc.
all_motion_data = cfc.all_motion_data

# Errors for non-ML predictions using simple physics models like 
# constant-velocity, constant-acceleration, etc.
min_norm_labels = cfc.min_norm_labels

# Labels for which of these physical models performed best for each vid frame.
err_norm_lists = cfc.err_norm_lists


# The above three data sequences have the following type: 
#     List[Dict[Combo, (Dict|NDArray)]]
# That is, we have a list of dictionaries which store the results per combo,
# where said "result" might be an NDArray (in the case of min_norm_labels) or
# another dict with "column" names.
# 
# The combo-keyed dict's index in the top-level list corresponds to the number 
# of frames we are skipping when we read the dataset. So the [0] dict is when
# not skipping any  frames, the [1] dict for when reading every 2nd frame only, 
# etc.

#%%
# The below are functions that convert the above lists of dicts into single
# train/test sets that we can pass into our model training.

# First, we concatenate together the data for a subset of combos and get
# a list of 3 items, where each list index again corresponds to the frame 
# skip amount but results are no longer separated by combo.
# Motivation: We may want to quickly filter out a skip amount for training.
def concatForComboSubset(data, combo_subset): #, front_trim: int = 0):
    ret_val = []
    for els_for_skip in data:
        subset_via_combos = [els_for_skip[ck[:2]] for ck in combo_subset]
        concated = None
        front_trim = 0 # May set this via param in future code.
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

# This converts List[Dict[Any, NDArray]] items, which are lists of result 
# dicts of NDarrays indexed by frame skip, into a single 2D NDArray.
# It also returns the dictionary keys in the order that the columns appear in
# the 2D array so that we know which column is which.
def get2DArrayFromDataStruct(data: typing.List[typing.Dict[typing.Any, NDArray]], 
                            ks: typing.List[MOTION_DATA_KEY_TYPE] = None,
                            stack_axis: int = 0):
    if ks is None:
        ks = list(data[0].keys())
    concated = {k: np.concatenate([d[k] for d in data]) for k in ks}
    stacked = np.stack([concated[k] for k in ks], stack_axis)
    return ks, stacked

# Filter out combos based on the 3D object ("body") subset chosen. 
def combosByBod(bods):
    return [c for c in combos if c[0] in bods]
            
# def concatForKeys(data, keys):
#     return np.stack((data[k] for k in keys))

#%%

# We'll split our data into train/test sets where the vids for a single body
# will either all be train vids or all be test vids. This way (a) we are
# guaranteed to have every motion "class" in our train and test sets, and (b)
# we'll know how well the models generalize to new 3D objects not trained on.
bod_arange = np.arange(len(gtc.BCOT_BODY_NAMES), dtype=int)
train_bodies, test_bodies = train_test_split(bod_arange, test_size = 0.2, random_state=0)

train_combos = combosByBod(train_bodies)
test_combos = combosByBod(test_bodies)

# The below gets the training and test data, but leaves them currently still
# separated by skip amount. Here, one can quickly slap a "[0]" at the end of
# each line to just look at data for one skip amount, for example.
train_data = concatForComboSubset(all_motion_data, train_combos)
train_labels = concatForComboSubset(min_norm_labels, train_combos)
train_errs = concatForComboSubset(err_norm_lists, train_combos)
test_labels = concatForComboSubset(min_norm_labels, test_combos)
test_data = concatForComboSubset(all_motion_data, test_combos)
test_errs = concatForComboSubset(err_norm_lists, test_combos)

# Get 2D NDArrays from the above.
concat_train_labels = np.concatenate(train_labels)
concat_test_labels = np.concatenate(test_labels)
# Get the "keys" as another return value so that we know the column names/order.
motion_data_keys, concat_train_data = get2DArrayFromDataStruct(train_data, stack_axis=-1)
_, concat_test_data = get2DArrayFromDataStruct(test_data, motion_data_keys, stack_axis=-1)

# We'll specify the column names/order manually for this one.
motion_mod_keys = [MOTION_MODEL(i) for i in range(1, len(MOTION_MODEL) + 1)] 
_, concat_train_errs = get2DArrayFromDataStruct(train_errs, motion_mod_keys, stack_axis=-1)
_, concat_test_errs = get2DArrayFromDataStruct(test_errs, motion_mod_keys, stack_axis=-1)

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

# Type hint for a dict with string keys and items that are either (int, int)
# intervals or a bool numpy array of indices.
IndDict = typing.Dict[
    str, typing.Union[typing.Tuple[int,int], NDArray[np.bool]]
]

# Gets the per-frame pose error in millimeters for a set of "labels" which 
# represent which physics-based motion model (non-regression-ML) chosen each 
# frame.
# Returns a dict that separates these MAEs based on frame category, e.g., skip
# amount.
def getErrorPerSkip(concat_errs: NDArray, pred_labels: NDArray, inds_dict: IndDict):
    pred_labels_rs = pred_labels.reshape(-1,1)
    taken_errs = np.take_along_axis(concat_errs, pred_labels_rs, axis=1)
    ret_dict: typing.Dict[str, NDArray] = dict()
    for k, inds in inds_dict.items():
        if inds is not None and isinstance(inds, tuple):
            ret_dict[k] = taken_errs[inds[0]:inds[1]]
        else:
            ret_dict[k] = taken_errs[inds]
        {k: taken_errs[inds] for k, inds in inds_dict.items()}
    return ret_dict

# Same as the above, but returns MAE per inds_dict category rather than a whole
# list of per-frame errors.
def assessPredError(concat_errs, pred_labels, inds_dict = None):
    if inds_dict is None:
        inds_dict = {"all:": None}
    all_errs_dict = getErrorPerSkip(concat_errs, pred_labels, inds_dict)
    mean_dict = {k: v.mean() for k, v in all_errs_dict.items()}
    return mean_dict
#%%

# Get the indices of each skip amount inside the concatenated 2D array we
# created above. There might be a "smarter" way to do this given how things were
# previously split into lists by skip amount, but whatever.
s_inds = []       # For test data
s_train_inds = [] # For trian data
ts_ind = motion_data_keys.index(MOTION_DATA.TIMESTEP)
for i in range(1,4): # We have data for frame steps of 1, 2, and 3.
    curr_s_inds = concat_test_data[:, ts_ind] == i
    s_inds.append(curr_s_inds)
    s_train_inds.append(concat_train_data[:, ts_ind] == i)

# Convert the above 3-item lists into dicts.
s_ind_dict = {"skip" + str(i): s_inds[i] for i in range(3)}
s_ind_dict["all"] = None # Because my_np_array[None] returns all elements.

s_train_ind_dict = {"skip" + str(i): s_train_inds[i] for i in range(3)}
s_train_ind_dict["all"] = None

mc = WeightedErrorCriterion(1, np.array([len(MOTION_MODEL)], dtype=np.intp))
y_errs_reshape = concat_train_errs.reshape((
    concat_train_errs.shape[0], 1, concat_train_errs.shape[1]
))
mc.set_y_errs(y_errs_reshape)

# Find the "lower bound" for error when we train a classifier to use the physics
# models for prediction, by finding pose MAE for perfect classification.
error_lim = assessPredError(concat_test_errs, concat_test_labels, s_ind_dict)
print("Classification MAE limit:", error_lim)

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
big_tree = big_tree.fit(concat_train_data, concat_train_labels)
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
    #         clf = clf.fit(concat_train_data, concat_train_labels)
    trimmed_tree = trim_to_depth(big_tree, d)
    graph_pred = trimmed_tree.predict(concat_test_data)
    scores_for_depth = assessPredError(concat_test_errs, graph_pred, s_ind_dict)
    for k, score_for_depth in scores_for_depth.items():
        scores[k][d-1] = score_for_depth

print("Scores for trimmed trees:")
print(scores)
#%%
import matplotlib.pyplot as plt

# Plot decision tree test score errors for the different depths.
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
#         mclf = mclf.fit(concat_train_data, concat_train_labels)
mclf = trim_to_depth(big_tree, 4)

mclfps = mclf.predict(concat_test_data).copy()

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

################################################################################
# Vanilla Regression Network Code!
################################################################################

import tensorflow as tf
import keras
import posemath as pm # Small "library" I wrote for vector operations.
from sklearn.preprocessing import StandardScaler

# We will start off the neural net code by constructing a regression network
# that predicts multipliers for velocity, acceleration, and jerk that we will
# use to construct the displacement from the current position to the position
# we predict for the next timestamp.

ComboList = typing.List[gtc.Combo]

# For each frame of video, we consider a coordinate frame where one axis is
# aligned with the object's velocity and another is aligned with the
# acceleration (or, at least, the part of it orthogonal to velocity).
# We then calculate and return the speed, acceleration, and jerk for the current
# time and the position at the next time in this frame.
# Returns a List[Dict[Combo, Dict[str, NDArray]]] that again separates things
# by frame skip amount and by combo.
def dataForCombosJAV(combos: ComboList, onlySkip0: bool = False):
    all_data = [dict() for _ in range(3)] # Empty dict for each skip amount.

    skip_end = 1 if onlySkip0 else 3
    for c in combos:
        calc_obj = BCOT_Data_Calculator(c.body_ind, c.seq_ind, 0)
        all_translations = calc_obj.getTranslationsGTNP(False)
        for skip in range(skip_end):
            step = skip + 1
            translations = all_translations[::step]
            vels = np.diff(translations, axis=0)
            # We need a velocity for the last timestep, but not an acceleration,
            # because we need the vectors that take each current position to
            # the next when calculating the "ground truth" for displacement
            # predictions. This is the velocity vector; acceleration vectors
            # are not needed for this; we only need "current" acceleration.
            accs = np.diff(vels[:-1], axis=0)
            jerks = np.diff(accs, axis=0)

            # To only calculate as much as we need, we clip the arrays' fronts
            # off when we can.
            speeds = np.linalg.norm(vels[2:-1], axis=-1)
            unit_vels = pm.safelyNormalizeArray(vels[2:-1], speeds[:, np.newaxis])
            # Find the magnitude of the velocity that is parallel to and
            # orthogonal to velocity.
            acc_p = pm.einsumDot(accs[1:], unit_vels) # Parallel magnitude
            acc_p_vecs = pm.scalarsVecsMul(acc_p, unit_vels) # Parallel vec3
            acc_o_vecs = accs[1:] - acc_p_vecs # Orthogonal vec3
            acc_o = np.linalg.norm(acc_o_vecs, axis=-1) # Orthogonal magnitude

            unit_acc_o = pm.safelyNormalizeArray(acc_o_vecs, acc_o[:, np.newaxis])

            # If velocity and the orthogonal acceleration form two axes of our
            # coordinate frame, their cross product forms the third.
            plane_orthos = np.cross(unit_vels, unit_acc_o)
            # We now have matrices to convert vectors in world space into
            # these local velocity-aligned frames.
            mats = np.stack([unit_vels, unit_acc_o, plane_orthos], axis=1)

            # Transform each jerk and to-next-frame into this frame via matmul.
            local_jerks = pm.einsumMatVecMul(mats, jerks)
            local_diffs = pm.einsumMatVecMul(mats, vels[3:])

            # We'll now return all of the data needed to convert velocity,
            # acceleration, and jerk multipliers into local vectors in these
            # new frames. To do this, we don't need to return the coordinate
            # frames themselves: we just need to know the velocity in this
            # frame (a vector [speed, 0, 0]), the acceleration in this frame
            # (i.e. [a_p, a_o, 0]), etc. And since we don't need to return 0s,
            # we can just return the following:
            c_res = {
                'v': speeds, 'a_p': acc_p, 'a_o': acc_o, 
                'j_v': local_jerks[:, 0], 'j_a': local_jerks[:, 1],
                'j_cross': local_jerks[:, 2], 'd_v': local_diffs[:, 0],
                'd_a': local_diffs[:, 1], 'd_cross': local_diffs[:, 2]
            }

            all_data[skip][c] = c_res
    return all_data

# Function that combines the results of the previous one into a 2D numpy
# array.
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

# Get the pose loss for a set of Jerk, Acceleration, & Velocity multipliers.
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
# A lot of the features we calculated might be collinear (especially since a lot 
# of very similar features were tried for the decision tree) we'll remove the
# collinear ones before training.
colin_thresh = 0.7 # Threshold for collinearity.

nonco_cols, co_mat = pm.non_collinear_features(concat_train_data, colin_thresh)

vel_bcs_ind = motion_data_keys.index(MOTION_DATA.VEL_BCS_RATIOS)

last_best_ind = motion_data_keys.index(MOTION_DATA.LAST_BEST_LABEL)
timestamp_ind = motion_data_keys.index(MOTION_DATA.TIMESTAMP)

max_bcs_co = max(np.max(co_mat[vel_bcs_ind]), np.max(co_mat[:, vel_bcs_ind]))

if not nonco_cols[vel_bcs_ind] or max_bcs_co > colin_thresh:
    raise Exception("Need to decide how to handle this case!")

nonco_cols[vel_bcs_ind] = False # It's the variable we want to predict.
nonco_cols[last_best_ind] = False # Needs one-hot encoding or similar.
nonco_cols[timestamp_ind] = False # Current frame number seems... unhelpful.

# select_cols = np.where(nonco_cols)[0][[0, 1, 2, 3, 13, 26, 27]]
# nonco_cols[:] = False
# nonco_cols[list(select_cols)] = True

# Get the data subset for the selected non-collinear columns.
nonco_train_data = concat_train_data[:, nonco_cols]
nonco_test_data = concat_test_data[:, nonco_cols]

# %%


# Custom importance weighting layer
# Class written by ChatGPT and then manually verified.
# (But I'm not a big tensorflow expert, so maybe my verification was faulty...)
class ImportanceLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, weight_decay=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.weight_decay = weight_decay

    def build(self, input_shape):
        # Define per-feature weights with L2 regularization (weight decay)
        self.importance_weights = self.add_weight(
            shape=(self.input_dim,), 
            initializer="ones",  # Start with all weights = 1
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True
        )

    def call(self, inputs):
        return inputs * self.importance_weights  # Element-wise multiplication


dropout_rate = 0.2
nodes_per_layer = 128
bcs_model = keras.Sequential([
    keras.layers.Input((nonco_train_data.shape[1],)),
    # ImportanceLayer(nonco_train_data.shape[1]),  # Custom importance layer
    keras.layers.Dense(nodes_per_layer, activation='relu'),
    keras.layers.Dropout(dropout_rate),
    keras.layers.Dense(nodes_per_layer, activation='relu'),
    keras.layers.Dropout(dropout_rate),
    keras.layers.Dense(nodes_per_layer, activation='relu'),
    keras.layers.Dense(3)
])

bcs_model.summary()
bcs_model.compile(loss=poseLossJAV, optimizer='adam')
# %%
# Get local-frame data.
bcs_train, bcs_test = dataForComboSplitJAV(
    nametup_combos, train_combos, test_combos
)

# Convert from numpy array to tf tensor.
bcs_train = tf.convert_to_tensor(bcs_train, dtype=tf.float32)
bcs_test = tf.convert_to_tensor(bcs_test, dtype=tf.float32)

# Z-scale each column to standard normal distribution.
bcs_scalar = StandardScaler()
z_nonco_train_data = bcs_scalar.fit_transform(nonco_train_data)
#%% Train the network.
bcs_hist = bcs_model.fit(z_nonco_train_data, bcs_train, epochs=32, shuffle=True)

#%% Evaluate network on test data.
z_nonco_test_data = bcs_scalar.transform(nonco_test_data)
bcs_pred = bcs_model.predict(z_nonco_test_data)
bcs_test_errs = poseLossJAV(bcs_test, bcs_pred)
#%% Print scores on test data.
bcs_test_scores = {k: np.mean(bcs_test_errs[v]) for k, v in s_ind_dict.items()}
print(bcs_test_scores)
#%%

#%% 

################################################################################
# Column Scrambling to Assess Feature Importance
################################################################################

# Get the column names for each of the kept columns.
nonco_featnames = [
    k.name for i, k in enumerate(motion_data_keys) if nonco_cols[i]
]

# Finds the errors for the model when one of the test data columns has its
# data scrambled, as per the advice of a StackOverflow post on how to figure out
# which columns are more important for the prediction.
def errsForColScramble(model: keras.Model, data: NDArray, col_ind: int, y_true: NDArray):
    data_scramble = data.copy() # So that original's not affected.

    # Column scramble:
    data_scramble[:, col_ind] = np.random.default_rng().choice(
        data_scramble[:, col_ind], len(data_scramble), False
    )

    # Getting new errors:
    preds = model.predict(data_scramble)
    errs = model.loss(y_true, preds)
    return errs

# For each column and for each skip amount, scramble the column and find the new
# score.
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

################################################################################
# Vanilla Classification Network
################################################################################

def customPoseLoss(y_true, y_pred):
    probs = tf.nn.softmax(y_pred, axis=1)
    return tf.reduce_sum(y_true * probs, axis=1)
# Example Usage
# tf_concat_train_errs = tf.convert_to_tensor(concat_train_errs, dtype=tf.float32)
tf_loss_fn = customPoseLoss # CustomLossWithErrors(concat_train_errs)

input_dim = concat_train_data.shape[1]
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
adam = keras.optimizers.Adam(0.01)

tfmodel.compile(optimizer=adam, loss=tf_loss_fn)
tfmodel.fit(concat_train_data, concat_train_errs, epochs=5, shuffle=True)
#%%
bgtrain = big_tree.predict(concat_train_data)
bgtrain_score = assessPredError(concat_train_errs, bgtrain, s_train_ind_dict)
print("Big tree train errs=", bgtrain_score)

mclf_train = mclf.predict(concat_train_data)
mclf_train_score = assessPredError(concat_train_errs, mclf_train, s_train_ind_dict)
print("Smaller tree train errs=", mclf_train_score)

tf_train_preds = np.argmax(tfmodel(concat_train_data).numpy(), axis=1)
tf_train_errs = assessPredError(concat_train_errs, tf_train_preds, s_train_ind_dict)
print("TF train errs=", tf_train_errs)
print()

tf_test_preds = np.argmax(tfmodel(concat_test_data).numpy(), axis=1)
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

