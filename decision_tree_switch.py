#%% Imports
import typing
import copy
import time
from collections import defaultdict
from enum import Enum

import datetime


import numpy as np
from numpy.typing import NDArray

# Decision tree imports ========================================================
from sklearn import tree as sk_tree
from sklearn.tree._tree import TREE_LEAF

from custom_tree.weighted_impurity import WeightedErrorCriterion

# Local code imports ===========================================================
# For reading the dataset into numpy arrays:
from gtCommon import PoseLoaderBCOT
import gtCommon as gtc

# Stuff needed for calculating the input features for the non-RNN models.
# MOTION_DATA is an enum representing input feature column "names", while
# MOTION_MODEL is an enum that represents some physical non-ML motion prediction
# schemes like constant-velocity, constant-acceleration, etc.
from motiontools.posefeatures import (
    MOTION_DATA, MOTION_MODEL, ANG_OR_MAG, JAV,     # Enums
    SpecifiedMotionData, OneHotMotionData,          # Other "labels" for columns
    NumpyForSkipAndID, OrderForJAV, PoseLoaderList, # Types
    dataForCombosJAV, getWorldFrameDisplacements,
    gtMultipliers6, getBaselineJAV6,
    CalcsForVideo                                   # Classes
)
    
from nn_utilities.nn_losses import (
    poseLossJAV, poseLossVec3
)
from nn_utilities.nn_loading import loadLatestModels

from motiontools.dataorg import (
    DataOrganizer, concatForComboSubset, UnitAwareScaler, SkipSubsetKind
)

from datatools.data_splitting import DataSubsetKind

TRAIN_NEW_MODEL = False

# Some consts used in calculating the input features.
OBJ_IS_STATIC_THRESH_MM = 10.0 # 10 millimeters; semi-arbitrary
STRAIGHT_LINE_ANG_THRESH_DEG = 30.0 # 30deg as arbitrary max "straight" angle.
CIRC_ERR_RADIUS_RATIO_THRESH = 0.10 # Threshold for if motion's circular.
MAX_MIN_JERK_OPT_ITERS = 33 # Max iters for min jerk optimization calcs.
MAX_SPLIT_MIN_JERK_OPT_ITERS = 33
ERR_NA_VAL = np.finfo(np.float32).max # A non-inf but inf-like value.

#%%
combos = PoseLoaderBCOT.getAllIDs(True)

#%%

# From the combo 3-tuples, construct nametuple versions containing only the
# uniquely-identifying parts. Some functions expect this instead of the 3-tuple. 
nametup_combos = [gtc.VidBCOT(*c[:2]) for c in combos]
bcot_loaders = [
    PoseLoaderBCOT(nc.body_ind, nc.seq_ind) for nc in nametup_combos
]

cfc = CalcsForVideo(
    obj_static_thresh_mm=OBJ_IS_STATIC_THRESH_MM, 
    straight_angle_thresh_deg=STRAIGHT_LINE_ANG_THRESH_DEG,
    err_na_val=ERR_NA_VAL, min_jerk_opt_iter_lim=MAX_MIN_JERK_OPT_ITERS,
    split_min_jerk_opt_iter_lim = MAX_SPLIT_MIN_JERK_OPT_ITERS,
    err_radius_ratio_thresh=CIRC_ERR_RADIUS_RATIO_THRESH,
    # exclude_axis_angs=False, exclude_bidir=False, exclude_circ_data=False,
    # exclude_onehots=False, exclude_past_muls=False, exclude_timescaled=False,
    # exclude_vel_deg2=False
)
results = cfc.getAll(bcot_loaders)

# Input features like velocity, acceleration, jerk, rotation speed, etc.
all_motion_data = cfc.all_motion_data

# Errors for non-ML predictions using simple physics models like 
# constant-velocity, constant-acceleration, etc.
min_norm_labels = cfc.min_norm_labels

# Labels for which of these physical models performed best for each vid frame.
err_norm_lists = cfc.err_norm_lists

#%%

# We'll split our data into train/test sets where the vids for a single body
# will either all be train vids or all be test vids. This way (a) we are
# guaranteed to have every motion "class" in our train and test sets, and (b)
# we'll know how well the models generalize to new 3D objects not trained on.
bcot_split = PoseLoaderBCOT.trainValidationTestByBody(
    validation_ratio=0.1, test_ratio=0.2, random_seed=0
)
bcot_train_combos, bcot_validation_combos, bcot_test_combos = bcot_split

train_combos_c2 = [c[:2] for c in bcot_train_combos]
test_combos_c2 = [c[:2] for c in bcot_test_combos] # Get the unique part of each.
val_combos_c2 = [c[:2] for c in bcot_validation_combos]
if len(val_combos_c2) == 0:
    val_combos_c2 = None

dog = DataOrganizer(
    all_motion_data, min_norm_labels, err_norm_lists,
    train_combos_c2, test_combos_c2, val_combos_c2
)

#%%
test_bodies = np.unique([c[0] for c in bcot_test_combos])
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
motion_mod_len = len(MOTION_MODEL)
mc = WeightedErrorCriterion(1, np.array([motion_mod_len], dtype=np.intp))
class_errs_shape = dog.concat_train_class_errs.shape
y_errs_reshape = dog.concat_train_class_errs.reshape((
    class_errs_shape[0], 1, class_errs_shape[1]
))
mc.set_y_errs(y_errs_reshape)

# Find the "lower bound" for error when we train a classifier to use the physics
# models for prediction, by finding pose MAE for perfect classification.
error_lim_all = dog.getClassErrsTest(dog.concat_test_labels)
error_lim = dog.getClassScoresTest(dog.concat_test_labels)
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
big_tree = big_tree.fit(dog.concat_train_data, dog.concat_train_labels)
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
scores = defaultdict(lambda : np.empty(max_depth))
depths = np.arange(1, max_depth + 1)
for d in depths:
    # Preiously, we trained new trees from scratch using the below code, but as
    # described above, we'll instead trim the "main" tree to get new ones.
    #         clf = sk_tree.DecisionTreeClassifier(max_depth=d, criterion=mc)
    #         clf = clf.fit(concat_train_data, concat_train_labels)
    trimmed_tree = trim_to_depth(big_tree, d)
    graph_pred = trimmed_tree.predict(dog.concat_test_data)
    scores_for_depth = dog.getClassScoresTest(graph_pred)
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

#%%
# Again, we'll replace old code with a trimming of our main tree.
#         mclf = sk_tree.DecisionTreeClassifier(max_depth=4, criterion=mc)
#         mclf = mclf.fit(concat_train_data, concat_train_labels)
mclf = trim_to_depth(big_tree, 4)

mclfps = mclf.predict(dog.concat_test_data).copy()

#%%

mc_errs = dog.getClassScoresTest(mclfps)
print("Error for my decision tree=", mc_errs)
print("Done!", mclfps)

def getTreeDataTrainScore(tree, data_organizer: DataOrganizer):
    p = tree.predict(data_organizer.concat_train_data)
    return data_organizer.getClassScoresTrain(p)
def getTreeDataTestScore(tree, data_organizer: DataOrganizer):
    p = tree.predict(data_organizer.concat_test_data)
    return data_organizer.getClassScoresTest(p)

#%%
from sklearn.tree import export_graphviz
import pathlib

tree_path = pathlib.Path(__file__).parent.resolve() / "results" / "tree.dot"
feature_names = [e.name for e in dog.motion_data_keys]
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


# Function that combines the results of the previous one into a 2D numpy
# array.
# List[Dict]
def dataForComboSplitJAV(train_combos: typing.List, test_combos: typing.List, 
                         validation_combos: typing.Optional[typing.List] = None,
                         *,
                         pose_loaders: typing.Optional[PoseLoaderList] = None, 
                         precalc_per_combo: typing.Optional[NumpyForSkipAndID] = None):   
    if pose_loaders is None and precalc_per_combo is None:
        raise ValueError(
            "Cannot have combos and precalc_per_combo both be None!"
        )
    elif pose_loaders is not None and precalc_per_combo is not None:
        raise ValueError(
            "Cannot provide values for both  combos and precalc_per_combo!"
        )
     
    all_data = precalc_per_combo
    if precalc_per_combo is None:
        all_data = dataForCombosJAV(
            pose_loaders, (JAV.JERK, JAV.ACCELERATION, JAV.VELOCITY)
        )
    

    train_res = np.concatenate(
        concatForComboSubset(all_data, train_combos), axis=0
    )
    test_res = np.concatenate(
        concatForComboSubset(all_data, test_combos), axis=0
    )
    if validation_combos is not None and len(validation_combos) > 0:
        val_res = np.concatenate(
            concatForComboSubset(all_data, validation_combos), axis=0
        )
        return train_res, test_res, val_res
    return train_res, test_res



def poseLossAngle(y_true, y_pred):
    # 2acos(abs([cos||x/2|| cos||y/2|| + <x>*<y> sin||x/2|| sin||y/2||))
    ang_true = np.linalg.norm(y_true, axis=-1) + 0.00000001
    ang_pred = np.linalg.norm(y_pred, axis=-1) + 0.00000001
    half_ang_true = ang_true / 2
    half_ang_pred = ang_pred / 2
    cos_true = np.cos(half_ang_true)
    cos_pred = np.cos(half_ang_pred)
    sin_true = np.sin(half_ang_true)
    sin_pred = np.sin(half_ang_pred)
    vec3_dots = pm.einsumDot(y_true, y_pred)
    unit_vec3_dots = vec3_dots / (ang_true * ang_pred)
    quat_dots = cos_true * cos_pred + unit_vec3_dots * sin_true * sin_pred
    quat_dots_1 = np.abs(np.clip(quat_dots, -1, 1))
    return 2 * np.arccos(quat_dots_1) #tf.convert_to_tensor(...)


#%%
# A lot of the features we calculated might be collinear (especially since a lot 
# of very similar features were tried for the decision tree) we'll remove the
# collinear ones before training.
colin_thresh = 0.7 # Threshold for collinearity.

nonco_cols, co_mat = pm.non_collinear_features(
    dog.concat_train_data, colin_thresh
)
def indsForKeysMD(keysMD: typing.List[MOTION_DATA]):
    ret = []
    for k in keysMD:
        f = -1
        try:
            f = dog.motion_data_keys.index(k)
        except ValueError:
            print("Key", k.name, "not found.")
        if f >= 0:
            ret.append(f)
    return ret

timestamp_ind = dog.motion_data_keys.index(MOTION_DATA.TIMESTAMP)
framenum_ind = dog.motion_data_keys.index(MOTION_DATA.FRAME_NUM)
onehot_inds = [
    i for i, k in enumerate(dog.motion_data_keys)
    if isinstance(k, OneHotMotionData)
]
GT_inds = [
    i for i, k in enumerate(dog.motion_data_keys)
    if len(k.name) == 3 and k.name[:2] == "GT"
]
ang_inds = [
    i for i, k in enumerate(dog.motion_data_keys)
    if isinstance(k, SpecifiedMotionData) and k.ang_or_mag == ANG_OR_MAG.ANG
]
bidir_inds = [
    i for i, k in enumerate(dog.motion_data_keys)
    if isinstance(k, SpecifiedMotionData) and k.bidirectional
]
circ_vec3_inds = [
    i for i, k in enumerate(dog.motion_data_keys)
    if isinstance(k, SpecifiedMotionData) and k.base_cat.name[:4].upper() == "CIRC"
]
veld_ra_inds = [
    i for i, k in enumerate(dog.motion_data_keys)
    if isinstance(k, SpecifiedMotionData) and k.axis == MOTION_DATA.VEL_DEG2_VEC3
]
veld2_dot_inds = [
    i for i, k in enumerate(dog.motion_data_keys)
    if "DEG2_DOT" in k.name.upper()
]
# plane_ra_inds = [
#     i for i, k in enumerate(dog.motion_data_keys)
#     if isinstance(k, SpecifiedMotionData) and k.axis == OTHER_DIRECTION.PLANE_ORTHO
# ]
all_circ_inds = [
    i for i, k in enumerate(dog.motion_data_keys) if "CIRC" in k.name.upper()
]
all_timescaled_inds = [
    i for i, k in enumerate(dog.motion_data_keys) if "TIMESCALED" in k.name.upper()
]
misc_rem_inds = indsForKeysMD([
    MOTION_DATA.INV_VEL_BCS_RATIOS, MOTION_DATA.RAD_DIFF,
    MOTION_DATA.SPEED_ACC_RATIO, MOTION_DATA.DISP_MAG_DIFF,
    MOTION_DATA.PLANE_NORMAL_DOT, MOTION_DATA.SPEED_ORTHO_ACC_RATIO
])

nonco_cols[:] = True
nonco_cols[timestamp_ind] = False # Current frame number seems... unhelpful.
nonco_cols[framenum_ind] = False

broad_exclusions = onehot_inds + GT_inds + ang_inds + bidir_inds 
broad_exclusions += circ_vec3_inds + all_circ_inds + all_timescaled_inds
broad_exclusions += veld_ra_inds + veld2_dot_inds

nonco_cols[broad_exclusions] = False
nonco_cols[misc_rem_inds] = False
nonco_cols[[
    i for i, k in enumerate(dog.motion_data_keys)
    if isinstance(k, MOTION_DATA) and k != MOTION_DATA.TIMESTEP
]] = False
# nonco_cols[plane_ra_inds] = False


AVD2_KEY = SpecifiedMotionData(
    MOTION_DATA.ACC_VEC3, MOTION_DATA.VEL_DEG1_VEC3, ANG_OR_MAG.ANG, False, True
)

bounce_ang_key = SpecifiedMotionData(
    MOTION_DATA.VEL_DEG1_VEC3, MOTION_DATA.VEL_DEG1_VEC3, ANG_OR_MAG.ANG,
    False, True
)

col_sub_keys = [
    AVD2_KEY, bounce_ang_key, MOTION_DATA.VEL_BCS_RATIOS,
    MOTION_DATA.CIRC_ACC, MOTION_DATA.DISP_MAG_DIFF, MOTION_DATA.TIMESTEP,
    MOTION_DATA.DISP_MAG_RATIO
]
# col_indices = indsForKeysMD(col_sub_keys)
# nonco_cols[col_indices] = True

nonco_col_nums = np.where(nonco_cols)[0]
# Get the column names for each of the kept columns.
nonco_col_ks = [k for i, k in enumerate(dog.motion_data_keys) if nonco_cols[i]]
nonco_featnames = np.array([k.name for k in nonco_col_ks])

# select_cols = np.where(nonco_cols)[0][[0, 1, 2, 3, 13, 26, 27]]
# nonco_cols[:] = False
# nonco_cols[list(select_cols)] = True


# Custom importance weighting layer suggested/described "in theory" by a friend.
# Then, I had the class written by ChatGPT and manually verified.
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

class OutVecMode(Enum):
    JAV_MULTIPLIERS = 1
    VEL_ALIGNED_VEC3 = 2
    WORLD_VEC3 = 3
    WORLD_DISP = 4
    ROT_ALIGNED_VEC3 = 5
    ROT_AA = 6
    ROT_VEL_AA = 7
    ROT_FIXED_AX = 8

WORLD_VEC_MODES = (OutVecMode.WORLD_VEC3, OutVecMode.WORLD_DISP)
ROT_VEC_MODES = (
    OutVecMode.ROT_AA, OutVecMode.ROT_VEL_AA, OutVecMode.ROT_FIXED_AX
)
#%%
chosen_mode = OutVecMode.JAV_MULTIPLIERS

def getUntrainedNN(in_dim: int, out_dim: int, loss = None, n_layers: int = 3,
                   nodes_per_layer: int = 128):

    dropout_rate = 0.2
    vel_nn_activation = 'sigmoid' # Works better than relu for this NN.
    in_shape = in_dim # + (6 if use_resid_data else 0)

    make_dense = lambda num_nodes = nodes_per_layer: keras.layers.Dense(
        num_nodes, activation=vel_nn_activation #, kernel_initializer=initer
    )

    in_layer = keras.layers.Input((in_shape,))

    # in_layer = keras.layers.GaussianNoise(0.99)(in_layer)

    x = make_dense()(in_layer)
    for _ in range(n_layers - 1):
        x = keras.layers.Dropout(dropout_rate)(x)
        x = make_dense()(x)
    
    out = keras.layers.Dense(out_dim)(x)

    model = keras.Model(inputs = in_layer, outputs = out)

    if loss is None:
        loss = poseLossJAV

    model.summary()
    optim = 'adam'
    model.compile(loss=loss, optimizer=optim)
    return model

sel_dim = 12
sel_loss = poseLossJAV
if chosen_mode != OutVecMode.JAV_MULTIPLIERS:
    if chosen_mode == OutVecMode.ROT_FIXED_AX:
        sel_dim = 1
        sel_loss = 'mae'
    else:
        sel_dim = 3
        sel_loss = 'mse' if chosen_mode in ROT_VEC_MODES else poseLossVec3 
    
JAV_order = (JAV.JERK, JAV.ACCELERATION, JAV.VELOCITY)[::-1]

# # Convert from numpy array to tf tensor.
# bcs_train = tf.convert_to_tensor(bcs_train, dtype=tf.float32)
# bcs_test = tf.convert_to_tensor(bcs_test, dtype=tf.float32)

# Z-scale each column to standard normal distribution.
bcs_scaler = UnitAwareScaler(nonco_col_ks) #, False)

class DataForJAV:
    def __init__(self, data_organizer: DataOrganizer, loaders, bcs_scaler, 
                 col_inds: NDArray, JAV_order: OrderForJAV,
                 outVecMode: OutVecMode,
                 skip: typing.Union[int,SkipSubsetKind] = SkipSubsetKind._all,
                 *, save_data_for_conf: bool = False):

        self.data_organizer = data_organizer
        self.data_organizer.setPickAndTransform(col_inds, bcs_scaler)
        self.outVecMode = outVecMode
        self.pos_scale = 0.0
        self.rot_scale = 0.0
        self.skip = skip
        if isinstance(bcs_scaler, UnitAwareScaler):
            self.pos_scale = bcs_scaler.pos_scale
            self.rot_scale = bcs_scaler.rot_scale
        elif outVecMode != OutVecMode.JAV_MULTIPLIERS:
            raise NotImplementedError(
                "Non-unit-aware scaler not yet supported!"
            )


        self.save_data_for_conf = save_data_for_conf
        self.jav_per_combo = None
        _rot_align = self.outVecMode == OutVecMode.ROT_ALIGNED_VEC3 
        _rot_output = self.outVecMode in ROT_VEC_MODES
        need_rot_vel = _rot_align or self.outVecMode in WORLD_VEC_MODES
        need_rot_vel |= _rot_output
        need_rot = need_rot_vel or _rot_align

        save_data_for_conf |= self.outVecMode in WORLD_VEC_MODES
        need_pos = save_data_for_conf or _rot_align
        jav_res = dataForCombosJAV(
            loaders, JAV_order, save_data_for_conf, need_pos,
            need_rot, need_rot_vel
        )
        # Default: assume all bools were False and no tuple returned.
        self.jav_per_combo = jav_res
        self.w2ls_JAV = None
        self.translations_JAV = None
        self._rmatsv9 = None
        self._aas_JAV = None
        self._rot_vels_JAV = None
        r_mats = None
        jav_tup_ind = 1
        if need_pos or need_rot:
            self.jav_per_combo = jav_res[0]
        if save_data_for_conf:
            self.w2ls_JAV = jav_res[jav_tup_ind]
            jav_tup_ind += 1
        if need_pos:
            self.translations_JAV = jav_res[jav_tup_ind]
            jav_tup_ind += 1
        if need_rot:
            r_mats = jav_res[jav_tup_ind]
            self._rmatsv9 = [
                {k: x.reshape(-1, 9) for k, x in d.items()} for d in r_mats
            ]
            self._aas_JAV = [
                {k: pm.axisAngleFromMatArray(x) for k, x in d.items()}
                for d in r_mats
            ]
            jav_tup_ind += 1
        if need_rot_vel:
            self._rot_vels_JAV = jav_res[jav_tup_ind]
            jav_tup_ind += 1
            if self.outVecMode == OutVecMode.ROT_FIXED_AX:
                prev_angs = [
                    {
                        k: np.linalg.norm(v, axis=-1, keepdims=True)
                        for k, v in d.items()
                    }
                    for d in self._rot_vels_JAV
                ]
                self._prev_vel_axes = [
                    {
                        k: pm.safelyNormalizeArray(v, prev_angs[i][k])
                        for k, v in d.items()
                    }
                    for i, d in enumerate(self._rot_vels_JAV)
                ]
                self._prev_vel_ax_concat = self._worldvec_concats(
                    1, diff_ord=0, scale=1.0, use_translation=False,
                    vecs=self._prev_vel_axes, curr_diff_ord=1
                )
                self._prev_ang_concat = self._worldvec_concats(
                    1, diff_ord=0, scale=self.rot_scale, use_translation=False,
                    vecs=prev_angs, curr_diff_ord=1
                )
        
        
        jav_split = dataForComboSplitJAV(
            data_organizer.subset_ids[DataSubsetKind.TRAIN],
            data_organizer.subset_ids[DataSubsetKind.TEST],
            data_organizer.subset_ids[DataSubsetKind.VALIDATION],
            precalc_per_combo=self.jav_per_combo
        )
        self.jav_train, self.jav_test = jav_split[:2]
        self.jav_validation = np.empty((0, ) + self.jav_train.shape[1:])
        if len(data_organizer.subset_ids[DataSubsetKind.VALIDATION]) > 0:
            self.jav_validation = jav_split[2]

        _javs_by_subset = {
            DataSubsetKind.TRAIN: self.jav_train,
            DataSubsetKind.TEST: self.jav_test,
            DataSubsetKind.VALIDATION: self.jav_validation
        }

        # Set default values for the neural network input values and ground
        # truth values.
        b, e = None, None # Beginning and ending JAV columns
        self.ref_prediction_name = "Const vel" if _rot_output else "Quadratic" 
        self.ref_predictions = { # Quadratic acc JAV multipliers.
            # TODO: These assume constant timesteps for now.
            k: np.repeat((1.0, 0.0), (3, 9)).reshape(1, -1)
            for k in DataSubsetKind.nonWholeValues()
        }
        if self.outVecMode == OutVecMode.VEL_ALIGNED_VEC3:
            b, e = 12, 15
            for k, _ref_javs in _javs_by_subset.items():
                _ref_preds = np.empty((len(_ref_javs), 3))
                # Constructing quadratic interpolation predictions.
                # TODO: These assume constant timesteps for now.
                _ref_preds[:, 0] = _ref_javs[:, 0] + _ref_javs[:, 1]
                _ref_preds[:, 1] = _ref_javs[:, 2]
                _ref_preds[:, 2] = 0.0
                self.ref_predictions[k] = _ref_preds
        _dog = self.data_organizer
        self._in_arrs = {
            DataSubsetKind.TRAIN: _dog.col_subset_train,
            DataSubsetKind.TEST: _dog.col_subset_test,
            DataSubsetKind.VALIDATION: _dog.col_subset_validation
        }
        self._gt_arrs = {k: j[:, b:e] for k, j in _javs_by_subset.items()}

            
        self.translationScaler = None
        if self.pos_scale != 0.0 and need_pos:
            self._fitWorldScaler(self.pos_scale)

        self.score_scaler = 0.0
        relevant_scale = self.rot_scale if _rot_output else self.pos_scale
        if self.outVecMode == OutVecMode.WORLD_VEC3:
            self.score_scaler = self.translationScaler
        elif self.outVecMode != OutVecMode.JAV_MULTIPLIERS:
            self.score_scaler = relevant_scale
        
        self.score_fn = poseLossJAV
        if outVecMode != OutVecMode.JAV_MULTIPLIERS:
            if outVecMode in ROT_VEC_MODES:
                self.score_fn = poseLossAngle 
            else:
                self.score_fn = poseLossVec3

        if self.outVecMode in WORLD_VEC_MODES or _rot_align or _rot_output:
            if self.outVecMode == OutVecMode.ROT_FIXED_AX:
                self._gt_rot_vels = self._worldvec_concats(
                    0, diff_ord=0, scale=self.rot_scale, use_translation=False,
                    vecs=self._rot_vels_JAV, curr_diff_ord=1
                )
            _w2l_mats_prev = {k: None for k in DataSubsetKind.nonWholeValues()}

            if _rot_align:
                _l2w_mats_prev = self._worldvec_concats(
                    1, 0, 0.0, vecs=r_mats
                )
                _w2l_mats_prev = {
                    k: np.swapaxes(v, -2, -1) for k, v in _l2w_mats_prev.items()
                }
            
            for k, _w2l_mats_prev_sub in _w2l_mats_prev.items():
                self._in_arrs[k] = self._world_coord_cols(
                    self._in_arrs[k], k, _w2l_mats_prev_sub
                )

            _diff_ord = 0; _scale = 0.0; _curr_diff_ord = 0
            _use_t = True; _vecs = None
            if _rot_align or self.outVecMode == OutVecMode.WORLD_DISP:
                _diff_ord = 1; _scale = self.pos_scale
            elif _rot_output:
                _scale = self.rot_scale
                _use_t = False
                if self.outVecMode == OutVecMode.ROT_VEL_AA:
                    _curr_diff_ord = 1
                    _vecs = self._rot_vels_JAV
                elif self.outVecMode == OutVecMode.ROT_FIXED_AX:
                    _curr_diff_ord = 2
                    _vecs = []
                    for i, r_mats_for_skip in enumerate(r_mats):
                        d = dict()
                        for k, rms in r_mats_for_skip.items():
                            d[k] = pm.closestAnglesAboutAxis(
                                rms[1:-1], rms[2:],
                                self._prev_vel_axes[i][k][:-1]
                            ).reshape(-1, 1)
                        _vecs.append(d)
                elif self.outVecMode == OutVecMode.ROT_AA:
                    _vecs = self._aas_JAV

            self._gt_arrs = self._worldvec_concats(
                0, diff_ord = _diff_ord, scale = _scale, use_translation=_use_t,
                vecs=_vecs, curr_diff_ord=_curr_diff_ord
            )
            if self.outVecMode == OutVecMode.ROT_FIXED_AX:
                for k, v in self._gt_arrs.items():
                    self._gt_arrs[k] = v / self._prev_ang_concat[k]
                    # From plotting the gt for the near-zero angles, it seems
                    # that a multiplier of 0.0 is the mean of a fairly normal
                    # -looking histogram. So 0.0 is probably the safest bet.
                    self._gt_arrs[k][self._prev_ang_concat[k] == 0.0] = 0.0
            if _rot_align:
                for k in DataSubsetKind.nonWholeValues():
                    self._gt_arrs[k] = pm.einsumMatVecMul(
                        _w2l_mats_prev[k], self._gt_arrs[k]
                    )

        # If we want to only use data for one skip value, we filter things here.
        skip_inds = {
            k: ... if skip == SkipSubsetKind._all else v[skip]
            for k, v in _dog.subset_skip_inds.items()
        }
        self._in_arrs = {k: v[skip_inds[k]] for k, v in self._in_arrs.items()}
        self._gt_arrs = {k: v[skip_inds[k]] for k, v in self._gt_arrs.items()}
        self.ref_predictions = {
            k: v[skip_inds[k]] for k, v in self.ref_predictions.items()
        }
        
        # These wouldn't have been scaled earlier. Might want to move/fix things
        # so that all scaling happens in the same place, though.
        if self.outVecMode == OutVecMode.VEL_ALIGNED_VEC3:
            for k in DataSubsetKind.nonWholeValues():
                self._gt_arrs[k] /= self.pos_scale
                self.ref_predictions[k] /= self.pos_scale


        self.in_train = self._in_arrs[DataSubsetKind.TRAIN]
        self.in_test = self._in_arrs[DataSubsetKind.TEST]
        self.in_validation = self._in_arrs[DataSubsetKind.VALIDATION]

        self.gt_train = self._gt_arrs[DataSubsetKind.TRAIN]
        self.gt_test = self._gt_arrs[DataSubsetKind.TEST]
        self.gt_validation = self._gt_arrs[DataSubsetKind.VALIDATION]

        
        # self.jav_train = np.concatenate(
        #     (partial_jav_train, jav_tr_append), axis=-1
        # )
        # self.jav_test = np.concatenate(
        #     (partial_jav_test, jav_te_append), axis=-1
        # )

    def _world_coord_cols(self, curr_cols: NDArray, subset_kind: DataSubsetKind,
                          w2l_rotations: typing.Optional[NDArray] = None):
        scs = self.pos_scale
        rs = self.rot_scale
        _ts = self.translations_JAV
        subset_ids = self.data_organizer.subset_ids[subset_kind]
        coords = self._worldvec_helper(subset_ids, 1, use_translation=True)
        coords_tm1 = self._worldvec_helper(subset_ids, 2, use_translation=True)
        coords_tm2 = self._worldvec_helper(subset_ids, 3, use_translation=True)
        coords_tm3 = self._worldvec_helper(subset_ids, 4, use_translation=True)
        vels = self._worldvec_helper(subset_ids, 1, 1, scale=scs, vecs = _ts)
        accs = self._worldvec_helper(subset_ids, 1, 2, scale=scs, vecs = _ts)
        jerks = self._worldvec_helper(subset_ids, 1, 3, scale=scs, vecs = _ts)
        aas = self._worldvec_helper(subset_ids, 1, vecs=self._aas_JAV)
        rmatv9s = self._worldvec_helper(subset_ids, 1, vecs=self._rmatsv9)
        rot_vels_unscaled = self._worldvec_helper(
            subset_ids, 1, current_diff_order=1, vecs=self._rot_vels_JAV
        )
        rot_vels = rot_vels_unscaled / rs
        rot_accs = self._worldvec_helper(
            subset_ids, 1, diff_order=1,
            current_diff_order=1, vecs=self._rot_vels_JAV, scale = rs
        )
        rot_jerks = self._worldvec_helper(
            subset_ids, 1, diff_order=2,
            current_diff_order=1, vecs=self._rot_vels_JAV, scale=rs
        )
        w2ls_concat: NDArray = np.concatenate(concatForComboSubset(
            self.w2ls_JAV, subset_ids
        ), axis=0)
        other_coords = (coords_tm1, coords_tm2, coords_tm3)
        other_vecs = (vels, accs, jerks, aas, rot_vels, rot_accs, rot_jerks)
        w2l_thing = (w2ls_concat.reshape(-1, 9), )
        if w2l_rotations is not None:
            other_coords = tuple(
                pm.einsumMatVecMul(w2l_rotations, c - coords)
                for c in other_coords
            )
            other_vecs = tuple(
                pm.einsumMatVecMul(w2l_rotations, v) for v in other_vecs
            )
            local_to_vel = pm.einsumMatMatMul(w2ls_concat, w2l_rotations)
            w2l_thing = (local_to_vel.reshape(-1, 9), )

        combined_tuple = (curr_cols, ) + other_coords + other_vecs + w2l_thing
        if w2l_rotations is None:
            combined_tuple += (coords, rmatv9s)

        # TODO: These assume constant timesteps for now.
        if self.outVecMode == OutVecMode.WORLD_VEC3:
            self.ref_predictions[subset_kind] = coords + vels + accs
        elif self.outVecMode == OutVecMode.WORLD_DISP:
            self.ref_predictions[subset_kind] = vels + accs
        elif self.outVecMode == OutVecMode.ROT_ALIGNED_VEC3:
            self.ref_predictions[subset_kind] = other_vecs[0] + other_vecs[1]
        elif self.outVecMode == OutVecMode.ROT_FIXED_AX:
            # self.ref_predictions[subset_kind] = \
            #     self._prev_ang_concat[subset_kind]
            self.ref_predictions[subset_kind] = np.ones((1, 1))
        elif self.outVecMode == OutVecMode.ROT_VEL_AA:
            self.ref_predictions[subset_kind] = rot_vels
        elif self.outVecMode == OutVecMode.ROT_AA:
            vel_qs = pm.quatsFromAxisAngleVec3s(rot_vels_unscaled)
            extrap_vel_qs = pm.multiplyQuatLists(
                vel_qs, pm.quatsFromAxisAngleVec3s(aas)
            )
            self.ref_predictions[subset_kind] = pm.axisAngleVec3sFromQuats(
                extrap_vel_qs, True
            ) / rs

        return np.concatenate(combined_tuple, axis=-1)
    
    def _fitWorldScaler(self, scale: float):
        scaler = StandardScaler()
        if self.translations_JAV is None:
            raise Exception("No translations stored to set scaler with!")
        
        scaler.fit(
            concatForComboSubset(
                self.translations_JAV,
                self.data_organizer.subset_ids[DataSubsetKind.TRAIN]
            )[0]
        )
        scaler.scale_[:] = scale
        
        self.translationScaler = scaler
    
    def _worldvec_concats(self, shift_from_gt: int, diff_ord: int, scale: float,
                          use_translation: bool = False, vecs = None,
                          curr_diff_ord: int = 0):
        ret_dict = dict()
        for k in DataSubsetKind.nonWholeValues():
            ret_dict[k] = self._worldvec_helper(
                self.data_organizer.subset_ids[k], shift_from_gt, diff_ord,
                scale=scale, vecs=vecs, use_translation=use_translation,
                current_diff_order=curr_diff_ord
            )
        return ret_dict
    
    def _worldvec_helper(self, subset_ids, shift_from_gt: int,
                         diff_order: int = 0, current_diff_order: int = 0,
                         scale: float = 0.0, use_translation: bool = False,
                         vecs: typing.Optional[typing.List[typing.Dict[typing.Any, NDArray]]] = None):
        start = 4 - diff_order - current_diff_order - shift_from_gt
        if use_translation:
            vecs = self.translations_JAV
        elif vecs is None:
            raise ValueError("No vectors specified!")
        concat = np.concatenate(concatForComboSubset(
            vecs, subset_ids, front_trim=start, end_trim=shift_from_gt,
            diff_order = diff_order
        ), axis=0)
        if use_translation and scale == 0.0:
            concat = self.translationScaler.transform(concat)
        elif scale != 0.0:
            concat /= scale
        return concat
    
    # TODO: move this to dataorg.py
    @staticmethod
    def _getSubsetVals(val_dict: typing.Dict[DataSubsetKind, NDArray],
                       subset_kind: DataSubsetKind):
        '''Concatenates subsets if necessary (e.g., if we ask for the whole)
        and otherwise just performs dictionary key access.'''
        if subset_kind == DataSubsetKind.WHOLE:
            return np.concatenate([
                val_dict[k] for k in DataSubsetKind.nonWholeValues()
            ], axis=0)
        return val_dict[subset_kind]

    def getRefPredictions(self, subset: DataSubsetKind):
        ret = self._getSubsetVals(self.ref_predictions, subset)
        # For "constant" predictions like with the JAV multipliers, where
        # we rely on broadcasting, we ensure the broadcasting doesn't break.
        if len(self.gt_train) == 1:
            raise ValueError("You only have one training sample???!")
        elif len(self.ref_predictions[DataSubsetKind.TRAIN]) == 1:
            # Unless something ELSE broke, a ref prediction of len 1 means that
            # our ref predictions are constant. So to make sure broadcasting
            # still works, we need to ensure we keep the len at 1.
            return ret[:1]
        return ret        

    def getScoresSubset(self, subset: DataSubsetKind, predictions: NDArray,
                        should_print: bool = True):

        _predictions = predictions
        gt_source = self._gt_arrs
        if self.outVecMode == OutVecMode.ROT_FIXED_AX:
            ref_axes = self._getSubsetVals(self._prev_vel_ax_concat, subset)
            ref_angs = self._getSubsetVals(self._prev_ang_concat, subset)
            _predictions = (predictions * ref_angs) * ref_axes
            gt_source = self._gt_rot_vels

        inds: typing.Dict[str, NDArray] = dict() 
        gt_param = self._getSubsetVals(gt_source, subset)
        if subset == DataSubsetKind.WHOLE:
            ttvks = DataSubsetKind.nonWholeValues()
            inds = {
                s: np.concatenate(
                    [self.data_organizer.subset_skip_inds[t][s] for t in ttvks]
                ) for s in SkipSubsetKind if s >= 0
            }
            inds[SkipSubsetKind._all] = ...
        else:
            inds = self.data_organizer.subset_skip_inds[subset]

        return self._scoreHelper(
            _predictions, gt_param, self.score_fn, self.score_scaler, inds,
            should_print, self.skip
        )
    
    
    def getScoresTrain(self, predictions: NDArray, should_print: bool = True):
        return self.getScoresSubset(
            DataSubsetKind.TRAIN, predictions, should_print
        )

    def getScoresTest(self, predictions: NDArray, should_print: bool = True):
        return self.getScoresSubset(
            DataSubsetKind.TEST, predictions, should_print
        )
    
    def getScoresValidation(self, predictions: NDArray, should_print: bool = True):
        return self.getScoresSubset(
            DataSubsetKind.VALIDATION, predictions, should_print
        )

    @staticmethod
    def _scoreHelper(preds: NDArray, gt_vals: NDArray,
                     score_fn: typing.Callable, sc, inds: typing.Dict,
                     should_print: bool, skip: SkipSubsetKind):
        _preds = preds
        _gts = gt_vals
        if sc is not None and sc != 0.0:
            if isinstance(sc, float):
                _preds = preds * sc
                _gts = gt_vals * sc
            else:         
                _preds = sc.inverse_transform(preds)
                _gts = sc.inverse_transform(gt_vals)

        if skip >= 0:
            inds = {SkipSubsetKind(skip) : ...}
            
        errs: NDArray = score_fn(_gts, _preds)
        if not isinstance(errs, np.ndarray):
            errs = errs.numpy()
        # Print scores on test data.
        scores = {k: np.mean(errs[v]) for k, v in inds.items()}
        if should_print:
            for k, v in scores.items():
                print(k.display_name + ":", v)
        return scores
    
bcotjav = DataForJAV(
    dog, bcot_loaders, bcs_scaler, nonco_cols, JAV_order, chosen_mode,
    save_data_for_conf=True, #skip=2
)
#%%
print("Reference scores for whole dataset (i.e., no train/test split).")
print("Reference method:", bcotjav.ref_prediction_name)
print("Reference scores:")
bcotjav.getScoresSubset(DataSubsetKind.WHOLE, bcotjav.getRefPredictions(DataSubsetKind.WHOLE), True)

#%%
import pickle
with open("./results/models/scaler.pickle", "wb") as f:
    pickle.dump(bcs_scaler, f)


#%%
bcs_model = getUntrainedNN(bcotjav.in_train.shape[1], sel_dim, sel_loss) #, 5, 256)



#%% Train the network.
if TRAIN_NEW_MODEL:
    val_param = None
    if bcot_validation_combos is not None and len(bcot_validation_combos) > 0:
        val_param = (bcotjav.in_validation, bcotjav.gt_validation)
    bcs_hist = bcs_model.fit(
        bcotjav.in_train, bcotjav.gt_train, epochs=32, shuffle=True,
        validation_data=val_param,
        # batch_size = 1024
    )
else:
    bcs_model = loadLatestModels((chosen_mode.name, ))[chosen_mode.name]
#%% Evaluate network on test data.

bcs_pred = bcs_model.predict(bcotjav.in_test, batch_size = 1024)
bcotjav.getScoresTest(bcs_pred) #scaledAAs(bcotjav.in_test[:, -30:-27]))

#%% Saving model to disk.
if TRAIN_NEW_MODEL:
    model_name = "results/models/{}-{:%Y-%m-%d_%H-%M-%S}.keras".format(
        chosen_mode.name, datetime.datetime.now()
    )
    bcs_model.save(model_name)

#%% Print scores on test data.
bcs_test_errs: NDArray = sel_loss(bcotjav.gt_test, bcs_pred).numpy()

#%%
bcot_test_ids = dog.subset_ids[DataSubsetKind.TEST]
med_test_scores = np.empty((3, len(bcot_test_ids)))
all_grouped_test_scores = []
for skip_amt in range(3):
    skip_bounds = dog.skip_bounds[DataSubsetKind.TEST][skip_amt:(skip_amt + 2)]
    skip_subset = bcs_test_errs[skip_bounds[0]:skip_bounds[1]]
    boundaries = dog.frame_bounds[DataSubsetKind.TEST][skip_amt]
    grouped_test_scores = []
    for i in range(len(bcot_test_ids)):
        row_idxs = boundaries[i:(i+2)]
        errs_for_id = skip_subset[row_idxs[0]:row_idxs[1]]
        med_test_scores[skip_amt, i] = np.median(errs_for_id)
        grouped_test_scores.append(errs_for_id)
    all_grouped_test_scores.append(grouped_test_scores)

#%%
def bcot_name(bcot_id):
    return (gtc.BCOT_SEQ_NAMES[bcot_id[1]], gtc.BCOT_BODY_NAMES[bcot_id[0]])
med_test_sort_idxs = np.argsort(med_test_scores, axis=1)
med_test_sort_ids = [
    [bcot_name(bcot_test_ids[i]) for i in mtsi] for mtsi in med_test_sort_idxs
] 

# print(bcs_test_scores)
#%%
import errorstats as es
motion_data_key_subset = [dog.motion_data_keys[i] for i in nonco_col_nums]

all_rotation_mats_T: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
for combo in combos:
    calculator = PoseLoaderBCOT(combo[0], combo[1])
    all_rotation_mats_T[combo[:2]] = np.swapaxes(
        calculator.getRotationMatsGTNP(), -1, -2
    ) # transposes
#%%
print("Starting the confidence interval stuff.")

# TODO: Replace motion-kind str with an enum in here and other files.
motion_kinds_plus = PoseLoaderBCOT.motion_kinds + ["all"]
reframed_JAV_errs: typing.List[typing.Dict[str, NDArray]] = [
    {mk: [] for mk in motion_kinds_plus} for _ in range(3)
]

lim_class_errs: typing.List[typing.Dict[str, NDArray]] = [
    {mk: [] for mk in motion_kinds_plus} for _ in range(3)
]

for skip in range(3):
    for combo in bcot_test_combos:
        c2 = combo[:2]
        curr_input_dict = all_motion_data[skip][c2]
        curr_input = np.stack(
            [curr_input_dict[k] for k in motion_data_key_subset], axis=-1
        )
        curr_input = bcs_scaler.transform(curr_input)

        curr_jav_pred = bcs_model.predict(
            curr_input, batch_size=1024, verbose=0
        )

        world_disp = getWorldFrameDisplacements(
            bcotjav.jav_per_combo[skip][c2], curr_jav_pred,
            bcotjav.w2ls_JAV[skip][c2]
        )
        curr_translations = bcotjav.translations_JAV[skip][c2]
        in_translations = curr_translations[-(len(world_disp) + 1):-1]
        jav_pred = in_translations + world_disp

        curr_rotation_mats = all_rotation_mats_T[c2][::(skip+1)]
        curr_jav_errs = es.localizeErrsInFrames(
            {"JAV": jav_pred}, curr_translations[1:], 
            curr_rotation_mats[1:]
        )["JAV"]

        # class_lim_start_ind = -(len(curr_min_norm_vecs) + 1)
        curr_min_norm_vecs = cfc.min_norm_vecs[skip][c2] \
            + curr_translations[-len(world_disp):]
        
        curr_class_lim_errs = es.localizeErrsInFrames(
            {"Class Lim":  curr_min_norm_vecs},
            curr_translations[1:], curr_rotation_mats[1:]
        )["Class Lim"]

        reframed_JAV_errs[skip][combo[-1]].append(curr_jav_errs)
        reframed_JAV_errs[skip]["all"].append(curr_jav_errs)
        lim_class_errs[skip][combo[-1]].append(curr_class_lim_errs)
        lim_class_errs[skip]["all"].append(curr_class_lim_errs)
        progress_str = "\rProgress: skip {}, combo ({:2},{:2})".format(
            skip, c2[0], c2[1]
        )
        print(progress_str, end = '', flush=True)
#%%
def printErrStats3D(errs: typing.List[typing.Dict[str, NDArray]]):
    world_field = es.LocalizedErrsCollection._fields[0]
    for skip in range(3):
        print("\nSkip {}:".format(skip))

        for mk in motion_kinds_plus:
            curr_errs = np.concatenate(errs[skip][mk], axis=1)

            stats = {
                f: es.getStats(curr_errs[i])
                for i, f in enumerate(es.LocalizedErrsCollection._fields)
            }
            print("  {} (score {:0.4f}):".format(mk, stats[world_field].mean_mag))
            for name, stat in stats.items():
                print(es.formattedErrStats(stat, name, 4, stats[world_field]))
            print()

printErrStats3D(reframed_JAV_errs)


#%% 

################################################################################
# Column Scrambling to Assess Feature Importance
################################################################################

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
    preds = model.predict(data_scramble, batch_size=1024, verbose=0)
    errs = model.loss(y_true, preds)
    return errs

# For each column and for each skip amount, scramble the column and find the new
# score.
scramble_scores = np.empty((dog.col_subset_test.shape[1], len(SkipSubsetKind)))
print()
 

num_nonco_cols = dog.col_subset_test.shape[1]
 

for col_ind in range(num_nonco_cols):
    print(
        "Testing column index {:03d}/{}.".format(col_ind + 1, num_nonco_cols),
        end='\r', flush=True
    )
    errs = errsForColScramble(
        bcs_model, dog.col_subset_test, col_ind, bcotjav.jav_test
    )
    for i, k in enumerate(SkipSubsetKind):
        scramble_scores[col_ind, i] = np.mean(
            errs[dog.subset_skip_inds[DataSubsetKind.TEST][k]]
        )
#%%
scramble_rank = np.argsort(scramble_scores, axis=0)[::-1]
print("Most important feature inds:", scramble_rank[:10], sep='\n')
    



#%%
import shap
default_rng = np.random.default_rng()
shap_bg = default_rng.choice(dog.col_subset_test, 100, False, axis=0)
shap_ex = shap.GradientExplainer(bcs_model, shap_bg)

#%%
shap_test_inds = default_rng.choice(len(dog.col_subset_test), 200, False, axis=0)
shap_test = dog.col_subset_test[shap_test_inds]
# If input shape is (n_rows, n_feats) and output shape is (n_rows, n_outs), shap
# values shape is (n_rows, n_feats, n_outs)
shap_values_tf = shap_ex(shap_test)
preds_shap_bg = bcs_model(shap_bg).numpy()
# SHAP DeepExplainer seems to leave base_values as None, which breaks certain
# plots, so I need to fill it in manually unless I use GradientExplainer.
# shap_values_tf.base_values = np.broadcast_to(
#     preds_shap_bg.mean(axis=0), (len(shap_test), preds_shap_bg.shape[-1])
# )
# We also set the feature_names so that plots include them.
shap_values_tf.feature_names = nonco_featnames
shap.initjs()
shap_sort = np.argsort(
    np.mean(np.abs(shap_values_tf.values), axis=0), axis=0
)[::-1].transpose()
#%%
output_shap_ind = 0
shap.summary_plot(shap_values_tf.values[..., output_shap_ind], shap_test, feature_names=nonco_featnames)

#%%
fig, ax = shap.partial_dependence_plot(
    77,
    lambda x: bcs_model.predict(x, verbose=0)[..., output_shap_ind],
    shap_test,
    model_expected_value=True,
    feature_expected_value=True,
    show=False,
    ice=False,
)

#%%
shap.plots.bar(shap_values_tf[..., output_shap_ind].abs.max(0))

#%%
shap.plots.scatter(
    shap_values_tf[:, 100, output_shap_ind],
    color=shap_values_tf[..., output_shap_ind]
)

#%%
clustering = shap.utils.hclust(shap_test) #, bcotjav.jav_test[shap_test_inds, 12:15])
#%%
shap.plots.bar(shap_values_tf[..., output_shap_ind], clustering=clustering)
#%% Graphing the SHAP and scramble importance rankings.

scramble_for_bars = scramble_scores - np.min(scramble_scores, axis=0)
scramble_for_bars /= (np.mean(scramble_for_bars, axis=0) + np.std(scramble_for_bars, axis=0))
num_features = len(nonco_col_nums)
nonco_arange = np.arange(num_features)
for skip in range(4):
    plt.bar(
        nonco_arange + skip / 4, scramble_for_bars[:, skip],
        width = 1/4, label="skip " + str(skip)
    )
plt.legend()
plt.xticks(nonco_arange[::5])
plt.xticks(nonco_arange, minor=True)
plt.xlabel("Feature number")
plt.ylabel("Scramble Importance")
plt.ylim(0, 1)
plt.show()

#%%
shap_accum = np.mean(np.abs(shap_values_tf.values), axis=0)
shap_accum /= (np.mean(shap_accum, axis=0) + np.std(shap_accum, axis=0))
num_muls = shap_values_tf.shape[-1]
for mul in range(num_muls):
    plt.bar(
        nonco_arange + mul / num_muls, shap_accum[:, mul],
        width = 1/12, label="mul " + str(mul)
    )
plt.legend()
plt.xticks(nonco_arange[::5])
plt.xticks(nonco_arange, minor=True)
plt.xlabel("Feature number")
plt.ylabel("SHAP Importance")
plt.ylim(0, 1)
plt.show()

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

input_dim = dog.col_subset_train.shape[1]
num_classes = len(MOTION_MODEL)

onehot_train_labels = tf.one_hot(dog.concat_train_labels, num_classes)

# I wanted to try to get the below to overfit to make sure I wasn't doing
# "something wrong" before worrying about regularization. Initial attempts to
# overfit failed, and train performance was no better than with a decision tree.
# I finally succeeded in getting it to start overfitting (though it also
# outperformed the tree slightly on test data) by using 3 hidden layers, 2048
# nodes each, relu activation, no regularization, sigmoid activation for final
# layer, CategoricalCrossentropy loss for training, using only noncolinear
# columns, default 'adam' optimizer, batch size 1024, and then training for a
# few sets of 5 epochs. These ideas generally came from:
# https://stats.stackexchange.com/questions/474738/how-do-i-intentionally-design-an-overfitting-neural-network
nncl_per_layer_num = 2048
nncl_act = 'relu'
tfmodel = keras.Sequential([
    keras.layers.Input((input_dim,)),
    keras.layers.Dense(nncl_per_layer_num, activation=nncl_act),
    # keras.layers.Dropout(0.2),
    keras.layers.Dense(nncl_per_layer_num, activation=nncl_act),
    keras.layers.Dense(nncl_per_layer_num, activation=nncl_act),
    # keras.layers.Dense(1024, activation='sigmoid'),
    # keras.layers.Dropout(0.2),
    # keras.layers.Dense(1, activation='sigmoid'),
    keras.layers.Dense(num_classes, activation='sigmoid')])

tfmodel.summary()

adam = 'adam' #keras.optimizers.Adam(0.01)
catcross = True
tf_loss_fn2 = keras.losses.CategoricalCrossentropy(from_logits=True) \
              if catcross else tf_loss_fn
ncll_y = onehot_train_labels if catcross else dog.concat_train_class_errs

tfmodel.compile(optimizer=adam, loss=tf_loss_fn2)
#%%
nn_class_hist = tfmodel.fit(
    dog.col_subset_train, ncll_y, epochs=5, batch_size = 1024, shuffle=True
)
#%%
bgtrain = big_tree.predict(dog.concat_train_data)
bgtrain_score = dog.getClassScoresTrain(bgtrain)
print("Big tree train errs=", bgtrain_score)

mclf_train = mclf.predict(dog.concat_train_data)
mclf_train_score = dog.getClassScoresTrain(mclf_train)
print("Smaller tree train errs=", mclf_train_score)

tf_train_preds = np.argmax(tfmodel(dog.col_subset_train).numpy(), axis=1)
tf_train_errs = dog.getClassScoresTrain(tf_train_preds)
print("TF train errs=", tf_train_errs)
print()

tf_test_preds = np.argmax(tfmodel(dog.col_subset_test).numpy(), axis=1)
tf_test_errs = dog.getClassScoresTest(tf_test_preds)
print("TF test errs=", tf_test_errs)


#%%
################################################################################
# Comparing Error Histograms
################################################################################
import matplotlib.pyplot as plt

big_tree_preds = big_tree.predict(dog.concat_test_data)
tree_errs_for_plt = dog.getClassErrsTest(big_tree_preds)

acc_only_preds = np.full(
    (len(dog.col_subset_test), 1),
    dog.motion_mod_keys.index(MOTION_MODEL.ACC_DEG2)
)
acc_errs_for_plt = dog.getClassErrsTest(acc_only_preds)


bar_skip_key = SkipSubsetKind.skip2
bar_data: typing.List[NDArray] = [
    tree_errs_for_plt[bar_skip_key], acc_errs_for_plt[bar_skip_key],
    bcs_test_errs[dog.subset_skip_inds[DataSubsetKind.TEST][bar_skip_key]],
    error_lim_all[bar_skip_key]
]
bar_labels = ['Tree', 'Acc Only', 'JAV NN', "Classification lim"]
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

#%%
from gtCommon import PoseLoaderTUDL

tudl_ids = PoseLoaderTUDL.getAllIDs(True)
tudl_loaders = [PoseLoaderTUDL(*t) for t in tudl_ids]

cfc.getAll(tudl_loaders)
#%%
tudl_train, tudl_test = PoseLoaderTUDL.trainTestIDs()
tdog = DataOrganizer(
    cfc.all_motion_data, cfc.min_norm_labels, cfc.err_norm_lists,
    tudl_train, tudl_test, dog.motion_data_keys
)
#
tjav = DataForJAV(tdog, tudl_loaders, bcs_scaler, nonco_cols, JAV_order)
#%%

tjavps = bcs_model.predict(tdog.col_subset_test)
tjav.getScoresTest(tjavps)
#%%
ajnn = getUntrainedNN()

bt_train = np.concatenate((dog.col_subset_test, tdog.col_subset_test), axis=0)
bt_jav = np.concatenate((bcotjav.jav_test, tjav.jav_test), axis=0)

#%%
ajnn.fit(bt_train, bt_jav, epochs=32, shuffle=True)

#%%
bcot_ajnn_pred = ajnn.predict(dog.col_subset_train, batch_size=1024)

tudl_ajnn_pred = ajnn.predict(tdog.col_subset_train, batch_size=1024)

bcotjav.getScoresTrain(bcot_ajnn_pred)
tjav.getScoresTrain(tudl_ajnn_pred)
#%%
ax = plt.figure().add_subplot(projection='3d')

data_to_3d_plot = bcot_ajnn_pred[:, :3]

inds_dict = dog.subset_skip_inds[DataSubsetKind.TRAIN]

for k in SkipSubsetKind:
    if k == SkipSubsetKind._all:
        continue
    dp = data_to_3d_plot[inds_dict[k]]
    inds = np.random.choice(len(dp), 1000)

    # print(dp[inds].T.shape)
    ax.scatter3D(*dp[inds].T, label="BCOT " + k.display_name)

data_to_3d_plot = tudl_ajnn_pred[:, :3]
inds_dict = tdog.subset_skip_inds[DataSubsetKind.TRAIN]

for k in SkipSubsetKind:
    if k == SkipSubsetKind._all:
        continue
    dp = data_to_3d_plot[inds_dict[k]]
    inds = np.random.choice(len(dp), 1000)

    # print(dp[inds].T.shape)
    ax.scatter3D(*dp[inds].T, label="TUDL " + k.display_name)

ax.set_xlabel("i")
ax.set_ylabel("j")
ax.set_zlabel("k")
ax.legend()
plt.show()

# 

#%%

def getUntrainedNNC():
    dropout_rate = 0.2
    nodes_per_layer = 128
    vel_nn_activation = 'sigmoid'

    input_layer = keras.Input(shape=(len(nonco_col_nums),))

    def build_branch():
        x = keras.layers.Dense(nodes_per_layer, activation=vel_nn_activation)(input_layer)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(nodes_per_layer, activation=vel_nn_activation)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(nodes_per_layer, activation=vel_nn_activation)(x)
        x = keras.layers.Dense(1)(x)  # Output a single float
        return x

    # Build 3 branches
    branch_outputs = [build_branch() for _ in range(3)]

    # Concatenate the three outputs
    output = keras.layers.Concatenate()(branch_outputs)

    model = keras.Model(inputs=input_layer, outputs=output)
    model.summary()
    model.compile(loss=poseLossJAV, optimizer='adam')
    return model

ajnnC = getUntrainedNNC()

ajnnC.fit(bt_train, bt_jav, epochs=32, shuffle=True)

#%%
ajnnC_pred = ajnnC.predict(tdog.col_subset_train)
tjav.getScoresTrain(ajnnC_pred)

# ajnnC_loss = poseLossJAV(bcs_test, ajnnC_pred)
# print({k: np.mean(ajnnC_loss[v]) for k, v in dog.skip_inds_dict.items()})

#%%
def getUntrainedNNSplit(input_shape1, input_shape2, input_shape3):
    dropout_rate = 0.2
    nodes_per_layer = 32
    vel_nn_activation = 'sigmoid'

    input1 = keras.Input(shape=(input_shape1,), name='input1')
    input2 = keras.Input(shape=(input_shape2,), name='input2')
    input3 = keras.Input(shape=(input_shape3,), name='input3')

    def build_branch(inp):
        x = keras.layers.Dense(nodes_per_layer, activation=vel_nn_activation)(inp)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(nodes_per_layer, activation=vel_nn_activation)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(nodes_per_layer, activation=vel_nn_activation)(x)
        x = keras.layers.Dense(1)(x)
        return x

    out1 = build_branch(input1)
    out2 = build_branch(input2)
    out3 = build_branch(input3)

    output = keras.layers.Concatenate()([out1, out2, out3])

    model = keras.Model(inputs=[input1, input2, input3], outputs=output)
    model.compile(loss=poseLossJAV, optimizer='adam')
    model.summary()
    return model

lnccn = len(nonco_col_nums)
split_nn = getUntrainedNNSplit(lnccn, lnccn - 1, lnccn - 2)
split_nn.fit([dog.col_subset_test, dog.col_subset_test[:, :-1], dog.col_subset_test[:, :-2]], bcotjav.jav_test, epochs=33, batch_size=128, shuffle=True)

#%%
tjav.getScoresTest(split_nn.predict([tdog.col_subset_test, tdog.col_subset_test[:, :-1], tdog.col_subset_test[:, :-2]]))

#%%
from scipy.optimize import lsq_linear
# Get the pose loss for a set of Jerk, Acceleration, & Velocity multipliers.
def gtMultipliersJAV(y_true, bounds = None, tol: float = 0.001, max_iter: int = 32, verbose=False):
    '''
    When we create the "JAV" data, we specify the permutation of
    (velocity, acceleration, jerk) to orthonormalize into frames. For simplicity
    below, assume the order is in fact velocity, then acceleration, then jerk.

    In that case, y_true contains the following columns, in order:
     - speed
     - accel parallel to velocity
     - accel ortho to velocity
     - jerk parallel to speed, ortho to speed but in acc plane, ortho to plane
     - correct pose displacement in the same "coordinate frame" as the jerk.

    In other words, we are working in an orthonormal coordinate frame where the 
    x-axis is aligned with velocity, the y with acceleration, and then z is
    orthogonal to both.
    
    Then, y_pred contains the multipliers for velocity, acceleration, and jerk, 
    respectively. The predicted "local" displacement is thus:
    [[speed, acc_x, jerk_x],       [vel_multiplier,
     [0,     acc_y, jerk_y],     x  acc_multiplier,
     [0,     0,     jerk_z]]        jerk_multiplier]

    Then after this matrix multiplication, we find the distance between it and
    the correct pose displacement, both vec3s.
    '''
    muls_to_pt_mats = np.zeros((len(y_true), 3, 3))
    muls_to_pt_mats[:, 0, 0] = y_true[:, 0]
    muls_to_pt_mats[:, :2, 1] = y_true[:, 1:3]
    muls_to_pt_mats[:, :, 2] = y_true[:, 3:6]

    if bounds is None:
        inv_mats = np.linalg.inv(muls_to_pt_mats)
        return pm.einsumMatVecMul(inv_mats, y_true[:, 12:15]) #6:9])
    
    n = len(y_true)
    res = np.empty((n, 3))
    n_digs = 0
    status_template = "Done iter {}/" + str(n) + " ({:0.2f})."
    if verbose:
        n_digs = int(np.ceil(np.log10(n)))
        status_template = status_template + (" " * n_digs)
        print(status_template.format(0, 0), end='')

    for i in range(n):
        lsq_res = lsq_linear(
            muls_to_pt_mats[i], y_true[i, 12:15], #6:9],
            bounds, tol=tol, max_iter=max_iter
        )
        res[i] = lsq_res.x
        if (verbose and (i + 1) % 1000 == 0) or i == (n - 1):
            print("\r" + status_template.format((i + 1), (i + 1)/n), end = '')
    if verbose:
        print()
    return res

bounds = (-3,3)
bounded_gt_bcot_tr = gtMultipliersJAV(bcotjav.jav_train, bounds, verbose=True)
bounded_gt_tudl_tr = gtMultipliersJAV(tjav.jav_train, bounds, verbose=True)

# %%
def gtScaledVelAligned(y_true, bounds = None):
    gathered = y_true[:, [0,2,5]]
    scaled = y_true[:, -3:] / gathered
    if bounds is not None:
        scaled = np.clip(scaled, *bounds)
    return scaled
gt_bcot = gtScaledVelAligned(bcotjav.jav_train, bounds)
gt_tudl = gtScaledVelAligned(tjav.jav_train, bounds)

bcotjav.getScoresTrain(gt_bcot)
tjav.getScoresTrain(gt_tudl)

#%%
def myGetClosestPoint(normals: NDArray, scaled_plane_offsets: NDArray, points: NDArray):
    norm_sq = pm.einsumDot(normals, normals)
    pn = None
    if points.ndim > 1 and len(points) > 1:
        pn = pm.einsumDot(normals, points)
    else: 
        pn = normals @ points.flatten()
    scalar = (scaled_plane_offsets - pn) / norm_sq
    disps = pm.scalarsVecsMul(scalar, normals)
    return points + disps

def testClosest(normals, offsets, pts):
    closest = myGetClosestPoint(normals, offsets, pts)
    assert np.allclose(pm.einsumDot(closest, normals), offsets)
    diffs = pts - closest
    diff_norms = np.linalg.norm(diffs, axis=-1)
    unit_diffs = pm.safelyNormalizeArray(diffs, diff_norms[..., np.newaxis])
    unit_norms = pm.normalizeAll(normals)
    dot_diffs = np.abs(pm.einsumDot(unit_diffs, unit_norms)) - 1.0
    dots_good = dot_diffs < 0.0001
    norms_good = diff_norms < 0.0001
    assert (np.all(dots_good | norms_good))
    return closest
#%%
def my_gtMultipliers6(y_true: NDArray, baseline_m6: NDArray):
    
    j_o = y_true[:, 8] / y_true[:, 5]

    flat = baseline_m6.flatten()
    a_a, j_a = myGetClosestPoint(
        y_true[:, [2, 4]], y_true[:, 7], flat[[2, 4]]
    ).transpose()

    v_v, a_v, j_v = myGetClosestPoint(
        y_true[:, [0, 1, 3]], y_true[:, 6], flat[[0, 1, 3]]
    ).transpose()

    return np.stack([v_v, a_v, a_a, j_v, j_a, j_o], axis=-1)

base = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
gt_bcot = my_gtMultipliers6(bcotjav.jav_train, base)
gt_tudl = my_gtMultipliers6(tjav.jav_train, bounds, verbose=True)

bcotjav.getScoresTrain(gt_bcot)
tjav.getScoresTrain(gt_tudl)

#%%
bcs_pred_tr = bcs_model.predict(dog.col_subset_train, batch_size=1024)

#%%

current_idx = 0

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

def getRandSubset(data, gt, n):
    inds = np.random.choice(len(data), n)
    return data[inds], gt[inds]

skis = [dog.subset_skip_inds[DataSubsetKind.TRAIN][i] for i in range(3)]
bcot_sub_data = []
bcot_sub_gt = []
bcot_sub_nn = []

for s in skis:
    skip_data = dog.concat_train_data[s]
    bcot_sub_inds = np.random.choice(len(skip_data), 500)
    bcot_sub_data.append(skip_data[bcot_sub_inds])
    # bcot_sub_gt.append(bcotjav.jav_train[s][bcot_sub_inds, 9:15])
    bcot_sub_gt.append(gt_bcot[s][bcot_sub_inds])
    bcot_sub_nn.append(bcs_pred_tr[s][bcot_sub_inds])
# p_sub_data, p_sub_gt = getRandSubset(pdog.concat_train_data, gt_p, 1000)


# The below function was written by Copilot/Claude but then modified by me.
# E.g., I added the "superbin_radius", median calc, etc.
def calculate_interval_means(x, y, lb, ub, n, superbin_radius):
    """
    Calculate means of y values corresponding to x values in n equally spaced intervals between lb and ub.
    
    Parameters:
    -----------
    x : np.ndarray
        Input array of values to be binned
    y : np.ndarray
        Array of values whose means will be calculated for each bin
    lb : float
        Lower bound of the interval
    ub : float
        Upper bound of the interval
    n : int
        Number of intervals to create
    
    Returns:
    --------
    tuple:
        - bin_centres: array of bin centres (length n)
        - means: array of means for each interval (length n)
        - medians: array of medians for each interval (length n)
    """
    # Create bin edges
    bin_edges = np.linspace(lb, ub, n + 1)
    
    # Use np.digitize to find which bin each x value belongs to
    # Subtract 1 from bin indices because np.digitize starts counting from 1
    bin_indices = np.digitize(x, bin_edges) - 1
    
    # Create mask for values within the bounds
    mask = (x >= lb) & (x <= ub)
    
    # Initialize arrays to store results
    means = np.zeros(n)
    medians = np.zeros(n)

    # Calculate means for each bin
    for i in range(n):
        bin_mask = (bin_indices >= i - superbin_radius) 
        bin_mask = bin_mask & (bin_indices <= i + superbin_radius)
        bin_mask = bin_mask & mask
        if np.any(bin_mask):
            bin_data = y[bin_mask]
            means[i] = np.mean(bin_data)
            medians[i] = np.median(bin_data)
        else:
            means[i] = np.nan
            medians[i] = np.nan

    bin_centres = (bin_edges[1:] + bin_edges[:-1])/2.0
    return bin_centres, means, medians


def plot_column(idx):
    fig.suptitle(f'(BCOT) Column: {feature_names[idx]} (#{idx})')
    
    all_column_data = np.concatenate(
        [bcot_sub_data[j][:, idx] for j in range(3)], axis=0
    )
    val_min = np.min(all_column_data)
    val_max = np.max(all_column_data)
    quants = np.quantile(all_column_data, (0.1, 0.9))

    na_inds = (all_column_data == ERR_NA_VAL)
    if np.any(na_inds):
        n_na_vals = all_column_data[~na_inds]
        val_min = np.min(n_na_vals)
        val_max = np.max(n_na_vals)

    xlims = (val_min, val_max)
    if (val_max - val_min)/(quants[1] - quants[0]) > 1.5:
        xlims = (quants)

    colours = ['blue', 'orange', 'green']
    for i in range(3):
        axs[i].cla()  # clear the axes
        for j in range(2,-1,-1): #3):
            c = colours[j]
            bsd = bcot_sub_data[j][:, idx]
            label = "skip" + str(j)
            axs[i].scatter(bsd, bcot_sub_gt[j][:, i], alpha=0.01, color=c)
            inter_data = calculate_interval_means(
                bsd, bcot_sub_gt[j][:, i], *xlims, 100, 2
            )
            axs[i].plot(inter_data[0], inter_data[1], color=c, ls="dashed", label=label)
            axs[i].plot(inter_data[0], inter_data[2], color=c, ls="dotted")
        # axs[i].scatter(p_sub_data[:, idx], p_sub_gt[:, i], alpha=0.6, label="Pauwels")
        # for j in range(3):
        #     axs[i].scatter(bcot_sub_data[j][:, idx], bcot_sub_nn[j][:, i], alpha=0.6, label="NN" + str(j))


        axs[i].set_ylim(-0.1, 1.1) #-2, 2)

        axs[i].set_xlabel(gtc.truncateName(feature_names[idx], 33))
        axs[i].set_xlim(*xlims)
    axs[0].set_ylim(0.8, 1.2)
    axs[0].legend()
    axs[0].set_ylabel("Multiplier")
    axs[0].set_title("Vel")
    axs[1].set_title("Acc parallel")
    axs[2].set_title("Acc ortho")


    plt.tight_layout()
    fig.canvas.draw_idle()

def on_key(event):
    global current_idx
    if event.key == 'right':
        current_idx = min(current_idx + 1, len(feature_names) - 1)
        plot_column(current_idx)
    elif event.key == 'left':
        current_idx = max(current_idx - 1, 0)
        plot_column(current_idx)

plot_column(current_idx)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
# %%

def testAllDiscreteJAV(aj_combos = None):
    # Figure out what are the best constant vel, acc, jerk multipliers out of the
    # "physics-based" options.
    acc_opts = [0.0, 0.5, 1.0] # 1.0 means deg2 velocity calculation
    
    # When you look at the degree-3 lagrange interpolating polynomial and its 
    # derivatives at "x=3", and letting v' represent the degree-1 velocity and 
    # letting a' represent degree-2 velocity, one can see that degree-3 
    # acceleration is (a' + j), and one can see that degree-3 velocity is 
    # (v' + a'/2 + j/3). So if we use jerk in no calculations, the total jerk
    # multiplier is 0. If we use it just in the jerk calculation, the total jerk
    # multipliers is 1/6. If we use it in just the jerk and accel calculations,
    # but not velocity, the total is (1/2 * 1) + (1/6 * 1) = 2/3. Finally, if
    # we use it in all calculations, the total is 1.0.
    # So the "most sensible" options for the jerk multiplier are 
    # [0, 1/6, 2/3, 1]. However, if we allow more "odd" combinations, like
    # using it for the velocity and jerk but not acceleration, we can have
    # all of the below 1/6 multiples:
    jerk_opts = [0.0, 1/6, 1/3, 1/2, 2/3, 5/6, 1.0]

    if aj_combos is None:
        aj_combos = []
        for a0 in range(3):
            for a1 in range(3):
                for j0 in range(7):
                    for j1 in range(7):
                        for j2 in range(7):
                            aj_combos.append((
                                1.0, acc_opts[a0], acc_opts[a1], 
                                jerk_opts[j0], jerk_opts[j1], jerk_opts[j2]
                            ))

    ajnp = np.empty((len(aj_combos), len(SkipSubsetKind)))
    for i, ajc in enumerate(aj_combos):
        ajscore = bcotjav.getScoresTrain(np.array([[*ajc]]), False)
        for j, sk in enumerate(SkipSubsetKind):
            ajnp[i, j] = ajscore[sk]
    return ajnp
#%%


base_a_opts = [0.0, 0.5, 1.0]
# See other commenting on how these possible multiplier totals are found
# when looking at the lagrange polynomial derivatives.
base_j_opts = [0, 1/6, 2/3, 1.0]

base2 = getBaselineJAV6(base_a_opts, base_j_opts)
gt_bcot = gtMultipliers6(bcotjav.jav_train, np.asarray(base2))

#%%
all_base_errs = np.empty((len(bcotjav.jav_train), len(base2)))
for i, b in enumerate(base2):
    all_base_errs[:, i] = poseLossJAV(bcotjav.jav_train, np.asarray(b).reshape(1, -1))

min_base_errs = np.min(all_base_errs, axis=-1)
for k, v in dog.subset_skip_inds[DataSubsetKind.TRAIN].items():
    print(k.display_name, ":", np.mean(min_base_errs[v]))

argmin_base_errs = np.argmin(all_base_errs, axis=-1)
plt.bar(*np.unique(argmin_base_errs, return_counts=True))
plt.show()

#%%
mcb = WeightedErrorCriterion(1, np.array([len(base2)], dtype=np.intp))
base_errs_reshape = all_base_errs.reshape((all_base_errs.shape[0], 1, all_base_errs.shape[1]))
mcb.set_y_errs(base_errs_reshape)


dumb_labels = np.zeros(len(dog.concat_train_data))
dumb_labels[:len(base2)] = np.arange(len(base2))

#%%
base2_tree = sk_tree.DecisionTreeClassifier(max_depth=8, criterion=mcb)
start_time = time.time()
base2_tree = base2_tree.fit(dog.concat_train_data, dumb_labels)
print("Done!")
print("Time spent:", time.time() - start_time)

#%%
from motiontools.dataorg import motionClassScores

test_base_errs = np.empty((len(bcotjav.jav_test), len(base2)))
for i, b in enumerate(base2):
    test_base_errs[:, i] = poseLossJAV(bcotjav.jav_test, np.asarray(b).reshape(1, -1))

#%%
base2_pred = base2_tree.predict(dog.concat_test_data)
base2_pred_res = motionClassScores(
    test_base_errs, base2_pred.astype(int), dog.skip_inds_dict
)

for k, v in base2_pred_res.items():
    print(k, ":", v)

#%%

class_resid_nn = getUntrainedNN(None, True)

cl_start_pt = np.empty((len(dog.concat_train_data), 6))

cl_start_tree = trim_to_depth(base2_tree, 4)
base2_tr_pred = cl_start_tree.predict(dog.concat_train_data).astype(int)

for i, cl_start_val in enumerate(base2_tr_pred):
    cl_start_pt[i] = base2[cl_start_val]

class_resid_nn.fit(
    np.concatenate((dog.col_subset_train, cl_start_pt), axis=-1),
    bcotjav.jav_train, # np.concatenate((bcotjav.jav_train, cl_start_pt), axis=-1),
    epochs=32, shuffle=True
)

#%%


cl_start_pt_test = np.empty((len(dog.concat_test_data), 6))

base2_te_pred = cl_start_tree.predict(dog.concat_test_data).astype(int)

for i, cl_start_val in enumerate(base2_te_pred):
    cl_start_pt_test[i] = base2[cl_start_val]

class_resid_test = class_resid_nn.predict(
    np.concatenate((dog.col_subset_test, cl_start_pt_test), axis=-1)
)

# class_resid_losses = poseLossResidualJAV(
#     np.concatenate((bcotjav.jav_test, cl_start_pt_test), axis=-1), class_resid_test
# )


class_resid_losses = poseLossJAV(
    bcotjav.jav_test, class_resid_test
)

for k, v in dog.subset_skip_inds[DataSubsetKind.TEST].items():
    print(k, np.mean(class_resid_losses[v]))
