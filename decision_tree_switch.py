#%% Imports
import typing
import copy
import time
from collections import defaultdict

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
from motiontools.posefeatures import MOTION_DATA, MOTION_MODEL, JAV # Enums
# Classes, functions, and type hints:
from motiontools.posefeatures import CalcsForVideo, dataForCombosJAV
from motiontools.posefeatures import PoseLoaderList, NumpyForSkipAndID, OrderForJAV

from motiontools.dataorg import DataOrganizer, concatForComboSubset

# Some consts used in calculating the input features.
OBJ_IS_STATIC_THRESH_MM = 10.0 # 10 millimeters; semi-arbitrary
STRAIGHT_LINE_ANG_THRESH_DEG = 30.0 # 30deg as arbitrary max "straight" angle.
CIRC_ERR_RADIUS_RATIO_THRESH = 0.10 # Threshold for if motion's circular.
MAX_MIN_JERK_OPT_ITERS = 33 # Max iters for min jerk optimization calcs.
MAX_SPLIT_MIN_JERK_OPT_ITERS = 33
ERR_NA_VAL = np.finfo(np.float32).max # A non-inf but inf-like value.

#%%
combos = PoseLoaderBCOT.getAllIDs()

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
    err_radius_ratio_thresh=CIRC_ERR_RADIUS_RATIO_THRESH
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
bcot_train_combos, bcot_test_combos = PoseLoaderBCOT.trainTestByBody(
    test_ratio=0.2, random_seed=0
)
train_combos_c2 = [c[:2] for c in bcot_train_combos]
test_combos_c2 = [c[:2] for c in bcot_test_combos] # Get the unique part of each.

dog = DataOrganizer(
    all_motion_data, min_norm_labels, err_norm_lists,
    train_combos_c2, test_combos_c2
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
print("TODO: Add single 'limit' legend entry!")

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
def dataForComboSplitJAV(train_combos: typing.List, test_combos: typing.List, *,
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
    return train_res, test_res

# Get the pose loss for a set of Jerk, Acceleration, & Velocity multipliers.
def poseLossJAV(y_true, y_pred):
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

    y_pred2 = y_pred + y_true[:, 9:15]

    pred_disp_0 = y_true[:, 0] * y_pred2[:, 0] + y_true[:, 1] * y_pred2[:, 1] \
        + y_true[:, 3] * y_pred2[:, 3]
    pred_disp_1 = y_true[:, 2] * y_pred2[:, 2] + y_true[:, 4] * y_pred2[:, 4]
    pred_disp_2 = y_true[:, 5] * y_pred2[:, 5]

    pred_disp = tf.stack([pred_disp_0, pred_disp_1, pred_disp_2], axis=-1)
    # pred_disp = tf.gather(y_true, (0,2,5), axis=-1) * y_pred

    true_disp = y_true[:, 6:9]

    err_vec3 = true_disp - pred_disp
    return tf.norm(err_vec3, axis=-1)

def getWorldFrameDisplacements(y_true, y_pred, world2locals):
    disp = np.empty((len(y_true), 3))

    y_pred2 = y_pred + y_true[:, 9:15]
    # Calculating the local displacement is the same as custom tf loss function.
    disp[:, 0] = y_true[:, 0] * y_pred2[:, 0] + y_true[:, 1] * y_pred2[:, 1] \
        + y_true[:, 3] * y_pred2[:, 2]
    disp[:, 1] = y_true[:, 2] * y_pred2[:, 1] + y_true[:, 4] * y_pred2[:, 2]
    disp[:, 2] = y_true[:, 5] * y_pred2[:, 2]

    # Convert local displacement into world displacement.
    local2worlds = np.swapaxes(world2locals, -1, -2)
    return pm.einsumMatVecMul(local2worlds, disp) 

#%%
# A lot of the features we calculated might be collinear (especially since a lot 
# of very similar features were tried for the decision tree) we'll remove the
# collinear ones before training.
colin_thresh = 0.7 # Threshold for collinearity.

nonco_cols, co_mat = pm.non_collinear_features(
    dog.concat_train_data, colin_thresh
)

last_best_ind = dog.motion_data_keys.index(MOTION_DATA.LAST_BEST_LABEL)
timestamp_ind = dog.motion_data_keys.index(MOTION_DATA.TIMESTAMP)
framenum_ind = dog.motion_data_keys.index(MOTION_DATA.FRAME_NUM)

# nonco_cols[:] = False
nonco_cols[last_best_ind] = False # Needs one-hot encoding or similar.
nonco_cols[timestamp_ind] = False # Current frame number seems... unhelpful.
nonco_cols[framenum_ind] = False

AVD2_KEY = SpecifiedMotionData(
    MOTION_DATA.ACC_VEC3, RELATIVE_AXIS.VEL_DEG1, ANG_OR_MAG.ANG, False, True
)
AVD2_DATA_IND = dog.motion_data_keys.index(AVD2_KEY)


col_sub_keys = [
    AVD2_KEY, MOTION_DATA.BOUNCE_ANGLE, MOTION_DATA.VEL_BCS_RATIOS,
    MOTION_DATA.CIRC_ACC, MOTION_DATA.DISP_MAG_DIFF, MOTION_DATA.TIMESTEP,
    MOTION_DATA.DISP_MAG_RATIO
]
col_indices = [dog.motion_data_keys.index(k) for k in col_sub_keys]
nonco_cols[col_indices] = True

nonco_col_nums = np.where(nonco_cols)[0]

# select_cols = np.where(nonco_cols)[0][[0, 1, 2, 3, 13, 26, 27]]
# nonco_cols[:] = False
# nonco_cols[list(select_cols)] = True


# %%


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

def getUntrainedNN():

    dropout_rate = 0.2
    nodes_per_layer = 128
    vel_nn_activation = 'sigmoid' # Works better than relu for this NN.
    model = keras.Sequential([
        keras.layers.Input((len(nonco_col_nums),)),
        # ImportanceLayer(nonco_train_data.shape[1]),  # Custom importance layer
        keras.layers.Dense(nodes_per_layer, activation=vel_nn_activation),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(nodes_per_layer, activation=vel_nn_activation),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(nodes_per_layer, activation=vel_nn_activation),
        keras.layers.Dense(6)
    ])

    model.summary()
    model.compile(loss=poseLossJAV, optimizer='adam')
    return model
bcs_model = getUntrainedNN()
# %%
JAV_order = (JAV.JERK, JAV.ACCELERATION, JAV.VELOCITY)[::-1]

# # Convert from numpy array to tf tensor.
# bcs_train = tf.convert_to_tensor(bcs_train, dtype=tf.float32)
# bcs_test = tf.convert_to_tensor(bcs_test, dtype=tf.float32)

# Z-scale each column to standard normal distribution.
bcs_scaler = StandardScaler()

from motiontools.posefeatures import SpecifiedMotionData, RELATIVE_AXIS, ANG_OR_MAG

class DataForJAV:
    def __init__(self, data_organizer: DataOrganizer, loaders, bcs_scaler, 
                 col_inds: NDArray, JAV_order: OrderForJAV,
                 save_data_for_conf: bool = False):

        self.data_organizer = data_organizer
        if len(self.data_organizer.col_subset_test) <= 0:
            self.data_organizer.setPickAndTransform(col_inds, bcs_scaler)

        self.save_data_for_conf = save_data_for_conf
        self.jav_per_combo = None
        if save_data_for_conf:
            res = dataForCombosJAV(loaders, JAV_order, True, True)
            self.jav_per_combo = res[0]
            self.w2ls_JAV = res[1]
            self.translations_JAV = res[2]
        else:
            self.jav_per_combo = dataForCombosJAV(loaders, JAV_order, False, False)
            self.w2ls_JAV = None
            self.translations_JAV = None

        # TODO: Currently we calculate both train and test while only printing one.
        # Need a small refactor to fix this.
        partial_jav_train, partial_jav_test = dataForComboSplitJAV(
            data_organizer.train_ids, data_organizer.test_ids,
            precalc_per_combo=self.jav_per_combo
        )

        jav_tr_append = self.sineThing(True)
        jav_te_append = self.sineThing(False)

        self.jav_train = np.concatenate(
            (partial_jav_train, jav_tr_append), axis=-1
        )
        self.jav_test = np.concatenate(
            (partial_jav_test, jav_te_append), axis=-1
        )


    def sineThing(self, is_train: bool):
        
        data = self.data_organizer.concat_train_data if is_train else self.data_organizer.concat_test_data
        res = np.zeros((len(data), 6))
        res[:, 0] = 1
        avd1 = data[:, AVD2_DATA_IND]
        res[:, 1] = 0.5 * np.sin(2 * avd1)
        res[:, 2] = 0.5 * (1.0 - np.cos(2 * avd1))
        return res

        
    def printScoresTrain(self, predictions: typing.Optional[NDArray] = None):
        if predictions is None:
            predictions = bcs_model.predict(self.data_organizer.col_subset_train)
        self._printHelper(
            predictions, self.jav_train, self.data_organizer.skip_train_inds_dict
        )

    def printScoresTest(self, predictions: typing.Optional[NDArray] = None):
        if predictions is None:
            predictions = bcs_model.predict(self.data_organizer.col_subset_test)
        self._printHelper(
            predictions, self.jav_test, self.data_organizer.skip_inds_dict
        )

    @staticmethod
    def _printHelper(preds: NDArray, jav_vals: NDArray, inds: typing.Dict):
        errs: NDArray = poseLossJAV(jav_vals, preds).numpy()
        # Print scores on test data.
        scores = {k: np.mean(errs[v]) for k, v in inds.items()}
        for k, v in scores.items():
            print(k + ":", v)

bcotjav = DataForJAV(dog, bcot_loaders, bcs_scaler, nonco_cols, JAV_order, True)


#%% Train the network.
bcs_hist = bcs_model.fit(dog.col_subset_train, bcotjav.jav_train, epochs=32, shuffle=True)

#%% Evaluate network on test data.

bcs_pred = bcs_model.predict(dog.col_subset_test)
bcs_test_errs: NDArray = poseLossJAV(bcotjav.jav_test, bcs_pred).numpy()
#%% Print scores on test data.
bcs_test_scores = {k: np.mean(bcs_test_errs[v]) for k, v in dog.skip_inds_dict.items()}
for k, v in bcs_test_scores.items():
    print(k + ":", v)



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

# Get the column names for each of the kept columns.
nonco_featnames = [
    k.name for i, k in enumerate(dog.motion_data_keys) if nonco_cols[i]
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
    preds = model.predict(data_scramble, batch_size=1024, verbose=0)
    errs = model.loss(y_true, preds)
    return errs

# For each column and for each skip amount, scramble the column and find the new
# score.
skip_keys = dog.skip_inds_dict.keys()
scramble_scores = np.empty((dog.col_subset_test.shape[1], len(skip_keys)))
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
    for i, k in enumerate(skip_keys):
        scramble_scores[col_ind, i] = np.mean(errs[dog.skip_inds_dict[k]])
#%%
scramble_rank = np.argsort(scramble_scores, axis=0)[::-1]
print("Most important feature inds:", scramble_rank[:10], sep='\n')
    



#%%
import shap
default_rng = np.random.default_rng()
rng_choice = default_rng.choice(dog.col_subset_test, 100, False, axis=0)
shap_ex = shap.DeepExplainer(bcs_model, rng_choice)

#%%
rng_single = default_rng.choice(rng_choice, 1, axis=0)
shap_values = shap_ex.shap_values(rng_single)#, samples=500)
shap.initjs()

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

input_dim = dog.concat_train_data.shape[1]
num_classes = len(MOTION_MODEL)

onehot_train_labels = tf.one_hot(dog.concat_train_labels, num_classes)

tfmodel = keras.Sequential([
    keras.layers.Input((input_dim,)),
    keras.layers.Dense(512, activation='sigmoid'),
    # keras.layers.Dropout(0.2),
    keras.layers.Dense(512, activation='sigmoid'),
    keras.layers.Dense(512, activation='sigmoid'),
    # keras.layers.Dense(1024, activation='sigmoid'),
    # keras.layers.Dropout(0.2),
    # keras.layers.Dense(1, activation='sigmoid'),
    keras.layers.Dense(num_classes, activation='sigmoid')])

tfmodel.summary()

tf_loss_fn2 = keras.losses.CategoricalCrossentropy(from_logits=True)
adam = keras.optimizers.Adam(0.01)

tfmodel.compile(optimizer=adam, loss=tf_loss_fn)
tfmodel.fit(
    dog.concat_train_data, dog.concat_train_class_errs, epochs=5, shuffle=True
)
#%%
bgtrain = big_tree.predict(dog.concat_train_data)
bgtrain_score = dog.getClassScoresTrain(bgtrain)
print("Big tree train errs=", bgtrain_score)

mclf_train = mclf.predict(dog.concat_train_data)
mclf_train_score = dog.getClassScoresTrain(mclf_train)
print("Smaller tree train errs=", mclf_train_score)

tf_train_preds = np.argmax(tfmodel(dog.concat_train_data).numpy(), axis=1)
tf_train_errs = dog.getClassScoresTrain(tf_train_preds)
print("TF train errs=", tf_train_errs)
print()

tf_test_preds = np.argmax(tfmodel(dog.concat_test_data).numpy(), axis=1)
tf_test_errs = dog.getClassScoresTest(tf_test_preds)
print("TF test errs=", tf_test_errs)


#%%
################################################################################
# Comparing Error Histograms
################################################################################
import matplotlib.pyplot as plt

big_tree_preds = big_tree.predict(dog.concat_test_data)
tree_errs_for_plt = dog.getClassErrsTest(big_tree_preds)

acc_only_preds = np.full_like(
    big_tree_preds, dog.motion_mod_keys.index(MOTION_MODEL.ACC_DEG2)
)
acc_errs_for_plt = dog.getClassErrsTest(acc_only_preds)


bar_skip_key = 'skip2'
bar_data: typing.List[NDArray] = [
    tree_errs_for_plt[bar_skip_key], acc_errs_for_plt[bar_skip_key],
    bcs_test_errs[dog.skip_inds_dict[bar_skip_key]], error_lim_all[bar_skip_key]
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

tudl_ids = PoseLoaderTUDL.getAllIDs()
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
tjav.printScoresTest(tjavps)
#%%
ajnn = getUntrainedNN()

bt_train = np.concatenate((dog.col_subset_test, tdog.col_subset_test), axis=0)
bt_jav = np.concatenate((bcotjav.jav_test, tjav.jav_test), axis=0)

#%%
ajnn.fit(bt_train, bt_jav, epochs=32, shuffle=True)

#%%
bcot_ajnn_pred = ajnn.predict(dog.col_subset_train, batch_size=1024)

tudl_ajnn_pred = ajnn.predict(tdog.col_subset_train, batch_size=1024)

bcotjav.printScoresTrain(bcot_ajnn_pred)
tjav.printScoresTrain(tudl_ajnn_pred)
#%%
ax = plt.figure().add_subplot(projection='3d')

data_to_3d_plot = bcot_ajnn_pred[:, :3]

inds_dict = dog.skip_train_inds_dict

for k in skip_keys:
    if k == 'all':
        continue
    dp = data_to_3d_plot[inds_dict[k]]
    inds = np.random.choice(len(dp), 1000)

    # print(dp[inds].T.shape)
    ax.scatter3D(*dp[inds].T, label="BCOT " + k)

data_to_3d_plot = tudl_ajnn_pred[:, :3]
inds_dict = tdog.skip_train_inds_dict

for k in skip_keys:
    if k == 'all':
        continue
    dp = data_to_3d_plot[inds_dict[k]]
    inds = np.random.choice(len(dp), 1000)

    # print(dp[inds].T.shape)
    ax.scatter3D(*dp[inds].T, label="TUDL " + k)

ax.set_xlabel("i")
ax.set_ylabel("j")
ax.set_zlabel("k")
ax.legend()
plt.show()

# 

#%%

from keras import layers ########################################jljlklijlij
def getUntrainedNNC():
    dropout_rate = 0.2
    nodes_per_layer = 128
    vel_nn_activation = 'sigmoid'

    input_layer = keras.Input(shape=(len(nonco_col_nums),))

    def build_branch():
        x = layers.Dense(nodes_per_layer, activation=vel_nn_activation)(input_layer)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(nodes_per_layer, activation=vel_nn_activation)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(nodes_per_layer, activation=vel_nn_activation)(x)
        x = layers.Dense(1)(x)  # Output a single float
        return x

    # Build 3 branches
    branch_outputs = [build_branch() for _ in range(3)]

    # Concatenate the three outputs
    output = layers.Concatenate()(branch_outputs)

    model = keras.Model(inputs=input_layer, outputs=output)
    model.summary()
    model.compile(loss=poseLossJAV, optimizer='adam')
    return model

ajnnC = getUntrainedNNC()

ajnnC.fit(bt_train, bt_jav, epochs=32, shuffle=True)

#%%
ajnnC_pred = ajnnC.predict(tdog.col_subset_train)
tjav.printScoresTrain(ajnnC_pred)

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
        x = layers.Dense(nodes_per_layer, activation=vel_nn_activation)(inp)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(nodes_per_layer, activation=vel_nn_activation)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(nodes_per_layer, activation=vel_nn_activation)(x)
        x = layers.Dense(1)(x)
        return x

    out1 = build_branch(input1)
    out2 = build_branch(input2)
    out3 = build_branch(input3)

    output = layers.Concatenate()([out1, out2, out3])

    model = keras.Model(inputs=[input1, input2, input3], outputs=output)
    model.compile(loss=poseLossJAV, optimizer='adam')
    model.summary()
    return model

lnccn = len(nonco_col_nums)
split_nn = getUntrainedNNSplit(lnccn, lnccn - 1, lnccn - 2)
split_nn.fit([dog.col_subset_test, dog.col_subset_test[:, :-1], dog.col_subset_test[:, :-2]], bcotjav.jav_test, epochs=33, batch_size=128, shuffle=True)

#%%
tjav.printScoresTest(split_nn.predict([tdog.col_subset_test, tdog.col_subset_test[:, :-1], tdog.col_subset_test[:, :-2]]))

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
        return pm.einsumMatVecMul(inv_mats, y_true[:, 6:9])
    
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
            muls_to_pt_mats[i], y_true[i, 6:9], bounds, tol=tol, 
            max_iter=max_iter
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

bcotjav.printScoresTrain(gt_bcot)
tjav.printScoresTrain(gt_tudl)

#%%
def gtMultipliers6(y_true, bounds, tol: float = 0.001, max_iter: int = 32, verbose=False):
    '''
    [[speed, acc_x, jerk_x],       [vel_multiplier,
     [0,     acc_y, jerk_y],     x  acc_multiplier,
     [0,     0,     jerk_z]]        jerk_multiplier]
    '''
    muls_to_pt_mats = np.zeros((len(y_true), 3, 3))#6))
    muls_to_pt_mats[:, 0, 0] = y_true[:, 0]
    muls_to_pt_mats[:, 1, 1:3] = y_true[:, 1:3]
    # muls_to_pt_mats[:, 2, 3:] = y_true[:, 3:6]
    
    n = len(y_true)
    res = np.empty((n, 3))#6))
    n_digs = 0
    status_template = "Done iter {}/" + str(n) + " ({:0.2f})."
    if verbose:
        n_digs = int(np.ceil(np.log10(n)))
        status_template = status_template + (" " * n_digs)
        print(status_template.format(0, 0), end='')

    for i in range(n):
        lsq_res = lsq_linear(
            muls_to_pt_mats[i], y_true[i, 6:9], bounds, tol=tol, 
            max_iter=max_iter
        )
        res[i] = lsq_res.x
        if (verbose and (i + 1) % 1000 == 0) or i == (n - 1):
            print("\r" + status_template.format((i + 1), (i + 1)/n), end = '')
    if verbose:
        print()
    return res

bounds = (-2,2)
gt_bcot = gtMultipliers6(bcotjav.jav_train, bounds, verbose=True)
gt_tudl = gtMultipliers6(tjav.jav_train, bounds, verbose=True)

bcotjav.printScoresTrain(gt_bcot)
tjav.printScoresTrain(gt_tudl)

#%%
bcs_pred_tr = bcs_model.predict(dog.col_subset_train, batch_size=1024)

#%%

current_idx = 0

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

def getRandSubset(data, gt, n):
    inds = np.random.choice(len(data), n)
    return data[inds], gt[inds]

skis = [dog.skip_train_inds_dict['skip{}'.format(i)] for i in range(3)]
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

sb = dog.skip_train_inds_dict['skip2']
def plot_column(idx):
    fig.suptitle(f'(BCOT) Column: {feature_names[idx]} (#{idx})')
    for i in range(3):
        axs[i].cla()  # clear the axes
        for j in range(3):
            axs[i].scatter(bcot_sub_data[j][:, idx], bcot_sub_gt[j][:, i], alpha=0.6, label="skip" + str(j))
        # axs[i].scatter(p_sub_data[:, idx], p_sub_gt[:, i], alpha=0.6, label="Pauwels")
        # for j in range(3):
        #     axs[i].scatter(bcot_sub_data[j][:, idx], bcot_sub_nn[j][:, i], alpha=0.6, label="NN" + str(j))
        axs[i].set_xlabel(gtc.truncateName(feature_names[idx], 33))
        
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

if JAV_order[0] != JAV.VELOCITY or JAV_order[1] != JAV.ACCELERATION:
    raise Exception("Below hardcoded indices will be wrong!")

    
st = bcotjav.sineThing(True)
bcotjav.printScoresTrain(st)
