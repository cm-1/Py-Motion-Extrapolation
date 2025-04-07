#%% Imports
import typing
import copy

import numpy as np
from numpy.typing import NDArray

from sklearn.model_selection import train_test_split

# Local code imports ===========================================================
# For reading the dataset into numpy arrays:
from gtCommon import BCOT_Data_Calculator
import gtCommon as gtc

# Stuff needed for calculating the input features for the non-RNN models.
# MOTION_DATA_KEY_TYPE is a class representing input feature column "names",
# while the MOTION_DATA enum is a "subset" of these.
from posefeatures import MOTION_DATA, MOTION_DATA_KEY_TYPE, CalcsForCombo

# Some consts used in calculating the input features.
OBJ_IS_STATIC_THRESH_MM = 10.0 # 10 millimeters; semi-arbitrary
STRAIGHT_LINE_ANG_THRESH_DEG = 30.0 # 30deg as arbitrary max "straight" angle.
CIRC_ERR_RADIUS_RATIO_THRESH = 0.10 # Threshold for if motion's circular.
MAX_MIN_JERK_OPT_ITERS = 0 # Max iters for min jerk optimization calcs.
MAX_SPLIT_MIN_JERK_OPT_ITERS = 0
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
# The above data has the following type: 
#     List[Dict[Combo, Dict[MOTION_DATA, NDArray]]]
# That is, we have a list of dictionaries which store the results per combo,
# where said "result" is another dict with "column" name enums as keys.
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
# ---
# I have other non-regression code that sometimes passes in a list of NDArrays
# instead of a list of dicts, hence the isinstabce() check.
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
test_data = concatForComboSubset(all_motion_data, test_combos)

# Get 2D NDArrays from the above.

# Get the "keys" as another return value so that we know the column names/order.
motion_data_keys, concat_train_data = get2DArrayFromDataStruct(train_data, stack_axis=-1)
_, concat_test_data = get2DArrayFromDataStruct(test_data, motion_data_keys, stack_axis=-1)

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

# Find the "lower bound" for error when we train a classifier to use the physics
# models for prediction, by finding pose MAE for perfect classification.
# error_lim = assessPredError(concat_test_errs, concat_test_labels, s_ind_dict)
# print("Classification MAE limit:", error_lim)



#%%

################################################################################
# Vanilla Regression Network Code!
################################################################################

import tensorflow as tf
import keras
import posemath as pm # Small "library" I wrote for vector operations.
from sklearn.preprocessing import StandardScaler
from enum import Enum

# We will start off the neural net code by constructing a regression network
# that predicts multipliers for velocity, acceleration, and jerk that we will
# use to construct the displacement from the current position to the position
# we predict for the next timestamp.

class JAV(Enum):
    VELOCITY = 1
    ACCELERATION = 2
    JERK = 3

ComboList = typing.List[gtc.Combo]
PerComboJAV = typing.List[typing.Dict[gtc.Combo, typing.Dict[str, NDArray]]]
OrderForJAV = typing.Tuple[JAV, JAV, JAV]

# For each frame of video, we consider a coordinate frame where one axis is
# aligned with the object's velocity and another is aligned with the
# acceleration (or, at least, the part of it orthogonal to velocity).
# We then calculate and return the speed, acceleration, and jerk for the current
# time and the position at the next time in this frame.
# Returns a List[Dict[Combo, Dict[str, NDArray]]] that again separates things
# by frame skip amount and by combo.
def dataForCombosJAV(combos: ComboList, return_world2locals: bool = False, 
                     return_translations: bool = False, 
                     vec_order: typing.Optional[OrderForJAV] = None):
    # Empty dict for each skip amount.
    all_data: PerComboJAV = [dict() for _ in range(3)]
    all_world2local_mats: typing.List[typing.Dict[gtc.Combo, NDArray]] = \
        [dict() for _ in range(3)]
    all_translations: typing.List[typing.Dict[gtc.Combo, NDArray]] = \
        [dict() for _ in range(3)]
    
    if len(vec_order) != 3 or {v.value for v in vec_order} != {1, 2, 3}:
        raise ValueError("Vector order must be a permutation of (velocity, acceleration, jerk)!")
    
    skip_end = 3#1 if onlySkip0 else 3
    for c in combos:
        calc_obj = BCOT_Data_Calculator(c.body_ind, c.seq_ind, 0)
        curr_translations = calc_obj.getTranslationsGTNP(False)
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

            unit_vecs_o1 = pm.safelyNormalizeArray(
                vecs_o1, mags_o1[:, np.newaxis]
            )

            unit_vecs2 = np.cross(unit_vecs0, unit_vecs_o1)
            # We now have matrices to convert vectors in world space into
            # these local vector-aligned frames.
            mats = np.stack([unit_vecs0, unit_vecs_o1, unit_vecs2], axis=1)

            # Transform each third vector and to-next-frame displacement into
            # this frame via matmul.
            local_vecs2 = pm.einsumMatVecMul(mats, ordered[2])
            local_diffs = pm.einsumMatVecMul(mats, vels[3:])

            # We'll now return all of the data needed to convert velocity,
            # acceleration, and jerk multipliers into local vectors in these
            # new frames. To do this, we don't need to return the coordinate
            # frames themselves: we just need to know the velocity in this
            # frame (a vector [speed, 0, 0]), the acceleration in this frame
            # (i.e. [a_p, a_o, 0]), etc. And since we don't need to return 0s,
            # we can just return the following:
            c_res = (
                mags0, mags_p1, mags_o1, *(local_vecs2.T), *(local_diffs.T)
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

# Function that combines the results of the previous one into a 2D numpy
# array.
def dataForComboSplitJAV(train_combos: ComboList, test_combos: ComboList, *,
                         combos: typing.Optional[ComboList] = None, 
                         precalc_per_combo: typing.Optional[PerComboJAV] = None):   
    if combos is None and precalc_per_combo is None:
        raise ValueError(
            "Cannot have combos and precalc_per_combo both be None!"
        )
    elif combos is not None and precalc_per_combo is not None:
        raise ValueError(
            "Cannot provide values for both  combos and precalc_per_combo!"
        )
     
    all_data = precalc_per_combo
    if precalc_per_combo is None:
        all_data = dataForCombosJAV(combos)
    

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

last_best_ind = motion_data_keys.index(MOTION_DATA.LAST_BEST_LABEL)
timestamp_ind = motion_data_keys.index(MOTION_DATA.TIMESTAMP)

nonco_cols[last_best_ind] = False # Needs one-hot encoding or similar.
nonco_cols[timestamp_ind] = False # Current frame number seems... unhelpful.

# select_cols = np.where(nonco_cols)[0][[0, 1, 2, 3, 13, 26, 27]]
# nonco_cols[:] = False
# nonco_cols[list(select_cols)] = True

# Get the data subset for the selected non-collinear columns.
nonco_train_data = concat_train_data[:, nonco_cols]
nonco_test_data = concat_test_data[:, nonco_cols]

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
bcs_per_combo, w2ls_JAV, translations_JAV = dataForCombosJAV(
    nametup_combos, True, True, (JAV.JERK, JAV.ACCELERATION, JAV.VELOCITY)
)

bcs_train, bcs_test = dataForComboSplitJAV(
    train_combos, test_combos, precalc_per_combo=bcs_per_combo
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
    



################################################################################
# World-Frame LSTM
################################################################################

# We'll use a MinMax scaler to "normalize" the data, as per a tutorial.
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
#%%
win_size = 4

# For type hint. Seems that sklearn's "Scalers" don't have a single superclass
# that has a transform() method, unless I'm missing something. So I'm doing a
# type union instead.
ScalerType = typing.Union[
    MinMaxScaler, sklearn.preprocessing.StandardScaler, 
    sklearn.preprocessing.RobustScaler, sklearn.preprocessing.MaxAbsScaler,
    sklearn.preprocessing.PowerTransformer, sklearn.preprocessing.Normalizer,
    sklearn.preprocessing.QuantileTransformer
]

# Function that gets the "windows" of consecutive poses from the data for a 
# subset of combos. 
def rnnDataWindows(data: NDArray, combo_subset: typing.List[typing.Tuple], 
                   window_size: int, scaler: ScalerType, skip: int):
    step = skip + 1
    data_in = []
    data_out = []
    for combo in combo_subset:
        combo_data = scaler.transform(data[combo[:2]][::step])
        for i in range(len(combo_data) - window_size):
            data_in.append(combo_data[i:(i + window_size)])
            data_out.append(combo_data[i + window_size])
    data_in_np = np.array(data_in)
    data_out_np = np.array(data_out)
    return (data_in_np, data_out_np)

# def rnnErrsForCombos(combo_subset, window_size, skip):
#     skip_subset = err_norm_lists[skip]
#     errs_all = []
#     for combo in combo_subset:
#         errs_dict = skip_subset[combo[:2]]
#         cut_amt = window_size - 4 # jerk preds already cut
#         assert cut_amt >= 0
#         errs_stack = np.stack([
#             errs_dict[k][cut_amt:] for k in motion_mod_keys
#         ], axis=-1)
#         errs_all.append(errs_stack)
#     return np.concatenate(errs_all, axis=0)

# lstm_train_errs = rnnErrsForCombos(train_combos, win_size, 2)


#%% Read all the translation data in and get the scaler for normalization.
all_translations: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
# all_rotations: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
# all_rotation_mats: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
for combo in combos:
    calculator = BCOT_Data_Calculator(combo[0], combo[1], 0)
    all_translations[combo[:2]] = calculator.getTranslationsGTNP(False)
    # aa_rotations = calculator.getRotationsGTNP(False)
    # quats = pm.quatsFromAxisAngleVec3s(aa_rotations)
    # all_rotations[combo[:2]] = quats
    # all_rotation_mats[combo[:2]] = calculator.getRotationMatsGTNP(False)

all_translations_concat = np.concatenate(
    [all_translations[c[:2]] for c in combos], axis=0
)
translation_scaler = MinMaxScaler(feature_range=(0,1))
translation_scaler.fit(all_translations_concat)

#%% Get the data windows for the RNN training.
lstm_skip = 2

train_translations_in, train_translations_out = rnnDataWindows(
    all_translations, train_combos, win_size, translation_scaler, lstm_skip
)

#%%
lstm_model = keras.Sequential([
    keras.layers.LSTM(50, return_sequences=True),
    keras.layers.LSTM(50),
    keras.layers.Dense(3)#num_classes, activation='sigmoid')
])

lstm_model.summary()

lstm_model.compile(optimizer='adam', loss='mse')#tf_loss_fn)
#%%
lstm_model.fit(train_translations_in, train_translations_out, epochs=32)


#%%
test_translations_in, test_translations_out = rnnDataWindows(
    all_translations, test_combos, win_size, translation_scaler, lstm_skip
)

lstm_test_pred = lstm_model.predict(test_translations_in)

# lstm_test_pred_labs = np.argmax(lstm_test_pred, axis=1)
# lstm_test_errs = rnnErrsForCombos(test_combos, win_size, 2)
# lstm_test_err_dists = assessPredError(lstm_test_errs, lstm_test_pred_labs)
# print(lstm_test_err_dists)

# Transform the coordinates back into non-normalized form to get the errors
# in millimeters.
unscaled_lstm_test_pred = translation_scaler.inverse_transform(lstm_test_pred)
unscaled_lstm_test_gt = translation_scaler.inverse_transform(test_translations_out)

lstm_test_errs = np.linalg.norm(
    unscaled_lstm_test_pred - unscaled_lstm_test_gt, axis=-1
)
print("LSTM score (mm):", lstm_test_errs.mean())

#%% 

# To show a weakness of trying to use an LSTM to predict coordinates in 
# "world space", we will imagine all input coordinates were shifted by a 
# constant vec3 and see that accuracy degrades.
shift_in = 1.0 - test_translations_in #+ 0.015 * np.arange(1,4)
shift_out = 1.0 - test_translations_out #+ 0.015 * np.arange(1,4)
shift_pred = lstm_model.predict(shift_in)

scaled_shift_pred = translation_scaler.inverse_transform(shift_pred)
scaled_shift_gt = translation_scaler.inverse_transform(shift_out)
shift_lstm_errs = np.linalg.norm(scaled_shift_pred - scaled_shift_gt, axis=-1)
print("LSTM err when all points translated by same amount:", shift_lstm_errs.mean())

#%%

################################################################################
# Velocity-Aligned-Frame LSTM
################################################################################

# As with the "vanilla" regression network, we'll get our data in a velocity
# -aligned frame. But this time, we'll start by keeping things separated by 
# combo so that we know the starting and ending points for a video's data and,
# thus, can create our data windows without having a window erroneously overlap
# two separate videos.
all_jav = dataForCombosJAV(nametup_combos)

# While we'll keep data separated by skip amount and by combo, we'll combine our
# per-combo columns, currently split up as separate dict entries, into single 
# numpy arrays.
all_jav = [{
    _combo: np.stack([col_data for _, col_data in _combo_data.items()], axis=-1) 
    for _combo, _combo_data in aj.items()
} for aj in all_jav]

jav_scalers = [MinMaxScaler(feature_range=(0,1)) for _ in range(3)]
jav_scalers_3 = [MinMaxScaler(feature_range=(0,1)) for _ in range(3)]

for skip in range(3):
    # We'll *temporarily* combine the data for all combos into a single numpy
    # array, but that will just be to fit the Scaler used to normalize the data.
    all_jav_concat = np.concatenate(
        [all_jav[skip][c[:2]] for c in combos], axis=0
    )
    # Scaler for all data columns.
    jav_scalers[skip].fit(all_jav_concat)
    # Scale all 3 axes by uniform amount and centre to 0 for later simplicity
    jav_scalers[skip].scale_[-3:] = jav_scalers[skip].scale_[-1]
    jav_scalers[skip].min_[-3:] = 0.0

    # Scaler for just the last three columns, the "displacement" ones.
    jav_scalers_3[skip].fit(all_jav_concat[:, -3:])
    # Ensure this scaler matches the other one.
    jav_scalers_3[skip].scale_[-3:] = jav_scalers[skip].scale_[-1]
    jav_scalers_3[skip].min_[-3:] = 0.0
    # We no longer need all_jav_concat, and it might take up a fair bit of RAM.
    del all_jav_concat

#%% Get data windows

jav_lstm_skip = 2

train_jav_in, train_jav_out = rnnDataWindows(
    all_jav[jav_lstm_skip], train_combos, win_size, jav_scalers[jav_lstm_skip], 
    0 # Always a skip of 0 in this case because the frame is rotating...
)

#%%
jav_lstm_model = keras.Sequential([
    # keras.layers.LSTM(128, return_sequences=True),
    keras.layers.LSTM(7),
    keras.layers.Dense(3) #num_classes, activation='sigmoid')
])

jav_lstm_model.summary()

jav_lstm_model.compile(optimizer='adam', loss='mse')#tf_loss_fn)
#%%
jav_lstm_model.fit(train_jav_in[..., -3:], train_jav_out[:, -3:], epochs=32)

#%% Test on test data.
test_jav_in, test_jav_out = rnnDataWindows(
    all_jav[jav_lstm_skip], test_combos, win_size, jav_scalers[jav_lstm_skip], 
    0 # Always a skip of 0 in this case because the frame is rotating...

)

jav_lstm_test_pred = jav_lstm_model.predict(test_jav_in[..., -3:])

def getErrsLSTM(scaler, unscaled_preds, scaled_gt):
    scaled_preds = scaler.inverse_transform(unscaled_preds)
    return np.linalg.norm(scaled_preds - scaled_gt, axis=-1)

# Transform the coordinates back into non-normalized form to get the errors
# in millimeters.
jav_scaled_lstm_gt_test = jav_scalers_3[jav_lstm_skip].inverse_transform(
    test_jav_out[:, -3:]
)

jav_scaled_lstm_gt_train = jav_scalers_3[jav_lstm_skip].inverse_transform(
    train_jav_out[:, -3:]
)
jav_lstm_test_errs = getErrsLSTM(
    jav_scalers_3[jav_lstm_skip], jav_lstm_test_pred, jav_scaled_lstm_gt_test
)
print("JAV LSTM score:", jav_lstm_test_errs.mean())

#%%
from sklearn.linear_model import LinearRegression

jav_train_flatter = train_jav_in[:, -1, -3:].reshape(train_jav_in.shape[0], 3)
jav_test_flatter = test_jav_in[:, -1, -3:].reshape(test_jav_in.shape[0], 3)
jav_lin_reg = LinearRegression()#fit_intercept=False)
jav_lin_reg.fit(jav_train_flatter, train_jav_out[:, -3:])

#%%
jav_lin_test_pred = jav_lin_reg.predict(jav_test_flatter)

jav_lin_test_errs = getErrsLSTM(
    jav_scalers_3[jav_lstm_skip], jav_lin_test_pred, jav_scaled_lstm_gt_test
)

print("JAV lin score:", jav_lin_test_errs.mean())

#%%

print("Linear JAV coefs:")
print(np.round(jav_lin_reg.coef_, 2))
print("Linear JAV intercept:")

print(np.round(jav_lin_reg.intercept_, 2))


#%%

vel_preds = np.zeros_like(jav_test_flatter)
vel_preds[:, 0] = np.linalg.norm(jav_test_flatter, axis=-1)


vel_scores = getErrsLSTM(
    jav_scalers_3[jav_lstm_skip], vel_preds, jav_scaled_lstm_gt_test
)

print(vel_scores.mean())


#%%

################################################################################
# Comparing Error Histograms
################################################################################

import matplotlib.pyplot as plt

bar_skip_key = 'skip2'
bar_data = [
    bcs_test_errs[s_ind_dict[bar_skip_key]], lstm_test_errs, jav_lstm_test_errs
]
bar_labels = ['JAV NN', 'Global LSTM', 'JAV LSTM']
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
