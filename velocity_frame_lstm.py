################################################################################
# Velocity-Aligned-Frame Networks
################################################################################
import typing
import enum

import numpy as np
from numpy.typing import NDArray

# StandardScaler seems to perform better than others, like MinMax.
from sklearn.preprocessing import StandardScaler

import rnn_models as rnnm

from nn_utilities.nn_losses import *

# Local code imports ===========================================================
# For reading the dataset into numpy arrays:
import gtCommon as gtc
# Functions to get RNN "windows" and velocity-aligned frame data, respectively.
from motiontools.posefeatures import dataForCombosJAV
# Functions to get all combos and to split combos into train/test sets:
from motiontools.posefeatures import JAV # Enum
from data_by_combo_functions import UnscaledDistanceLogger # Custom callback
from data_by_combo_functions import rnnDataWindows, scaleWindows

# Input configuration:
WIN_SIZE = 6
USE_CONTEXT = False
USE_VEL_IN = False


# This variable was controlling whether the 15-column JAV was used as input.
# However, if re-enabling, would have to reduce window size so that total num
# of train/test samples is consistent. Or pad it somehow...
ONE_FRAME_PER_WINDOW: bool = True

# Output configuration:
class LSTM_OUT_MODE(enum.Enum):
    DEFAULT = 0
    JAV15 = 1
    DISP = 2

CHOSEN_OUT = LSTM_OUT_MODE.DISP

# General configuration:
FORCE_VEL_SCALING = True

# How many frames to skip when going over the pose dataset; skipping more frames
# simulates larger motions or smaller fps. Typical values are 0, 1, 2.
DATASET_SKIP_FRAMES = 2

APPLY_NOISE = False

print("Getting train-test split.")
all_combos = gtc.PoseLoaderBCOT.trainValidationTestByBody(test_ratio=0.2, random_seed=0)
train_combos = gtc.PoseLoaderBCOT.prepIDsForConstructor(all_combos[0])
val_combos = gtc.PoseLoaderBCOT.prepIDsForConstructor(all_combos[1])
test_combos = gtc.PoseLoaderBCOT.prepIDsForConstructor(all_combos[2])

#%%
################################################################################
# READING THE DATA
################################################################################

# From the combo 3-tuples, we get nametuple versions containing only the
# uniquely-identifying parts. Some functions expect this instead of the 3-tuple. 
# We also get a list of pose loaders, one for each BCOT video.
combos, loaders = gtc.PoseLoaderBCOT.getAllMinimalIDsAndLoaders(True)

print("Reading/normalizing data.")
# As with the "vanilla" regression network, we'll get our data in a velocity
# -aligned frame. But this time, we'll start by keeping things separated by 
# combo so that we know the starting and ending points for a video's data and,
# thus, can create our data windows without having a window erroneously overlap
# two separate videos.
all_jav, w2ls, jav_translations = dataForCombosJAV(
    loaders, (JAV.VELOCITY, JAV.ACCELERATION, JAV.JERK), APPLY_NOISE, True, True
)

#%% 
################################################################################
# CALCULATING THE DATA WINDOWS
################################################################################

print("Calculating data's sliding \"windows\" for the network.")

jav_lstm_skip = DATASET_SKIP_FRAMES # Renaming var.

train_jav_in: NDArray
train_jav_out: NDArray
test_jav_in: NDArray
test_jav_out: NDArray
val_jav_in: NDArray
val_jav_out: NDArray

def getWindowsForVelLSTM(combo_subset, use_vel_frame: bool):
    w2ls_param = None
    centring_param = False
    if use_vel_frame:
        w2ls_param = w2ls[jav_lstm_skip]
        centring_param = True
    return rnnDataWindows(
        jav_translations[jav_lstm_skip], combo_subset, WIN_SIZE,
        0, # Always a window "skip" of 0 in this case because the frame is rotating...
        None, w2ls_param, centring_param
    )
def getWindowsJAV(combo_subset, window_size = WIN_SIZE):
    return rnnDataWindows(
        all_jav[jav_lstm_skip], combo_subset, window_size, 
        0, # Always a window "skip" of 0 in this case because the frame is rotating...
        None
    )

if not ONE_FRAME_PER_WINDOW:
    # TODO: try an architecture that uses JAV vec15s as input. However, a window
    # size of W would need either >w prev positions known or some padding....
    raise NotImplementedError("ONE_FRAME_PER_WINDOW=False is not implemented yet.")

train_jav_in, train_jav_out = getWindowsForVelLSTM(train_combos, USE_VEL_IN)
test_jav_in, test_jav_out = getWindowsForVelLSTM(test_combos, USE_VEL_IN)
val_jav_in, val_jav_out = getWindowsForVelLSTM(val_combos, USE_VEL_IN)
train_velframe_in = train_jav_in
if not USE_VEL_IN and FORCE_VEL_SCALING:
    train_velframe_in, _ = getWindowsForVelLSTM(train_combos, True)
if CHOSEN_OUT == LSTM_OUT_MODE.JAV15:
    _, train_jav_out = getWindowsJAV(train_combos, 2)
    _, test_jav_out = getWindowsJAV(test_combos, 2)
    _, val_jav_out = getWindowsJAV(val_combos, 2)
#%%
################################################################################
# NORMALIZING THE DATA
################################################################################

# For world-frame translations, would like to just use *all* translations to
# set scaler so that scaler is same regardless of skip-sampling.
# For velocity frame, this is not possible. Instead, would like to get all the
# values that the neural network will see from the training data.
# To avoid duplicates, this would be the end of every window plus... nvm, I
# realized just adding the front of window [0] won't cut it because this won't
# include the first frames for any video but the first, nor will it include the
# final "out" frames... depending on how the paper goes, this is something I'll
# need to fix eventually but there's no point spending time to fix it in advance
# now, since it will barely change the scaling and these results may not even
# make it into the paper. The most memory-efficient fix that I can think of is
# to have the RNN window function return the indices of windows where new videos
# start and then use those indices to get necessary slices.


# Rather than concatenate all data for scaling into a numpy array of the
# required (n, d) shape, it's more memory-efficient to just pass in a list
# of old arrays or slices, and then use partial_fit for those. 
def scalerForWindows(translation_groups, dim):
    ret = StandardScaler()
    new_shape = (-1, dim)
    ret.fit(np.reshape(translation_groups[0], new_shape))
    for tg in translation_groups[1:]:
        ret.partial_fit(np.reshape(tg, new_shape))
    return ret

ts_vel = [train_velframe_in[:, -1], train_velframe_in[0, :-1]]
ts_world: typing.List[NDArray] = []
# all_rotations: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
# all_rotation_mats: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
for combo in train_combos:
    calculator = gtc.PoseLoaderBCOT(combo[0], combo[1], 0)
    ts_world.append(calculator.getTranslationsGTNP())

ts_main = ts_vel if USE_VEL_IN else ts_world
ts_mag = ts_vel if FORCE_VEL_SCALING else ts_main

temp_mag_scaler = scalerForWindows(ts_mag, dim=1)
scaler_obj = scalerForWindows(ts_main, dim=3)

scaler_obj.scale_[-3:] = temp_mag_scaler.scale_
if USE_VEL_IN:
    scaler_obj.mean_[-3:] = 0.0

train_jav_in = scaleWindows(train_jav_in, scaler_obj)
test_jav_in = scaleWindows(test_jav_in, scaler_obj)
val_jav_in = scaleWindows(val_jav_in, scaler_obj)

if CHOSEN_OUT != LSTM_OUT_MODE.JAV15:
    train_jav_out = scaler_obj.transform(train_jav_out)
    test_jav_out = scaler_obj.transform(test_jav_out)
    val_jav_out = scaler_obj.transform(val_jav_out)

    if CHOSEN_OUT == LSTM_OUT_MODE.DISP:
        train_jav_out -= train_jav_in[:, -1]
        test_jav_out -= test_jav_in[:, -1]
        val_jav_out -= val_jav_in[:, -1]

# The context must use the same frame as rnnDataWindows applies to each window,
# i.e. w2ls index (i + WIN_SIZE - shift) for window i, where shift is the number
# of leading frames that w2ls lacks relative to the translations. The "4" here
# encodes that shift, so the first window's frame is w2ls[WIN_SIZE - 4].
# If the shift ever changes, the length assertion below will fail.
w2l_trim = WIN_SIZE - 4
curr_step_w2ls = w2ls[jav_lstm_skip]
def getContextData(combo_subset):
    concat_w2ls = np.concatenate([curr_step_w2ls[c][w2l_trim:] for c in combo_subset])
    concat_w2ls = concat_w2ls.reshape(-1, 9)
    return concat_w2ls

train_in = train_jav_in
val_in = val_jav_in
test_in = test_jav_in
if USE_CONTEXT:
    train_context = getContextData(train_combos)
    val_context = getContextData(val_combos)
    test_context = getContextData(test_combos)
    assert len(train_context) == len(train_jav_out)
    assert len(val_context) == len(val_jav_out)
    assert len(test_context) == len(test_jav_out)
    train_in = (train_jav_in, train_context)
    val_in = (val_jav_in, val_context)
    test_in = (test_jav_in, test_context)
#%%
################################################################################
# BUILDING THE NETWORK
################################################################################
print("Building the NN.")

_mode_is_jav15 = (CHOSEN_OUT == LSTM_OUT_MODE.JAV15)
out_dims = 12 if _mode_is_jav15 else 3
loss_to_use = poseLossJAV if _mode_is_jav15 else poseLossVec3
input_win_len = WIN_SIZE
if USE_VEL_IN:
    input_win_len = WIN_SIZE - 1
LSTM_UNITS = (100, 100)
if USE_CONTEXT:
    jav_lstm_model = rnnm.get_frame_context_lstm(
        loss_to_use, input_win_len, 9, units=LSTM_UNITS, out_size=out_dims
    )
else:
    jav_lstm_model = rnnm.get_simple_lstm(
        loss_to_use, out_dims, 'adam', units=LSTM_UNITS
    )

# Log the loss in mm, rather than just scaled coordinates.
logger_scaler = None if _mode_is_jav15 else scaler_obj
jav_lstm_logger = UnscaledDistanceLogger(
    train_in, train_jav_out, logger_scaler, loss_func=loss_to_use
)
print("Note: The default reported \"loss\" will not be in millimeters, because training data was normalized.")
print("Actual MAE in millimeters will be printed explicitly/separately.")

#%%
################################################################################
# TRAINING THE NETWORK
################################################################################

early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                      restore_best_weights=True)
plateau = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)

jav_lstm_hist = jav_lstm_model.fit(
    train_in, train_jav_out, epochs=100,
    validation_data=[val_in, val_jav_out],
    callbacks=[jav_lstm_logger, early, plateau]
)

#%%
################################################################################
# EVALUATING THE NETWORK
################################################################################



jav_lstm_test_pred = jav_lstm_model.predict(test_in)

def getErrsLSTM(scaler, unscaled_preds, scaled_gt):

    scaled_preds = unscaled_preds
    if CHOSEN_OUT != LSTM_OUT_MODE.JAV15:
        scaled_preds = scaler.inverse_transform(unscaled_preds)

    return loss_to_use(scaled_gt, scaled_preds)

# Transform the coordinates back into non-normalized form to get the errors
# in millimeters.
jav_scaled_lstm_gt_test = test_jav_out
if CHOSEN_OUT != LSTM_OUT_MODE.JAV15:
    jav_scaled_lstm_gt_test = scaler_obj.inverse_transform(
        test_jav_out
    )

const_acc_pred = np.repeat((1.0, 0.0), (3, 9)).reshape(1, -1)
if CHOSEN_OUT != LSTM_OUT_MODE.JAV15:
    if not USE_VEL_IN:
        const_acc_pred = 3 * test_jav_in[:, -1] - 3 * test_jav_in[:, -2] + test_jav_in[:, -3]
        if CHOSEN_OUT == LSTM_OUT_MODE.DISP:
            const_acc_pred = const_acc_pred - test_jav_in[:, -1]
    else:
        scaler_at_0 = -3 * scaler_obj.mean_ / scaler_obj.scale_
        const_acc_pred = scaler_at_0 -3 * test_jav_in[:, -1, -3:] + test_jav_in[:, -2, -3:]
        if CHOSEN_OUT == LSTM_OUT_MODE.DISP:
            const_acc_pred = const_acc_pred - test_jav_in[:, -1]

const_acc_errs = getErrsLSTM(
    scaler_obj, const_acc_pred, jav_scaled_lstm_gt_test
)

# jav_scaled_lstm_gt_train = scaler_obj.inverse_transform(
#     train_jav_out[:, -3:]
# )
jav_lstm_test_errs = getErrsLSTM(
    scaler_obj, jav_lstm_test_pred, jav_scaled_lstm_gt_test
)
print("\n\nAcc score (millimeters) on test data:", np.mean(const_acc_errs))
print("\n\nJAV LSTM score (millimeters) on test data:", np.mean(jav_lstm_test_errs))
print("Above score is for skip{} data.".format(jav_lstm_skip))
#%%

################################################################################
# Train loss graph.
################################################################################
import matplotlib.pyplot as plt

plt.plot(jav_lstm_logger.mean_distances)

plt.xlabel("Epoch")
plt.ylabel("MAE (mm)")
plt.title("Loss for skip{} dataset train".format(jav_lstm_skip))
plt.grid(True)
plt.show()

print("Done program.")
