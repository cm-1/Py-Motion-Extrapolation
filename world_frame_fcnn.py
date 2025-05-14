################################################################################
# Trying out a network largely based on:
# Wang, Juxing, and Shen, Linyong. "Semi-Adaptable Human Hand Motion Prediction 
# Based on Neural Networks and Kalman Filter." Journal of Physics: Conference 
# Series. Vol. 2029. No. 1. IOP Publishing, 2021.
################################################################################
import typing

import numpy as np

import keras

# We'll use a MinMax scaler to "normalize" the data, as per a tutorial.
from sklearn.preprocessing import MinMaxScaler

# Local code imports ===========================================================
# For reading the dataset into numpy arrays:
from gtCommon import PoseLoaderBCOT
# Functions to get RNN "windows".
from data_by_combo_functions import rnnDataWindows
# Functions to get all combos and to split combos into train/test sets:
from data_by_combo_functions import UnscaledDistanceLogger # Custom callback


WIN_SIZE = 4

# How many frames to skip when going over the pose dataset; skipping more frames
# simulates larger motions or smaller fps. Typical values are 0, 1, 2.
DATASET_SKIP_FRAMES = 2

combos = PoseLoaderBCOT.getAllIDs()

print("Getting train-test split.")
train_combos, test_combos = PoseLoaderBCOT.trainTestByBody(test_ratio=0.2, random_seed=0)

#%%
################################################################################
# READING/NORMALIZING THE DATA
################################################################################

print("Reading/normalizing data.")

#%% Read all the translation data in and get the scaler for normalization.
all_poses: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
# all_rotations: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
# all_rotation_mats: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
for combo in combos:
    calculator = PoseLoaderBCOT(combo[0], combo[1], 0)
    curr_translations = calculator.getTranslationsGTNP()
    curr_rotations = calculator.getRotationsGTNP()
    all_poses[combo[:2]] = np.concatenate(
        (curr_translations, curr_rotations), axis=-1
    )
    # aa_rotations = calculator.getRotationsGTNP(False)
    # quats = pm.quatsFromAxisAngleVec3s(aa_rotations)
    # all_rotations[combo[:2]] = quats
    # all_rotation_mats[combo[:2]] = calculator.getRotationMatsGTNP(False)

all_poses_concat = np.concatenate(
    [all_poses[c[:2]] for c in combos], axis=0
)
translation_scaler = MinMaxScaler(feature_range=(0,1))
translation_scaler.fit(all_poses_concat)
# Scale all 3 axes by uniform amount and centre to 0 for later simplicity

translation_scaler.scale_[:3] = np.max(translation_scaler.scale_)
translation_scaler.scale_[3:] = 1.0
translation_scaler.min_[:] = 0.0

translation_scaler3 = MinMaxScaler(feature_range=(0,1))
translation_scaler3.fit(all_poses_concat[:, :3])
translation_scaler3.scale_[:] = translation_scaler.scale_[:3]
translation_scaler3.min_[:] = 0.0

#%%
################################################################################
# CALCULATING THE DATA WINDOWS
################################################################################

print("Calculating data's sliding \"windows\" for the network.")

lstm_skip = DATASET_SKIP_FRAMES # "Renaming" var.

train_translations_in, train_translations_out = rnnDataWindows(
    all_poses, train_combos, WIN_SIZE, translation_scaler, lstm_skip
)

#%%
################################################################################
# BUILDING THE NETWORK
################################################################################
print("Building the NN.")

lstm_model = keras.Sequential([
    keras.layers.Input((WIN_SIZE * 6,)),
    keras.layers.Dense(12, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    # keras.layers.Dense(18, activation='relu'),
    keras.layers.Dense(3)#num_classes, activation='sigmoid')
])

lstm_model.summary()

# Log the loss in mm, rather than just scaled coordinates.
lstm_logger = UnscaledDistanceLogger(
    train_translations_in.reshape(-1, WIN_SIZE*6), train_translations_out[..., :3],
    translation_scaler3
)
print("Note: The default reported \"loss\" will not be in millimeters, because training data was normalized.")
print("Actual MAE in millimeters will be printed explicitly/separately.")


adam = keras.optimizers.Adam(0.0001)
lstm_model.compile(optimizer=adam, loss='mse')#tf_loss_fn)
#%%
################################################################################
# TRAINING THE NETWORK
################################################################################

lstm_hist = lstm_model.fit(
    train_translations_in.reshape(-1, WIN_SIZE*6),
    train_translations_out[:, :3], epochs=600, callbacks=[lstm_logger]
)


#%%
################################################################################
# EVALUATING THE NETWORK
################################################################################

test_translations_in, test_translations_out = rnnDataWindows(
    all_poses, test_combos, WIN_SIZE, translation_scaler, lstm_skip
)

lstm_test_pred = lstm_model.predict(test_translations_in.reshape(-1, WIN_SIZE*6))

# lstm_test_pred_labs = np.argmax(lstm_test_pred, axis=1)
# lstm_test_errs = rnnErrsForCombos(test_combos, WIN_SIZE, 2)
# lstm_test_err_dists = assessPredError(lstm_test_errs, lstm_test_pred_labs)
# print(lstm_test_err_dists)

# Transform the coordinates back into non-normalized form to get the errors
# in millimeters.
unscaled_lstm_test_pred = translation_scaler3.inverse_transform(lstm_test_pred)
unscaled_lstm_test_gt = translation_scaler.inverse_transform(test_translations_out)

lstm_test_errs = np.linalg.norm(
    unscaled_lstm_test_pred - unscaled_lstm_test_gt[:, :3], axis=-1
)
print("\n\nLSTM score (millimeters) on test data:", lstm_test_errs.mean())
print("Above score is for skip{} data.".format(lstm_skip))

#%% 

# To show a weakness of trying to use an LSTM to predict coordinates in 
# "world space", we will imagine all input coordinates were shifted by a 
# constant vec3 and see that accuracy degrades.
shift_amt = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
shift_in = test_translations_in + shift_amt #+ 0.015 * np.arange(1,4)
shift_out = test_translations_out + shift_amt #+ 0.015 * np.arange(1,4)
shift_pred = lstm_model.predict(shift_in.reshape(-1, WIN_SIZE*6))

scaled_shift_pred = translation_scaler3.inverse_transform(shift_pred)
scaled_shift_gt = translation_scaler.inverse_transform(shift_out)
shift_lstm_errs = np.linalg.norm(scaled_shift_pred - scaled_shift_gt[:, :3], axis=-1)
print("LSTM err when all inputs translated by constant amount:", shift_lstm_errs.mean())

#%%

################################################################################
# Train loss graph.
################################################################################
import matplotlib.pyplot as plt

plt.plot(lstm_logger.mean_distances)
plt.xlabel("Epoch")
plt.ylabel("MAE (mm)")
plt.title("Loss for skip{} train".format(lstm_skip))
plt.show()

print("Done program.")
