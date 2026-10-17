################################################################################
# World-Frame Rotation LSTM
################################################################################

import typing

import numpy as np

import tensorflow as tf
import keras

from sklearn.preprocessing import StandardScaler

# Local code imports ===========================================================
# For reading the dataset into numpy arrays:
from gtCommon import PoseLoaderBCOT
# Functions to get RNN "windows".
from data_by_combo_functions import rnnDataWindows
# Functions to get all combos and to split combos into train/test sets:
from data_by_combo_functions import UnscaledDistanceLogger # Custom callback

import rnn_models as rnnm

from nn_utilities.nn_losses import poseLossRotVec3CL, poseLossRotVec3CL2, poseLossRotVec3NP
import posemath as pm

WIN_SIZE = 6

# How many frames to skip when going over the pose dataset; skipping more frames
# simulates larger motions or smaller fps. Typical values are 0, 1, 2.
DATASET_SKIP_FRAMES = 2

USE_RESIDUAL = True

combos = PoseLoaderBCOT.getAllIDs()

print("Getting train-test split.")
all_combos = PoseLoaderBCOT.trainValidationTestByBody(test_ratio=0.2, random_seed=0)
train_combos = PoseLoaderBCOT.prepIDsForConstructor(all_combos[0])
val_combos = PoseLoaderBCOT.prepIDsForConstructor(all_combos[1])
test_combos = PoseLoaderBCOT.prepIDsForConstructor(all_combos[2])

#%%
################################################################################
# READING/NORMALIZING THE DATA
################################################################################

print("Reading/normalizing data.")

#%% Read all the rotation data in and get the scaler for normalization.
all_rotations: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
for combo in combos:
    calculator = PoseLoaderBCOT(combo[0], combo[1], 0)
    all_rotations[combo[:2]] = calculator.getRotationsGTNP()
    # For future experiments (translations via calculator.getTranslationsGTNP()
    # for vec6/vec12 inputs):
    # aa_rotations = calculator.getRotationsGTNP(False)
    # quats = pm.quatsFromAxisAngleVec3s(aa_rotations)
    # all_rotation_quats[combo[:2]] = quats
    # all_rotation_mats[combo[:2]] = calculator.getRotationMatsGTNP(False)

# %%
# Rotation values are currently left unscaled: the scaler is an identity
# transform, kept so rnnDataWindows and UnscaledDistanceLogger can use the same
# interface as for scaled data.
rotation_scaler = StandardScaler()

# TODO: Do actual fitting!
rotation_scaler.fit(all_rotations[(0,0)].reshape(-1, 3))
rotation_scaler.scale_[:] = 1.0
rotation_scaler.mean_[:] = 0.0

def outputModeTransform(vecs_in, vecs_out):
    ret_vecs_out = vecs_out
    if USE_RESIDUAL:
        prev_vec_quats = pm.quatsFromAxisAngleVec3s(vecs_in[:, -1])
        out_quats = pm.quatsFromAxisAngleVec3s(vecs_out)
        resid_out = pm.quatsToFrom(out_quats, prev_vec_quats)
        ret_vecs_out = pm.axisAngleVec3sFromQuats(resid_out, True)
    return ret_vecs_out

#%%
################################################################################
# CALCULATING THE DATA WINDOWS
################################################################################

print("Calculating data's sliding \"windows\" for the network.")

lstm_skip = DATASET_SKIP_FRAMES # "Renaming" var.

train_rotations_in, train_rotations_out = rnnDataWindows(
    all_rotations, train_combos, WIN_SIZE, lstm_skip, rotation_scaler
)
train_rotations_out = outputModeTransform(train_rotations_in, train_rotations_out)

val_rotations_in, val_rotations_out = rnnDataWindows(
    all_rotations, val_combos, WIN_SIZE, lstm_skip, rotation_scaler
)
val_rotations_out = outputModeTransform(val_rotations_in, val_rotations_out)


#%%
################################################################################
# BUILDING THE NETWORK
################################################################################
print("Building the NN.")

loss = poseLossRotVec3CL2

lstm_model = rnnm.get_simple_lstm(loss, 3, 'adam', units=(100, 100))


# Log the loss in mm, rather than just scaled coordinates.
lstm_logger = UnscaledDistanceLogger(
    train_rotations_in, train_rotations_out, rotation_scaler
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

lstm_hist = lstm_model.fit(
    train_rotations_in, train_rotations_out, epochs=100,
    validation_data=(val_rotations_in, val_rotations_out),
    callbacks=[lstm_logger, early, plateau]
)


#%%
################################################################################
# EVALUATING THE NETWORK
################################################################################

test_rotations_in, test_rotations_out = rnnDataWindows(
    all_rotations, test_combos, WIN_SIZE, lstm_skip, rotation_scaler
)
test_rotations_out = outputModeTransform(test_rotations_in, test_rotations_out)

lstm_test_pred = lstm_model.predict(test_rotations_in)

quats_k = pm.quatsFromAxisAngleVec3s(test_rotations_in[:, -1])
quats_km1 = pm.quatsFromAxisAngleVec3s(test_rotations_in[:, -2])
phys_pred_qs = pm.quatsToFrom(quats_k, quats_km1)
if not USE_RESIDUAL:
    phys_pred_qs = pm.multiplyQuatLists(phys_pred_qs, quats_k)

phys_pred = pm.axisAngleVec3sFromQuats(phys_pred_qs, False)
# Transform back into non-normalized form. (Currently an identity transform for
# rotations, but this keeps the evaluation correct if scaling is ever added.)
unscaled_lstm_test_pred = rotation_scaler.inverse_transform(lstm_test_pred)
unscaled_phys_pred = rotation_scaler.inverse_transform(phys_pred)
unscaled_lstm_test_gt = rotation_scaler.inverse_transform(test_rotations_out)

# %%
phys_errs = poseLossRotVec3NP(unscaled_lstm_test_gt, unscaled_phys_pred)
lstm_test_errs2 = loss(unscaled_lstm_test_gt, unscaled_lstm_test_pred)
_lstm_test_errs = poseLossRotVec3NP(unscaled_lstm_test_gt, unscaled_lstm_test_pred)
np_test_score = np.mean(np.rad2deg(_lstm_test_errs))
tf_score = np.rad2deg(float(tf.reduce_mean(lstm_test_errs2)))
phys_score = np.rad2deg(np.mean(phys_errs))
unit_str = "degs"

print("\n\nSanity check phys-based score ({}) on test data: {}".format(unit_str, phys_score))
print("\n\nTF LSTM score ({}) on test data: {}".format(unit_str, tf_score))
print("NP LSTM score ({}) on test data: {}".format(unit_str, np_test_score))
print("Above scoring is for skip{} data.".format(lstm_skip))

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
