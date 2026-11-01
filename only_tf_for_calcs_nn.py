# %%
import typing
import pickle

import numpy as np
from numpy.typing import NDArray

import tensorflow as tf
import keras

# tf.config.run_functions_eagerly(True) 

from gtCommon import PoseLoaderBCOT

import posemath as pm

from data_by_combo_functions import rnnDataWindows

from motiontools.dataorg import UnitAwareScaler, SUBSET_PRESET

from motiontools.hypothetical_inputs_calc import HypotheticalInputsForNN

from nn_utilities.nn_loading import loadLatestModels
from nn_utilities.nn_export import AbstractModelWrapper
from nn_utilities.nn_inference import (
    PointsToInputsConstStep, ang_vel_displace,
    tfGetDisplacementQuats, tfAxisAngleVec3sToQuats, tfMulQuatLists,
    tfQuatsToAxisAngleVec3s
)
from nn_utilities.nn_modification import create_hardcoded_model
from nn_utilities.nn_losses import getVelFrameDisplacements
from nn_utilities.lstm_io import (
    loadRun, findRun, comboHash, LSTM_OUT_MODE
)

CUSTOM_LAYER_WINDOW_SIZE = 6  # Number of consecutive pose frames
CUSTOM_LAYER_SKIP = 2         # Frame skip (0=all frames, 1=every other, 2=every 3rd)
USE_NOISE = False

SAVE_RUNS_DIR = "results/lstms"

print(f"Loading test pose windows...")
print(f"  Window size: {CUSTOM_LAYER_WINDOW_SIZE}")
print(f"  Skip: {CUSTOM_LAYER_SKIP}")

# Get train/test IDs
all_ids = PoseLoaderBCOT.trainValidationTestByBody(test_ratio=0.2, random_seed=0)
train_ids, val_ids, test_ids = [
    sorted(PoseLoaderBCOT.prepIDsForConstructor(ids)) for ids in all_ids
]

all_poses: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
# all_rotations: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
# all_rotation_mats: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
custom_step = CUSTOM_LAYER_SKIP + 1
for combo in train_ids + test_ids:
    calculator = PoseLoaderBCOT(combo[0], combo[1], 0)
    curr_translations = calculator.getTranslationsGTNP()[::custom_step]
    curr_rotations = calculator.getRotationsGTNP()[::custom_step]
    if USE_NOISE:
        curr_translations = calculator.getNoisyTranslation(4.5)[::custom_step]
        curr_rotations = calculator.getNoisyRotation(3.0)[::custom_step]
    curr_poses = np.concatenate(
        (curr_translations, curr_rotations), axis=-1
    )
    # TODO: Better-comment this part, maybe make a switch to turn it on/off,
    # etc.
    # curr_poses = np.pad(curr_poses, ((2, 0), (0, 0)), mode="edge")

    all_poses[combo[:2]] = curr_poses

test_windows, test_gt_pts = rnnDataWindows(
    all_poses, test_ids, CUSTOM_LAYER_WINDOW_SIZE, skip=0
)
test_windows_flatter = test_windows.reshape(
    test_windows.shape[0], CUSTOM_LAYER_WINDOW_SIZE * 6
)
print("Test shape:", test_windows.shape)
print("Test reshape:", test_windows_flatter.shape)

# Load scaler to get ref_keys
loaded_scaler: UnitAwareScaler
with open("./results/models/scaler.pickle", "rb") as f:
    loaded_scaler = pickle.load(f)
ref_keys = loaded_scaler.column_keys
# Get the column permutation using the static method from HypotheticalInputsForNN
_orig_key_order = HypotheticalInputsForNN._generated_key_order()
permutation_np = np.asarray([_orig_key_order.index(k) for k in ref_keys])



# Create custom layer that, from last poses, creates columns for FCNN.
custom_layer = PointsToInputsConstStep(
    step=custom_step,
    scale_means=loaded_scaler.mean_, scale_scales=loaded_scaler.scale_,
    to_nn_permut=permutation_np,
    zero_angle_thresh=0.01
)

print("Creating combined model with custom layer...")

# %%
# Load saved FCNN .keras file.
model_key = "JAV_MULTIPLIERS"
bcs_model = loadLatestModels((model_key, ), SUBSET_PRESET.MINIMAL, USE_NOISE)[model_key]
hardcoded_model = create_hardcoded_model(bcs_model, [
    1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])

run_meta = {
    "script": "lstms_translation", "win_size": CUSTOM_LAYER_WINDOW_SIZE,
    "skip": CUSTOM_LAYER_SKIP, "use_vel_in": False, "use_context": False,
    "out_mode": LSTM_OUT_MODE.DISP.name, "force_vel_scaling": True,
    "apply_noise": USE_NOISE, "lstm_units": [128, 128, 128],
    "train_split": comboHash(train_ids),
    "val_split": comboHash(val_ids),
    "test_split": comboHash(test_ids),
}

load_lstm_dir = findRun(SAVE_RUNS_DIR, run_meta)
if load_lstm_dir is None:
    raise Exception("No saved LSTM matches the current settings.")

jav_lstm_model = loadRun(load_lstm_dir, True, run_meta)

# Define the getWorldFrameDisplacements function as a TensorFlow operation
def getWorldFrameDisplacements(y_true, y_pred, world2locals):
    disp = getVelFrameDisplacements(y_true, y_pred)
    # Conversion to TFLite converts einsum version of this into a batched matmul
    # in an incorrect way, so I need to "manually" do it the correct way.
    disp_mats = disp[..., tf.newaxis] # pyright:ignore
    mm = tf.matmul(world2locals, disp_mats, transpose_a=True)
    return tf.reshape(mm, [-1, 3])

class AbstractKalmanWrapper(AbstractModelWrapper):
    """
    Shared logic for models that map a state of N/6 stacked vec6 poses to the
    next state (shift the history by one pose, append a newly predicted pose).

    Subclasses implement predict_pose (returning [B, 6]).
    """

    def __init__(self, input_shape):
        super().__init__(input_shape)
        self.num_prev_cells: int = input_shape[-1] - 6

    # ---- hooks ----
    def predict_pose(self, x: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def forward(self, x: tf.Tensor):
        x.set_shape(self.input_shape)
        prev_pts = x[:, 6:]
        full_new_state = tf.concat((prev_pts, self.predict_pose(x)), axis=1)
        return full_new_state


    def jacobian_naive(self, x: tf.Tensor):
        x.set_shape(self.input_shape)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            y = self.forward(x)
        unflat_jacobian = tape.jacobian(y, x)
        return tf.reshape(unflat_jacobian, [-1, tf.shape(unflat_jacobian)[-1]])


    def jacobian(self, x: tf.Tensor):
        """
        Less naive Jacobian: Runs only what is needed for each block.
        """
        x.set_shape(self.input_shape)

        # The part that comes from shifting the five most recent poses within
        # the state vector.
        eye_block = tf.eye(self.num_prev_cells, dtype=tf.float32)
        zeros_block = tf.zeros((self.num_prev_cells, 6), dtype=tf.float32)
        j_shift = tf.concat([zeros_block, eye_block], axis=-1)

        # New-pose block: differentiate only predict_pose, not the whole forward.
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            y_new = self.predict_pose(x)
        unflat = tape.jacobian(y_new, x)
        j_new = tf.reshape(unflat, [-1, tf.shape(unflat)[-1]])

        return tf.concat([j_shift, j_new], axis=0)

def constant_angvel_rotation(x: tf.Tensor) -> tf.Tensor:
    """Default rotation model: constant angular velocity from the last two poses."""
    aas1 = x[:, -3:]
    aas0 = x[:, -9:-6]
    ws0, vs0 = tfAxisAngleVec3sToQuats(aas0)
    ws1, vs1 = tfAxisAngleVec3sToQuats(aas1)
    wd, vd = tfGetDisplacementQuats(ws0, vs0, ws1, vs1)
    wf, vf = tfMulQuatLists(wd, vd, ws1, vs1)
    const_angvel = tfQuatsToAxisAngleVec3s(wf, vf)
    return const_angvel


class FcnnKalmanWrapper(AbstractKalmanWrapper):
    """Your original preproc + FCNN-multipliers pipeline."""

    def __init__(self, preproc_layer, fcnn_model, input_shape):
        super().__init__(input_shape)
        self.preproc = preproc_layer
        self.fcnn = fcnn_model

    def predict_pose(self, x):
        features = self.preproc(x)

        multiplier_predictions = self.fcnn(features[0])
        if isinstance(multiplier_predictions, list):
            # I should've commented this better; I think switching tf versions
            # requires this workaround? I forget the exact reason now....
            # print("Loaded model prediction was a list for whatever reason?")
            multiplier_predictions = multiplier_predictions[0]

        y_true, world2locals = features[1], features[2]
        orig_pos = x[:, -6:-3]
        disp = getWorldFrameDisplacements(y_true, multiplier_predictions, world2locals)
        return tf.concat((orig_pos + disp, constant_angvel_rotation(x)), axis=-1)

class LstmKalmanWrapper(AbstractKalmanWrapper):
    def __init__(self, lstm_model, input_shape, lstm_vec_dim: int,
                 predicts_delta: bool, rotation_fn=constant_angvel_rotation):
        super().__init__(input_shape)
        self.lstm = lstm_model
        self.lstm_vec_dim = lstm_vec_dim
        self.predicts_delta = predicts_delta
        self.rotation_fn = rotation_fn
        self.out_dim = lstm_model.output_shape[-1]
        assert self.out_dim in (3, 6), f"Unexpected LSTM output dim {self.out_dim}"
        assert self.lstm_vec_dim in (3, 6), f"Unexpected LSTM vec dim {self.lstm_vec_dim}"
        self.timesteps = input_shape[-1] // 6

    def _run_lstm(self, x):
        seq_vec6 = tf.reshape(x, (-1, self.timesteps, 6))
        seq = seq_vec6 if self.lstm_vec_dim == 6 else seq_vec6[..., -6:-3]
        out = self.lstm(seq)
        return out[0] if isinstance(out, list) else out

    def predict_pose(self, x):
        out = self._run_lstm(x)
        orig_pos = x[:, -6:-3]
        orig_rot = x[:, -3:]

        if self.out_dim == 6:
            out_t = out[..., -6:-3]
            out_r = out[..., -3:]
            trans = out_t + orig_pos if self.predicts_delta else out_t
            rot = ang_vel_displace(orig_rot, out_r) if self.predicts_delta else out_r
            return tf.concat((trans, rot), axis=-1)
        elif self.out_dim == 3:
            trans = out + orig_pos if self.predicts_delta else out       
            # vec3 output: translation from the LSTM, rotation from rotation_fn
            return tf.concat((trans, self.rotation_fn(x)), axis=-1)
        else:
            raise Exception("Bad out dimension!")

# %%

wrapper = FcnnKalmanWrapper(custom_layer, bcs_model, (1, 36))
wrapper_batch = FcnnKalmanWrapper(custom_layer, bcs_model, (None, 36))

wrapper_l = LstmKalmanWrapper(jav_lstm_model, (1, 36), 3, False)
wrapper_batch_l = LstmKalmanWrapper(jav_lstm_model, (None, 36), 3, False)

print("="*60)
print("Final model setup complete")
print("="*60 + "\n")

tf_const_input = tf.constant(test_windows_flatter.astype(np.float32)) 
batch_out = wrapper_batch.forward(tf_const_input)
print("First batch out:", batch_out[0])
final_model_forward = wrapper_batch.get_frozen_func(wrapper_batch.forward)
final_model_forward1 = wrapper.get_frozen_func(wrapper.forward)
final_model_forward1_l = wrapper_l.get_frozen_func(wrapper_l.forward)
test_out = final_model_forward(tf_const_input)[0]
out_pts = test_out[..., -6:-3]
out_aas = test_out[..., -3:]
#%%
test_errs = out_pts - test_gt_pts[:, :3]

test_score = np.mean(np.linalg.norm(test_errs, axis=-1))
print("Translation test score:", test_score)

# %%
test_aa_errs = pm.poseLossAngle(test_gt_pts[:, 3:], out_aas)
print("AA test score:", np.rad2deg(np.mean(test_aa_errs)))

# %%
# import datetime
# model_name = "results/models/{}-{:%Y-%m-%d_%H-%M-%S}.keras".format(
#     "state_transition", datetime.datetime.now()
# )
# final_model.save(model_name)

#%%
wrapper.save_as_savedmodel("./results/models/e2e_saved_models_skip{}/".format(CUSTOM_LAYER_SKIP))
wrapper.save_jacobian("./results/models/jacobian_graph.pb")
wrapper.save_forward("./results/models/forward_graph.pb")

# %%
arange_dat = tf.constant(np.arange(36).reshape(wrapper.input_shape).astype(np.float32))
print(final_model_forward1(arange_dat))
#%%
x = tf.constant(np.random.uniform(-1, 1, (1, 36)).astype(np.float32))
j = wrapper.jacobian(x) #tf.constant(arange_dat, dtype=tf.float32))
print("First Jacobian tested...")
j2 = wrapper.jacobian_naive(x)
print("Jacobians same:", np.allclose(j.numpy(), j2.numpy()))
print("Max difference:", np.max(np.abs(j.numpy() - j2.numpy())))

#%%
jac_fn = "./results/models/jac.tflite"
# wrapper.save_tflite(wrapper.forward, "./results/models/f.tflite", False)
wrapper.save_tflite(wrapper.jacobian, jac_fn, True)
# %%
import nn_standalones.tflite_attempt as nnt

nnt.test_tflite_export(wrapper.jacobian, jac_fn, wrapper.input_shape)
