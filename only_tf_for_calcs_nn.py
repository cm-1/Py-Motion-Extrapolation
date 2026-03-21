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

from motiontools.dataorg import UnitAwareScaler

from motiontools.hypothetical_inputs_calc import HypotheticalInputsForNN

from nn_utilities.nn_loading import loadLatestModels
from nn_utilities.nn_export import AbstractModelWrapper
from nn_utilities.nn_inference import PointsToInputsConstStep, ang_vel_extrapolate

CUSTOM_LAYER_WINDOW_SIZE = 6  # Number of consecutive pose frames
CUSTOM_LAYER_SKIP = 2         # Frame skip (0=all frames, 1=every other, 2=every 3rd)

def create_hardcoded_model(refModel):
    shp = refModel.get_config()['layers'][0]['config']['batch_shape'][1:]
    oshp = refModel.compute_output_shape(shp)
    inputs = tf.keras.Input(shape=shp)
    
    const_func = lambda x: (x * 0.0)[..., :12] + tf.constant([[
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]])

    outputs = tf.keras.layers.Lambda(const_func, output_shape=oshp)(inputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Custom Layer Mode: Attach preprocessing layer to model
print("\n" + "="*60)
print("CUSTOM LAYER MODE ACTIVATED")
print("="*60)


# Get train/test IDs
train_ids, test_ids = PoseLoaderBCOT.trainTestByBody(test_ratio=0.2, random_seed=0)
train_ids = sorted(PoseLoaderBCOT.prepIDsForConstructor(train_ids))
test_ids = sorted(PoseLoaderBCOT.prepIDsForConstructor(test_ids))

print(f"Loading test pose windows...")
print(f"  Window size: {CUSTOM_LAYER_WINDOW_SIZE}")
print(f"  Skip: {CUSTOM_LAYER_SKIP}")

all_poses: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
# all_rotations: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
# all_rotation_mats: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
custom_step = CUSTOM_LAYER_SKIP + 1
for combo in train_ids + test_ids:
    calculator = PoseLoaderBCOT(combo[0], combo[1], 0)
    curr_translations = calculator.getTranslationsGTNP()[::custom_step]
    curr_rotations = calculator.getRotationsGTNP()[::custom_step]
    curr_poses = np.concatenate(
        (curr_translations, curr_rotations), axis=-1
    )
    # TODO: Better-comment this part, maybe make a switch to turn it on/off,
    # etc.
    curr_poses = np.pad(curr_poses, ((2, 0), (0, 0)), mode="edge")

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
bcs_model = loadLatestModels((model_key, ))[model_key]
hardcoded_model = create_hardcoded_model(bcs_model)

# Define the getVelFrameDisplacements function as a TensorFlow operation
def getVelFrameDisplacements(y_true, y_pred):


    # y_pred2 = y_pred + y_true[:, 9:15]
    # Calculating the local displacement is the same as custom tf loss function.
    disp_0 = y_true[:, 0] * y_pred[:, 0] + y_true[:, 1] * y_pred[:, 1] \
        + y_true[:, 3] * y_pred[:, 3] + y_true[:, 6] * y_pred[:, 6] \
        + y_true[:, 9] * y_pred[:, 9]
    disp_1 = y_true[:, 2] * y_pred[:, 2] + y_true[:, 4] * y_pred[:, 4] \
        + y_true[:, 7] * y_pred[:, 7] + y_true[:, 10] * y_pred[:, 10]
    disp_2 = y_true[:, 5] * y_pred[:, 5] + y_true[:, 8] * y_pred[:, 8] \
        + y_true[:, 11] * y_pred[:, 11]

    return tf.stack((disp_0, disp_1, disp_2), axis=-1)

# Define the getWorldFrameDisplacements function as a TensorFlow operation
def getWorldFrameDisplacements(y_true, y_pred, world2locals):
    disp = getVelFrameDisplacements(y_true, y_pred)
    # Convert local displacement into world displacement.
    local2worlds = tf.transpose(world2locals, perm=[0, 2, 1])
    # Conversion to TFLite converts einsum version of this into a batched matmul
    # in an incorrect way, so I need to "manually" do it the correct way.
    disp_mats = disp[..., tf.newaxis] # pyright:ignore
    mm = tf.matmul(local2worlds, disp_mats)
    return tf.reshape(mm, [-1, 3])

class ModularKalmanWrapper(AbstractModelWrapper):
    def __init__(self, preproc_layer, fcnn_model, input_shape):
        super().__init__(input_shape)
        # Store components separately
        self.preproc = preproc_layer
        self.fcnn = fcnn_model
        self.num_prev_cells: int = input_shape[-1] - 6
        

    def predict_translation(self, x: tf.Tensor):
        """
        Reconstructs the full forward pass. 
        """
        
        # 1. Preprocessing (PointsToInputsConstStep)
        features = self.preproc(x)
        
        # 2. FCNN Inference (JAV_MULTIPLIERS)
        # Apply loaded model
        multiplier_predictions = self.fcnn(features[0])
        if isinstance(multiplier_predictions, list):
            multiplier_predictions = multiplier_predictions[0]
            # I should've commented this better; I think switching tf versions
            # requires this workaround? I forget the exact reason now....
            print("Loaded model prediction was a list for whatever reason?")
        
        # Define the getVelFrameDisplacements function as a TensorFlow operation
        y_true = features[1] #tf.cast(inputs[0], tf.float32)
        y_pred = multiplier_predictions #tf.cast(inputs[1], tf.float32)
        world2locals = features[2] #tf.cast(inputs[2], tf.float32)
        
        orig_pos = x[:, -6:-3]

        disp = getWorldFrameDisplacements(y_true, y_pred, world2locals)
        ret_pos = orig_pos + disp
        return ret_pos

    def predict_rotation(self, x: tf.Tensor):
        orig_aas_0 = x[:, -9:-6]
        orig_aas_1 = x[:, -3:]
        ret_aas = ang_vel_extrapolate(orig_aas_0, orig_aas_1)
        return ret_aas
        
    def forward(self, x: tf.Tensor):
        x.set_shape(self.input_shape)

        ret_pos = self.predict_translation(x)
        ret_aas = self.predict_rotation(x)
        last_pose = tf.concat((ret_pos, ret_aas), axis=-1) 

        prev_pts = x[:, 6:]


        full_new_state = tf.concat((prev_pts, last_pose), axis=1)

        return full_new_state

    def jacobian_naive(self, x: tf.Tensor):
        x.set_shape(self.input_shape)

        """Jacobian subgraph"""
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            y = self.forward(x)
        unflat_jacobian = tape.jacobian(y, x)
        return tf.reshape(unflat_jacobian, [-1, tf.shape(unflat_jacobian)[-1]])


    def jacobian(self, x: tf.Tensor):
        """
        Optimal Jacobian: Runs only what is needed for each block.
        """
        x.set_shape(self.input_shape)

        # The part that comes from shifting the five most recent poses within
        # the state vector.
        eye_block = tf.eye(
            self.num_prev_cells, dtype=tf.float32
        )
        zeros_block = tf.zeros((self.num_prev_cells, 6), dtype=tf.float32)
        j_shift = tf.concat([zeros_block, eye_block], axis=-1)

        # The part that comes from the new translation prediction.
        with tf.GradientTape(persistent=True) as tape_f:
            tape_f.watch(x)
            y_t = self.predict_translation(x)
        unflat_jac_t = tape_f.jacobian(y_t, x)
        j_t = tf.reshape(unflat_jac_t, [-1, tf.shape(unflat_jac_t)[-1]])

        # The part that comes from the new rotation prediction.
        with tf.GradientTape(persistent=True) as tape_j:
            tape_j.watch(x)
            y_r = self.predict_rotation(x)
        unflat_jac_r = tape_j.jacobian(y_r, x)
        j_r = tf.reshape(unflat_jac_r, [-1, tf.shape(unflat_jac_r)[-1]])

        # Clean up
        del tape_f
        del tape_j

        # Concatenate into square matrix.
        return tf.concat([j_shift, j_t, j_r], axis=0)

# %%

wrapper = ModularKalmanWrapper(custom_layer, bcs_model, (1, 36))
wrapper_batch = ModularKalmanWrapper(custom_layer, bcs_model, (None, 36))

print("="*60)
print("Final model setup complete")
print("="*60 + "\n")

tf_const_input = tf.constant(test_windows_flatter.astype(np.float32)) 
final_model_forward = wrapper_batch.get_frozen_func(wrapper_batch.forward)
test_out = final_model_forward(tf_const_input)[0]
out_pts = test_out[..., -6:-3]
out_aas = test_out[..., -3:]
#%%
test_errs = out_pts - test_gt_pts[:, :3]

test_score = np.mean(np.linalg.norm(test_errs, axis=-1))
print("Translation test score:", test_score)

# %%
test_aa_errs = pm.poseLossAngle(test_gt_pts[:, 3:], out_aas)
print("AA test score:", np.mean(test_aa_errs))

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
arange_dat = np.arange(36).reshape(wrapper.input_shape).astype(np.float32)
print(final_model_forward(arange_dat))
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
