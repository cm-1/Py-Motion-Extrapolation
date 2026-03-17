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

from nn_utilities.nn_inference import PointsToInputsConstStep, ang_vel_extrapolate

CUSTOM_LAYER_WINDOW_SIZE = 6  # Number of consecutive pose frames
CUSTOM_LAYER_SKIP = 2          # Frame skip (0=all frames, 1=every other, 2=every 3rd)

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



# Create custom layer
custom_layer = PointsToInputsConstStep(
    step=custom_step,
    scale_means=loaded_scaler.mean_, scale_scales=loaded_scaler.scale_,
    to_nn_permut=permutation_np,
    zero_angle_thresh=0.01
)

print("Creating combined model with custom layer...")

# Build new model: Input -> Custom Layer -> Original Model -> Output
window_input = keras.layers.Input(
    shape=test_windows_flatter.shape[1:],
    name='pose_window_input', dtype=tf.float32
)

# Apply custom preprocessing layer
features = custom_layer(window_input)
# %%
model_key = "JAV_MULTIPLIERS"
bcs_model = loadLatestModels((model_key, ))[model_key]
hardcoded_model = create_hardcoded_model(bcs_model)

# Apply loaded model
predictions = bcs_model(features[0])
if isinstance(predictions, list):
    predictions = predictions[0]
    print("Loaded model prediction was a list for whatever reason?")

# Create combined model
combined_model = keras.Model(
    inputs=window_input,
    outputs=predictions,
    name='model_with_preprocessing'
)

print("Combined model architecture:")
combined_model.summary()
print("="*60)
print("Custom layer setup complete")
print("="*60 + "\n")

# %%
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

# Create a new layer that wraps getWorldFrameDisplacements
class WorldFrameDisplacementsLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WorldFrameDisplacementsLayer, self).__init__(**kwargs)

    def call(self, inputs):
        y_true = inputs[0] #tf.cast(inputs[0], tf.float32)
        y_pred = inputs[1] #tf.cast(inputs[1], tf.float32)
        world2locals = inputs[2] #tf.cast(inputs[2], tf.float32)
        orig_inputs = tf.reshape(inputs[3], [-1, 6, 6])
        
        orig_aas_0 = orig_inputs[:, -2, 3:]
        orig_aas_1 = orig_inputs[:, -1, 3:]
        orig_pos = orig_inputs[:, -1, :3]

        disp = getWorldFrameDisplacements(y_true, y_pred, world2locals)
        ret_pos = orig_pos + disp
        ret_aas = ang_vel_extrapolate(orig_aas_0, orig_aas_1)
        return tf.concat((ret_pos, ret_aas), axis=-1) 

# Append the new layer to the combined model
world_frame_displacements_layer = WorldFrameDisplacementsLayer(name="wfdl")
combined_model_output = combined_model.output
world_frame_displacements_output = world_frame_displacements_layer(
    [features[1], combined_model_output, features[2], window_input]
)



# Create the final model
final_model = keras.Model(
    inputs=combined_model.input,
    outputs=world_frame_displacements_output, #features[0], features[1], features[2], combined_model_output], 
    name='final_model_with_world_frame_displacements'
)

print("Final model architecture:")
final_model.summary()
print("="*60)
print("Final model setup complete")
print("="*60 + "\n")

test_out = final_model.predict(test_windows_flatter, batch_size=1024)
out_pts = test_out[..., :3]
out_aas = test_out[..., 3:]
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

# %%
from nn_utilities.nn_export import ModelExportWrapper
wrapper = ModelExportWrapper(final_model, (1, 36))

#%%
wrapper.save_as_savedmodel("./results/models/e2e_saved_models/")
wrapper.save_jacobian("./results/models/jacobian_graph.pb")
wrapper.save_forward("./results/models/forward_graph.pb")

# %%
arange_dat = np.arange(36).reshape(wrapper.input_shape).astype(np.float32)
print(final_model.predict(arange_dat))
#%%
x = tf.constant(np.random.uniform(-1, 1, (1, 36)).astype(np.float32))
j = wrapper.jacobian_orig(x) #tf.constant(arange_dat, dtype=tf.float32))
print("First Jacobian tested...")
j2 = wrapper.jacobian(x)
print("Jacobians same:", np.allclose(j.numpy(), j2[0].numpy()))
print("Max difference:", np.max(np.abs(j.numpy() - j2[0].numpy())))

#%%
jac_fn = "./results/models/jac.tflite"
# wrapper.save_tflite(wrapper.forward, "./results/models/f.tflite", False)
wrapper.save_tflite(wrapper.jacobian, jac_fn, False)
# %%
import nn_standalones.tflite_attempt as nnt

nnt.test_tflite_export(wrapper.jacobian, jac_fn, wrapper.input_shape)
