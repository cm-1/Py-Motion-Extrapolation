import pickle
import glob

import numpy as np
import keras

from motiontools.posefeatures import (
    CalcsForVideo, dataForPositionsJAV, JAV, getWorldFrameDisplacements
)
from motiontools.dataorg import UnitAwareScaler

# Needed when loading model, even though IDE syntax highlighting marks this as
# "unused"!
from nn_utilities.nn_losses import poseLossJAV


cfc = CalcsForVideo()

# We need 7 input points to get crackle calculations because my code currently
# assumes the last one is ground truth for which it shouldn't generate any
# predictions, and we need 6 input points to calculate nonzero crackle.
rand_pts = np.random.uniform(-33, 33, (7, 3))
default_aas = np.ones_like(rand_pts) 
# default_aas += np.random.normal(scale=0.01, size=rand_pts.shape) # Avoid NaN
rand_all_cols = cfc.getInputFeatures(
    rand_pts, default_aas, max_step = 1
).motion_data[0]

scaler: UnitAwareScaler
with open("./results/models/scaler.pickle", "rb") as f:
    scaler = pickle.load(f)
selected_keys = scaler.column_keys

matching_models = glob.glob("./results/models/JAV_MULTIPLIERS*.keras")
chosen_model = sorted(matching_models)[-1]
print("Loading model:", chosen_model)
loaded_model = keras.models.load_model(chosen_model)

rand_all_cols_np = np.stack(
    [rand_all_cols[k] for k in selected_keys], axis=-1
)
rand_inputs = scaler.transform(rand_all_cols_np)
# print(rand_inputs)
rand_out_JAV = loaded_model.predict(rand_inputs, verbose=0)

jav_order = (JAV.VELOCITY, JAV.ACCELERATION, JAV.JERK)
rand_in_jav, _, w2ls = dataForPositionsJAV(jav_order, rand_pts, None, 1)

rand_out_vec3 = getWorldFrameDisplacements(rand_in_jav, rand_out_JAV, w2ls)
print(rand_out_vec3)

