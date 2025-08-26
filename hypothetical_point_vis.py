import typing
import pickle
import glob

import numpy as np
from numpy.typing import NDArray
import keras

from motiontools.posefeatures import (
    CalcsForVideo, dataForPositionsJAV, JAV, getWorldFrameDisplacements
)
from motiontools.dataorg import UnitAwareScaler

# Needed when loading model, even though IDE syntax highlighting marks this as
# "unused"!
from nn_utilities.nn_losses import poseLossJAV

scaler: UnitAwareScaler
with open("./results/models/scaler.pickle", "rb") as f:
    scaler = pickle.load(f)
selected_keys = scaler.column_keys

matching_models = glob.glob("./results/models/JAV_MULTIPLIERS*.keras")
chosen_model = sorted(matching_models)[-1]
print("Loading model:", chosen_model)
loaded_model = keras.models.load_model(chosen_model, custom_objects={"poseLossJAV": poseLossJAV})

cfc = CalcsForVideo(min_jerk_opt_iter_lim=0, split_min_jerk_opt_iter_lim=0)

jav_order = (JAV.VELOCITY, JAV.ACCELERATION, JAV.JERK)
def getSingleInput(rand_pts: NDArray, rand_aas: NDArray):
    # default_aas += np.random.normal(scale=0.01, size=rand_pts.shape) # Avoid NaN
    rand_all_cols = cfc.getInputFeatures(
        rand_pts, rand_aas, max_step = 1
    ).motion_data[0]


    rand_all_cols_np = [rand_all_cols[k][-1] for k in selected_keys]
    # print(rand_inputs)
    rand_in_jav, _, w2ls = dataForPositionsJAV(jav_order, rand_pts, None, 1)

    return (rand_all_cols_np, rand_in_jav[-1], w2ls[-1], )
    
def processAllInputs(tupleList: typing.Tuple[NDArray, NDArray, NDArray]):
    unscaled_cols = np.stack([t[0] for t in tupleList], axis=0)
    rand_in_jav = np.stack([t[1] for t in tupleList], axis=0)
    w2ls = np.stack([t[2] for t in tupleList], axis=0)
    rand_inputs = scaler.transform(unscaled_cols)
    rand_out_JAV = loaded_model.predict(rand_inputs, batch_size=1024, verbose=0)

    rand_out_vec3 = getWorldFrameDisplacements(rand_in_jav, rand_out_JAV, w2ls)
    return rand_out_vec3

# We need 7 input points to get crackle calculations because my code currently
# assumes the last one is ground truth for which it shouldn't generate any
# predictions, and we need 6 input points to calculate nonzero crackle.
rand_pts = np.random.normal(0.0, 0.0001, (7,3))# zeros((7,3))
default_aas = np.ones_like(rand_pts) 
rand_pts[-3, 0] = 1.0
GRAPH_RES = 50
print(end="")
outlist = []
for xi in range(GRAPH_RES):
    x = (xi / GRAPH_RES) * 2 - 1
    print("\rx index =", xi, end="", flush=True)
    for yi in range(GRAPH_RES):
        y = (yi / GRAPH_RES) * 2 - 1
        rand_pts[-2, :2] = (x, y)
        outlist.append(getSingleInput(rand_pts, default_aas))

rand_out_vec3s = processAllInputs(outlist).reshape(GRAPH_RES, GRAPH_RES, 3)
#%%
import matplotlib.pyplot as plt

fig = plt.figure(0)
fig.clear()

ax = fig.add_subplot(111, projection='3d')


# Loop through each 2D slice in the 3D array and plot it as a surface
for level in range(3):
    # Get the 2D slice at the current level    
    Z = rand_out_vec3s[..., level]
    
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))

    surf = ax.plot_surface(
        X, Y, Z, alpha=0.7, label=f'k={level}'
    )

plt.show()

