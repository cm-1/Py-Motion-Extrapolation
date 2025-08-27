#%%
import typing
import pickle
import glob

import numpy as np
from numpy.typing import NDArray
import keras

from motiontools.posefeatures import (
    CalcsForVideo, dataForPositionsJAV, JAV, getWorldFrameDisplacements,
    PositionDerivativeCollection
)
from motiontools.dataorg import UnitAwareScaler
import posemath as pm

# Needed when loading model, even though IDE syntax highlighting marks this as
# "unused"!
from nn_utilities.nn_losses import poseLossJAV

GRAPH_RES = 50
DISP_RADIUS = 32.0

scaler: UnitAwareScaler
with open("./results/models/scaler.pickle", "rb") as f:
    scaler = pickle.load(f)
selected_keys = scaler.column_keys

matching_models = glob.glob("./results/models/hot_1024*.keras")
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

#%%
# We need 7 input points to get crackle calculations because my code currently
# assumes the last one is ground truth for which it shouldn't generate any
# predictions, and we need 6 input points to calculate nonzero crackle.
rand_pts = np.random.normal(0.0, 0.0010, (7,3))# zeros((7,3))
rand_pts[:-4] = 0
origin_pt = rand_pts[-4]
origin_pt[:] = 0
last_fixed_pt = rand_pts[-3]
last_fixed_pt[:] = (15, 0, 0)

default_aas = np.ones_like(rand_pts)


from gtCommon import PoseLoaderBCOT
all_bcot_ids = PoseLoaderBCOT.getAllIDs(True)
train_bcot_ids, val_bcot_ids, test_bcot_ids = \
    PoseLoaderBCOT.trainValidationTestByBody(0.1, 0.2, 0)
test_loaders = [PoseLoaderBCOT(*v[:2]) for v in test_bcot_ids]
found_ind = -1
for tl in test_loaders:
    tl_diffs = np.diff(tl.getTranslationsGTNP(), 1, axis=0)
    tl_diff_subset = tl_diffs[:6]
    for i in range(0, len(tl_diffs) - 6 + 1):
        tl_diff_subset = tl_diffs[i:(i+6)]
        if np.sum(np.linalg.norm(tl_diff_subset[:4], axis=-1)) < 4.0:
            found_ind = i
            # break
    if found_ind >= 0:
        tl_diff_subset = tl_diffs[found_ind:(found_ind + 6)]
        default_aas = tl.getRotationsGTNP()[found_ind:(found_ind+7)]
        diff_mat = pm.getOrthonormalFrames(
            True, tl_diff_subset[-3:-2], tl_diff_subset[-2:-1]
        )[1][0]
        rand_pts = tl.getTranslationsGTNP()[found_ind:found_ind + 7]
        new_frame_pts = (rand_pts @ diff_mat.T)
        rand_pts = new_frame_pts - new_frame_pts[-3]
        break
last_fixed_pt = rand_pts[-3]
#%%

outlist = []
all_in_disp_mags = (np.arange(GRAPH_RES) / GRAPH_RES) * 2 - 1
all_in_disp_mags *= max(DISP_RADIUS, np.linalg.norm(rand_pts[-3] - rand_pts[-2]) + 5)
x_pts = last_fixed_pt[0] + all_in_disp_mags
y_pts = last_fixed_pt[1] + all_in_disp_mags
print(end="")
for xi, x in enumerate(x_pts):
    print("\rx index =", xi, end="", flush=True)
    for yi, y in enumerate(y_pts):
        rand_pts[-2, :2] = (x, y)
        outlist.append(getSingleInput(rand_pts, default_aas))
#%%
Y, X = np.meshgrid(y_pts, x_pts)
xyz_ins = np.dstack((X, Y, np.broadcast_to(last_fixed_pt[2], X.shape)))
rand_out_vec3s_ca = 3 * xyz_ins - 3 * last_fixed_pt + origin_pt
rand_out_vec3s = xyz_ins + processAllInputs(outlist).reshape(GRAPH_RES, GRAPH_RES, 3)
#%%
# import matplotlib.pyplot as plt

# fig = plt.figure(0)
# fig.clear()

# ax = fig.add_subplot(111, projection='3d')
# ax.clear()
# for level in range(3):
#     surf = ax.plot_wireframe(
#         X, Y, rand_out_vec3s[..., level], label='xyz'[level],
#         color="C" + str(level)
#     )
# surf_n = ax.plot_wireframe(
#     X, Y, np.linalg.norm(rand_out_vec3s - xyz_ins, axis=-1),
#     color="C3", label='|d|'
# )
# ax.plot(*(rand_pts[:-2].T),'x-')
# for i, pt in enumerate(rand_pts[:-2]):
#     ax.text(x=pt[0], y=pt[1], z=pt[2], s="x" + str(i))
# ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
# ax.legend()

# plt.show()

#%%
import plotly.graph_objects as go

fig = go.Figure()

# Add the three xyz surfaces

for level, name in enumerate("xyz"):
    fig.add_trace(go.Surface(
        x=X, y=Y, z=rand_out_vec3s[..., level],
        opacity=0.32, name=name, showscale=False
    ))


disp_mags = np.linalg.norm(rand_out_vec3s - xyz_ins, axis=-1)
# Add |d| surface
fig.add_trace(go.Surface(
    x=X, y=Y, z=disp_mags,
    opacity=0.32, name="|d|", showscale=False, legendrank=2
))


disp_mags2 = np.linalg.norm(xyz_ins - last_fixed_pt, axis=-1).flatten()
out_vec3s_list = rand_out_vec3s.reshape(-1, 3)
disp_cols = 20 * disp_mags2/disp_mags2.max()
fig.add_trace(go.Scatter3d(
    x=out_vec3s_list[:, 0], y=out_vec3s_list[:, 1], z=np.zeros_like(out_vec3s_list[:,0]), #out_vec3s_list[:, 2],
    mode='markers',
    marker=dict(
        size=1,
        color=disp_cols.flatten(),                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
))
out_vec3s_list = rand_out_vec3s_ca.reshape(-1, 3)
fig.add_trace(go.Scatter3d(
    x=out_vec3s_list[:, 0], y=out_vec3s_list[:, 1], z=np.zeros_like(out_vec3s_list[:,0]), #out_vec3s_list[:, 2],
    mode='markers',
    marker=dict(
        size=1,
        color=disp_cols.flatten(),                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)) 

# Add polyline and markers
fig.add_trace(go.Scatter3d(
    x=rand_pts[:-2, 0], y=rand_pts[:-2, 1], z=rand_pts[:-2, 2],
    mode="lines+markers+text",
    text=[f"x{i}" for i in range(len(rand_pts) - 2)],
    textposition="top center",
    line=dict(color="black"),
    marker=dict(size=5, symbol="x")
))

fig.update_layout(
    scene=dict(
        xaxis_title="x", yaxis_title="y", zaxis_title="z",
        aspectmode="data"
    )
)
fig.update_traces(showlegend=True)#, showscale=False)

fig.show()
