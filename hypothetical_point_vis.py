#%% Imports and const definitions.
import typing
import pickle
import functools

import numpy as np
from numpy.typing import NDArray

from motiontools.posefeatures import HypotheticalInputsForNN

from motiontools.dataorg import UnitAwareScaler
from motiontools.shared_constants import *
from nn_utilities.nn_loading import loadLatestModels
from nn_utilities.nn_inference import getHypotheticalOutputsNN
import posemath as pm

GRAPH_RES = 50
DISP_RADIUS = 32.0

#%% Load things from disk.

# Load the unit scaler that's been saved to disk.
scaler: UnitAwareScaler
with open("./results/models/scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

# Load the NNs that have been saved to disk; save them in a dict.
model_prefixes = ("JAV_MULTIPLIER", "HOT3D_JM")
model_loads = loadLatestModels(model_prefixes)


#%% Choose transforms for the first frames.
from gtCommon import PoseLoaderBCOT
_, __, test_bcot_ids = PoseLoaderBCOT.trainValidationTestByBody(0.1, 0.2, 0)

rand_pts, default_aas = PoseLoaderBCOT.getStaticStartSample(
    [v[:2] for v in test_bcot_ids], True
)

rand_pts.setflags(write=False)
default_aas.setflags(write=False)

last_fixed_pt = rand_pts[LAST_FIXED_PT_IND]
orig_dynamic_pt = rand_pts[DYNAMIC_PT_IND]

#%% Construct object for quickly calculating outputs for hypothetical inputs.
hc = HypotheticalInputsForNN(
    rand_pts[:DYNAMIC_PT_IND].copy(),
    pm.matsFromScaledAxisAngleArray(default_aas[:GT_PT_IND].copy()),
    1, scaler.column_keys
)

max_mag = min(
    DISP_RADIUS, np.linalg.norm(orig_dynamic_pt - last_fixed_pt) * 1.1
)

inferBCOT = functools.partial(
    getHypotheticalOutputsNN, model_loads[model_prefixes[0]], scaler
)

hyp_dyn_pts_list = hc.getInputGridOfVec3s(GRAPH_RES, max_mag, orig_dynamic_pt)

nn_out_list = inferBCOT(hc, hyp_dyn_pts_list)

#%%
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

fig = go.FigureWidget()

def getColourMags(input_pts: NDArray, ref_pt: NDArray):
    mags = np.linalg.norm(input_pts - ref_pt, axis=-1).flatten()
    return mags / np.max(mags)

def getScatterMarkers(input_color_vals: typing.Optional[NDArray] = None,
                      colorscale = 'Viridis', opacity=0.8, size=1):
    marker_spec = dict(size=size, opacity=opacity)
    if input_color_vals is not None:
        marker_spec["color"] = input_color_vals.flatten()
        marker_spec["colorscale"] = colorscale 
    return marker_spec

def getLines(name: str, pts: NDArray, min_label_ind: int = 0, color="black",
             size=5):
    labels = [f"x{i}" for i in range(min_label_ind, min_label_ind + len(pts))]
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines+markers+text",
        name=name, text=labels, textposition="top center",
        line=dict(color=color), marker=dict(size=size, symbol="x"),
    )

def getScatter(name: str, pts: NDArray,
               input_color_vals: typing.Optional[NDArray] = None,
               colorscale = 'Viridis', opacity=0.8, size=1):
    
    marker_spec = getScatterMarkers(input_color_vals, colorscale, opacity, size)
    return go.Scatter3d(
        name=name, x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers',
        marker=marker_spec
    )

disp_cols = getColourMags(hyp_dyn_pts_list, last_fixed_pt)
fig.add_trace(getScatter("nn_out", nn_out_list, disp_cols))

fig.add_trace(
    getScatter("const_acc", hc.getConstAccPreds(hyp_dyn_pts_list), disp_cols)
) 

# Add polyline and markers
fig.add_trace(getLines("fixed_pts", rand_pts[:DYNAMIC_PT_IND]))
fig.add_trace(
    getLines("gt_pts", rand_pts[DYNAMIC_PT_IND:], DYNAMIC_PT_IND, "green")
)
sel_out_vec3 = inferBCOT(hc, orig_dynamic_pt)[0]
nn_single_vis_pts = np.stack([rand_pts[DYNAMIC_PT_IND], sel_out_vec3], axis=0)
fig.add_trace(
    getLines("nn_single_pts", nn_single_vis_pts, DYNAMIC_PT_IND, "red")
)

fig.update_layout(
    scene=dict(
        xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"
    )
)

x_slider = widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description="X")
y_slider = widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description="Y")
z_slider = widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description="Z")

rand_pts_copy = rand_pts.copy()

model_selector = widgets.ToggleButtons(
    options=model_prefixes,
    description="Model:",
    style={"description_width": "initial"}
)

# mutable reference to current inference function
current_model_name = model_prefixes[0]

def set_model(change):
    global inferBCOT, current_model_name
    current_model_name = change["new"]
    inferBCOT = functools.partial(
        getHypotheticalOutputsNN, model_loads[current_model_name], scaler
    )
    # force a redraw with new model immediately
    update_plot(None)

model_selector.observe(set_model, names="value")

def update_plot(value):
    """When sliders move, update selected point + recompute x6 with selected model."""
    idx = DYNAMIC_PT_IND
    rand_pts_copy[idx] = np.array([x_slider.value, y_slider.value, z_slider.value])
    main_hyp_pt = rand_pts_copy[idx]

    # with fig.batch_update():
    new_hyp_dyn_pts = hc.getInputGridOfVec3s(GRAPH_RES, max_mag, main_hyp_pt)
    col_mags = getColourMags(new_hyp_dyn_pts, main_hyp_pt)
    markers = getScatterMarkers(col_mags)

    # NN output from currently selected model
    new_nn_outs = inferBCOT(hc, new_hyp_dyn_pts)
    fig.update_traces(
        x=new_nn_outs[:, 0], y=new_nn_outs[:, 1], z=new_nn_outs[:, 2],
        marker=markers, selector=({"name": "nn_out"})
    )

    # constant-acc comparison
    new_ca_outs = hc.getConstAccPreds(new_hyp_dyn_pts)
    fig.update_traces(
        x=new_ca_outs[:, 0], y=new_ca_outs[:, 1], z=new_ca_outs[:, 2],
        marker=markers, selector=({"name": "const_acc"})
    )
    

for s in (x_slider, y_slider, z_slider):
    s.observe(update_plot, names="value")

# Initialize sliders with point 0
update_plot(None)


ui = widgets.VBox([model_selector, fig, x_slider, y_slider, z_slider])
display(ui)

# fig.update_traces(showlegend=True)#, showscale=False)


#%% Matplotlib surface plotter.
import matplotlib.pyplot as plt

hyp_dyn_pts_grid = hyp_dyn_pts_list.reshape(GRAPH_RES, GRAPH_RES, 3)
nn_out_grid = nn_out_list.reshape(GRAPH_RES, GRAPH_RES, 3)

mfig = plt.figure(0)
mfig.clear()

ax = mfig.add_subplot(111, projection='3d')
ax.clear()
for level in range(3):
    surf = ax.plot_wireframe(
        hyp_dyn_pts_grid[..., 0], hyp_dyn_pts_grid[..., 1], nn_out_grid[..., level], label='xyz'[level],
        color="C" + str(level)
    )
surf_n = ax.plot_wireframe(
    hyp_dyn_pts_grid[..., 0], hyp_dyn_pts_grid[..., 1],
    np.linalg.norm(nn_out_grid - last_fixed_pt, axis=-1),
    color="C3", label='|d|'
)
ax.plot(*(rand_pts[:-2].T),'x-')
for i, pt in enumerate(rand_pts[:-2]):
    ax.text(x=pt[0], y=pt[1], z=pt[2], s="x" + str(i))
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.legend()

plt.show()
