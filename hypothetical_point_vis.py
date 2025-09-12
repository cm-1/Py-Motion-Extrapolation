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

NUM_BCOT_SAMPLES = 32

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

# Get the video IDs for each set
train_bcot_ids, val_bcot_ids, test_bcot_ids = PoseLoaderBCOT.trainValidationTestByBody(0.1, 0.2, 0)

# Dictionary to store sequences from each set
sequence_sets = {
    'Train': [v[:2] for v in train_bcot_ids],
    'Validation': [v[:2] for v in val_bcot_ids],
    'Test': [v[:2] for v in test_bcot_ids]
}

# Get sequences for each set
sequences_by_set = {
    set_name: PoseLoaderBCOT.getStaticStartSample([v[
        :2] for v in vid_ids], True, NUM_BCOT_SAMPLES, True
    )
    for set_name, vid_ids in sequence_sets.items()
}

# Initialize with test set, first sequence
current_set = 'Test'
current_sequences = sequences_by_set[current_set]
current_seq_idx = 0

rand_pts, default_aas = current_sequences[0]  # Start with first sequence
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
        marker_spec.update(
            {"color": input_color_vals.flatten(), "colorscale": colorscale}
        )
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

#%% Interactive controls setup
# Add set selector and sequence selector before the existing controls
set_selector = widgets.RadioButtons(
    options=['Train', 'Validation', 'Test'],
    value='Test',
    description='Dataset:',
    style={'description_width': 'initial'}
)

# Create sequence selector - will update its max value based on set selection
seq_selector = widgets.IntSlider(
    value=0,
    min=0,
    max=len(sequences_by_set['Test']) - 1,  # Initial max for test set
    step=1,
    description='Sequence:',
    style={'description_width': 'initial'}
)


sliders = [
    widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description=axis)
    for axis in "XYZ"
]

rand_pts_copy = rand_pts.copy()

model_selector = widgets.ToggleButtons(
    options=model_prefixes,
    description="Model:",
    style={"description_width": "initial"}
)

current_model_name = model_prefixes[0]

def update_sequence_selector(change):
    """Update available sequences when dataset changes"""
    global current_set, current_sequences, rand_pts, default_aas, last_fixed_pt
    current_set = change['new']

    current_sequences = sequences_by_set[current_set]
    
    # Update sequence selector range
    seq_selector.max = len(current_sequences) - 1
    seq_selector.value = 0
    
    # Clear and redraw all sequence traces
    fig.data = []  # Clear all traces
    
    # Add background traces for all sequences in gray
    for pts, _ in current_sequences:
        fig.add_trace(getLines("fixed_pts_bg", pts[:DYNAMIC_PT_IND], color="lightgray"))
        fig.add_trace(getLines("gt_pts_bg", pts[DYNAMIC_PT_IND:], DYNAMIC_PT_IND, "lightgray"))
    
    # Add main visualization traces
    rand_pts, default_aas = current_sequences[0]
    last_fixed_pt = rand_pts[LAST_FIXED_PT_IND]
    
    # Add scatter plots for nn_out and const_acc
    disp_cols = getColourMags(hyp_dyn_pts_list, last_fixed_pt)
    fig.add_trace(getScatter("nn_out", nn_out_list, disp_cols))
    fig.add_trace(getScatter("const_acc", hc.getConstAccPreds(hyp_dyn_pts_list), disp_cols))
    
    # Add highlighted sequence traces
    fig.add_trace(getLines("fixed_pts", rand_pts[:DYNAMIC_PT_IND], color="black"))
    fig.add_trace(getLines("gt_pts", rand_pts[DYNAMIC_PT_IND:], DYNAMIC_PT_IND, "green"))
    
    # Add single point visualization
    sel_out_vec3 = inferBCOT(hc, rand_pts[DYNAMIC_PT_IND])[0]
    nn_single_vis_pts = np.stack([rand_pts[DYNAMIC_PT_IND], sel_out_vec3], axis=0)
    fig.add_trace(getLines("nn_single_pts", nn_single_vis_pts, DYNAMIC_PT_IND, "red"))
    
    # Reset sliders to match new sequence
    update_sliders_from_point(rand_pts[DYNAMIC_PT_IND])

def update_selected_sequence(change):
    """Update visualization when sequence index changes"""
    global rand_pts, default_aas, last_fixed_pt
    idx = change['new']
    rand_pts, default_aas = current_sequences[idx]
    last_fixed_pt = rand_pts[LAST_FIXED_PT_IND]
    
    # Update highlighted sequence traces only
    fig.update_traces(
        x=rand_pts[:DYNAMIC_PT_IND, 0],
        y=rand_pts[:DYNAMIC_PT_IND, 1],
        z=rand_pts[:DYNAMIC_PT_IND, 2],
        selector={"name": "fixed_pts"}
    )
    fig.update_traces(
        x=rand_pts[DYNAMIC_PT_IND:, 0],
        y=rand_pts[DYNAMIC_PT_IND:, 1],
        z=rand_pts[DYNAMIC_PT_IND:, 2],
        selector={"name": "gt_pts"}
    )
    
    # Update sliders to match new sequence
    update_sliders_from_point(rand_pts[DYNAMIC_PT_IND])
    # This will trigger update_plot which will update nn_out and const_acc

def update_sliders_from_point(point):
    """Update slider values without triggering callbacks"""
    for i in range(3):
        sliders[i].value = point[i]
    update_plot(None)

def set_model(change):
    global inferBCOT, current_model_name
    current_model_name = change["new"]
    inferBCOT = functools.partial(
        getHypotheticalOutputsNN, model_loads[current_model_name], scaler
    )
    update_plot(None)


def update_plot(value):
    """When sliders move, update selected point + recompute x6 with selected model."""
    idx = DYNAMIC_PT_IND
    rand_pts_copy[idx] = np.array([s.value for s in sliders])
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

# Connect callbacks    
set_selector.observe(update_sequence_selector, names='value')
seq_selector.observe(update_selected_sequence, names='value')
model_selector.observe(set_model, names="value")
for s in sliders:
    s.observe(update_plot, names="value")

# Initialize sliders with point 0
update_plot(None)


# Display the interactive visualization
ui = widgets.VBox([
    widgets.HBox([set_selector, seq_selector]),
    model_selector,
    fig
] + sliders)
display(ui)

