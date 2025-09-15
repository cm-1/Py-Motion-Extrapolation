#%% Imports and const definitions.
import typing
import pickle
import functools

import numpy as np
from numpy.typing import NDArray

from motiontools.posefeatures import HypotheticalInputsForNN

from motiontools.dataorg import UnitAwareScaler, joinArrays
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
sequences_by_set = dict()
for set_name, vid_ids in sequence_sets.items():
    tup_list = PoseLoaderBCOT.getStaticStartSample(
        [v[:2] for v in vid_ids], True, NUM_BCOT_SAMPLES, True
    )
    pose_info = np.stack(tup_list, axis=0)
    pose_info.setflags(write=False)
    sequences_by_set[set_name] = pose_info 

# Initialize with test set, first sequence
current_set = 'Test'
current_sequences = sequences_by_set[current_set]
current_seq_idx = 0


#%% Construct object for quickly calculating outputs for hypothetical inputs.
all_statics = np.concatenate(list(sequences_by_set.values()), axis=0)

all_last_fixed = all_statics[:, 0, LAST_FIXED_PT_IND]
all_orig_dynamic = all_statics[:, 0, DYNAMIC_PT_IND]
radii = np.linalg.norm(all_orig_dynamic - all_last_fixed, axis=-1)

max_mag = min(DISP_RADIUS, np.max(radii) * 1.1)
max_mag = 0.1


def updateHC(hc: HypotheticalInputsForNN, pts_and_aas: NDArray):
    pts, aas = pts_and_aas.copy()
    hc.updatePrecalcs(
        pts[:DYNAMIC_PT_IND], pm.matsFromScaledAxisAngleArray(aas[:GT_PT_IND])
    )

init_pts, init_aas = current_sequences[0].copy()

hc = HypotheticalInputsForNN(
    init_pts[:DYNAMIC_PT_IND],
    pm.matsFromScaledAxisAngleArray(init_aas[:GT_PT_IND]),
    1, scaler.column_keys
)


inferBCOT = functools.partial(
    getHypotheticalOutputsNN, model_loads[model_prefixes[0]], scaler
)

#%%
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display


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
             size=1, use_labels: bool = True):
    labels = None
    mode = "lines+markers"
    if use_labels:
        if np.any(np.isnan(pts)):
            raise NotImplementedError(
                "Labels for disjoint lines not supported!"
            )
        labels = [
            f"x{i}" for i in range(min_label_ind, min_label_ind + len(pts))
        ]
        mode += "+text"
    
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode=mode, name=name,
        text=labels, textposition="top center", line=dict(color=color),
        marker=dict(size=size, symbol="x"),
    )

def getScatter(name: str, pts: NDArray,
               input_color_vals: typing.Optional[NDArray] = None,
               colorscale = 'Viridis', opacity=0.8, size=1):
    
    marker_spec = getScatterMarkers(input_color_vals, colorscale, opacity, size)
    return go.Scatter3d(
        name=name, x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers',
        marker=marker_spec
    )

# Interactive controls setup
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

model_selector = widgets.ToggleButtons(
    options=model_prefixes,
    description="Model:",
    style={"description_width": "initial"}
)
#%%
fig = go.FigureWidget()
fig.update_layout(
    scene=dict(
        xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"
    )
)


current_model_name = model_prefixes[0]

def get_nn_ca_marker_outputs(hc: HypotheticalInputsForNN, form_markers: bool):
    global max_mag, GRAPH_RES
    main_hyp_pt = np.array([s.value for s in sliders])

    # with fig.batch_update():
    new_hyp_dyn_pts = hc.getInputGridOfVec3s(GRAPH_RES, max_mag, main_hyp_pt)
    col_mags = getColourMags(new_hyp_dyn_pts, main_hyp_pt)
    if form_markers:
        col_mags = getScatterMarkers(col_mags)

    # NN output from currently selected model
    new_nn_outs = inferBCOT(hc, new_hyp_dyn_pts)
    new_ca_outs = hc.getConstAccPreds(new_hyp_dyn_pts)

    return (new_nn_outs, new_ca_outs, col_mags)

def get_pred_orig_dyn(hc: HypotheticalInputsForNN, pts: NDArray):
    
    d_pt = pts[DYNAMIC_PT_IND]
    sel_out_vec3 = inferBCOT(hc, d_pt)[0]
    nn_single_vis_pts = np.stack([d_pt, sel_out_vec3], axis=0)
    return nn_single_vis_pts

def update_sequence_selector(change):
    """Update available sequences when dataset changes"""
    global current_set, current_sequences, hc
    current_set = change['new']

    current_sequences = sequences_by_set[current_set]
    
    # Update sequence selector range
    seq_selector.max = len(current_sequences) - 1
    seq_selector.value = 0
    
    # Clear and redraw all sequence traces
    fig.data = []  # Clear all traces
    
    # Add background traces for all sequences in gray
    bg_fixed_pts = joinArrays([c[0][:DYNAMIC_PT_IND] for c in current_sequences])
    bg_gt_pts = joinArrays([c[0][DYNAMIC_PT_IND:] for c in current_sequences])
    
    fig.add_trace(getLines(
        "fixed_pts_bg", bg_fixed_pts, color="lightgray", size=1,
        use_labels=False
    ))
    fig.add_trace(getLines(
        "gt_pts_bg", bg_gt_pts, DYNAMIC_PT_IND,
        color="gray", size=1, use_labels=False
    ))
    
    # Add main visualization traces
    pts = current_sequences[0][0]
    updateHC(hc, current_sequences[0])

    nn_out_list, ca, disp_cols = get_nn_ca_marker_outputs(hc, False)
    # Add scatter plots for nn_out and const_acc
    fig.add_trace(getScatter("nn_out", nn_out_list, disp_cols))
    fig.add_trace(getScatter("const_acc",ca, disp_cols))
    
    # Add highlighted sequence traces
    fig.add_trace(getLines("fixed_pts", pts[:DYNAMIC_PT_IND], color="black"))
    fig.add_trace(getLines("gt_pts", pts[DYNAMIC_PT_IND:], DYNAMIC_PT_IND, "green"))
    
    # Add single point visualization
    nn_for_orig_dyn = get_pred_orig_dyn(hc, pts)
    fig.add_trace(getLines(
        "nn_single_pts", nn_for_orig_dyn, DYNAMIC_PT_IND, "red"
    ))
    
    # Reset sliders to match new sequence
    update_sliders_from_point(pts[DYNAMIC_PT_IND])

def update_selected_sequence(change):
    """Update visualization when sequence index changes"""
    global hc
    idx = change['new']
    pts, _ = current_sequences[idx]
    
    # Update highlighted sequence traces only
    fig.update_traces(
        x=pts[:DYNAMIC_PT_IND, 0],
        y=pts[:DYNAMIC_PT_IND, 1],
        z=pts[:DYNAMIC_PT_IND, 2],
        selector={"name": "fixed_pts"}
    )
    fig.update_traces(
        x=pts[DYNAMIC_PT_IND:, 0],
        y=pts[DYNAMIC_PT_IND:, 1],
        z=pts[DYNAMIC_PT_IND:, 2],
        selector={"name": "gt_pts"}
    )
    nnps = get_pred_orig_dyn(hc, pts)
    fig.update_traces(
        x=nnps[:, 0], y=nnps[:, 1], z=nnps[:, 2],
        selector={"name": "nn_single_pts"}
    )

    # Update sliders to match new sequence
    # update_sliders_from_point(rand_pts[DYNAMIC_PT_IND])
    # This will trigger update_plot which will update nn_out and const_acc
    update_plot(None)

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
    global hc
    new_nn_outs, new_ca_outs, markers = get_nn_ca_marker_outputs(hc, True)
    fig.update_traces(
        x=new_nn_outs[:, 0], y=new_nn_outs[:, 1], z=new_nn_outs[:, 2],
        marker=markers, selector=({"name": "nn_out"})
    )

    # constant-acc comparison
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
update_sequence_selector({"new": current_set})
update_plot(None)


# Display the interactive visualization
ui = widgets.VBox([
    widgets.HBox([set_selector, seq_selector]),
    model_selector,
    fig
] + sliders)
display(ui)

