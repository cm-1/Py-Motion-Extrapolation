#%% Imports and const definitions.
import typing
import pickle
import functools

import numpy as np
from numpy.typing import NDArray

from datatools.data_splitting import DataSubsetKind

from motiontools.posefeatures import HypotheticalInputsForNN

from motiontools.dataorg import UnitAwareScaler, joinArrays
from motiontools.shared_constants import *
from nn_utilities.nn_loading import loadLatestModels
from nn_utilities.nn_inference import getHypotheticalOutputsNN
import posemath as pm

from gtCommon import PoseLoaderBCOT

GRAPH_RES = 50
DISP_RADIUS = 32.0

NUM_BCOT_SAMPLES = 64

#%% Load things from disk.

# Load the unit scaler that's been saved to disk.
scaler: UnitAwareScaler
with open("./results/models/scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

# Load the NNs that have been saved to disk; save them in a dict.
model_prefixes = ("JAV_MULTIPLIER", "HOT3D_JM")
model_loads = loadLatestModels(model_prefixes)


#%% Choose transforms for the first frames.

# Get the video IDs for each set
bcot_id_split = PoseLoaderBCOT.trainValidationTestByBody(0.1, 0.2, 0)

sequences_by_set = PoseLoaderBCOT.getGroupedStaticStartSamples(
    *bcot_id_split, True, NUM_BCOT_SAMPLES, True
) 



#%% Construct object for quickly calculating outputs for hypothetical inputs.
all_statics = np.concatenate(list(sequences_by_set.values()), axis=0)

all_last_fixed = all_statics[:, 0, LAST_FIXED_PT_IND]
all_orig_dynamic = all_statics[:, 0, DYNAMIC_PT_IND]
radii = np.linalg.norm(all_orig_dynamic - all_last_fixed, axis=-1)

max_mag = min(DISP_RADIUS, np.max(radii) * 1.1)
# max_mag = 0.1


# Initialize with test set, first sequence
current_set = DataSubsetKind.TEST
current_sequences = sequences_by_set[current_set]
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

from plottools.plotly_gens import (
    getLines, getScatter, getColourMags, getScatterMarkers
)

#%%
# Interactive controls setup
# Add set selector and sequence selector before the existing controls
set_selector = widgets.RadioButtons(
    options=[(k.name, k) for k in DataSubsetKind.nonWholeValues()],
    value=current_set,
    description='Dataset:',
    style={'description_width': 'initial'}
)

# Create sequence selector - will update its max value based on set selection
seq_selector = widgets.IntSlider(
    value=0,
    min=0,
    max=len(sequences_by_set[current_set]) - 1,  # Initial max for test set
    step=1,
    description='Sequence:',
    style={'description_width': 'initial'}
)

print("TODO: Update slider min and max based on currently selected sequences!")
sliders = [
    widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description=axis)
    for axis in "XYZ"
]

model_selector = widgets.ToggleButtons(
    options=model_prefixes,
    description="Model:",
    style={"description_width": "initial"}
)

current_model_name = model_prefixes[0]

fig = go.FigureWidget()
fig.update_layout(
    # autosize=False,
    width=700, height=700,
    scene=dict(
        xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"
    )
)

def get_nn_ca_marker_outputs(hc: HypotheticalInputsForNN, form_markers: bool):
    global max_mag, GRAPH_RES
    main_hyp_pt = np.array([s.value for s in sliders])

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
    bg_fixed_pts = joinArrays([c[0][:DYNAMIC_PT_IND + 1] for c in current_sequences])
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
    pts, aas = current_sequences[0]
    hc.updatePrecalcs(
        pts[:DYNAMIC_PT_IND], pm.matsFromScaledAxisAngleArray(aas[:GT_PT_IND])
    )

    nn_out_list, ca, disp_cols = get_nn_ca_marker_outputs(hc, False)
    # Add scatter plots for nn_out and const_acc
    fig.add_trace(getScatter("nn_out", nn_out_list, disp_cols))
    fig.add_trace(getScatter("const_acc",ca, disp_cols, "Oranges"))
    # fig.add_trace(getLines("nn_out", nn_out_list, color="cyan", size=0, use_labels=False))
    # fig.add_trace(getLines("const_acc", ca, color="orange", size=0, use_labels=False))
    
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

def update_trace_args(vec3s: NDArray, name: str):
    return dict(
        x=vec3s[:, 0], y=vec3s[:, 1], z=vec3s[:, 2], selector={"name": name}
    )

def update_trace_pts(fig, vec3s: NDArray, name: str):
    fig.update_traces(**update_trace_args(vec3s, name))

def update_trace_pts_markers(fig, vec3s: NDArray, name: str, markers):
    d = update_trace_args(vec3s, name)
    d.update(marker=markers)
    fig.update_traces(**d)

def update_selected_sequence(change):
    """Update visualization when sequence index changes"""
    global hc
    idx = change['new']
    pts, _ = current_sequences[idx]
    
    # with fig.batch_update():

    # Update highlighted sequence traces only
    update_trace_pts(fig, pts[:DYNAMIC_PT_IND], "fixed_pts")
    update_trace_pts(fig, pts[DYNAMIC_PT_IND:], "gt_pts")

    nnps = get_pred_orig_dyn(hc, pts)
    update_trace_pts(fig, nnps, "nn_single_pts")

    # Update sliders to match new sequence
    update_sliders_from_point(pts[DYNAMIC_PT_IND])
    # This will trigger update_plot which will update nn_out and const_acc
    # update_plot(None)

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
    # with fig.batch_update():
    update_trace_pts_markers(fig, new_nn_outs, "nn_out", markers)

    markers["colorscale"] = "Oranges"
    # constant-acc comparison
    update_trace_pts_markers(fig, new_ca_outs, "const_acc", markers)


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

