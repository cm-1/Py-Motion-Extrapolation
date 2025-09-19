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

scaler: UnitAwareScaler
with open("./results/models/scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

# Load the NNs that have been saved to disk; save them in a dict.
model_prefixes = ("JAV_MULTIPLIER", "HOT3D_JM")
models = loadLatestModels(model_prefixes)


#%% Choose transforms for the first frames.

bcot_id_split = PoseLoaderBCOT.trainValidationTestByBody(0.1, 0.2, 0)

sequences_by_subset = PoseLoaderBCOT.getGroupedStaticStartSamples(
    *bcot_id_split, True, NUM_BCOT_SAMPLES, True
) 



#%% Construct object for quickly calculating outputs for hypothetical inputs.
all_statics = np.concatenate(list(sequences_by_subset.values()), axis=0)

all_last_fixed = all_statics[:, 0, LAST_FIXED_PT_IND]
all_orig_dynamic = all_statics[:, 0, DYNAMIC_PT_IND]
radii = np.linalg.norm(all_orig_dynamic - all_last_fixed, axis=-1)

max_mag = min(DISP_RADIUS, np.max(radii) * 1.1)
# max_mag = 0.1


class PlottingState:
    def __init__(self,
                 sequences_by_subset: typing.Dict[DataSubsetKind, NDArray],
                 models: typing.Dict[str, typing.Any], scaler: UnitAwareScaler):
        # Default initialization values
        self.subset = DataSubsetKind.TEST
        self.model_prefix: str = next(iter(models.keys()))
        
        self._sequences_by_subset = sequences_by_subset
        self.current_sequences = sequences_by_subset[self.subset]
        self._models = models
        self._scaler = scaler

        self.infer = functools.partial(
            getHypotheticalOutputsNN,
            models[self.model_prefix], scaler
        )

    def change_set(self, new_set: DataSubsetKind):
        self.subset = new_set
        self.current_sequences = self._sequences_by_subset[new_set]
    
    def change_model(self, model_prefix: str):
        self.model_prefix = model_prefix
        ps.infer = functools.partial(
            getHypotheticalOutputsNN, self._models[model_prefix], self._scaler
        )


ps = PlottingState(sequences_by_subset, models, scaler)
init_pts, init_aas = ps.current_sequences[0].copy()

hc = HypotheticalInputsForNN(
    init_pts[:DYNAMIC_PT_IND],
    pm.matsFromScaledAxisAngleArray(init_aas[:GT_PT_IND]),
    1, scaler.column_keys
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
# Add subset selector and sequence selector
subset_selector = widgets.RadioButtons(
    options=[(k.name, k) for k in DataSubsetKind.nonWholeValues()],
    value=ps.subset, description='Dataset:',
    style={'description_width': 'initial'}
)

# Create sequence selector - will update its max value based on subset selection
seq_selector = widgets.IntSlider(
    value=0, step=1, description='Sequence:',
    min=0, max=len(sequences_by_subset[ps.subset]) - 1, # Max for initial subset
    style={'description_width': 'initial'}
)

print("TODO: Update slider min and max based on currently selected sequences!")
sliders = [
    widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description=axis)
    for axis in "XYZ"
]

model_selector = widgets.ToggleButtons(
    options=model_prefixes, value=model_prefixes[0], description="Model:",
    style={"description_width": "initial"}
)

fig = go.FigureWidget()
fig.update_layout(
    # autosize=False,
    width=700, height=700,
    scene=dict(
        xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"
    )
)

def get_nn_ca_marker_outputs(hc: HypotheticalInputsForNN, form_markers: bool):
    main_hyp_pt = np.array([s.value for s in sliders])

    new_hyp_dyn_pts = hc.getInputGridOfVec3s(GRAPH_RES, max_mag, main_hyp_pt)
    col_mags = getColourMags(new_hyp_dyn_pts, main_hyp_pt)
    if form_markers:
        col_mags = getScatterMarkers(col_mags)

    # NN output from currently selected model
    new_nn_outs = ps.infer(hc, new_hyp_dyn_pts)

    new_ca_outs = hc.getConstAccPreds(new_hyp_dyn_pts)

    return (new_nn_outs, new_ca_outs, col_mags)

def get_pred_orig_dyn(hc: HypotheticalInputsForNN, pts: NDArray):
    
    d_pt = pts[DYNAMIC_PT_IND]
    sel_out_vec3 = ps.infer(hc, d_pt)[0]
    return np.stack([d_pt, sel_out_vec3], axis=0)

def subset_callback(change):
    """Called when dataset subset changes"""
    ps.change_set(change['new'])
    
    # Update sequence selector range
    seq_selector.max = len(ps.current_sequences) - 1
    seq_selector.value = 0
    
    fig.data = []  # Clear all traces
    
    # Add background traces for all sequences in gray
    bg_fixed = joinArrays([c[0][:DYNAMIC_PT_IND + 1] for c in ps.current_sequences])
    bg_gt = joinArrays([c[0][DYNAMIC_PT_IND:] for c in ps.current_sequences])
    
    fig.add_trace(getLines("fixed_pts_bg", bg_fixed, color="lightgray"))
    fig.add_trace(getLines("gt_pts_bg", bg_gt, color="gray"))
    
    # Add main visualization traces
    pts, aas = ps.current_sequences[0]
    hc.updatePrecalcs(
        pts[:DYNAMIC_PT_IND], pm.matsFromScaledAxisAngleArray(aas[:GT_PT_IND])
    )

    nn_outs, ca_outs, disp_cols = get_nn_ca_marker_outputs(hc, False)
    # Add scatter plots for nn_out and const_acc
    fig.add_trace(getScatter("nn_out", nn_outs, disp_cols))
    fig.add_trace(getScatter("const_acc", ca_outs, disp_cols, "Oranges"))
    
    # Add highlighted sequence traces
    fig.add_trace(getLines("fixed_pts", pts[:DYNAMIC_PT_IND], 0, "black"))
    fig.add_trace(
        getLines("gt_pts", pts[DYNAMIC_PT_IND:], DYNAMIC_PT_IND, "green")
    )
    
    # Add single point visualization
    fig.add_trace(getLines(
        "nn_single_pts", get_pred_orig_dyn(hc, pts), DYNAMIC_PT_IND, "red"
    ))
    
    # Reset sliders to match new sequence
    set_xyz_sliders(pts[DYNAMIC_PT_IND])

def update_trace_pts(fig, vec3s: NDArray, name: str, **kwargs):
    fig.update_traces(
        x=vec3s[:, 0], y=vec3s[:, 1], z=vec3s[:, 2], selector={"name": name},
        **kwargs
    )
    
def seq_callback(change):
    """Called when sequence index changes"""
    idx = change['new']
    pts, _ = ps.current_sequences[idx]
    
    # with fig.batch_update():

    # Update highlighted sequence traces only
    update_trace_pts(fig, pts[:DYNAMIC_PT_IND], "fixed_pts")
    update_trace_pts(fig, pts[DYNAMIC_PT_IND:], "gt_pts")

    nnps = get_pred_orig_dyn(hc, pts)
    update_trace_pts(fig, nnps, "nn_single_pts")

    # Update sliders to match new sequence
    set_xyz_sliders(pts[DYNAMIC_PT_IND])
    
def set_xyz_sliders(point):
    
    for i, s in enumerate(sliders):
    # Update slider values without triggering callbacks.
        s.unobserve(update_plot, names="value")
        s.value = point[i]
        s.observe(update_plot, names="value")
    update_plot(None)

def model_callback(change):
    ps.change_model(change["new"])
    update_plot(None)

def update_plot(value):
    """Update grids of points, e.g. when 3D slider changes."""
    new_nn_outs, new_ca_outs, markers = get_nn_ca_marker_outputs(hc, True)
    # with fig.batch_update():
    update_trace_pts(fig, new_nn_outs, "nn_out", marker=markers)

    markers["colorscale"] = "Oranges"
    # constant-acc comparison
    update_trace_pts(fig, new_ca_outs, "const_acc", marker=markers)


# Connect callbacks   
subset_selector.observe(subset_callback, names='value')
seq_selector.observe(seq_callback, names='value')
model_selector.observe(model_callback, names="value")
for s in sliders:
    s.observe(update_plot, names="value")

# Initialize plot for default selections.
subset_callback({"new": ps.subset})

# Display the interactive visualization
ui = widgets.VBox([
    widgets.HBox([subset_selector, seq_selector]),
    model_selector,
    fig
] + sliders)
display(ui)

