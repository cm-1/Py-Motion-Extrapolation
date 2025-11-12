#%% Imports and const definitions.
import typing
import pickle

import numpy as np
from numpy.typing import NDArray

from datatools.data_splitting import DataSubsetKind

from motiontools.posefeatures import HypotheticalInputsForNN

from motiontools.dataorg import UnitAwareScaler, joinArrays
from motiontools.shared_constants import *
from nn_utilities.nn_loading import loadLatestModels
from nn_utilities.nn_inference import getOutputsNN
import nn_utilities.physics_JAV_models as pjm
import posemath as pm

from gtCommon import PoseLoaderBCOT

GRAPH_RES = 50
DISP_RADIUS = 32.0
STEP = 3

NUM_BCOT_SAMPLES = 32
NUM_IDS_IF_APPLICABLE = 1

#%% Load things from disk.

scaler: UnitAwareScaler
with open("./results/models/scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

# Load the NNs that have been saved to disk; save them in a dict.
load_model_prefixes = ("JAV_MULTIPLIERS", "HOT3D_JM")
models = loadLatestModels(load_model_prefixes)

models["ConstAcc"] = pjm.ConstAccMultipliersModel()

model_prefixes = tuple(models.keys())


#%% Choose transforms for the first frames.

bcot_id_split = list(PoseLoaderBCOT.trainValidationTestByBody(0.1, 0.2, 0))

overlap = True
crit = "none"
bcot_scores_npz = np.load(
    "./results/models/scores_on_bcot.npz", allow_pickle=False
)
load_id_order = bcot_scores_npz["id_order"]
med_static_scores = bcot_scores_npz["median_static"]
med_bcot_nn_scores = bcot_scores_npz["median_JAV_MULTIPLIERS"]
med_hot3d_nn_scores = bcot_scores_npz["median_HOT3D_JM"]

relative_scores = (med_bcot_nn_scores - med_hot3d_nn_scores) / med_static_scores
sorted_score_inds = np.argsort(relative_scores, axis=1)


for i in range(3):
    bcot_id_split[i] = [
        tuple(int(i) for i in load_id_order[t])
        for t in sorted_score_inds[STEP-1, :NUM_IDS_IF_APPLICABLE]
    ]
sequences_by_subset = PoseLoaderBCOT.getGroupedSamplesByCriteria(
    *bcot_id_split, True, overlap, crit, STEP, NUM_BCOT_SAMPLES, verbose=True
)

#%% Construct object for quickly calculating outputs for hypothetical inputs.
all_statics = np.concatenate(list(sequences_by_subset.values()), axis=0)

all_last_fixed = all_statics[:, 0, LAST_FIXED_PT_IND]
all_orig_dynamic = all_statics[:, 0, DYNAMIC_PT_IND]
radii = np.linalg.norm(all_orig_dynamic - all_last_fixed, axis=-1)

max_radius = np.max(radii)
rad_margin = max_radius * 0.1
dynamic_mins = np.min(all_orig_dynamic, axis=0) - rad_margin
dynamic_maxs = np.max(all_orig_dynamic, axis=0) + rad_margin

max_mag = min(DISP_RADIUS, max_radius * 1.1)


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
        
        ArraysByDSK: typing.TypeAlias = typing.Dict[DataSubsetKind, NDArray]
        self.all_seq_nn_outs: typing.Dict[str, ArraysByDSK] = dict()

        self.all_seq_errors: typing.Dict[str, ArraysByDSK] = dict()
        self.scaled_input_storage: ArraysByDSK = dict()
        temp_hc = None
        for model_num, (model_prefix, model) in enumerate(self._models.items()):
            sub_dict: ArraysByDSK = dict()
            errs_sub_dict: ArraysByDSK = dict()
            for dsk, data in self._sequences_by_subset.items():
                # Our "data" is shaped like a list of (pts, rotation) 2-tuples.
                pts = np.swapaxes(data[:, 0], 0, 1)
                prev_pts = pts[:DYNAMIC_PT_IND]
                last_pts = pts[DYNAMIC_PT_IND]
                aas = np.swapaxes(data[:, 1], 0, 1)[:GT_PT_IND]
                rmats = pm.matsFromScaledAxisAngleArray(aas)
                if temp_hc is None:
                    temp_hc = HypotheticalInputsForNN(
                        prev_pts, rmats, STEP, self._scaler.column_keys
                    )
                    prev_pts = None
                    rmats = None
                # Efficiency-TODO: The below uses redundant temp_hc
                # calculations each time. Should just calculate the scaled
                # inputs in outer loop (iterate by DataSubsetKind before model)
                # and then just call the model's predict(...) method here!
                outs_nn, scaled_inputs = getOutputsNN(
                    model, self._scaler, temp_hc, last_pts, prev_pts, rmats,
                    "scaled"
                )
                if model_num == 0:
                    self.scaled_input_storage[dsk] = scaled_inputs
                out_segs = np.stack((outs_nn, last_pts), axis=-2)
                sub_dict[dsk] = out_segs
                errs_sub_dict[dsk] = np.linalg.norm(
                    outs_nn - pts[GT_PT_IND], axis=-1
                )

            self.all_seq_nn_outs[model_prefix] = sub_dict
            self.all_seq_errors[model_prefix] = errs_sub_dict

    def change_set(self, new_set: DataSubsetKind):
        self.subset = new_set
        self.current_sequences = self._sequences_by_subset[new_set]
    
    def infer(self, hc, pts):
        return getOutputsNN(
            self._models[self.model_prefix], self._scaler, hc, pts
        )

ps = PlottingState(sequences_by_subset, models, scaler)
init_pts, init_aas = ps.current_sequences[0].copy()

hc = HypotheticalInputsForNN(
    init_pts[:DYNAMIC_PT_IND],
    pm.matsFromScaledAxisAngleArray(init_aas[:GT_PT_IND]),
    STEP, scaler.column_keys
)

#%%
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

from plottools.plotly_gens import getColourMags, TraceManager

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
dstep = np.round(np.clip(max_mag/32, 0.1, 1.0), 1)
sliders = [
    widgets.FloatSlider(value=0, min=low, max=high, step=dstep, description=xyz)
    for low, high, xyz in zip(dynamic_mins, dynamic_maxs, "XYZ")
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

tm = TraceManager(fig)

def get_nn_ca_marker_outputs(hc: HypotheticalInputsForNN):
    main_hyp_pt = np.array([s.value for s in sliders])

    new_hyp_dyn_pts = hc.getInputGridOfVec3s(GRAPH_RES, max_mag, main_hyp_pt)

    col_mags = getColourMags(new_hyp_dyn_pts, main_hyp_pt)

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
    
    tm.clear()

    # Add background traces for all sequences in gray
    bg_fixed = joinArrays([c[0][:DYNAMIC_PT_IND + 1] for c in ps.current_sequences])
    bg_gt = joinArrays([c[0][DYNAMIC_PT_IND:] for c in ps.current_sequences])
    
    tm.set_line_trace("fixed_pts_bg", bg_fixed, color="lightgray")
    tm.set_line_trace("gt_pts_bg", bg_gt, color="gray")
    
    model_callback({"new": ps.model_prefix}, update_after=False)

    # Add main visualization traces
    seq_callback({"new": 0}) # Sequence slider index has been set to 0.
    
def seq_callback(change):
    """Called when sequence index changes"""
    idx = change['new']
    pts, aas = ps.current_sequences[idx]

    hc.updatePrecalcs(
        pts[:DYNAMIC_PT_IND], pm.matsFromScaledAxisAngleArray(aas[:GT_PT_IND])
    )
    
    # with fig.batch_update():

    # Update highlighted sequence traces only
    tm.set_line_trace("fixed_pts", pts[:DYNAMIC_PT_IND], 0, "black")
    tm.set_line_trace("gt_pts", pts[DYNAMIC_PT_IND:], DYNAMIC_PT_IND, "green")

    tm.set_line_trace(
        "nn_single_pts", get_pred_orig_dyn(hc, pts), DYNAMIC_PT_IND, "red"
    )

    lfp = pts[LAST_FIXED_PT_IND]
    acc_line = np.stack((lfp, lfp + hc.prev_acc), axis=0)
    jerk_line = np.stack((lfp, lfp + hc.prev_jerk), axis=0)
    tm.set_line_trace("prev_acc", acc_line, color="aqua")
    tm.set_line_trace("prev_jerk", jerk_line, color="magenta")

    # Update sliders to match new sequence
    set_xyz_sliders(pts[DYNAMIC_PT_IND])
    
def set_xyz_sliders(point):
    
    for i, s in enumerate(sliders):
    # Update slider values without triggering callbacks.
        s.unobserve(update_plot, names="value")
        s.value = point[i]
        s.observe(update_plot, names="value")
    update_plot(None)

def model_callback(change, update_after: bool = True):
    ps.model_prefix = (change["new"])
    nn_fixed = joinArrays(ps.all_seq_nn_outs[ps.model_prefix][ps.subset])
    tm.set_line_trace("nn_fixed", nn_fixed, color="purple")
    if update_after:
        update_plot(None)

def update_plot(value):
    """Update grids of points, e.g. when 3D slider changes."""
    new_nn_outs, new_ca_outs, disp_mags = get_nn_ca_marker_outputs(hc)
    with fig.batch_update():
        markers = tm.set_scatter_trace("nn_out", new_nn_outs, disp_mags)

        markers["colorscale"] = "Oranges"
        # constant-acc comparison
        tm.set_scatter_trace("const_acc", new_ca_outs, marker=markers)


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

