################################################################################
# Velocity-Aligned-Frame Networks
################################################################################
import json
import typing

import numpy as np
from numpy.typing import NDArray

# StandardScaler seems to perform better than others, like MinMax.
from sklearn.preprocessing import StandardScaler

import rnn_models as rnnm

from nn_utilities.nn_losses import *
from nn_utilities.lstm_io import (
    saveRun, loadRun, findRun, makeRunDir, comboHash, LSTM_OUT_MODE,
    exportWorldFrameModel, listRuns
)

# Local code imports ===========================================================
# For reading the dataset into numpy arrays:
import gtCommon as gtc
# Functions to get RNN "windows" and velocity-aligned frame data, respectively.
from motiontools.posefeatures import dataForCombosJAV
# Functions to get all combos and to split combos into train/test sets:
from motiontools.posefeatures import JAV # Enum
from data_by_combo_functions import UnscaledDistanceLogger # Custom callback
from data_by_combo_functions import rnnDataWindows, scaleWindows

# Input configuration:
WIN_SIZE = 6
USE_CONTEXT = False
USE_VEL_IN = False


# This variable was controlling whether the 15-column JAV was used as input.
# However, if re-enabling, would have to reduce window size so that total num
# of train/test samples is consistent. Or pad it somehow...
ONE_FRAME_PER_WINDOW: bool = True


CHOSEN_OUT = LSTM_OUT_MODE.DISP

# General configuration:
FORCE_VEL_SCALING = True

# How many frames to skip when going over the pose dataset; skipping more frames
# simulates larger motions or smaller fps. Typical values are 0, 1, 2.
DATASET_SKIP_FRAMES = 2

APPLY_NOISE = False

# Saving/loading trained models (see nn_utilities/lstm_io.py). Newly trained
# models are saved under SAVE_RUNS_DIR (None disables saving and loading). If
# LOAD_SAVED_RUN is True and a saved run there matches the current settings,
# it's loaded instead of training. If several match, the newest is used (or the
# newest one trained with SEED, if SEED isn't None).
LOAD_SAVED_RUN = True
SAVE_RUNS_DIR = "results/lstms"

# Random seed for model initialization and training. None picks a new seed each
# run; it's printed, included in the run's name, and saved with the run.
SEED = None

# If True, also saves an export wrapper (export_model.keras, in the run's
# directory) that takes raw positions in mm and returns the next position in mm,
# with the scaling inside the model. See exportWorldFrameModel. Only valid for
# world-frame, DISP-output, no-context models.
EXPORT_MODEL = True

print("Getting train-test split.")
all_combos = gtc.PoseLoaderBCOT.trainValidationTestByBody(test_ratio=0.2, random_seed=0)
train_combos = gtc.PoseLoaderBCOT.prepIDsForConstructor(all_combos[0])
val_combos = gtc.PoseLoaderBCOT.prepIDsForConstructor(all_combos[1])
test_combos = gtc.PoseLoaderBCOT.prepIDsForConstructor(all_combos[2])

#%%
################################################################################
# READING THE DATA
################################################################################

# From the combo 3-tuples, we get nametuple versions containing only the
# uniquely-identifying parts. Some functions expect this instead of the 3-tuple. 
# We also get a list of pose loaders, one for each BCOT video.
combos, loaders = gtc.PoseLoaderBCOT.getAllMinimalIDsAndLoaders(True)

print("Reading/normalizing data.")
# As with the "vanilla" regression network, we'll get our data in a velocity
# -aligned frame. But this time, we'll start by keeping things separated by 
# combo so that we know the starting and ending points for a video's data and,
# thus, can create our data windows without having a window erroneously overlap
# two separate videos.
all_jav, w2ls, jav_translations = dataForCombosJAV(
    loaders, (JAV.VELOCITY, JAV.ACCELERATION, JAV.JERK), APPLY_NOISE, True, True
)

#%% 
################################################################################
# CALCULATING THE DATA WINDOWS
################################################################################

print("Calculating data's sliding \"windows\" for the network.")

jav_lstm_skip = DATASET_SKIP_FRAMES # Renaming var.

train_jav_in: NDArray
train_jav_out: NDArray
test_jav_in: NDArray
test_jav_out: NDArray
val_jav_in: NDArray
val_jav_out: NDArray

def getWindowsForVelLSTM(combo_subset, use_vel_frame: bool):
    w2ls_param = None
    centring_param = False
    if use_vel_frame:
        w2ls_param = w2ls[jav_lstm_skip]
        centring_param = True
    return rnnDataWindows(
        jav_translations[jav_lstm_skip], combo_subset, WIN_SIZE,
        0, # Always a window "skip" of 0 in this case because the frame is rotating...
        None, w2ls_param, centring_param
    )
def getWindowsJAV(combo_subset, window_size = WIN_SIZE):
    return rnnDataWindows(
        all_jav[jav_lstm_skip], combo_subset, window_size, 
        0, # Always a window "skip" of 0 in this case because the frame is rotating...
        None
    )

if not ONE_FRAME_PER_WINDOW:
    # TODO: try an architecture that uses JAV vec15s as input. However, a window
    # size of W would need either >w prev positions known or some padding....
    raise NotImplementedError("ONE_FRAME_PER_WINDOW=False is not implemented yet.")

train_jav_in, train_jav_out = getWindowsForVelLSTM(train_combos, USE_VEL_IN)
test_jav_in, test_jav_out = getWindowsForVelLSTM(test_combos, USE_VEL_IN)
val_jav_in, val_jav_out = getWindowsForVelLSTM(val_combos, USE_VEL_IN)
train_velframe_in = train_jav_in
if not USE_VEL_IN and FORCE_VEL_SCALING:
    train_velframe_in, _ = getWindowsForVelLSTM(train_combos, True)
if CHOSEN_OUT == LSTM_OUT_MODE.JAV15:
    _, train_jav_out = getWindowsJAV(train_combos, 2)
    _, test_jav_out = getWindowsJAV(test_combos, 2)
    _, val_jav_out = getWindowsJAV(val_combos, 2)
#%%
################################################################################
# NORMALIZING THE DATA
################################################################################

# For world-frame translations, would like to just use *all* translations to
# set scaler so that scaler is same regardless of skip-sampling.
# For velocity frame, this is not possible. Instead, would like to get all the
# values that the neural network will see from the training data.
# To avoid duplicates, this would be the end of every window plus... nvm, I
# realized just adding the front of window [0] won't cut it because this won't
# include the first frames for any video but the first, nor will it include the
# final "out" frames... depending on how the paper goes, this is something I'll
# need to fix eventually but there's no point spending time to fix it in advance
# now, since it will barely change the scaling and these results may not even
# make it into the paper. The most memory-efficient fix that I can think of is
# to have the RNN window function return the indices of windows where new videos
# start and then use those indices to get necessary slices.


# Rather than concatenate all data for scaling into a numpy array of the
# required (n, d) shape, it's more memory-efficient to just pass in a list
# of old arrays or slices, and then use partial_fit for those. 
def scalerForWindows(translation_groups, dim):
    ret = StandardScaler()
    new_shape = (-1, dim)
    ret.fit(np.reshape(translation_groups[0], new_shape))
    for tg in translation_groups[1:]:
        ret.partial_fit(np.reshape(tg, new_shape))
    return ret

ts_vel = [train_velframe_in[:, -1], train_velframe_in[0, :-1]]
ts_world = [jav_translations[0][c] for c in train_combos]
# all_rotations: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
# all_rotation_mats: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
# for combo in train_combos:
#     calculator = gtc.PoseLoaderBCOT(combo[0], combo[1], 0)
#     ts_world.append(calculator.getTranslationsGTNP())
#%%
ts_main = ts_vel if USE_VEL_IN else ts_world
ts_mag = ts_vel if FORCE_VEL_SCALING else ts_main

temp_mag_scaler = scalerForWindows(ts_mag, dim=1)
scaler_obj = scalerForWindows(ts_main, dim=3)

scaler_obj.scale_[-3:] = temp_mag_scaler.scale_
if USE_VEL_IN:
    scaler_obj.mean_[-3:] = 0.0

# Keep the unscaled windows (train, val, test) so that the data can be re-scaled
# with a different scaler; see the input-scale sweep at the end of this file.
raw_jav_ins = (train_jav_in, val_jav_in, test_jav_in)
raw_jav_outs = (train_jav_out, val_jav_out, test_jav_out)

def scaleDataSplits(scaler):
    '''Returns the scaled [train, val, test] inputs and the scaled [train, val]
    targets for the given scaler, built from the unscaled windows. The test
    inputs must be scaled to be fed to the model, but the test targets are
    deliberately NOT scaled: they stay in mm (test_gt_mm), and predictions are
    converted to mm (see predictionsMM) before being compared with them.'''
    ins = [scaleWindows(x, scaler, ensureCopy=True) for x in raw_jav_ins]
    outs = list(raw_jav_outs[:2])
    if CHOSEN_OUT != LSTM_OUT_MODE.JAV15:
        outs = [scaler.transform(o) for o in outs]
        if CHOSEN_OUT == LSTM_OUT_MODE.DISP:
            outs = [o - i[:, -1] for o, i in zip(outs, ins[:2])]
    return ins, outs

(train_jav_in, val_jav_in, test_jav_in), \
    (train_jav_out, val_jav_out) = scaleDataSplits(scaler_obj)
# Test ground truth, in mm (for DISP, the next position, not the displacement).
test_gt_mm = raw_jav_outs[2]

# The context must use the same frame as rnnDataWindows applies to each window,
# i.e. w2ls index (i + WIN_SIZE - shift) for window i, where shift is the number
# of leading frames that w2ls lacks relative to the translations. The "4" here
# encodes that shift, so the first window's frame is w2ls[WIN_SIZE - 4].
# If the shift ever changes, the length assertion below will fail.
w2l_trim = WIN_SIZE - 4
curr_step_w2ls = w2ls[jav_lstm_skip]
def getContextData(combo_subset):
    concat_w2ls = np.concatenate([curr_step_w2ls[c][w2l_trim:] for c in combo_subset])
    concat_w2ls = concat_w2ls.reshape(-1, 9)
    return concat_w2ls

# The context doesn't depend on the input scaling, so it's computed only once.
contexts = (None, None, None)
if USE_CONTEXT:
    train_context = getContextData(train_combos)
    val_context = getContextData(val_combos)
    test_context = getContextData(test_combos)
    assert len(train_context) == len(train_jav_out)
    assert len(val_context) == len(val_jav_out)
    assert len(test_context) == len(test_gt_mm)
    contexts = (train_context, val_context, test_context)

def packInputs(jav_ins):
    '''Turns [train, val, test] scaled windows into the model's inputs (adding
    the context if used).'''
    if not USE_CONTEXT:
        return list(jav_ins)
    return list(zip(jav_ins, contexts))

train_in, val_in, test_in = packInputs((train_jav_in, val_jav_in, test_jav_in))
#%%
################################################################################
# BUILDING THE NETWORK
################################################################################
print("Building the NN.")

_mode_is_jav15 = (CHOSEN_OUT == LSTM_OUT_MODE.JAV15)
out_dims = 12 if _mode_is_jav15 else 3
loss_to_use = poseLossJAV if _mode_is_jav15 else poseLossVec3
input_win_len = WIN_SIZE
if USE_VEL_IN:
    input_win_len = WIN_SIZE - 1
LSTM_UNITS = (128, 128, 128)

def buildLSTMModel():
    if USE_CONTEXT:
        return rnnm.get_frame_context_lstm(
            loss_to_use, input_win_len, 9, units=LSTM_UNITS, out_size=out_dims
        )
    return rnnm.get_simple_lstm(
        loss_to_use, out_dims, 'adam', units=LSTM_UNITS
    )

# Settings that define the data and model. Saved with each run and checked when
# loading one. (The random seed is recorded with the run but not checked.)
run_meta = {
    "script": "lstms_translation",
    "win_size": WIN_SIZE,
    "skip": DATASET_SKIP_FRAMES,
    "use_vel_in": USE_VEL_IN,
    "use_context": USE_CONTEXT,
    "out_mode": CHOSEN_OUT.name,
    "force_vel_scaling": FORCE_VEL_SCALING,
    "apply_noise": APPLY_NOISE,
    "lstm_units": list(LSTM_UNITS),
    # Hashes of the video IDs in each split, to catch changes to the splitting.
    "train_split": comboHash(train_combos),
    "val_split": comboHash(val_combos),
    "test_split": comboHash(test_combos),
}
# Scaler parameters; also checked when loading a run.
run_arrays = {"scaler_mean": scaler_obj.mean_, "scaler_scale": scaler_obj.scale_}

loaded_run_dir = None
if LOAD_SAVED_RUN and SAVE_RUNS_DIR:
    loaded_run_dir = findRun(SAVE_RUNS_DIR, run_meta, run_arrays, seed=SEED)
    if loaded_run_dir is None:
        print("No saved run matches the current settings; training a new model.")

if loaded_run_dir:
    jav_lstm_model = loadRun(loaded_run_dir, False, run_meta, run_arrays)
else:
    if SEED is None:
        SEED = int(np.random.default_rng().integers(2**31))
    keras.utils.set_random_seed(SEED)
    print("Random seed:", SEED)

    _frame_str = "vel" if USE_VEL_IN else "world"
    _ctx_str = "_ctx" if USE_CONTEXT else ""
    _units_str = "x".join(str(u) for u in LSTM_UNITS)
    run_tag = "trans_{}in_skip{}_{}{}_u{}_s{}".format(
        _frame_str, DATASET_SKIP_FRAMES, CHOSEN_OUT.name.lower(), _ctx_str,
        _units_str, SEED
    )

    jav_lstm_model = buildLSTMModel()

print("Note: The default reported \"loss\" will not be in millimeters, because training data was normalized.")
print("Actual MAE in millimeters will be printed explicitly/separately.")

#%%
################################################################################
# TRAINING THE NETWORK
################################################################################

def fitLSTM(model, tr_in, tr_out, va_in, va_out, scaler):
    '''Trains model on the given (already scaled) data. Returns the Keras
    history and the logger, which records the training loss in mm.'''
    # Log the loss in mm, rather than just scaled coordinates.
    logger_scaler = None if _mode_is_jav15 else scaler
    logger = UnscaledDistanceLogger(
        tr_in, tr_out, logger_scaler, loss_func=loss_to_use
    )
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=7,
                                          restore_best_weights=True)
    plateau = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    hist = model.fit(
        tr_in, tr_out, epochs=500, validation_data=[va_in, va_out],
        callbacks=[logger, early, plateau]
    )
    return hist, logger

run_dir = loaded_run_dir # Stays None if a new model is trained but not saved.
if not loaded_run_dir:
    jav_lstm_hist, jav_lstm_logger = fitLSTM(
        jav_lstm_model, train_in, train_jav_out, val_in, val_jav_out, scaler_obj
    )
    if SAVE_RUNS_DIR:
        run_dir = makeRunDir(SAVE_RUNS_DIR, run_tag)
        saveRun(
            run_dir, jav_lstm_model, run_meta, run_arrays, info={"seed": SEED}
        )

#%%
################################################################################
# EXPORTING THE NETWORK
################################################################################

def unrollLSTMs(model, win_len, dims=3):
    cfg = model.get_config()
    for lc in cfg["layers"]:
        if lc["class_name"] == "LSTM":
            lc["config"]["unroll"] = True
    new_model = keras.Sequential.from_config(cfg)
    new_model.build((None, win_len, dims))
    new_model.set_weights(model.get_weights())
    return new_model

if EXPORT_MODEL:
    if USE_VEL_IN or USE_CONTEXT or CHOSEN_OUT != LSTM_OUT_MODE.DISP:
        raise NotImplementedError(
            "Export is only implemented for world-frame, DISP, no-context models."
        )
    if run_dir is None:
        raise ValueError("EXPORT_MODEL needs SAVE_RUNS_DIR set, to know where to save.")
    export_model = exportWorldFrameModel(unrollLSTMs(
        jav_lstm_model, 6, 3), scaler_obj, input_win_len, run_dir / "scaled_model.keras"
    )

#%%
################################################################################
# EVALUATING THE NETWORK
################################################################################



jav_lstm_test_pred = jav_lstm_model.predict(test_in)

raw_test_jav_in = raw_jav_ins[2]

def unscalePredictionsToMM(scaler, scaled_preds, raw_test_in):
    '''Runs model on the (scaled) test inputs and returns its predictions in mm,
    the same units as test_gt_mm. raw_test_in is the unscaled test windows, needed
    to turn predicted displacements into positions.'''
    if CHOSEN_OUT == LSTM_OUT_MODE.JAV15:
        return scaled_preds # Already unscaled.
    if CHOSEN_OUT == LSTM_OUT_MODE.DISP:
        # Scaled displacement d -> mm is d * scale (the mean cancels).
        return raw_test_in[:, -1] + scaled_preds * scaler.scale_
    return scaler.inverse_transform(scaled_preds)

mm_test_preds = unscalePredictionsToMM(
    scaler_obj, jav_lstm_test_pred, raw_test_jav_in
)

# Constant-acceleration baseline, computed directly on the unscaled windows.
const_acc_pred = np.zeros(12)
const_acc_pred[[0, 1, 5]] = 1.0
if CHOSEN_OUT != LSTM_OUT_MODE.JAV15:
    if not USE_VEL_IN:
        const_acc_pred = 3 * test_jav_in[:, -1] - 3 * test_jav_in[:, -2] + test_jav_in[:, -3]
        if CHOSEN_OUT == LSTM_OUT_MODE.DISP:
            const_acc_pred = const_acc_pred - test_jav_in[:, -1]
    else:
        scaler_at_0 = -3 * scaler_obj.mean_ / scaler_obj.scale_
        const_acc_pred = scaler_at_0 -3 * test_jav_in[:, -1, -3:] + test_jav_in[:, -2, -3:]
        if CHOSEN_OUT == LSTM_OUT_MODE.DISP:
            const_acc_pred = const_acc_pred - test_jav_in[:, -1]

const_acc_pred_mm = unscalePredictionsToMM(
    scaler_obj, const_acc_pred, raw_test_jav_in
)
const_acc_errs = loss_to_use(test_gt_mm, const_acc_pred_mm)
jav_lstm_test_errs = loss_to_use(test_gt_mm, mm_test_preds)
print("\n\nAcc score (millimeters) on test data:", np.mean(const_acc_errs))
print("\n\nJAV LSTM score (millimeters) on test data:", np.mean(jav_lstm_test_errs))
print("Above score is for skip{} data.".format(jav_lstm_skip))
#%%

################################################################################
# Train loss graph.
################################################################################
import matplotlib.pyplot as plt

if not loaded_run_dir: # No training curve when a saved model was loaded.
    plt.plot(jav_lstm_logger.mean_distances)

    plt.xlabel("Epoch")
    plt.ylabel("MAE (mm)")
    plt.title("Loss for skip{} dataset train".format(jav_lstm_skip))
    plt.grid(True)
    plt.show()

print("Done program.")

#%%
################################################################################
# INPUT-SCALE SENSITIVITY SWEEP
################################################################################
# Trains (or loads) one LSTM per value in SWEEP_SCALES (per seed in
# SWEEP_SEEDS). Each uses the same data and settings as the main model, except
# that the scaler's scale_ (the number the inputs are divided by) is replaced by
# the sweep value, for all three axes; the means are those of scaler_obj. Runs
# are saved to SWEEP_RUNS_DIR, with the scale in meta.json ("input_scale") and
# the test MAE in mm in "info" ("test_mae_mm"), so that loadSweepResults() can
# rebuild the table for a graph later without loading any models.
import copy

RUN_SCALE_SWEEP = True
SWEEP_RUNS_DIR = "results/lstms_scale_sweep"
# Scale values (mm per scaled unit). Here, multiples of the ground-truth scale
# (1.0 is the normal model); any array of positive numbers works.
# scaler_naive = scalerForWindows(ts_main, dim=1)
SWEEP_SCALES = [2.9754, 25.2044, 100, 200, 300, 400, 500]
# A fixed seed per model keeps the comparison between scales fair. Add more
# seeds to see the run-to-run variation.
SWEEP_SEEDS = [0]

def scalerWithScale(base_scaler, scale):
    s = copy.deepcopy(base_scaler)
    s.scale_ = np.full_like(base_scaler.scale_, scale)
    s.var_ = s.scale_ ** 2
    return s

def testErrorMM(model, scaler, test_inputs):
    unscaledPreds = model.predict(test_inputs)
    predsMM = unscalePredictionsToMM(scaler, unscaledPreds, raw_test_jav_in)
    return float(np.mean(loss_to_use(test_gt_mm, predsMM)))

def loadSweepResults(runs_dir=SWEEP_RUNS_DIR):
    '''Returns a list of (input_scale, seed, test_mae_mm, run_dir) for the
    saved sweep runs whose other settings match the current ones.'''
    rows = []
    for d, saved_meta, info in listRuns(runs_dir):
        others = {k: v for k, v in saved_meta.items() if k != "input_scale"}
        if "input_scale" not in saved_meta or others != json.loads(json.dumps(run_meta)):
            continue
        rows.append((saved_meta["input_scale"], info.get("seed"),
                     info.get("test_mae_mm"), d))
    return sorted(rows, key=lambda r: (r[0], r[1] if r[1] is not None else -1))

if RUN_SCALE_SWEEP:
    for sweep_seed in SWEEP_SEEDS:
        for scale_ind, sweep_scale in enumerate(SWEEP_SCALES):
            sweep_scale = float(sweep_scale)
            print("\n=== Run {}, Input scale {:.6g} (seed {}) ===".format(
                scale_ind + 1, sweep_scale, sweep_seed))
            sweep_scaler = scalerWithScale(scaler_obj, sweep_scale)
            sweep_meta = dict(run_meta, input_scale=sweep_scale)
            sweep_arrays = {"scaler_mean": sweep_scaler.mean_,
                            "scaler_scale": sweep_scaler.scale_}

            sw_ins, sw_outs = scaleDataSplits(sweep_scaler)
            sw_train_in, sw_val_in, sw_test_in = packInputs(sw_ins)

            sweep_run = None
            if LOAD_SAVED_RUN:
                sweep_run = findRun(SWEEP_RUNS_DIR, sweep_meta, sweep_arrays,
                                    seed=sweep_seed)
            if sweep_run:
                sweep_model = loadRun(sweep_run, False, sweep_meta, sweep_arrays)
                sweep_mae = testErrorMM(sweep_model, sweep_scaler, sw_test_in)
            else:
                keras.utils.set_random_seed(sweep_seed)
                sweep_model = buildLSTMModel()
                fitLSTM(sweep_model, sw_train_in, sw_outs[0], sw_val_in,
                        sw_outs[1], sweep_scaler)
                sweep_mae = testErrorMM(sweep_model, sweep_scaler, sw_test_in)
                saveRun(
                    makeRunDir(SWEEP_RUNS_DIR, "trans_scalesweep_{:.4g}_s{}".format(
                        sweep_scale, sweep_seed)),
                    sweep_model, sweep_meta, sweep_arrays,
                    info={"seed": sweep_seed, "input_scale": sweep_scale,
                          "test_mae_mm": sweep_mae}
                )
            print("Test MAE (mm): {:.4f}".format(sweep_mae))
            del sweep_model, sw_ins, sw_outs, sw_train_in, sw_val_in, sw_test_in
            keras.backend.clear_session()

    # Summary table and quick plot, from what's on disk.
    sweep_rows = loadSweepResults()
    print("\nscale (mm)   seed   test MAE (mm)")
    for sc_, sd_, mae_, _ in sweep_rows:
        print("{:10.5g}   {}   {:.4f}".format(sc_, sd_, mae_))
    if sweep_rows:
        plt.plot([r[0] for r in sweep_rows], [r[2] for r in sweep_rows], "o-")
        plt.axhline(np.mean(const_acc_errs), ls="--", c="gray",
                    label="Const. acceleration")
        plt.xlabel("Input scale (mm per scaled unit)")
        plt.ylabel("Test MAE (mm)")
        plt.title("LSTM sensitivity to input scale (skip{})".format(jav_lstm_skip))
        plt.legend()
        plt.grid(True)
        plt.show()
