'''
Helpers for saving and reloading trained LSTM runs (LSTM scripts only; the FCNN
has its own save/load code).

A run directory contains:
 - model.keras: architecture, weights, and optimizer state.
 - meta.json: "meta" (settings that define the data and model, checked on
   load) and "info" (recorded only, e.g. the random seed; never checked).
 - arrays.npz (optional): numeric parameters, such as scaler means/scales,
   that must match the freshly computed ones when loading.

findRun() searches a directory of runs for the newest one matching the current
settings, and loadRun() refuses to load a run that doesn't match, since
evaluating a model on differently-built data would run without errors but give
wrong numbers.

The loss functions are registered with keras.saving.register_keras_serializable
in nn_losses, so that module must be imported before loadRun() is called.
'''
import enum

import datetime
import hashlib
import json
import pathlib
import typing

import numpy as np
import keras

# Output configuration:
class LSTM_OUT_MODE(enum.Enum):
    DEFAULT = 0
    JAV15 = 1
    DISP = 2

def comboHash(combos) -> str:
    '''Short hash of a collection of combo IDs, independent of their order.
    Used to detect changes to the train/val/test splits.'''
    rows = sorted(",".join(str(x) for x in c) for c in combos)
    return hashlib.sha256("|".join(rows).encode()).hexdigest()[:12]


def makeRunDir(base_dir, tag: str) -> pathlib.Path:
    '''Returns base_dir/<tag>_<timestamp>; does not create it.'''
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return pathlib.Path(base_dir) / "{}_{}".format(tag, stamp)


def saveRun(run_dir, model, meta: dict,
            arrays: typing.Optional[typing.Dict[str, np.ndarray]] = None,
            info: typing.Optional[dict] = None) -> None:
    d = pathlib.Path(run_dir)
    d.mkdir(parents=True, exist_ok=False)
    model.save(str(d / "model.keras"))
    if arrays:
        np.savez(d / "arrays.npz", **arrays)
    (d / "meta.json").write_text(
        json.dumps({"meta": meta, "info": info or {}}, indent=2)
    )
    print("Saved run to", d)


def _readMeta(d: pathlib.Path) -> typing.Optional[dict]:
    try:
        saved = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(saved, dict) or "meta" not in saved:
        return None
    saved.setdefault("info", {})
    return saved


def _mismatch(d: pathlib.Path, expected_meta: dict,
              expected_arrays: typing.Optional[typing.Dict[str, np.ndarray]],
              rtol: float, atol: float) -> typing.Optional[str]:
    '''Returns a description of why run d doesn't match, or None if it does.'''
    saved = _readMeta(d)
    if saved is None:
        return "no readable meta.json in the expected format"
    if not (d / "model.keras").exists():
        return "no model.keras"

    saved_meta = saved["meta"]
    # Round-trip through JSON so tuples/enums compare the same way as saved.
    expected = json.loads(json.dumps(expected_meta))
    diffs = {
        k: (saved_meta.get(k), expected.get(k))
        for k in saved_meta.keys() | expected.keys()
        if saved_meta.get(k) != expected.get(k)
    }
    if diffs:
        return ("settings differ from current ones "
                "(key: (saved, current)): {}".format(diffs))

    if expected_arrays:
        arrays_path = d / "arrays.npz"
        if not arrays_path.exists():
            return "no arrays.npz to compare against"
        with np.load(arrays_path) as saved_arrays:
            for k, cur in expected_arrays.items():
                if k not in saved_arrays:
                    return "missing array '{}'".format(k)
                old = saved_arrays[k]
                cur = np.asarray(cur)
                if old.shape != cur.shape or not np.allclose(
                        old, cur, rtol=rtol, atol=atol):
                    return ("array '{}' differs from the current one "
                            "(saved {}, current {})".format(k, old, cur))
    return None


def findRun(base_dir, expected_meta: dict,
            expected_arrays: typing.Optional[typing.Dict[str, np.ndarray]] = None,
            seed: typing.Optional[int] = None,
            rtol: float = 1e-6, atol: float = 1e-8
            ) -> typing.Optional[pathlib.Path]:
    '''Returns the newest run in base_dir whose saved settings (and arrays)
    match the expected ones, or None if there isn't one. If seed is not None,
    only runs trained with that seed are considered.'''
    base = pathlib.Path(base_dir)
    if not base.is_dir():
        return None
    matches = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if _mismatch(d, expected_meta, expected_arrays, rtol, atol) is not None:
            continue
        if seed is not None and _readMeta(d)["info"].get("seed") != seed:
            continue
        matches.append(d)
    if not matches:
        return None
    return max(matches, key=lambda d: (d / "model.keras").stat().st_mtime)


def loadRun(run_dir, load_scaled: bool, expected_meta: dict,
            expected_arrays: typing.Optional[typing.Dict[str, np.ndarray]] = None,
            rtol: float = 1e-6, atol: float = 1e-8):
    '''Loads the model from run_dir, raising ValueError if the saved meta or
    arrays differ from the expected ones.'''
    d = pathlib.Path(run_dir)
    reason = _mismatch(d, expected_meta, expected_arrays, rtol, atol)
    if reason is not None:
        raise ValueError("Saved run {} doesn't match: {}".format(d, reason))
    print("Loading run from {} (info: {})".format(d, _readMeta(d)["info"]))
    scale_str = "scaled_" if load_scaled else ""
    return keras.models.load_model(str(d / "{}model.keras".format(scale_str)))

def exportWorldFrameModel(model, scaler, win_len, save_path=None, dims=3):
    '''
    Wraps a trained world-frame model (USE_VEL_IN=False, CHOSEN_OUT=DISP, no
    context) so that it takes raw positions (mm) and returns the next raw
    position (mm), with all scaling inside the model. Works for uniform and
    per-axis scalers. If save_path is given, the wrapper is saved there (a
    .keras path). Returns the wrapper.
    '''
    scale_vec = np.asarray(scaler.scale_, dtype=np.float32)
    mean = np.asarray(scaler.mean_, dtype=np.float32)

    # In DISP mode the scaled target is (next - last) / scale per axis, so the
    # mean cancels: the real displacement is just d * scale (no mean added).
    raw_in = keras.Input((win_len, dims), name="positions")
    x = keras.layers.Normalization(
        axis=-1, mean=mean, variance=scale_vec ** 2
    )(raw_in)
    d = model(x)
    d_mm = keras.layers.Rescaling(scale_vec.tolist())(d)
    next_pos = raw_in[:, -1, :] + d_mm
    export_model = keras.Model(raw_in, next_pos, name="world_frame_export")

    if save_path is not None:
        export_model.save(str(save_path))
    return export_model