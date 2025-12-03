#%%
################################################################################
# Vanilla Regression Network Code!
################################################################################
import typing

import datetime

import numpy as np
from numpy.typing import NDArray

import tensorflow as tf
import keras

from sklearn.preprocessing import StandardScaler

# Local code imports ===========================================================

# For reading the dataset into numpy arrays:
from gtCommon import PoseLoaderBCOT
import gtCommon as gtc

import posemath as pm # Small "library" I wrote for vector operations.

import motiontools.shared_constants

from nn_utilities.nn_modes import OutVecMode

from nn_utilities.nn_losses import (
    poseLossJAV, poseLossVec3
)
from nn_utilities.nn_loading import loadLatestModels
from nn_utilities.data_cache import (
    gen_cache_key, save_cached_data, load_cached_data, CacheKeys
)

from datatools.data_splitting import DataSubsetKind

# Stuff needed for calculating the input features for the non-RNN models.
# MOTION_DATA is an enum representing input feature column "names", while
# MOTION_MODEL is an enum that represents some physical non-ML motion prediction
# schemes like constant-velocity, constant-acceleration, etc.
from motiontools.key_and_vec_specs import (
    MOTION_DATA, MOTION_MODEL, ANG_OR_MAG,          # Enums
    SpecifiedMotionData, OneHotMotionData,          # Other "labels" for columns
)

from motiontools.posefeatures import (
    JAV,                                            # Enum
    OrderForJAV, NumpyForSkipAndID,                 # Type alias
    dataForCombosJAV,                               # Functions
    getWorldFrameDisplacements,
    gtMultipliers6, getBaselineJAV6
)

from motiontools.dataorg import (
    DataOrganizer, RowsAndColsHandler, UnitAwareScaler, SkipSubsetKind,
    concatForComboSubset
)

# End of imports
# ==============================================================================

# Global parameters.
TRAIN_NEW_MODEL = False


print("Starting to load data!")
dog = DataOrganizer.load(PoseLoaderBCOT) # Load our data.

bcot_test_ids = dog.subset_ids[DataSubsetKind.TEST] # Find test data subset.

# We will start off the neural net code by constructing a regression network
# that predicts multipliers for velocity, acceleration, and jerk that we will
# use to construct the displacement from the current position to the position
# we predict for the next timestamp.


def poseLossAngle(y_true, y_pred):
    # 2acos(abs([cos||x/2|| cos||y/2|| + <x>*<y> sin||x/2|| sin||y/2||))
    ang_true = np.linalg.norm(y_true, axis=-1) + 0.00000001
    ang_pred = np.linalg.norm(y_pred, axis=-1) + 0.00000001
    half_ang_true = ang_true / 2
    half_ang_pred = ang_pred / 2
    cos_true = np.cos(half_ang_true)
    cos_pred = np.cos(half_ang_pred)
    sin_true = np.sin(half_ang_true)
    sin_pred = np.sin(half_ang_pred)
    vec3_dots = pm.einsumDot(y_true, y_pred)
    unit_vec3_dots = vec3_dots / (ang_true * ang_pred)
    quat_dots = cos_true * cos_pred + unit_vec3_dots * sin_true * sin_pred
    quat_dots_1 = np.abs(np.clip(quat_dots, -1, 1))
    return 2 * np.arccos(quat_dots_1) #tf.convert_to_tensor(...)


#%%
# A lot of the features we calculated might be collinear (especially since a lot 
# of very similar features were tried for the decision tree) we'll remove the
# collinear ones before training.
colin_thresh = 0.7 # Threshold for collinearity.

nonco_cols, co_mat = pm.non_collinear_features(
    dog.concat_train_data, colin_thresh
)
def indsForKeysMD(keysMD: typing.List[MOTION_DATA]):
    ret = []
    for k in keysMD:
        f = -1
        try:
            f = dog.motion_data_keys.index(k)
        except ValueError:
            print("Key", k.name, "not found.")
        if f >= 0:
            ret.append(f)
    return ret

timestamp_ind = dog.motion_data_keys.index(MOTION_DATA.TIMESTAMP)
framenum_ind = dog.motion_data_keys.index(MOTION_DATA.FRAME_NUM)
onehot_inds = [
    i for i, k in enumerate(dog.motion_data_keys)
    if isinstance(k, OneHotMotionData)
]
GT_inds = [
    i for i, k in enumerate(dog.motion_data_keys)
    if len(k.name) == 3 and k.name[:2] == "GT"
]
ang_inds = [
    i for i, k in enumerate(dog.motion_data_keys)
    if isinstance(k, SpecifiedMotionData) and k.ang_or_mag == ANG_OR_MAG.ANG
]
bidir_inds = [
    i for i, k in enumerate(dog.motion_data_keys)
    if isinstance(k, SpecifiedMotionData) and k.bidirectional
]
circ_vec3_inds = [
    i for i, k in enumerate(dog.motion_data_keys)
    if isinstance(k, SpecifiedMotionData) and k.base_cat.name[:4].upper() == "CIRC"
]
veld_ra_inds = [
    i for i, k in enumerate(dog.motion_data_keys)
    if isinstance(k, SpecifiedMotionData) and k.axis == MOTION_DATA.VEL_DEG2_VEC3
]
veld2_dot_inds = [
    i for i, k in enumerate(dog.motion_data_keys)
    if "DEG2_DOT" in k.name.upper()
]
# plane_ra_inds = [
#     i for i, k in enumerate(dog.motion_data_keys)
#     if isinstance(k, SpecifiedMotionData) and k.axis == OTHER_DIRECTION.PLANE_ORTHO
# ]
all_circ_inds = [
    i for i, k in enumerate(dog.motion_data_keys) if "CIRC" in k.name.upper()
]
all_timescaled_inds = [
    i for i, k in enumerate(dog.motion_data_keys) if "TIMESCALED" in k.name.upper()
]
misc_rem_inds = indsForKeysMD([
    MOTION_DATA.INV_VEL_BCS_RATIOS, MOTION_DATA.RAD_DIFF,
    MOTION_DATA.SPEED_ACC_RATIO, MOTION_DATA.DISP_MAG_DIFF,
    MOTION_DATA.PLANE_NORMAL_DOT, MOTION_DATA.SPEED_ORTHO_ACC_RATIO
])

nonco_cols[:] = True
nonco_cols[timestamp_ind] = False # Current frame number seems... unhelpful.
nonco_cols[framenum_ind] = False

broad_exclusions = onehot_inds + GT_inds + ang_inds + bidir_inds 
broad_exclusions += circ_vec3_inds + all_circ_inds + all_timescaled_inds
broad_exclusions += veld_ra_inds + veld2_dot_inds

nonco_cols[broad_exclusions] = False
nonco_cols[misc_rem_inds] = False
nonco_cols[[
    i for i, k in enumerate(dog.motion_data_keys)
    if isinstance(k, MOTION_DATA) and k != MOTION_DATA.TIMESTEP
]] = False
# nonco_cols[plane_ra_inds] = False


AVD2_KEY = SpecifiedMotionData(
    MOTION_DATA.ACC_VEC3, MOTION_DATA.VEL_DEG1_VEC3, ANG_OR_MAG.ANG, False, True
)

bounce_ang_key = SpecifiedMotionData(
    MOTION_DATA.VEL_DEG1_VEC3, MOTION_DATA.VEL_DEG1_VEC3, ANG_OR_MAG.ANG,
    False, True
)

col_sub_keys = [
    AVD2_KEY, bounce_ang_key, MOTION_DATA.VEL_BCS_RATIOS,
    MOTION_DATA.CIRC_ACC, MOTION_DATA.DISP_MAG_DIFF, MOTION_DATA.TIMESTEP,
    MOTION_DATA.DISP_MAG_RATIO
]
# col_indices = indsForKeysMD(col_sub_keys)
# nonco_cols[col_indices] = True

nonco_col_nums = np.where(nonco_cols)[0]
# Get the column names for each of the kept columns.
nonco_col_ks = [k for i, k in enumerate(dog.motion_data_keys) if nonco_cols[i]]
nonco_featnames = np.array([k.name for k in nonco_col_ks])

# select_cols = np.where(nonco_cols)[0][[0, 1, 2, 3, 13, 26, 27]]
# nonco_cols[:] = False
# nonco_cols[list(select_cols)] = True


# Custom importance weighting layer suggested/described "in theory" by a friend.
# Then, I had the class written by ChatGPT and manually verified.
# (But I'm not a big tensorflow expert, so maybe my verification was faulty...)
class ImportanceLayer(keras.layers.Layer):
    def __init__(self, input_dim, weight_decay=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.weight_decay = weight_decay

    def build(self, input_shape):
        # Define per-feature weights with L2 regularization (weight decay)
        self.importance_weights = self.add_weight(
            shape=(self.input_dim,), 
            initializer="ones",  # Start with all weights = 1
            regularizer=keras.regularizers.l2(self.weight_decay),
            trainable=True
        )

    def call(self, inputs):
        return inputs * self.importance_weights  # Element-wise multiplication


WORLD_VEC_MODES = (OutVecMode.WORLD_VEC3, OutVecMode.WORLD_DISP)
ROT_VEC_MODES = (
    OutVecMode.ROT_AA, OutVecMode.ROT_VEL_AA, OutVecMode.ROT_FIXED_AX
)
#%%
chosen_mode = OutVecMode.JAV_MULTIPLIERS

def getUntrainedNN(in_dim: int, out_dim: int, loss = None, n_layers: int = 3,
                   nodes_per_layer: int = 128):

    dropout_rate = 0.2
    vel_nn_activation = 'sigmoid' # Works better than relu for this NN.
    in_shape = in_dim # + (6 if use_resid_data else 0)

    make_dense = lambda num_nodes = nodes_per_layer: keras.layers.Dense(
        num_nodes, activation=vel_nn_activation #, kernel_initializer=initer
    )

    in_layer = keras.layers.Input((in_shape,))

    # in_layer = keras.layers.GaussianNoise(0.99)(in_layer)

    x = make_dense()(in_layer)
    for _ in range(n_layers - 1):
        x = keras.layers.Dropout(dropout_rate)(x)
        x = make_dense()(x)
    
    out = keras.layers.Dense(out_dim)(x)

    model = keras.Model(inputs = in_layer, outputs = out)

    if loss is None:
        loss = poseLossJAV

    model.summary()
    optim = 'adam'
    model.compile(loss=loss, optimizer=optim)
    return model

sel_dim = 12
sel_loss = poseLossJAV
if chosen_mode != OutVecMode.JAV_MULTIPLIERS:
    if chosen_mode == OutVecMode.ROT_FIXED_AX:
        sel_dim = 1
        sel_loss = 'mae'
    else:
        sel_dim = 3
        sel_loss = 'mse' if chosen_mode in ROT_VEC_MODES else poseLossVec3 
    
JAV_order = (JAV.JERK, JAV.ACCELERATION, JAV.VELOCITY)[::-1]

# # Convert from numpy array to tf tensor.
# bcs_train = tf.convert_to_tensor(bcs_train, dtype=tf.float32)
# bcs_test = tf.convert_to_tensor(bcs_test, dtype=tf.float32)

# Z-scale each column to standard normal distribution.
bcs_scaler = UnitAwareScaler(nonco_col_ks) #, False)

class DataForJAV:
    class _CacheHelper(typing.NamedTuple):
        w2ls_JAV: NumpyForSkipAndID
        translations_JAV: NumpyForSkipAndID
        rmatsv9: NumpyForSkipAndID
        aas_JAV: NumpyForSkipAndID
        rot_vels_JAV: NumpyForSkipAndID
        prev_vel_axes: NumpyForSkipAndID
        concatted_ids: typing.List[typing.Any]

    def __init__(self, data_organizer: DataOrganizer, bcs_scaler, 
                 col_inds: NDArray, JAV_order: OrderForJAV,
                 outVecMode: OutVecMode,
                 skip: typing.Union[int,SkipSubsetKind] = SkipSubsetKind._all,
                 *, save_data_for_conf: bool = False,
                 use_cache: bool = False):

        self.data_organizer = data_organizer
        self.data_organizer.setPickAndTransform(
            col_inds, bcs_scaler, free_orig_mem=True
        )
        _dog = self.data_organizer
        self._val_start = len(_dog.col_subset_train)
        self._test_start = self._val_start + len(_dog.col_subset_validation)

        self._gt_arrs: typing.Dict[DataSubsetKind, NDArray] = {}
        self._in_arrs: typing.Dict[DataSubsetKind, NDArray] = {}

        self.translationScaler = None

        self.outVecMode = outVecMode
        self.pos_scale = 0.0
        self.rot_scale = 0.0
        if isinstance(bcs_scaler, UnitAwareScaler):
            self.pos_scale = bcs_scaler.pos_scale
            self.rot_scale = bcs_scaler.rot_scale
        elif outVecMode != OutVecMode.JAV_MULTIPLIERS:
            raise NotImplementedError(
                "Non-unit-aware scaler not yet supported!"
            )

        # save_data_for_conf |= self.outVecMode in WORLD_VEC_MODES
        _rot_align = self.outVecMode == OutVecMode.ROT_ALIGNED_VEC3 
        _rot_output = self.outVecMode in ROT_VEC_MODES
        _is_world_vec_mode = (self.outVecMode in WORLD_VEC_MODES)
        need_more_cols = _is_world_vec_mode or _rot_output or _rot_align
        # need_rot_vecs = _rot_align or self.outVecMode in WORLD_VEC_MODES
        # need_rot_vecs |= _rot_output

        needed_keys: typing.List[CacheKeys] = []
        if need_more_cols:
            _extra_cols_key = CacheKeys.EXTRA_NO_ROT_ALIGN_COLS_KEY
            if _rot_align:
                _extra_cols_key = CacheKeys.EXTRA_ROT_ALIGN_COLS_KEY
            needed_keys.append(_extra_cols_key)
        if self.outVecMode == OutVecMode.ROT_FIXED_AX:
            needed_keys += [
                CacheKeys.PREV_VEL_AX_KEY, CacheKeys.PREV_ANG_KEY,
                CacheKeys.GT_ROT_VELS_KEY
            ]
        if save_data_for_conf:
            needed_keys.append(CacheKeys.W2L_MATS_KEY)

        # Generate cache key for expensive operations
        cache_key = None
        cached_data: typing.Dict[CacheKeys, NDArray] = {}
        all_gt_data: typing.Dict[OutVecMode, NDArray] = {}
        cache_available = False
        if use_cache:
            print("Generating cache key....")
            cache_key = gen_cache_key(
                data_organizer.LoaderClass.datasetName(),
                data_organizer.subset_ids
            )
            print("Cache key:", cache_key)

            for nk in needed_keys:
                cached_data[nk] = load_thing(dog.supported_skips, etc)
            # Disable cache loading until we fix it.
            # _cache_load = load_cached_data(
            #     cache_key, outVecMode, self.pos_scale, self.rot_scale
            # )
            # if _cache_load is not None:
            #     print(f"Using cached data (key: {cache_key})")
            #     cache_available = True
            #     cached_data = _cache_load
            print("TODO: skip filtering in cache load.")

        if cache_available:
            raise NotImplementedError("Still need to work on cache loading!")
        else:
            # Because we'll be relying on the DataOrganizer to create our cache,
            # and because we want the cache to be complete, we'll need to make
            # sure we didn't load our DataOrganizer with only a subset of our
            # data. Because the cache key checks for video IDs, at this point we
            # just need to check for the thing independent of that: the skip.
            # So we test to make sure we support all possible skip values.
            if len(set(_dog.supported_skips)) != len(SkipSubsetKind):
                raise Exception(
                    "Shouldn't create cache with skip-filtered load!"
                )
            all_true_ids = _dog.LoaderClass.prepIDsForConstructor(_dog.getAllIDs())
            loaders = [_dog.LoaderClass(*true_id) for true_id in all_true_ids] 
            jav_res = dataForCombosJAV(
                loaders, JAV_order, True, True, True, True
            )

            # In this section, we'll be creating some values that we only really
            # need if we're creating the to-cache data on our first run.
            # These values begin with "i_" for "init"
            jav_per_combo = jav_res[0]
            i_w2ls_JAV = typing.cast(NumpyForSkipAndID, jav_res[1])
            i_translations_JAV = typing.cast(NumpyForSkipAndID, jav_res[2])
            r_mats = typing.cast(NumpyForSkipAndID, jav_res[3])
            i_rmatsv9 = [
                {k: x.reshape(-1, 9) for k, x in d.items()} for d in r_mats
            ]
            i_aas_JAV = [
                {k: pm.axisAngleFromMatArray(x) for k, x in d.items()}
                for d in r_mats
            ]
            i_rot_vels_JAV = typing.cast(NumpyForSkipAndID, jav_res[4])
            
            prev_angs = [
                {
                    k: np.linalg.norm(v, axis=-1, keepdims=True)
                    for k, v in d.items()
                }
                for d in i_rot_vels_JAV
            ]
            i_prev_vel_axes = [
                {
                    k: pm.safelyNormalizeArray(v, prev_angs[i][k])
                    for k, v in d.items()
                }
                for i, d in enumerate(i_rot_vels_JAV)
            ]
            
            self._cache_help = DataForJAV._CacheHelper(
                i_w2ls_JAV, i_translations_JAV, i_rmatsv9, i_aas_JAV,
                i_rot_vels_JAV, i_prev_vel_axes, self.data_organizer.getAllIDs()
            )

            self._fitWorldScaler(self.pos_scale)

            _prev_vel_ax_concat = self._worldvec_helper(
                1, diff_order=0, scale=1.0, use_translation=False,
                vecs=i_prev_vel_axes, current_diff_order=1
            )
            _prev_ang_concat = self._worldvec_helper(
                1, diff_order=0, scale=self.rot_scale, use_translation=False,
                vecs=prev_angs, current_diff_order=1
            )

                
            jav_concat = self._concat_per_id_data_by_dsk(jav_per_combo)
            all_gt_data[OutVecMode.JAV_MULTIPLIERS] = jav_concat

            w2l_concat = self._concat_per_id_data_by_dsk(i_w2ls_JAV)
            cached_data[CacheKeys.W2L_MATS_KEY] = w2l_concat

            _gt_rot_vels = self._worldvec_helper(
                0, diff_order=0, scale=self.rot_scale,
                use_translation=False, vecs=self._cache_help.rot_vels_JAV,
                current_diff_order=1
            )

            cached_data[CacheKeys.PREV_VEL_AX_KEY] = _prev_vel_ax_concat
            cached_data[CacheKeys.PREV_ANG_KEY] = _prev_ang_concat
            cached_data[CacheKeys.GT_ROT_VELS_KEY] = _gt_rot_vels

            _l2w_mats_prev = self._worldvec_helper(1, 0, scale=0.0, vecs=r_mats)
            _w2l_mats_prev = np.swapaxes(_l2w_mats_prev, -2, -1)
            
            _extra_wo, _extra_w = self._world_coord_cols(_w2l_mats_prev)
            cached_data[CacheKeys.EXTRA_NO_ROT_ALIGN_COLS_KEY] = _extra_wo
            cached_data[CacheKeys.EXTRA_ROT_ALIGN_COLS_KEY] = _extra_w

            for m in OutVecMode:
                if m == OutVecMode.JAV_MULTIPLIERS:
                    continue
                if m == OutVecMode.VEL_ALIGNED_VEC3:
                    continue

                m_rot_align = m == OutVecMode.ROT_ALIGNED_VEC3 
                m_rot_output = m in ROT_VEC_MODES
                _diff_ord = 0; _scale = 0.0; _curr_diff_ord = 0
                _use_t = True; _vecs = None
                if m_rot_align or m == OutVecMode.WORLD_DISP:
                    _diff_ord = 1; _scale = self.pos_scale
                elif m_rot_output:
                    _scale = self.rot_scale
                    _use_t = False
                    if m == OutVecMode.ROT_VEL_AA:
                        _curr_diff_ord = 1
                        _vecs = self._cache_help.rot_vels_JAV
                    elif m == OutVecMode.ROT_FIXED_AX:
                        _curr_diff_ord = 2
                        _vecs = []
                        for i, r_mats_for_skip in enumerate(r_mats):
                            d = dict()
                            for k, rms in r_mats_for_skip.items():
                                d[k] = pm.closestAnglesAboutAxis(
                                    rms[1:-1], rms[2:],
                                    self._cache_help.prev_vel_axes[i][k][:-1]
                                ).reshape(-1, 1)
                            _vecs.append(d)
                    elif m == OutVecMode.ROT_AA:
                        _vecs = self._cache_help.aas_JAV

                gt_for_m = self._worldvec_helper(
                    0, diff_order = _diff_ord, scale = _scale,
                    use_translation=_use_t, vecs=_vecs,
                    current_diff_order=_curr_diff_ord
                )
                if m == OutVecMode.ROT_FIXED_AX:
                    # From plotting the gt for near-zero angles, it seems
                    # a multiplier of 0.0 is the mean of a fairly normal
                    # -looking histogram. So 0.0 is probably the safest bet.
                    gt_for_m = pm.safeDivideElseZero(gt_for_m, _prev_ang_concat)
                if m_rot_align:
                    gt_for_m = pm.einsumMatVecMul(_w2l_mats_prev, gt_for_m)
                
                all_gt_data[m] = gt_for_m

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # The actual saving of cached data would go here, I guess.
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            


            # Once caching is done, delete the variables created for it.
            del self._cache_help
            del _prev_vel_ax_concat
            del _prev_ang_concat
            del _gt_rot_vels

            # We need to create a "separate" list before iteration because we're
            # deleting elements during said iteration.
            _cached_data_keys = list(cached_data.keys())
            for k in _cached_data_keys:
                if k not in needed_keys:
                    del cached_data[k]

        # endif cached_data is None
        #-----------------------------------------------------------------------
        
        self.cache_data_by_dsk = {
            k: self._getTrainValTestSplit(v) for k, v in cached_data.items()
        }
        # TODO: All reference predictions, for all modes, assume constant
        # timesteps for now.
        self.ref_prediction_name = "Const vel" if _rot_output else "Quadratic" 
        self.ref_predictions = { # Quadratic acc JAV multipliers.
            k: np.repeat((1.0, 0.0), (3, 9)).reshape(1, -1)
            for k in DataSubsetKind
        }
        # TODO: Left off here!
        if self.outVecMode == OutVecMode.VEL_ALIGNED_VEC3:
            raise NotImplementedError("Need to fix this!")
            b, e = 12, 15
            for k, _ref_javs in _javs_by_subset.items():
                _ref_preds = np.empty((len(_ref_javs), 3))
                # Constructing quadratic interpolation predictions.
                # TODO: These assume constant timesteps for now.
                _ref_preds[:, 0] = _ref_javs[:, 0] + _ref_javs[:, 1]
                _ref_preds[:, 1] = _ref_javs[:, 2]
                _ref_preds[:, 2] = 0.0
                self.ref_predictions[k] = _ref_preds
        elif self.outVecMode == OutVecMode.ROT_FIXED_AX:
            # self.ref_predictions[subset_kind] = \
            #     self._prev_ang_concat[subset_kind]
            self.ref_predictions = {k: np.ones((1, 1)) for k in DataSubsetKind}
        else:
            def vec3_selector(arr: NDArray, i3: int):
                start = i3*3
                end = None
                if i3 != -1:
                    end = start + 3
                return arr[start:end]
            # vels = 
            if self.outVecMode == OutVecMode.WORLD_VEC3:
                self.ref_predictions[subset_kind] = coords + vels + accs
            elif self.outVecMode == OutVecMode.WORLD_DISP:
                self.ref_predictions[subset_kind] = vels + accs
            elif self.outVecMode == OutVecMode.ROT_ALIGNED_VEC3:
                self.ref_predictions[subset_kind] = derivs[0] + derivs[1]
            elif self.outVecMode == OutVecMode.ROT_VEL_AA:
                self.ref_predictions[subset_kind] = rot_vels
            elif self.outVecMode == OutVecMode.ROT_AA:
                rot_vels_unscaled = self.rot_scale * rot_vels
                vel_qs = pm.quatsFromAxisAngleVec3s(rot_vels_unscaled)
                extrap_vel_qs = pm.multiplyQuatLists(
                    vel_qs, pm.quatsFromAxisAngleVec3s(aas)
                )
                self.ref_predictions[subset_kind] = pm.axisAngleVec3sFromQuats(
                    extrap_vel_qs, True
                ) / rs


        full_gt: NDArray
        if self.outVecMode == OutVecMode.VEL_ALIGNED_VEC3:
            full_gt = all_gt_data[OutVecMode.JAV_MULTIPLIERS][..., 12:15]
            
            # We'll delete it now because the divison will create a new array
            # and use more RAM.
            del all_gt_data
            
            full_gt /= self.pos_scale

        else:
            full_gt = all_gt_data[self.outVecMode]
            del all_gt_data
        self._gt_arrs = self._getTrainValTestSplit(full_gt)

        all_cols_data = _dog.col_subset_whole
        if need_more_cols:
            extra_cols: NDArray
            if _rot_align:
                extra_cols = cached_data[CacheKeys.EXTRA_ROT_ALIGN_COLS_KEY]
            else:
                extra_cols = cached_data[CacheKeys.EXTRA_NO_ROT_ALIGN_COLS_KEY]
            all_cols_data = np.concatenate((all_cols_data, extra_cols), axis=1)
            _dog.reset_col_subset_slices(all_cols_data)
        self._in_arrs = self._getTrainValTestSplit(all_cols_data)


        need_scaler_fit = save_data_for_conf or _rot_align or (self.outVecMode in WORLD_VEC_MODES) 
        if self.pos_scale != 0.0 and need_scaler_fit and self.translationScaler is None:
            self._fitWorldScaler(self.pos_scale)

        self.score_scaler = 0.0
        relevant_scale = self.rot_scale if _rot_output else self.pos_scale
        if self.outVecMode == OutVecMode.WORLD_VEC3:
            self.score_scaler = self.translationScaler
        elif self.outVecMode != OutVecMode.JAV_MULTIPLIERS:
            self.score_scaler = relevant_scale
        
        self.score_fn = poseLossJAV
        if self.outVecMode != OutVecMode.JAV_MULTIPLIERS:
            if self.outVecMode in ROT_VEC_MODES:
                self.score_fn = poseLossAngle 
            else:
                self.score_fn = poseLossVec3

        
        # These wouldn't have been scaled earlier. Might want to move/fix things
        # so that all scaling happens in the same place, though.
        # TODO: move this scaling earlier!!!
        if self.outVecMode == OutVecMode.VEL_ALIGNED_VEC3:
            for k in DataSubsetKind.nonWholeValues():
                self.ref_predictions[k] /= self.pos_scale


        self.in_train = self._in_arrs[DataSubsetKind.TRAIN]
        self.in_test = self._in_arrs[DataSubsetKind.TEST]
        self.in_validation = self._in_arrs[DataSubsetKind.VALIDATION]

        self.gt_train = self._gt_arrs[DataSubsetKind.TRAIN]
        self.gt_test = self._gt_arrs[DataSubsetKind.TEST]
        self.gt_validation = self._gt_arrs[DataSubsetKind.VALIDATION]

        
        # jav_train = np.concatenate(
        #     (partial_jav_train, jav_tr_append), axis=-1
        # )
        # jav_test = np.concatenate(
        #     (partial_jav_test, jav_te_append), axis=-1
        # )

    def _concat_per_id_data_by_dsk(self, per_id_data):
        
        temp_dict = {}
        for k, v in self.data_organizer.subset_ids.items():
            if k == DataSubsetKind.WHOLE:
                continue
            elif k == DataSubsetKind.VALIDATION and len(v) == 0:
                temp_dict[k] = temp_dict[DataSubsetKind.TRAIN][:0]
            else:
                temp_dict[k] = np.concatenate(
                    concatForComboSubset(per_id_data, v), axis=0
                )
        
        ret_concat = np.concatenate((
            temp_dict[DataSubsetKind.TRAIN],
            temp_dict[DataSubsetKind.VALIDATION],
            temp_dict[DataSubsetKind.TEST]
        ), axis=0)

        del temp_dict

        return ret_concat

    def _getTrainValTestSplit(self, array: NDArray):
        tr = array[:self._val_start]
        va = array[self._val_start:self._test_start]
        te = array[self._test_start:]
        return {
            DataSubsetKind.TRAIN: tr, DataSubsetKind.VALIDATION: va,
            DataSubsetKind.TEST: te, DataSubsetKind.WHOLE: array
        }

    def _world_coord_cols(self, w2l_rotations: typing.Optional[NDArray]):
        scs = self.pos_scale
        rs = self.rot_scale
        _ts = self._cache_help.translations_JAV
        coords = self._worldvec_helper(1, use_translation=True)
        coords_tm1 = self._worldvec_helper(2, use_translation=True)
        coords_tm2 = self._worldvec_helper(3, use_translation=True)
        coords_tm3 = self._worldvec_helper(4, use_translation=True)
        vels = self._worldvec_helper(1, 1, scale=scs, vecs = _ts)
        accs = self._worldvec_helper(1, 2, scale=scs, vecs = _ts)
        jerks = self._worldvec_helper(1, 3, scale=scs, vecs = _ts)
        aas = self._worldvec_helper(1, vecs=self._cache_help.aas_JAV)
        rot_vels_unscaled = self._worldvec_helper(
            1, current_diff_order=1, vecs=self._cache_help.rot_vels_JAV
        )
        rot_vels = rot_vels_unscaled / rs
        rot_accs = self._worldvec_helper(
            1, diff_order=1,
            current_diff_order=1, vecs=self._cache_help.rot_vels_JAV, scale = rs
        )
        rot_jerks = self._worldvec_helper(
            1, diff_order=2,
            current_diff_order=1, vecs=self._cache_help.rot_vels_JAV, scale=rs
        )
        w2ls_concat: NDArray = np.concatenate(concatForComboSubset(
            self._cache_help.w2ls_JAV, self._cache_help.concatted_ids
        ), axis=0)
        other_coords = (coords_tm1, coords_tm2, coords_tm3)
        derivs = (vels, accs, jerks, rot_vels, rot_accs, rot_jerks)
        aas_tup = (aas, )
        other_vecs = derivs + aas_tup
        w2l_thing = (w2ls_concat.reshape(-1, 9), )

        rmatv9s = self._worldvec_helper(1, vecs=self._cache_help.rmatsv9)
        no_rot_cols = other_vecs + other_coords + w2l_thing + (rmatv9s, coords)
        
        other_coords_rotated = tuple(
            pm.einsumMatVecMul(w2l_rotations, c - coords)
            for c in other_coords
        )
        other_vecs_rotated = tuple(
            pm.einsumMatVecMul(w2l_rotations, v) for v in other_vecs
        )
        local_to_vel = pm.einsumMatMatMul(w2ls_concat, w2l_rotations)
        w2l_thing_rotated = (local_to_vel.reshape(-1, 9), )
        rot_cols = other_vecs_rotated + other_coords_rotated + w2l_thing_rotated
        
        return no_rot_cols, rot_cols
    
    def _fitWorldScaler(self, scale: float):
        scaler = StandardScaler()
        if self._cache_help.translations_JAV is None:
            raise Exception("No translations stored to set scaler with!")
        
        scaler.fit(
            concatForComboSubset(
                self._cache_help.translations_JAV,
                self.data_organizer.subset_ids[DataSubsetKind.TRAIN]
            )[0]
        )
        scaler.scale_[:] = scale
        
        self.translationScaler = scaler
    
    
    def _worldvec_helper(self, shift_from_gt: int,
                         diff_order: int = 0, current_diff_order: int = 0,
                         scale: float = 0.0, use_translation: bool = False,
                         vecs: typing.Optional[typing.List[typing.Dict[typing.Any, NDArray]]] = None):
        start = 4 - diff_order - current_diff_order - shift_from_gt
        if use_translation:
            vecs = self._cache_help.translations_JAV
        elif vecs is None:
            raise ValueError("No vectors specified!")
        concat = np.concatenate(typing.cast(NDArray, concatForComboSubset(
            vecs, self._cache_help.concatted_ids, front_trim=start,
            end_trim=shift_from_gt, diff_order = diff_order
        )), axis=0)
        if use_translation and scale == 0.0:
            concat = typing.cast(
                NDArray, self.translationScaler.transform(concat)
            )
        elif scale != 0.0:
            concat /= scale
        return concat
    

    def getScoresSubset(self, subset: DataSubsetKind, predictions: NDArray,
                        should_print: bool = True):

        _predictions = predictions
        gt_param = self._gt_arrs[subset]
        if self.outVecMode == OutVecMode.ROT_FIXED_AX:
            ref_axes = self.cache_data_by_dsk[CacheKeys.PREV_VEL_AX_KEY][subset]
            ref_angs = self.cache_data_by_dsk[CacheKeys.PREV_ANG_KEY][subset]
            _predictions = (predictions * ref_angs) * ref_axes
            gt_param = self.cache_data_by_dsk[CacheKeys.GT_ROT_VELS_KEY][subset]

        return self._scoreHelper(
            _predictions, gt_param, self.score_fn, self.score_scaler, subset,
            self.data_organizer, should_print
        )
    
    
    def getScoresTrain(self, predictions: NDArray, should_print: bool = True):
        return self.getScoresSubset(
            DataSubsetKind.TRAIN, predictions, should_print
        )

    def getScoresTest(self, predictions: NDArray, should_print: bool = True):
        return self.getScoresSubset(
            DataSubsetKind.TEST, predictions, should_print
        )
    
    def getScoresValidation(self, predictions: NDArray, should_print: bool = True):
        return self.getScoresSubset(
            DataSubsetKind.VALIDATION, predictions, should_print
        )

    @staticmethod
    def _scoreHelper(preds: NDArray, gt_vals: NDArray,
                     score_fn: typing.Callable, sc, subset: DataSubsetKind,
                     row_handler: RowsAndColsHandler, should_print: bool):
        _preds = preds
        _gts = gt_vals
        if sc is not None and sc != 0.0:
            if isinstance(sc, float):
                _preds = preds * sc
                _gts = gt_vals * sc
            else:         
                _preds = sc.inverse_transform(preds)
                _gts = sc.inverse_transform(gt_vals)
        
        errs: NDArray = score_fn(_gts, _preds)
        if not isinstance(errs, np.ndarray):
            errs = errs.numpy()
        
        subsets = (subset, )
        if subset == DataSubsetKind.WHOLE:
            subsets = DataSubsetKind.nonWholeValues()
        
        scores: typing.Dict[
            SkipSubsetKind, typing.Union[float, np.floating]
        ] = {}
        # Print scores on test data.
        for sk in row_handler.supported_skips:
            if sk == SkipSubsetKind._all:
                scores[sk] = np.mean(errs)
                continue
            curr_sum = 0.0
            curr_count = 0
            for dsk in subsets:
                err_subset = row_handler.getSelectionData(errs, dsk, sk)
                curr_sum += np.sum(err_subset)
                curr_count += len(err_subset)
            scores[sk] = np.nan
            if curr_count > 0:
                scores[sk] = curr_sum / curr_count
        if should_print:
            for k, v in scores.items():
                print(k.display_name + ":", v)
        return scores
    
bcotjav = DataForJAV(
    dog, bcs_scaler, nonco_cols, JAV_order, chosen_mode,
    save_data_for_conf=True, #skip=2
)
#%%
print("Reference scores for whole dataset (i.e., no train/test split).")
print("Reference method:", bcotjav.ref_prediction_name)
print("Reference scores:")
bcotjav.getScoresSubset(DataSubsetKind.WHOLE, bcotjav.ref_predictions[DataSubsetKind.WHOLE], True)

#%%
import pickle
with open("./results/models/scaler.pickle", "wb") as f:
    pickle.dump(bcs_scaler, f)


#%%
bcs_model = getUntrainedNN(bcotjav.in_train.shape[1], sel_dim, sel_loss) #, 5, 256)



#%% Train the network.
latest_models = loadLatestModels((chosen_mode.name, "HOT3D_JM"))
if TRAIN_NEW_MODEL:
    val_param = None
    bcot_validation_ids = dog.subset_ids[DataSubsetKind.VALIDATION]
    if bcot_validation_ids is not None and len(bcot_validation_ids) > 0:
        val_param = (bcotjav.in_validation, bcotjav.gt_validation)
    bcs_hist = bcs_model.fit(
        bcotjav.in_train, bcotjav.gt_train, epochs=32, shuffle=True,
        validation_data=val_param,
        # batch_size = 1024
    )
    latest_models[chosen_mode.name] = bcs_model
else:
    bcs_model = latest_models[chosen_mode.name]
#%% Evaluate network on test data.

bcs_pred = bcs_model.predict(bcotjav.in_test, batch_size = 1024)
bcotjav.getScoresTest(bcs_pred) #scaledAAs(bcotjav.in_test[:, -30:-27]))

#%% Saving model to disk.
if TRAIN_NEW_MODEL:
    model_name = "results/models/{}-{:%Y-%m-%d_%H-%M-%S}.keras".format(
        chosen_mode.name, datetime.datetime.now()
    )
    bcs_model.save(model_name)

#%% Granular per-model scores on test data.
