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
# JAV_MULTIPLIERS, VEL_ALIGNED_VEC3, WORLD_VEC3, WORLD_DISP, ROT_ALIGNED_VEC3
# ROT_AA, ROT_VEL_AA, ROT_FIXED_AX

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
        if self.outVecMode == OutVecMode.VEL_ALIGNED_VEC3:
            needed_keys.append(CacheKeys.JAV_12)

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
                i_rot_vels_JAV, i_prev_vel_axes
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

            if CacheKeys.JAV_12 in needed_keys:
                cached_data[CacheKeys.JAV_12] = all_gt_data[OutVecMode.JAV_MULTIPLIERS]
        # endif cached_data is None
        #-----------------------------------------------------------------------
        all_cols_data = _dog.col_subset_whole
        extra_cols: typing.Optional[NDArray] = None
        if need_more_cols:
            if _rot_align:
                extra_cols = cached_data[CacheKeys.EXTRA_ROT_ALIGN_COLS_KEY]
            else:
                extra_cols = cached_data[CacheKeys.EXTRA_NO_ROT_ALIGN_COLS_KEY]
            all_cols_data = np.concatenate((all_cols_data, extra_cols), axis=1)
            _dog.reset_col_subset_slices(all_cols_data)
        
        self.cache_data_by_dsk = {
            k: self._getSplitByDSK(v) for k, v in cached_data.items()
        }

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
        self._gt_arrs = self._getSplitByDSK(full_gt)

        self._in_arrs = self._getSplitByDSK(all_cols_data)
        
        # TODO: All reference predictions, for all modes, assume constant
        # timesteps for now.
        self.ref_prediction_name = "Const vel" if _rot_output else "Quadratic" 
        self.ref_predictions = { # Quadratic acc JAV multipliers.
            k: np.repeat((1.0, 0.0), (3, 9)).reshape(1, -1)
            for k in DataSubsetKind
        }
        # TODO: Left off here!
        if self.outVecMode == OutVecMode.VEL_ALIGNED_VEC3:
            all_javs = cached_data[CacheKeys.JAV_12]
            # Constructing quadratic interpolation predictions.
            # TODO: These assume constant timesteps for now.
            _ref_preds = np.zeros_like(all_javs[:, :3])
            _ref_preds[:, 0] = all_javs[:, 0] + all_javs[:, 1]
            _ref_preds[:, 1] = all_javs[:, 2]
            _ref_preds[:, 2] = 0.0
            _ref_preds /= self.pos_scale

            self.ref_predictions = self._getSplitByDSK(_ref_preds)
        elif self.outVecMode == OutVecMode.ROT_FIXED_AX:
            # self.ref_predictions[subset_kind] = \
            #     self._prev_ang_concat[subset_kind]
            self.ref_predictions = {k: np.ones((1, 1)) for k in DataSubsetKind}
        elif self.outVecMode != OutVecMode.JAV_MULTIPLIERS:
            if extra_cols is None:
                raise Exception("Should have set extra_cols in this case!")
            def vec3_selector(arr: NDArray, i3: int):
                start = i3*3
                end = None
                if i3 != -1:
                    end = start + 3
                return arr[:, start:end]
            # vels = 
            vels = vec3_selector(extra_cols, 0)
            accs = vec3_selector(extra_cols, 1)
            rot_vels = vec3_selector(extra_cols, 3)
            ref_preds_whole: NDArray
            if self.outVecMode == OutVecMode.WORLD_VEC3:
                coords = vec3_selector(extra_cols, -1)
                ref_preds_whole = coords + vels + accs
            elif self.outVecMode == OutVecMode.WORLD_DISP:
                ref_preds_whole = vels + accs
            elif self.outVecMode == OutVecMode.ROT_ALIGNED_VEC3:
                ref_preds_whole = vels + accs
            elif self.outVecMode == OutVecMode.ROT_VEL_AA:
                ref_preds_whole = rot_vels
            elif self.outVecMode == OutVecMode.ROT_AA:
                rot_vels_unscaled = self.rot_scale * rot_vels
                vel_qs = pm.quatsFromAxisAngleVec3s(rot_vels_unscaled)
                _aas = vec3_selector(extra_cols, 6)
                extrap_vel_qs = pm.multiplyQuatLists(
                    vel_qs, pm.quatsFromAxisAngleVec3s(_aas)
                )
                ref_preds_whole = pm.axisAngleVec3sFromQuats(
                    extrap_vel_qs, True
                ) / self.rot_scale
            else:
                raise Exception(
                    self.outVecMode.name + " missing ref predictions!"
                )
            self.ref_predictions = self._getSplitByDSK(ref_preds_whole)




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
                self.score_fn = pm.poseLossAngle 
            else:
                self.score_fn = poseLossVec3

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

    def _concat_per_id_data_by_dsk(self, per_id_data, **kwargs):
        # TODO: instead of concat, might save RAM to set in empty.
        temp_dict = {}
        for k, v in self.data_organizer.subset_ids.items():
            if k == DataSubsetKind.WHOLE:
                continue
            elif k == DataSubsetKind.VALIDATION and len(v) == 0:
                temp_dict[k] = temp_dict[DataSubsetKind.TRAIN][:0]
            else:
                temp_dict[k] = np.concatenate(
                    concatForComboSubset(per_id_data, v, **kwargs), axis=0
                )
        
        ret_concat = np.concatenate((
            temp_dict[DataSubsetKind.TRAIN],
            temp_dict[DataSubsetKind.VALIDATION],
            temp_dict[DataSubsetKind.TEST]
        ), axis=0)

        del temp_dict

        return ret_concat

    def _getSplitByDSK(self, array: NDArray):
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
        w2ls_concat: NDArray = self._concat_per_id_data_by_dsk(
            self._cache_help.w2ls_JAV
        )
        other_coords = (coords_tm1, coords_tm2, coords_tm3)
        derivs = (vels, accs, jerks, rot_vels, rot_accs, rot_jerks)
        aas_tup = (aas, )
        other_vecs = derivs + aas_tup
        w2l_thing = (w2ls_concat.reshape(-1, 9), )

        rmatv9s = self._worldvec_helper(1, vecs=self._cache_help.rmatsv9)
        no_rot_cols = other_vecs + other_coords + w2l_thing + (rmatv9s, coords)
        no_rot_cols_np = np.concatenate(no_rot_cols, axis=-1)

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
        
        # Might need to refine deletion order a bit if I encounter RAM issues.
        del no_rot_cols
        rot_cols_np = np.concatenate(rot_cols, axis=-1)
        del rot_cols

        return no_rot_cols_np, rot_cols_np
    
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
        concat = self._concat_per_id_data_by_dsk(
            vecs, front_trim=start, end_trim=shift_from_gt,
            diff_order = diff_order
        )
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
# import tikzplotlib
# import matplotlib.pyplot as plt
# acc_ratios = np.linspace(0, 1, 101)
# res = np.zeros((3, 101))
# for ir, r in enumerate(acc_ratios):
#     ratio_errs = bcotjav.getScoresSubset(
#         DataSubsetKind.TEST,
#         r * np.array([[1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), False
#     )
#     for i in range(3):
#         res[i, ir] = ratio_errs[i]

# # Source: https://stackoverflow.com/questions/75900239/attributeerror-occurs-with-tikzplotlib-when-legend-is-plotted
# def tikzplotlib_fix_ncols(obj):
#     """
#     workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
#     """
#     if hasattr(obj, "_ncols"):
#         obj._ncol = obj._ncols
#     for child in obj.get_children():
#         tikzplotlib_fix_ncols(child)

# start_ratio = 70
# fig = plt.figure()
# plt.plot(acc_ratios[start_ratio:], res[0, start_ratio:], label="skip 0")
# plt.plot(acc_ratios[start_ratio:], res[1, start_ratio:], label="skip 1")
# plt.plot(acc_ratios[start_ratio:], res[2, start_ratio:], label="skip 2")
# plt.legend()
# tikzplotlib_fix_ncols(fig)
# tikzplotlib.save("mytikz.tex")

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
median_store_shape = (3, len(bcot_test_ids))
med_static_str = "median_static"
model_medians = {med_static_str: np.empty(median_store_shape)}

# Get static prediction errors.
static_test_errs: NDArray = sel_loss(
    bcotjav.gt_test, np.zeros((1, 12))
).numpy()
for m, (score_model_prefix, score_model) in enumerate(latest_models.items()):
    med_prefix = "median_"+ score_model_prefix
    # Get neural net errors.
    nn_preds = score_model.predict(bcotjav.in_test, batch_size=1024)
    nn_errs: NDArray = sel_loss(bcotjav.gt_test, nn_preds).numpy()

    model_medians[med_prefix] = np.empty(median_store_shape)

    for skip_amt in range(3):
        for i, vid_id in enumerate(bcot_test_ids):
            errs_for_id = dog.getSelectionData(
                nn_errs, DataSubsetKind.TEST, skip_amt, vid_id
            )

            model_medians[med_prefix][skip_amt, i] = np.median(errs_for_id)

            if m == 0:
                static_errs_for_id = dog.getSelectionData(
                    static_test_errs, DataSubsetKind.TEST, skip_amt, vid_id
                )
                model_medians[med_static_str][skip_amt, i] = np.median(
                    static_errs_for_id
                )

#%%
np.savez_compressed(
    "./results/models/scores_on_bcot.npz", id_order=bcot_test_ids,
    **model_medians
)

# print(bcs_test_scores)
#%%
import errorstats as es
motion_data_key_subset = [dog.motion_data_keys[i] for i in nonco_col_nums]

all_rotation_mats_T: typing.Dict[typing.Tuple[int, int], np.ndarray] = dict()
for combo in dog.getAllIDs():
    calculator = PoseLoaderBCOT(combo[0], combo[1])
    all_rotation_mats_T[combo[:2]] = np.swapaxes(
        calculator.getRotationMatsGTNP(), -1, -2
    ) # transposes
#%%
print("Starting the confidence interval stuff.")

# TODO: Replace motion-kind str with an enum in here and other files.
motion_kinds_plus = PoseLoaderBCOT.motion_kinds + ["all"]
reframed_JAV_errs: typing.List[typing.Dict[str, NDArray]] = [
    {mk: [] for mk in motion_kinds_plus} for _ in range(3)
]

lim_class_errs: typing.List[typing.Dict[str, NDArray]] = [
    {mk: [] for mk in motion_kinds_plus} for _ in range(3)
]

bcot_test_ids_with_mk = [
    (*c, PoseLoaderBCOT.getMotionKind(c[1])) for c in bcot_test_ids
]
for skip in range(3):
    for combo in bcot_test_ids_with_mk:
        c2 = combo[:2]
        curr_dsk = None
        for d in DataSubsetKind.nonWholeValues():
            if c2 in dog.subset_ids[d]:
                curr_dsk = d
                break
        if curr_dsk is None:
            raise Exception("Combo not found in train, val, or test!")
        curr_data = dog.col_subset_train
        if curr_dsk == DataSubsetKind.VALIDATION:
            curr_data = dog.col_subset_validation
        elif curr_dsk == DataSubsetKind.TEST:
            curr_data = dog.col_subset_test
        curr_input = dog.getSelectionData(curr_data, curr_dsk, skip, c2)
        curr_input = bcs_scaler.transform(curr_input)

        curr_jav_pred = bcs_model.predict(
            curr_input, batch_size=1024, verbose=0
        )

        # FIXME_JAV_CACHE: These attributes need to be created as instance vars
        world_disp = getWorldFrameDisplacements(
            bcotjav.jav_per_combo[skip][c2], curr_jav_pred,
            bcotjav.w2ls_JAV[skip][c2]
        )
        curr_translations = bcotjav.translations_JAV[skip][c2]
        curr_d1_vels = np.diff(curr_translations[1:-1], 1, axis=0)
        curr_accs = np.diff(curr_d1_vels, 1, axis=0)
        curr_d2_vels = curr_d1_vels[1:] + (curr_accs / 2.0) 
        in_translations = curr_translations[-(len(world_disp) + 1):-1]
        jav_pred = in_translations + world_disp

        curr_rotation_mats = all_rotation_mats_T[c2][::(skip+1)]
        curr_jav_errs = es.localizeErrsInFrames(
            {"JAV": jav_pred},  curr_translations[1:], curr_rotation_mats[1:],
            deg1_vels=curr_d1_vels, deg2_vels=curr_d2_vels, deg2_acc=curr_accs
        )["JAV"]

        # class_lim_start_ind = -(len(curr_min_norm_vecs) + 1)
        # curr_min_norm_vecs = cfc.min_norm_vecs[skip][c2] \
        #     + curr_translations[-len(world_disp):]
        
        # curr_class_lim_errs = es.localizeErrsInFrames(
        #     {"Class Lim":  curr_min_norm_vecs},  curr_translations[1:],
        #     curr_rotation_mats[1:], deg1_vels=curr_d1_vels,
        #     deg2_vels=curr_d2_vels, deg2_acc=curr_accs
        # )["Class Lim"]

        reframed_JAV_errs[skip][combo[-1]].append(curr_jav_errs)
        reframed_JAV_errs[skip]["all"].append(curr_jav_errs)
        # lim_class_errs[skip][combo[-1]].append(curr_class_lim_errs)
        # lim_class_errs[skip]["all"].append(curr_class_lim_errs)
        progress_str = "\rProgress: skip {}, combo ({:2},{:2})".format(
            skip, c2[0], c2[1]
        )
        print(progress_str, end = '', flush=True)
#%%
def printErrStats3D(errs: typing.List[typing.Dict[str, NDArray]]):
    world_field = es.LocalizedErrsCollection._fields[0]
    for skip in range(3):
        print("\nSkip {}:".format(skip))

        for mk in motion_kinds_plus:
            curr_errs = np.concatenate(errs[skip][mk], axis=1)

            stats = {
                f: es.getStats(curr_errs[i])
                for i, f in enumerate(es.LocalizedErrsCollection._fields)
            }
            print("  {} (score {:0.4f}):".format(mk, stats[world_field].mean_mag))
            for name, stat in stats.items():
                print(es.formattedErrStats(stat, name, 4, stats[world_field]))
            print()

printErrStats3D(reframed_JAV_errs)


#%% 

################################################################################
# Column Scrambling to Assess Feature Importance
################################################################################

# Finds the errors for the model when one of the test data columns has its
# data scrambled, as per the advice of a StackOverflow post on how to figure out
# which columns are more important for the prediction.
def errsForColScramble(model: keras.Model, data: NDArray, col_ind: int, y_true: NDArray):
    data_scramble = data.copy() # So that original's not affected.

    # Column scramble:
    data_scramble[:, col_ind] = np.random.default_rng().choice(
        data_scramble[:, col_ind], len(data_scramble), False
    )

    # Getting new errors:
    preds = model.predict(data_scramble, batch_size=1024, verbose=0)
    errs = model.loss(y_true, preds)
    return errs

# For each column and for each skip amount, scramble the column and find the new
# score.
scramble_scores = np.empty((dog.col_subset_test.shape[1], len(SkipSubsetKind)))
print()
 

num_nonco_cols = dog.col_subset_test.shape[1]
 

# For each column and for each skip amount, scramble the column and find the new
# score.
for col_ind in range(num_nonco_cols):
    print(
        "Testing column index {:03d}/{}.".format(col_ind + 1, num_nonco_cols),
        end='\r', flush=True
    )
    errs = errsForColScramble(
        bcs_model, dog.col_subset_test, col_ind, bcotjav.jav_test
    )
    for i, k in enumerate(SkipSubsetKind):
        scramble_scores[col_ind, i] = np.mean(
            dog.getSelectionData(errs, DataSubsetKind.TEST, k)
        )
#%%
scramble_rank = np.argsort(scramble_scores, axis=0)[::-1]
head_num = 10 # How many "best" to print.
head_best = scramble_rank[:head_num]
print("Most important feature inds:", head_best, sep='\n')

print("Names:")
for hb in head_best:
    print([nonco_featnames[i] for i in hb])
    



#%%
import shap
default_rng = np.random.default_rng()
shap_bg = default_rng.choice(dog.col_subset_test, 100, False, axis=0)
shap_ex = shap.GradientExplainer(bcs_model, shap_bg)

#%%
shap_test_inds = default_rng.choice(len(dog.col_subset_test), 200, False, axis=0)
shap_test = dog.col_subset_test[shap_test_inds]
# If input shape is (n_rows, n_feats) and output shape is (n_rows, n_outs), shap
# values shape is (n_rows, n_feats, n_outs)
shap_values_tf = shap_ex(shap_test)
preds_shap_bg = bcs_model(shap_bg).numpy()
# SHAP DeepExplainer seems to leave base_values as None, which breaks certain
# plots, so I need to fill it in manually unless I use GradientExplainer.
# shap_values_tf.base_values = np.broadcast_to(
#     preds_shap_bg.mean(axis=0), (len(shap_test), preds_shap_bg.shape[-1])
# )
# We also set the feature_names so that plots include them.
shap_values_tf.feature_names = nonco_featnames
shap.initjs()
shap_sort = np.argsort(
    np.mean(np.abs(shap_values_tf.values), axis=0), axis=0
)[::-1].transpose()
#%%
output_shap_ind = 0
shap.summary_plot(shap_values_tf.values[..., output_shap_ind], shap_test, feature_names=nonco_featnames)

#%%
fig, ax = shap.partial_dependence_plot(
    77,
    lambda x: bcs_model.predict(x, verbose=0)[..., output_shap_ind],
    shap_test,
    model_expected_value=True,
    feature_expected_value=True,
    show=False,
    ice=False,
)

#%%
shap.plots.bar(shap_values_tf[..., output_shap_ind].abs.max(0))

#%%
shap.plots.scatter(
    shap_values_tf[:, 100, output_shap_ind],
    color=shap_values_tf[..., output_shap_ind]
)

#%%
clustering = shap.utils.hclust(shap_test) #, bcotjav.jav_test[shap_test_inds, 12:15])
#%%
shap.plots.bar(shap_values_tf[..., output_shap_ind], clustering=clustering)
#%% Graphing the SHAP and scramble importance rankings.
import matplotlib.pyplot as plt

scramble_for_bars = scramble_scores - np.min(scramble_scores, axis=0)
scramble_for_bars /= (np.mean(scramble_for_bars, axis=0) + np.std(scramble_for_bars, axis=0))
num_features = len(nonco_col_nums)
nonco_arange = np.arange(num_features)
for skip in range(4):
    plt.bar(
        nonco_arange + skip / 4, scramble_for_bars[:, skip],
        width = 1/4, label="skip " + str(skip)
    )
plt.legend()
plt.xticks(nonco_arange[::5])
plt.xticks(nonco_arange, minor=True)
plt.xlabel("Feature number")
plt.ylabel("Scramble Importance")
plt.ylim(0, 1)
plt.show()

#%%
shap_accum = np.mean(np.abs(shap_values_tf.values), axis=0)
shap_accum /= (np.mean(shap_accum, axis=0) + np.std(shap_accum, axis=0))
num_muls = shap_values_tf.shape[-1]
for mul in range(num_muls):
    plt.bar(
        nonco_arange + mul / num_muls, shap_accum[:, mul],
        width = 1/12, label="mul " + str(mul)
    )
plt.legend()
plt.xticks(nonco_arange[::5])
plt.xticks(nonco_arange, minor=True)
plt.xlabel("Feature number")
plt.ylabel("SHAP Importance")
plt.ylim(0, 1)
plt.show()

#%%

################################################################################
# Vanilla Classification Network
################################################################################

def customPoseLoss(y_true, y_pred):
    probs = tf.nn.softmax(y_pred, axis=1)
    return tf.reduce_sum(y_true * probs, axis=1)
# Example Usage
# tf_concat_train_errs = tf.convert_to_tensor(concat_train_errs, dtype=tf.float32)
tf_loss_fn = customPoseLoss # CustomLossWithErrors(concat_train_errs)

input_dim = dog.col_subset_train.shape[1]
num_classes = len(MOTION_MODEL)

onehot_train_labels = tf.one_hot(dog.concat_train_labels, num_classes)

# I wanted to try to get the below to overfit to make sure I wasn't doing
# "something wrong" before worrying about regularization. Initial attempts to
# overfit failed, and train performance was no better than with a decision tree.
# I finally succeeded in getting it to start overfitting (though it also
# outperformed the tree slightly on test data) by using 3 hidden layers, 2048
# nodes each, relu activation, no regularization, sigmoid activation for final
# layer, CategoricalCrossentropy loss for training, using only noncolinear
# columns, default 'adam' optimizer, batch size 1024, and then training for a
# few sets of 5 epochs. These ideas generally came from:
# https://stats.stackexchange.com/questions/474738/how-do-i-intentionally-design-an-overfitting-neural-network
nncl_per_layer_num = 2048
nncl_act = 'relu'
tfmodel = keras.Sequential([
    keras.layers.Input((input_dim,)),
    keras.layers.Dense(nncl_per_layer_num, activation=nncl_act),
    # keras.layers.Dropout(0.2),
    keras.layers.Dense(nncl_per_layer_num, activation=nncl_act),
    keras.layers.Dense(nncl_per_layer_num, activation=nncl_act),
    # keras.layers.Dense(1024, activation='sigmoid'),
    # keras.layers.Dropout(0.2),
    # keras.layers.Dense(1, activation='sigmoid'),
    keras.layers.Dense(num_classes, activation='sigmoid')])

tfmodel.summary()

adam = 'adam' #keras.optimizers.Adam(0.01)
catcross = True
tf_loss_fn2 = keras.losses.CategoricalCrossentropy(from_logits=True) \
              if catcross else tf_loss_fn
ncll_y = onehot_train_labels if catcross else dog.concat_train_class_errs

tfmodel.compile(optimizer=adam, loss=tf_loss_fn2)
#%%
nn_class_hist = tfmodel.fit(
    dog.col_subset_train, ncll_y, epochs=5, batch_size = 1024, shuffle=True
)
#%%
bgtrain = big_tree.predict(dog.concat_train_data)
bgtrain_score = dog.getClassScoresTrain(bgtrain)
print("Big tree train errs=", bgtrain_score)

mclf_train = mclf.predict(dog.concat_train_data)
mclf_train_score = dog.getClassScoresTrain(mclf_train)
print("Smaller tree train errs=", mclf_train_score)

tf_train_preds = np.argmax(tfmodel(dog.col_subset_train).numpy(), axis=1)
tf_train_errs = dog.getClassScoresTrain(tf_train_preds)
print("TF train errs=", tf_train_errs)
print()

tf_test_preds = np.argmax(tfmodel(dog.col_subset_test).numpy(), axis=1)
tf_test_errs = dog.getClassScoresTest(tf_test_preds)
print("TF test errs=", tf_test_errs)


#%%
################################################################################
# Comparing Error Histograms
################################################################################
import matplotlib.pyplot as plt

big_tree_preds = big_tree.predict(dog.concat_test_data)
tree_errs_for_plt = dog.getClassErrsTest(big_tree_preds)

acc_only_preds = np.full(
    (len(dog.col_subset_test), 1),
    dog.motion_mod_keys.index(MOTION_MODEL.ACC_DEG2)
)
acc_errs_for_plt = dog.getClassErrsTest(acc_only_preds)


bar_skip_key = SkipSubsetKind.skip2
bar_data: typing.List[NDArray] = [
    tree_errs_for_plt[bar_skip_key], acc_errs_for_plt[bar_skip_key],
    dog.getSelectionData(bcs_test_errs, DataSubsetKind.TEST, bar_skip_key),
    error_lim_all[bar_skip_key]
]
bar_labels = ['Tree', 'Acc Only', 'JAV NN', "Classification lim"]
for i, arr in enumerate(bar_data):
    if arr.ndim > 1 and arr.shape[1] == 1:
        bar_data[i] = arr.flatten()
    elif arr.ndim > 1:
        raise Exception("Unexpected shape!")

bar_percentiles = [np.percentile(b, 90) for b in bar_data]
bar_pmax = np.max(bar_percentiles)

fig, ax = plt.subplots()
ax.hist(
    bar_data, histtype='step', stacked=False, fill=False, label=bar_labels,
    bins=35, density=True, range=(0, bar_pmax)
)
ax.legend()
ax.set_ylabel("Proportion")
ax.set_xlabel("Translation Error (mm)")
plt.show()

#%%
from gtCommon import PoseLoaderTUDL

tudl_ids = PoseLoaderTUDL.getAllIDs(True)
tudl_loaders = [PoseLoaderTUDL(*t) for t in tudl_ids]

cfc.getAll(tudl_loaders)
#%%
tudl_train, tudl_test = PoseLoaderTUDL.trainTestIDs()
tdog = DataOrganizer.FromCalcs(
    PoseLoaderTUDL,
    cfc.all_motion_data, cfc.min_norm_labels, cfc.err_norm_lists,
    tudl_train, tudl_test, dog.motion_data_keys
)
#
tjav = DataForJAV(tdog, bcs_scaler, nonco_cols, JAV_order)
#%%

tjavps = bcs_model.predict(tdog.col_subset_test)
tjav.getScoresTest(tjavps)
#%%
ajnn = getUntrainedNN()

bt_train = np.concatenate((dog.col_subset_test, tdog.col_subset_test), axis=0)
bt_jav = np.concatenate((bcotjav.jav_test, tjav.jav_test), axis=0)

#%%
ajnn.fit(bt_train, bt_jav, epochs=32, shuffle=True)

#%%
bcot_ajnn_pred = ajnn.predict(dog.col_subset_train, batch_size=1024)

tudl_ajnn_pred = ajnn.predict(tdog.col_subset_train, batch_size=1024)

bcotjav.getScoresTrain(bcot_ajnn_pred)
tjav.getScoresTrain(tudl_ajnn_pred)
#%%
ax = plt.figure().add_subplot(projection='3d')

data_to_3d_plot = bcot_ajnn_pred[:, :3]


for k in SkipSubsetKind:
    if k == SkipSubsetKind._all:
        continue
    dp = dog.getSelectionData(data_to_3d_plot, DataSubsetKind.TRAIN, k)
    inds = np.random.choice(len(dp), 1000)

    # print(dp[inds].T.shape)
    ax.scatter3D(*dp[inds].T, label="BCOT " + k.display_name)

data_to_3d_plot = tudl_ajnn_pred[:, :3]

for k in SkipSubsetKind:
    if k == SkipSubsetKind._all:
        continue
    dp = tdog.getSelectionData(data_to_3d_plot, DataSubsetKind.TRAIN, k)
    inds = np.random.choice(len(dp), 1000)

    # print(dp[inds].T.shape)
    ax.scatter3D(*dp[inds].T, label="TUDL " + k.display_name)

ax.set_xlabel("i")
ax.set_ylabel("j")
ax.set_zlabel("k")
ax.legend()
plt.show()

# 

#%%

def getUntrainedNNC():
    dropout_rate = 0.2
    nodes_per_layer = 128
    vel_nn_activation = 'sigmoid'

    input_layer = keras.Input(shape=(len(nonco_col_nums),))

    def build_branch():
        x = keras.layers.Dense(nodes_per_layer, activation=vel_nn_activation)(input_layer)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(nodes_per_layer, activation=vel_nn_activation)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(nodes_per_layer, activation=vel_nn_activation)(x)
        x = keras.layers.Dense(1)(x)  # Output a single float
        return x

    # Build 3 branches
    branch_outputs = [build_branch() for _ in range(3)]

    # Concatenate the three outputs
    output = keras.layers.Concatenate()(branch_outputs)

    model = keras.Model(inputs=input_layer, outputs=output)
    model.summary()
    model.compile(loss=poseLossJAV, optimizer='adam')
    return model

ajnnC = getUntrainedNNC()

ajnnC.fit(bt_train, bt_jav, epochs=32, shuffle=True)

#%%
ajnnC_pred = ajnnC.predict(tdog.col_subset_train)
tjav.getScoresTrain(ajnnC_pred)

# ajnnC_loss = poseLossJAV(bcs_test, ajnnC_pred)
# print({k: np.mean(ajnnC_loss[v]) for k, v in dog.skip_inds_dict.items()})

#%%
def getUntrainedNNSplit(input_shape1, input_shape2, input_shape3):
    dropout_rate = 0.2
    nodes_per_layer = 32
    vel_nn_activation = 'sigmoid'

    input1 = keras.Input(shape=(input_shape1,), name='input1')
    input2 = keras.Input(shape=(input_shape2,), name='input2')
    input3 = keras.Input(shape=(input_shape3,), name='input3')

    def build_branch(inp):
        x = keras.layers.Dense(nodes_per_layer, activation=vel_nn_activation)(inp)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(nodes_per_layer, activation=vel_nn_activation)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(nodes_per_layer, activation=vel_nn_activation)(x)
        x = keras.layers.Dense(1)(x)
        return x

    out1 = build_branch(input1)
    out2 = build_branch(input2)
    out3 = build_branch(input3)

    output = keras.layers.Concatenate()([out1, out2, out3])

    model = keras.Model(inputs=[input1, input2, input3], outputs=output)
    model.compile(loss=poseLossJAV, optimizer='adam')
    model.summary()
    return model

lnccn = len(nonco_col_nums)
split_nn = getUntrainedNNSplit(lnccn, lnccn - 1, lnccn - 2)
split_nn.fit([dog.col_subset_test, dog.col_subset_test[:, :-1], dog.col_subset_test[:, :-2]], bcotjav.jav_test, epochs=33, batch_size=128, shuffle=True)

#%%
tjav.getScoresTest(split_nn.predict([tdog.col_subset_test, tdog.col_subset_test[:, :-1], tdog.col_subset_test[:, :-2]]))

#%%
from scipy.optimize import lsq_linear
# Get the pose loss for a set of Jerk, Acceleration, & Velocity multipliers.
def gtMultipliersJAV(y_true, bounds = None, tol: float = 0.001, max_iter: int = 32, verbose=False):
    '''
    When we create the "JAV" data, we specify the permutation of
    (velocity, acceleration, jerk) to orthonormalize into frames. For simplicity
    below, assume the order is in fact velocity, then acceleration, then jerk.

    In that case, y_true contains the following columns, in order:
     - speed
     - accel parallel to velocity
     - accel ortho to velocity
     - jerk parallel to speed, ortho to speed but in acc plane, ortho to plane
     - correct pose displacement in the same "coordinate frame" as the jerk.

    In other words, we are working in an orthonormal coordinate frame where the 
    x-axis is aligned with velocity, the y with acceleration, and then z is
    orthogonal to both.
    
    Then, y_pred contains the multipliers for velocity, acceleration, and jerk, 
    respectively. The predicted "local" displacement is thus:
    [[speed, acc_x, jerk_x],       [vel_multiplier,
     [0,     acc_y, jerk_y],     x  acc_multiplier,
     [0,     0,     jerk_z]]        jerk_multiplier]

    Then after this matrix multiplication, we find the distance between it and
    the correct pose displacement, both vec3s.
    '''
    muls_to_pt_mats = np.zeros((len(y_true), 3, 3))
    muls_to_pt_mats[:, 0, 0] = y_true[:, 0]
    muls_to_pt_mats[:, :2, 1] = y_true[:, 1:3]
    muls_to_pt_mats[:, :, 2] = y_true[:, 3:6]

    if bounds is None:
        inv_mats = np.linalg.inv(muls_to_pt_mats)
        return pm.einsumMatVecMul(inv_mats, y_true[:, 12:15]) #6:9])
    
    n = len(y_true)
    res = np.empty((n, 3))
    n_digs = 0
    status_template = "Done iter {}/" + str(n) + " ({:0.2f})."
    if verbose:
        n_digs = int(np.ceil(np.log10(n)))
        status_template = status_template + (" " * n_digs)
        print(status_template.format(0, 0), end='')

    for i in range(n):
        lsq_res = lsq_linear(
            muls_to_pt_mats[i], y_true[i, 12:15], #6:9],
            bounds, tol=tol, max_iter=max_iter
        )
        res[i] = lsq_res.x
        if (verbose and (i + 1) % 1000 == 0) or i == (n - 1):
            print("\r" + status_template.format((i + 1), (i + 1)/n), end = '')
    if verbose:
        print()
    return res

bounds = (-3,3)
bounded_gt_bcot_tr = gtMultipliersJAV(bcotjav.jav_train, bounds, verbose=True)
bounded_gt_tudl_tr = gtMultipliersJAV(tjav.jav_train, bounds, verbose=True)

# %%
def gtScaledVelAligned(y_true, bounds = None):
    gathered = y_true[:, [0,2,5]]
    scaled = y_true[:, -3:] / gathered
    if bounds is not None:
        scaled = np.clip(scaled, *bounds)
    return scaled
gt_bcot = gtScaledVelAligned(bcotjav.jav_train, bounds)
gt_tudl = gtScaledVelAligned(tjav.jav_train, bounds)

bcotjav.getScoresTrain(gt_bcot)
tjav.getScoresTrain(gt_tudl)

#%%
def myGetClosestPoint(normals: NDArray, scaled_plane_offsets: NDArray, points: NDArray):
    norm_sq = pm.einsumDot(normals, normals)
    pn = None
    if points.ndim > 1 and len(points) > 1:
        pn = pm.einsumDot(normals, points)
    else: 
        pn = normals @ points.flatten()
    scalar = (scaled_plane_offsets - pn) / norm_sq
    disps = pm.scalarsVecsMul(scalar, normals)
    return points + disps

def testClosest(normals, offsets, pts):
    closest = myGetClosestPoint(normals, offsets, pts)
    assert np.allclose(pm.einsumDot(closest, normals), offsets)
    diffs = pts - closest
    diff_norms = np.linalg.norm(diffs, axis=-1)
    unit_diffs = pm.safelyNormalizeArray(diffs, diff_norms[..., np.newaxis])
    unit_norms = pm.normalizeAll(normals)
    dot_diffs = np.abs(pm.einsumDot(unit_diffs, unit_norms)) - 1.0
    dots_good = dot_diffs < 0.0001
    norms_good = diff_norms < 0.0001
    assert (np.all(dots_good | norms_good))
    return closest
#%%
def my_gtMultipliers6(y_true: NDArray, baseline_m6: NDArray):
    
    j_o = y_true[:, 8] / y_true[:, 5]

    flat = baseline_m6.flatten()
    a_a, j_a = myGetClosestPoint(
        y_true[:, [2, 4]], y_true[:, 7], flat[[2, 4]]
    ).transpose()

    v_v, a_v, j_v = myGetClosestPoint(
        y_true[:, [0, 1, 3]], y_true[:, 6], flat[[0, 1, 3]]
    ).transpose()

    return np.stack([v_v, a_v, a_a, j_v, j_a, j_o], axis=-1)

base = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
gt_bcot = my_gtMultipliers6(bcotjav.jav_train, base)
gt_tudl = my_gtMultipliers6(tjav.jav_train, bounds, verbose=True)

bcotjav.getScoresTrain(gt_bcot)
tjav.getScoresTrain(gt_tudl)

#%%
bcs_pred_tr = bcs_model.predict(dog.col_subset_train, batch_size=1024)

#%%

current_idx = 0

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

def getRandSubset(data, gt, n):
    inds = np.random.choice(len(data), n)
    return data[inds], gt[inds]

bcot_sub_data = []
bcot_sub_gt = []
bcot_sub_nn = []

for s in SkipSubsetKind:
    if s == SkipSubsetKind._all:
        continue
    skip_data = dog.getSelectionData(
        dog.concat_train_data, DataSubsetKind.TRAIN, s
    )
    bcot_sub_inds = np.random.choice(len(skip_data), 500)
    bcot_sub_data.append(skip_data[bcot_sub_inds])
    # bcot_sub_gt.append(bcotjav.jav_train[s][bcot_sub_inds, 9:15])
    bcot_sub_gt.append(
        dog.getSelectionData(gt_bcot, DataSubsetKind.TRAIN, s)[bcot_sub_inds]
    )
    bcot_sub_nn.append(
        dog.getSelectionData(bcs_pred_tr, DataSubsetKind.TRAIN, s)[bcot_sub_inds]
    )
# p_sub_data, p_sub_gt = getRandSubset(pdog.concat_train_data, gt_p, 1000)


# The below function was written by Copilot/Claude but then modified by me.
# E.g., I added the "superbin_radius", median calc, etc.
def calculate_interval_means(x, y, lb, ub, n, superbin_radius):
    """
    Calculate means of y values corresponding to x values in n equally spaced intervals between lb and ub.
    
    Parameters:
    -----------
    x : np.ndarray
        Input array of values to be binned
    y : np.ndarray
        Array of values whose means will be calculated for each bin
    lb : float
        Lower bound of the interval
    ub : float
        Upper bound of the interval
    n : int
        Number of intervals to create
    
    Returns:
    --------
    tuple:
        - bin_centres: array of bin centres (length n)
        - means: array of means for each interval (length n)
        - medians: array of medians for each interval (length n)
    """
    # Create bin edges
    bin_edges = np.linspace(lb, ub, n + 1)
    
    # Use np.digitize to find which bin each x value belongs to
    # Subtract 1 from bin indices because np.digitize starts counting from 1
    bin_indices = np.digitize(x, bin_edges) - 1
    
    # Create mask for values within the bounds
    mask = (x >= lb) & (x <= ub)
    
    # Initialize arrays to store results
    means = np.zeros(n)
    medians = np.zeros(n)

    # Calculate means for each bin
    for i in range(n):
        bin_mask = (bin_indices >= i - superbin_radius) 
        bin_mask = bin_mask & (bin_indices <= i + superbin_radius)
        bin_mask = bin_mask & mask
        if np.any(bin_mask):
            bin_data = y[bin_mask]
            means[i] = np.mean(bin_data)
            medians[i] = np.median(bin_data)
        else:
            means[i] = np.nan
            medians[i] = np.nan

    bin_centres = (bin_edges[1:] + bin_edges[:-1])/2.0
    return bin_centres, means, medians


def plot_column(idx):
    fig.suptitle(f'(BCOT) Column: {feature_names[idx]} (#{idx})')
    
    all_column_data = np.concatenate(
        [bcot_sub_data[j][:, idx] for j in range(3)], axis=0
    )
    val_min = np.min(all_column_data)
    val_max = np.max(all_column_data)
    quants = np.quantile(all_column_data, (0.1, 0.9))

    na_inds = (all_column_data == motiontools.shared_constants.ERR_NA_VAL)
    if np.any(na_inds):
        n_na_vals = all_column_data[~na_inds]
        val_min = np.min(n_na_vals)
        val_max = np.max(n_na_vals)

    xlims = (val_min, val_max)
    if (val_max - val_min)/(quants[1] - quants[0]) > 1.5:
        xlims = (quants)

    colours = ['blue', 'orange', 'green']
    for i in range(3):
        axs[i].cla()  # clear the axes
        for j in range(2,-1,-1): #3):
            c = colours[j]
            bsd = bcot_sub_data[j][:, idx]
            label = "skip" + str(j)
            axs[i].scatter(bsd, bcot_sub_gt[j][:, i], alpha=0.01, color=c)
            inter_data = calculate_interval_means(
                bsd, bcot_sub_gt[j][:, i], *xlims, 100, 2
            )
            axs[i].plot(inter_data[0], inter_data[1], color=c, ls="dashed", label=label)
            axs[i].plot(inter_data[0], inter_data[2], color=c, ls="dotted")
        # axs[i].scatter(p_sub_data[:, idx], p_sub_gt[:, i], alpha=0.6, label="Pauwels")
        # for j in range(3):
        #     axs[i].scatter(bcot_sub_data[j][:, idx], bcot_sub_nn[j][:, i], alpha=0.6, label="NN" + str(j))


        axs[i].set_ylim(-0.1, 1.1) #-2, 2)

        axs[i].set_xlabel(gtc.truncateName(feature_names[idx], 33))
        axs[i].set_xlim(*xlims)
    axs[0].set_ylim(0.8, 1.2)
    axs[0].legend()
    axs[0].set_ylabel("Multiplier")
    axs[0].set_title("Vel")
    axs[1].set_title("Acc parallel")
    axs[2].set_title("Acc ortho")


    plt.tight_layout()
    fig.canvas.draw_idle()

def on_key(event):
    global current_idx
    if event.key == 'right':
        current_idx = min(current_idx + 1, len(feature_names) - 1)
        plot_column(current_idx)
    elif event.key == 'left':
        current_idx = max(current_idx - 1, 0)
        plot_column(current_idx)

plot_column(current_idx)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
# %%

def testAllDiscreteJAV(aj_combos = None):
    # Figure out what are the best constant vel, acc, jerk multipliers out of the
    # "physics-based" options.
    acc_opts = [0.0, 0.5, 1.0] # 1.0 means deg2 velocity calculation
    
    # When you look at the degree-3 lagrange interpolating polynomial and its 
    # derivatives at "x=3", and letting v' represent the degree-1 velocity and 
    # letting a' represent degree-2 velocity, one can see that degree-3 
    # acceleration is (a' + j), and one can see that degree-3 velocity is 
    # (v' + a'/2 + j/3). So if we use jerk in no calculations, the total jerk
    # multiplier is 0. If we use it just in the jerk calculation, the total jerk
    # multipliers is 1/6. If we use it in just the jerk and accel calculations,
    # but not velocity, the total is (1/2 * 1) + (1/6 * 1) = 2/3. Finally, if
    # we use it in all calculations, the total is 1.0.
    # So the "most sensible" options for the jerk multiplier are 
    # [0, 1/6, 2/3, 1]. However, if we allow more "odd" combinations, like
    # using it for the velocity and jerk but not acceleration, we can have
    # all of the below 1/6 multiples:
    jerk_opts = [0.0, 1/6, 1/3, 1/2, 2/3, 5/6, 1.0]

    if aj_combos is None:
        aj_combos = []
        for a0 in range(3):
            for a1 in range(3):
                for j0 in range(7):
                    for j1 in range(7):
                        for j2 in range(7):
                            aj_combos.append((
                                1.0, acc_opts[a0], acc_opts[a1], 
                                jerk_opts[j0], jerk_opts[j1], jerk_opts[j2]
                            ))

    ajnp = np.empty((len(aj_combos), len(SkipSubsetKind)))
    for i, ajc in enumerate(aj_combos):
        ajscore = bcotjav.getScoresTrain(np.array([[*ajc]]), False)
        for j, sk in enumerate(SkipSubsetKind):
            ajnp[i, j] = ajscore[sk]
    return ajnp
#%%


base_a_opts = [0.0, 0.5, 1.0]
# See other commenting on how these possible multiplier totals are found
# when looking at the lagrange polynomial derivatives.
base_j_opts = [0, 1/6, 2/3, 1.0]

base2 = getBaselineJAV6(base_a_opts, base_j_opts)
gt_bcot = gtMultipliers6(bcotjav.jav_train, np.asarray(base2))

#%%
all_base_errs = np.empty((len(bcotjav.jav_train), len(base2)))
for i, b in enumerate(base2):
    all_base_errs[:, i] = poseLossJAV(bcotjav.jav_train, np.asarray(b).reshape(1, -1))

min_base_errs = np.min(all_base_errs, axis=-1)
for k in SkipSubsetKind:
    print(
        k.display_name, ":",
        np.mean(dog.getSelectionData(min_base_errs, DataSubsetKind.TRAIN, k))
    )

argmin_base_errs = np.argmin(all_base_errs, axis=-1)
plt.bar(*np.unique(argmin_base_errs, return_counts=True))
plt.show()

#%%
mcb = WeightedErrorCriterion(1, np.array([len(base2)], dtype=np.intp))
base_errs_reshape = all_base_errs.reshape((all_base_errs.shape[0], 1, all_base_errs.shape[1]))
mcb.set_y_errs(base_errs_reshape)


dumb_labels = np.zeros(len(dog.concat_train_data))
dumb_labels[:len(base2)] = np.arange(len(base2))

#%%
base2_tree = sk_tree.DecisionTreeClassifier(max_depth=8, criterion=mcb)
start_time = time.time()
base2_tree = base2_tree.fit(dog.concat_train_data, dumb_labels)
print("Done!")
print("Time spent:", time.time() - start_time)

#%%
from motiontools.dataorg import motionClassScores

test_base_errs = np.empty((len(bcotjav.jav_test), len(base2)))
for i, b in enumerate(base2):
    test_base_errs[:, i] = poseLossJAV(bcotjav.jav_test, np.asarray(b).reshape(1, -1))

#%%
base2_pred = base2_tree.predict(dog.concat_test_data)
base2_pred_res = motionClassScores(
    test_base_errs, base2_pred.astype(int), dog.skip_inds_dict
)

for k, v in base2_pred_res.items():
    print(k, ":", v)

#%%

class_resid_nn = getUntrainedNN(None, True)

cl_start_pt = np.empty((len(dog.concat_train_data), 6))

cl_start_tree = trim_to_depth(base2_tree, 4)
base2_tr_pred = cl_start_tree.predict(dog.concat_train_data).astype(int)

for i, cl_start_val in enumerate(base2_tr_pred):
    cl_start_pt[i] = base2[cl_start_val]

class_resid_nn.fit(
    np.concatenate((dog.col_subset_train, cl_start_pt), axis=-1),
    bcotjav.jav_train, # np.concatenate((bcotjav.jav_train, cl_start_pt), axis=-1),
    epochs=32, shuffle=True
)

#%%


cl_start_pt_test = np.empty((len(dog.concat_test_data), 6))

base2_te_pred = cl_start_tree.predict(dog.concat_test_data).astype(int)

for i, cl_start_val in enumerate(base2_te_pred):
    cl_start_pt_test[i] = base2[cl_start_val]

class_resid_test = class_resid_nn.predict(
    np.concatenate((dog.col_subset_test, cl_start_pt_test), axis=-1)
)

# class_resid_losses = poseLossResidualJAV(
#     np.concatenate((bcotjav.jav_test, cl_start_pt_test), axis=-1), class_resid_test
# )


class_resid_losses = poseLossJAV(
    bcotjav.jav_test, class_resid_test
)

for k in SkipSubsetKind:
    print(
        k,
        np.mean(
            dog.getSelectionData(class_resid_losses, DataSubsetKind.TEST, k)
        )
    )
