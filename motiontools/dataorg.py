from io import BufferedWriter
import typing
import builtins # For ellipsis type pre Python 3.10
from enum import IntEnum
import os
import pathlib

import pickle

import numpy as np
from numpy.typing import NDArray

import gtCommon as gtc
from motiontools.key_and_vec_specs import (
    MOTION_MODEL, MOTION_DATA, SpecifiedMotionData, ANG_OR_MAG, OTHER_DIRECTION,
    OneHotMotionData, MOTION_DATA_KEY_TYPE
)
from datatools.data_splitting import DataSubsetKind

# We frequently work with data sequences that have the following type: 
#     List[Dict[Combo, (Dict|NDArray)]]
# That is, we have a list of dictionaries which store the results per video,
# where said "result" might be an NDArray (in the case of best-class labels) or
# another dict with "column" names and per-column NDArray data.
# 
# The vid-ID-keyed dict's index in the top-level list corresponds to the number 
# of frames we are skipping when we read the dataset. So the [0] dict is when
# not skipping any  frames, the [1] dict for when reading every 2nd frame only, 
# etc.

# The below are functions that convert the above lists of dicts into single
# train/test sets that we can pass into our model training.

# First, we concatenate together the data for a subset of combos and get
# a list of 3 items, where each list index again corresponds to the frame 
# skip amount but results are no longer separated by combo.
# Motivation: We may want to quickly filter out a skip amount for training.
def concatForComboSubset(data, vid_ids, front_trim: int = 0, end_trim: int = 0,
                         diff_order: int = 0, del_original_data: bool = False,
                         return_indices: bool = False):
    ret_val: typing.List[typing.Union[typing.Dict, NDArray]] = []
    num_vids = len(vid_ids)
    id_index_maps = []
    single_vid_bounds: typing.Dict[SkipSubsetKind, NDArray] = {}

    # process skips in reverse order
    for skip_ind in range(len(data) - 1, -1, -1):
        els_for_skip = data[skip_ind]
        # print("  - subset_via_ids creation")
        
        subset_via_ids = [els_for_skip[vi] for vi in vid_ids]

        concated = None
        # front_trim = 0 # May set this via param in future code.
        end = -end_trim if end_trim > 0 else None
        frame_counts = None
        if not subset_via_ids:
            break
        elif isinstance(subset_via_ids[0], dict):
            # print("  - dict() creation")
            if diff_order > 0:
                concated = {
                    k: np.concatenate([
                        np.diff(s[k][front_trim:end], diff_order, axis=0)
                        for s in subset_via_ids
                    ]) for k in subset_via_ids[0].keys()
                }
            else:
                concated = {
                    k: np.concatenate([
                        s[k][front_trim:end] for s in subset_via_ids
                    ]) for k in subset_via_ids[0].keys()
                }

            if return_indices:
                frame_counts = np.asarray([
                    len(next(iter(d.values()))) for d in subset_via_ids
                ])
        else:
            if diff_order > 0:
                concated = np.concatenate([
                    np.diff(svc[front_trim:end], diff_order, axis=0)
                    for svc in subset_via_ids
                ])
            else:
                concated = np.concatenate([
                    svc[front_trim:end] for svc in subset_via_ids
                ]) 
            
            if return_indices:
                frame_counts = np.asarray([len(d) for d in subset_via_ids])
        ret_val.insert(0, concated)

        if return_indices:
            frame_counts = typing.cast(NDArray, frame_counts) - diff_order
            id_index_maps.insert(
                0, np.repeat(np.arange(num_vids), frame_counts)
            )
            single_vid_bounds[skip_ind] = np.pad(
                np.cumsum(frame_counts), (1, 0)
            )

        if del_original_data:
            for ck in vid_ids:
                del els_for_skip[ck]
    if not return_indices:
        return ret_val
    vid_id_to_idx = {vid: i for i, vid in enumerate(vid_ids)}
    return ret_val, vid_ids, vid_id_to_idx, id_index_maps, single_vid_bounds



# This converts List[Dict[Any, NDArray]] items, which are lists of result 
# dicts of NDarrays indexed by frame skip, into a single 2D NDArray.
# It also returns the dictionary keys in the order that the columns appear in
# the 2D array so that we know which column is which.
def get2DArrayFromDataStruct(data: typing.List[typing.Dict[typing.Any, NDArray]], 
                            ks: typing.Optional[typing.List[MOTION_DATA_KEY_TYPE]] = None,
                            stack_axis: int = 0):
    if ks is None:
        ks = list(data[0].keys())
    concated = {k: np.concatenate([d[k] for d in data]) for k in ks}
    stacked = np.stack([concated[k] for k in ks], stack_axis)
    return ks, stacked

def _isFitted(transformer):
    return hasattr(transformer, "n_features_in_") and transformer.n_features_in_ > 0

# Type hint for a dict with items that are either (int, int) intervals or a bool
# numpy array of indices.
# For the ellipses type hint, the solution lies in the comments to a question
# posted 2022-03-04 by Kyle (1574952/kyle) titled "Type hint for Ellipsis":
# https://stackoverflow.com/questions/71355085/type-hint-for-ellipsis
# The solution's in a comment by juanpa.arrivillaga (5014455/juanpa-arrivillaga)
# and notes that pre Python 3.10, a solution is to `import builtins` and use the
# type "builtins.ellipsis", *and to make sure it's in quotes*!
IndDict: typing.TypeAlias = typing.Dict[
    typing.Any,
    typing.Union[typing.Tuple[int,int], NDArray, "builtins.ellipsis"]
]

# Gets the per-frame pose error in millimeters for a set of "labels" which 
# represent which physics-based motion model (non-regression-ML) chosen each 
# frame.
# Returns a dict that separates these MAEs based on frame category, e.g., skip
# amount.
def motionClassErrs(per_class_errs: NDArray, pred_labels: NDArray, inds_dict: IndDict):
    pred_labels_rs = pred_labels.reshape(-1,1)
    taken_errs = np.take_along_axis(per_class_errs, pred_labels_rs, axis=1)
    ret_dict: typing.Dict[typing.Any, NDArray] = dict()
    for k, inds in inds_dict.items():
        if inds is not None and isinstance(inds, tuple):
            ret_dict[k] = taken_errs[inds[0]:inds[1]]
        else:
            ret_dict[k] = taken_errs[inds]
        # {k: taken_errs[inds] for k, inds in inds_dict.items()}
    return ret_dict

# Same as the above, but returns MAE per inds_dict category rather than a whole
# list of per-frame errors.
def motionClassScores(per_class_errs: NDArray, pred_labels,
                      inds_dict: typing.Optional[IndDict] = None):
    if inds_dict is None:
        inds_dict = {SkipSubsetKind._all: ...}
    all_errs_dict = motionClassErrs(per_class_errs, pred_labels, inds_dict)
    mean_dict: typing.Dict[typing.Any, float] = {
        k: v.mean() for k, v in all_errs_dict.items()
    }
    return mean_dict

class UnitAwareScaler:
    _pos_vec3_keys = (
        MOTION_DATA.VEL_DEG1_VEC3, MOTION_DATA.VEL_DEG2_VEC3,
        MOTION_DATA.ACC_VEC3, MOTION_DATA.JERK_VEC3,
        MOTION_DATA.JERK_ERR_VEC3, MOTION_DATA.CRACKLE_VEC3
    )
    _rot_vec3_keys = (
        MOTION_DATA.ROTATION_VEC3, MOTION_DATA.ROT_ACC_VEC3,
        MOTION_DATA.ROT_JERK_VEC3
    )
    
    def __init__(self, data_column_keys, scale_rots: bool = True):
        self.column_keys = data_column_keys
        self.n_features_in_ = 0 # Because we haven't fitted yet.
        self.pos_scale = 0.0
        self.rot_scale = 0.0
        self.scale_rots = scale_rots

    @staticmethod
    def _is_positional(key):
        if isinstance(key, MOTION_DATA):
            return key in UnitAwareScaler._pos_vec3_keys
        elif isinstance(key, OTHER_DIRECTION):
            return True
        return False
    
    @staticmethod
    def _is_rotational(key):
        if isinstance(key, MOTION_DATA):
            return key in UnitAwareScaler._rot_vec3_keys
        return False

    def fit(self, X):
        n_keys = len(self.column_keys)
        if len(X) <= 0:
            raise ValueError("No data provided to fit for!")
        if len(X[0]) != n_keys:
            raise ValueError("Number of data columns does not match keys!")
        
        smd_cols = [
            (i, k) for i, k in enumerate(self.column_keys)
            if isinstance(k, SpecifiedMotionData)
        ]

        nc_pos_proj_inds = [
            i for i, k in smd_cols
            if (
                k.base_cat in UnitAwareScaler._pos_vec3_keys
                and k.ang_or_mag == ANG_OR_MAG.MAG_PROJ
            )
        ]
        nc_rot_proj_inds = [
            i for i, k in smd_cols
            if (
                k.base_cat in UnitAwareScaler._rot_vec3_keys
                and k.ang_or_mag == ANG_OR_MAG.MAG_PROJ
            )
        ]
        nc_pos_pos_inds = [
            i for i, k in smd_cols
            if (
                k.base_cat in UnitAwareScaler._pos_vec3_keys
                and self._is_positional(k.axis)
                and k.ang_or_mag == ANG_OR_MAG.MAG_DOT
            )
        ]
        nc_rot_rot_inds = [
            i for i, k in smd_cols
            if (
                k.base_cat in UnitAwareScaler._rot_vec3_keys
                and self._is_rotational(k.axis)
                and k.ang_or_mag == ANG_OR_MAG.MAG_DOT
            )
        ]
        nc_pos_rot_inds = [
            i for i, k in smd_cols
            if (
                (
                    (k.base_cat in UnitAwareScaler._pos_vec3_keys)
                    != self._is_positional(k.axis)
                )
                and k.ang_or_mag == ANG_OR_MAG.MAG_DOT
            )
        ]
        nc_ang_inds = [i for i, k in smd_cols if k.ang_or_mag == ANG_OR_MAG.ANG]


        self.pos_scale = np.mean(np.std(X[:, nc_pos_proj_inds], axis=0))
        self.rot_scale = 1.0
        if self.scale_rots:
            self.rot_scale = np.mean(np.std(X[:, nc_rot_proj_inds], axis=0))

        self.scale_ = np.empty(n_keys)
        self.scale_[nc_ang_inds] = 1.0
        self.scale_[nc_pos_proj_inds] = self.pos_scale
        self.scale_[nc_rot_proj_inds] = self.rot_scale
        self.scale_[nc_pos_pos_inds] = self.pos_scale * self.pos_scale
        self.scale_[nc_pos_rot_inds] = self.pos_scale * self.rot_scale
        self.scale_[nc_rot_rot_inds] = self.rot_scale * self.rot_scale

        self.mean_ = np.zeros(n_keys)

        scale_not_filled = np.full(n_keys, True)
        filled_int_inds = np.concatenate((
            nc_ang_inds, nc_pos_proj_inds, nc_rot_proj_inds, nc_pos_pos_inds,
            nc_pos_rot_inds, nc_rot_rot_inds
        )).astype(int)
        scale_not_filled[filled_int_inds] = False

        self.mean_[scale_not_filled] = np.mean(X[:, scale_not_filled], axis=0)
        self.scale_[scale_not_filled] = np.std(X[:, scale_not_filled], axis=0)
        self.n_features_in_ = n_keys

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        return (X * self.scale_) + self.mean_
    
class SkipSubsetKind(IntEnum):
    skip0 = 0
    skip1 = 1
    skip2 = 2
    _all = -1

    @staticmethod
    def max_step():
        return max(SkipSubsetKind) + 1
    
    @property
    def display_name(self):
        return "all" if self is SkipSubsetKind._all else self.name



class RowsAndColsHandler:
    _ordered_dsks = (
        DataSubsetKind.TRAIN,
        DataSubsetKind.VALIDATION,
        DataSubsetKind.TEST
    )

    def __init__(
        self,
        single_vid_bounds: typing.Dict[
            DataSubsetKind, typing.Dict[SkipSubsetKind, NDArray]
        ],
        skip_bounds: typing.Dict[DataSubsetKind, NDArray],
        motion_data_keys: typing.List[MOTION_DATA_KEY_TYPE],
        motion_mod_keys: typing.List[MOTION_MODEL],
        subset_ids: typing.Dict[DataSubsetKind, typing.List],
        rows_per_subset: typing.Dict[DataSubsetKind, int],
        supported_skips: typing.Optional[typing.List[SkipSubsetKind]] = None
    ):
        self.single_vid_bounds = single_vid_bounds
        self.skip_bounds = skip_bounds
        self.motion_data_keys = motion_data_keys
        self.motion_mod_keys = motion_mod_keys
        self.subset_ids = subset_ids
        self.rows_per_subset = rows_per_subset

        if supported_skips is None:
            supported_skips = list(SkipSubsetKind)
        self.supported_skips = supported_skips

        # Temporary initializations before calling _init_remaining:
        self.train_val_border_ind = 0
        self.val_test_border_ind = 0
        self.skip_bounds_structured: typing.Dict[
            DataSubsetKind, typing.Dict[SkipSubsetKind, typing.Tuple[int, int]]
        ] = {}

        self._init_remaining()

    def _init_remaining(self):
        self.train_val_border_ind = self.rows_per_subset[DataSubsetKind.TRAIN]
        self.val_test_border_ind = self.train_val_border_ind \
            + self.rows_per_subset[DataSubsetKind.VALIDATION]

        self.skip_bounds_structured.clear()

        for dsk, skip_arr in self.skip_bounds.items():
            d_for_dsk: typing.Dict[SkipSubsetKind, typing.Tuple[int, int]] = {}
            
            for sk in SkipSubsetKind:
                if sk == SkipSubsetKind._all:
                    d_for_dsk[sk] = (skip_arr[0], skip_arr[-1])
                else:
                    d_for_dsk[sk] = (skip_arr[sk.value], skip_arr[sk.value + 1])
            self.skip_bounds_structured[dsk] = d_for_dsk

    def getAllIDs(self):
        # Using list unpacking here so that (a) the order is explicit and (b) if
        # some code accidentally uses NDArrays as IDs, then I don't get
        # accidental summations instead of concatenation by using "+".
        return [
            *self.subset_ids[DataSubsetKind.TRAIN],
            *self.subset_ids[DataSubsetKind.VALIDATION],
            *self.subset_ids[DataSubsetKind.TEST]
        ]
    
    def _slice_by_subset(self, arr: NDArray, subset_kind: DataSubsetKind):
        if subset_kind == DataSubsetKind.TRAIN:
            return arr[:self.train_val_border_ind]
        elif subset_kind == DataSubsetKind.VALIDATION:
            return arr[self.train_val_border_ind:self.val_test_border_ind]
        elif subset_kind == DataSubsetKind.TEST:
            return arr[self.val_test_border_ind:]
        elif subset_kind == DataSubsetKind.WHOLE:
            return arr
        else:
            raise ValueError(
                "Unrecognized DataSubsetKind: {}".format(subset_kind)
            )

    def _all_slices_by_subset(self, arr: NDArray):
        return {
            k: self._slice_by_subset(arr, k)
            for k in DataSubsetKind.nonWholeValues()
        }
    
    def whole_data_skip_slices(self, arr: NDArray, skip: SkipSubsetKind):
        tr_va_te_dict = self._all_slices_by_subset(arr)
        
        return [
            self.getSelectionData(tr_va_te_dict[dsk], dsk, skip)
            for dsk in RowsAndColsHandler._ordered_dsks
        ]

    
    def skip_selection(self, arr_dict: typing.Dict[str, NDArray],
                       skip: SkipSubsetKind, *, delete_orig: bool = False):
        '''Function for removing data for all skip values except one.
        Used, for example, when loading selectively from disk.'''
        if skip == SkipSubsetKind._all:
            return arr_dict
        
        new_counts: typing.Dict[DataSubsetKind, int] = {}

        ret_dict: typing.Dict[str, NDArray] = {}

        for i, (orig_k, orig_data) in enumerate(arr_dict.items()):
            sliced_arrs = self.whole_data_skip_slices(orig_data, skip)
            ret_dict[orig_k] = np.concatenate(sliced_arrs, axis=0)
            if i == 0:
                ordered_lens = [len(s) for s in sliced_arrs]
                new_counts = dict(
                    zip(RowsAndColsHandler._ordered_dsks, ordered_lens)
                    )
                new_counts[DataSubsetKind.WHOLE] = sum(ordered_lens)
            if delete_orig:
                arr_dict[orig_k] = np.empty(0)
                del orig_data
        if delete_orig:
            del arr_dict
        
        # Fix "self" based on the fact that we're looking at a single skip only.
        self.single_vid_bounds = {
            dsk: {skip: sub_dict[skip]}
            for dsk, sub_dict in self.single_vid_bounds.items()
        }
        bounds_dtype = next(iter(self.skip_bounds.values())).dtype
        self.skip_bounds.clear()
        
        bounds_len = 1 + SkipSubsetKind.max_step()

        for dsk in DataSubsetKind.nonWholeValues():
            curr_skip_bounds = np.zeros(bounds_len, dtype=bounds_dtype)
            curr_skip_bounds[(skip + 1):] = new_counts[dsk]
            self.skip_bounds[dsk] = curr_skip_bounds

        self.rows_per_subset = new_counts
        
        self.supported_skips = [skip]
        
        self._init_remaining()

        return ret_dict


    def getSelectionData(self, full_arr: NDArray, subset_kind: DataSubsetKind,
                         skip_amt: SkipSubsetKind, vid_id = None):
        '''NOTE: The upper bound returned is exclusive, so that you can just
        use them as indices arr[bounds[0]:bounds[1]] directly.'''
        # if skip_amt == SkipSubsetKind._all:
        #     # Would it even make sense to support this? Maybe ValueError?
        #     raise NotImplementedError("Skip selection of _all not supported.")
        if subset_kind == DataSubsetKind.WHOLE:
            # Same here.
            raise NotImplementedError("WHOLE selection not supported!")
        if skip_amt not in self.supported_skips:
            return full_arr[:0]

        dsk_subset_arr = full_arr
        if len(full_arr) != self.rows_per_subset[subset_kind]:
            if len(full_arr) == self.rows_per_subset[DataSubsetKind.WHOLE]:
                dsk_subset_arr = self._slice_by_subset(full_arr, subset_kind)
            else:
                raise ValueError("Bad row count!")

        if skip_amt == SkipSubsetKind._all:
            per_skip_bounds = (None, None)
        else:
            per_skip_bounds = self.skip_bounds[subset_kind][skip_amt:(skip_amt + 2)]
        skip_subset = dsk_subset_arr[per_skip_bounds[0]:per_skip_bounds[1]]

        if vid_id is None:
            return skip_subset
        
        bounds_in_skip = self.single_vid_bounds[subset_kind][skip_amt]
        
        id_ind = self.subset_ids[subset_kind].index(vid_id)
        row_idxs = bounds_in_skip[id_ind:(id_ind + 2)]
        return skip_subset[row_idxs[0]:row_idxs[1]] # Values for this video.


class DataOrganizer(RowsAndColsHandler):
    class _SubsetConcats(typing.NamedTuple):
        ids: typing.List
        concat_data: NDArray
        concat_labels: typing.Optional[NDArray]
        concat_class_errs: typing.Optional[NDArray]
        single_vid_bounds: typing.Dict[SkipSubsetKind, NDArray]
        skip_bounds: NDArray
    
    def __init__(self, row_col_handler: RowsAndColsHandler,
                 concat_whole_data: NDArray,
                 concat_whole_class_errs: typing.Optional[NDArray],
                 concat_whole_labels: typing.Optional[NDArray],
                 loader_class: typing.Type[gtc.PoseLoader]
                 ):
        super(DataOrganizer, self).__init__(
            single_vid_bounds=row_col_handler.single_vid_bounds,
            skip_bounds=row_col_handler.skip_bounds,
            motion_data_keys=row_col_handler.motion_data_keys,
            motion_mod_keys=row_col_handler.motion_mod_keys,
            subset_ids=row_col_handler.subset_ids,
            rows_per_subset=row_col_handler.rows_per_subset,
            supported_skips=row_col_handler.supported_skips
        )

        # Now for the numpy arrays.
        self.concat_whole_data = concat_whole_data
        slices = self._all_slices_by_subset(concat_whole_data)
        self.concat_train_data = slices[DataSubsetKind.TRAIN]
        self.concat_validation_data = slices[DataSubsetKind.VALIDATION]
        self.concat_test_data = slices[DataSubsetKind.TEST]
        
        if concat_whole_class_errs is not None:
            self.concat_whole_class_errs = concat_whole_class_errs
            slices = self._all_slices_by_subset(concat_whole_class_errs)
            self.concat_train_class_errs = slices[DataSubsetKind.TRAIN]
            self.concat_validation_class_errs = slices[DataSubsetKind.VALIDATION]
            self.concat_test_class_errs = slices[DataSubsetKind.TEST]

        if concat_whole_labels is not None:
            self.concat_whole_labels = concat_whole_labels
            slices = self._all_slices_by_subset(concat_whole_labels)
            self.concat_train_labels = slices[DataSubsetKind.TRAIN]
            self.concat_validation_labels = slices[DataSubsetKind.VALIDATION]
            self.concat_test_labels = slices[DataSubsetKind.TEST]

        self.LoaderClass = loader_class

        # Optional attributes to be set in later code.
        self.col_subset_whole = np.empty(0)
        self.col_subset_train = np.empty(0)
        self.col_subset_test = np.empty(0)
        self.col_subset_validation = np.empty(0)
        self.motion_data_key_subset = []


    def reset_col_subset_slices(self, new_source_data: NDArray):
        num_orig_cols = self.col_subset_whole.shape[1]
        if num_orig_cols > new_source_data.shape[1]:
            raise ValueError("Replacement data has FEWER columns!")

        del self.col_subset_whole
        del self.col_subset_train
        del self.col_subset_validation
        del self.col_subset_test

        self.col_subset_whole = new_source_data[:, :num_orig_cols]

        self._set_col_subset_slices()
        return 
    
    def _set_col_subset_slices(self):
        slices = self._all_slices_by_subset(self.col_subset_whole)
        self.col_subset_train = slices[DataSubsetKind.TRAIN]
        self.col_subset_validation = slices[DataSubsetKind.VALIDATION]
        self.col_subset_test = slices[DataSubsetKind.TEST]
        return

    @ staticmethod
    def FromCalcs(loader_class: typing.Type[gtc.PoseLoader], all_motion_data,
                  min_norm_labels, err_norm_lists, 
                  train_ids, test_ids, validation_ids = None,
                  motion_data_keys: typing.Optional[typing.List[MOTION_DATA_KEY_TYPE]]=None,
                  del_original_data: bool = False):
        
        # It is fine if motion_data_keys is None for now because, if it is, it
        # will be overwritten later.
        if motion_data_keys is None:
            arbitrary_seq_dict = next(iter(all_motion_data[0].values()))
            motion_data_keys = list(arbitrary_seq_dict.keys())
        
        _timestep_ind = motion_data_keys.index(MOTION_DATA.TIMESTEP)
        
        # We'll specify the column names/order manually for this one.
        motion_mod_keys = [
            MOTION_MODEL(i) for i in range(1, len(MOTION_MODEL) + 1)
        ] 
 
        DSK = DataSubsetKind
        subset_ids: typing.Dict[DataSubsetKind, typing.List] = dict()
        single_vid_bounds: typing.Dict[DataSubsetKind, typing.List[NDArray]] \
            = dict()
        skip_bounds: typing.Dict[DataSubsetKind, NDArray] = dict()

        # print("Starting training data:")
        # Training data:
        sd = DataOrganizer._splitAndConcatSubset(
            all_motion_data, min_norm_labels, err_norm_lists,
            _timestep_ind, motion_data_keys, motion_mod_keys, train_ids,
            del_original_data
        )
        subset_ids[DSK.TRAIN], concat_train_data = sd[:2]
        concat_train_labels, concat_train_class_errs = sd[2:4]
        single_vid_bounds[DSK.TRAIN] = sd[4]
        skip_bounds[DSK.TRAIN] = sd[5]

        # print("Starting testing data:")
        # Test data:
        sd = DataOrganizer._splitAndConcatSubset(
            all_motion_data, min_norm_labels, err_norm_lists,
            _timestep_ind, motion_data_keys, motion_mod_keys, test_ids,
            del_original_data
        )
        subset_ids[DSK.TEST], concat_test_data = sd[:2]
        concat_test_labels, concat_test_class_errs = sd[2:4]
        single_vid_bounds[DSK.TEST] = sd[4]
        skip_bounds[DSK.TEST] = sd[5]

        # print("Starting vals data:")
        # Validation data:
        sd = DataOrganizer._splitAndConcatSubset(
            all_motion_data, min_norm_labels, err_norm_lists,
            _timestep_ind, motion_data_keys, motion_mod_keys, validation_ids,
            del_original_data
        )
        subset_ids[DSK.VALIDATION], concat_validation_data = sd[:2]
        concat_validation_labels, concat_validation_class_errs = sd[2:4]
        single_vid_bounds[DSK.VALIDATION] = sd[4]
        skip_bounds[DSK.VALIDATION] = sd[5]

        rows_per_dsk: typing.Dict[DataSubsetKind, int] = {}
        rows_per_dsk[DataSubsetKind.TRAIN] = len(concat_train_data)
        rows_per_dsk[DataSubsetKind.VALIDATION] = len(concat_validation_data)
        rows_per_dsk[DataSubsetKind.TEST] = len(concat_test_data)
        rows_per_dsk[DataSubsetKind.WHOLE] = sum(rows_per_dsk.values())

        concat_whole_data = np.concatenate((
            concat_train_data, concat_validation_data, concat_test_data
        ), axis=0)
        del concat_train_data
        del concat_validation_data
        del concat_test_data

        if err_norm_lists is not None:
            concat_whole_class_errs = np.concatenate((
                concat_train_class_errs, concat_validation_class_errs,
                concat_test_class_errs
            ), axis=0)
            del concat_train_class_errs
            del concat_validation_class_errs
            del concat_test_class_errs

        if min_norm_labels is not None:
            concat_whole_labels = np.concatenate((
                concat_train_labels, concat_validation_labels, concat_test_labels
            ), axis=0)
            del concat_train_labels
            del concat_validation_labels
            del concat_test_labels

        non_np = RowsAndColsHandler(
            single_vid_bounds, skip_bounds, motion_data_keys, motion_mod_keys,
            subset_ids, rows_per_dsk
        )

        return DataOrganizer(
            non_np, concat_whole_data, concat_whole_class_errs,
            concat_whole_labels, loader_class
        )

        # self.untransformed_col_subset_train = empty_np
        # self.untransformed_col_subset_test = empty_np


    @staticmethod
    def _splitAndConcatSubset(all_motion_data, min_norm_labels, err_norm_lists,
                              timestep_ind: int,
                              motion_data_keys: typing.List[MOTION_DATA_KEY_TYPE],
                              motion_mod_keys: typing.List[MOTION_MODEL],
                              subset_ids: typing.Optional[typing.List],
                              del_original_data: bool):
        skip_ul = 1 + max(SkipSubsetKind).value
        if subset_ids is None:
            d_empty = np.empty((0, len(motion_data_keys)))
            lab_empty = None
            if min_norm_labels is not None:
                lab_empty = np.empty((0, ), dtype=int)
            e_empty = None
            if err_norm_lists is not None:
                e_empty = np.empty((0, len(motion_mod_keys)))
            vid_bounds_empty = {
                s: np.empty((0, ), dtype=int)
                for s in SkipSubsetKind if s >= 0
            }
            skip_bounds_na = np.zeros(skip_ul + 1, dtype=int)
            return DataOrganizer._SubsetConcats(
                [], d_empty, lab_empty, e_empty, vid_bounds_empty,
                skip_bounds_na
            )

        better_ids = sorted(set(subset_ids))
        
        # The below gets the data subset, but leaves them currently still 
        # separated by skip amount and by column key (if present). E.g., one can
        # quickly slap a "[0]" at the end of each line to just look at data for
        # one skip amount.
        labels = None
        if min_norm_labels is not None:
            # print("- Starting labels.")
            labels = concatForComboSubset(
                min_norm_labels, better_ids, del_original_data=del_original_data
            )
        errs = None
        if err_norm_lists is not None:
            # print("- Starting errs.")
            errs = concatForComboSubset(
                err_norm_lists, better_ids, del_original_data=del_original_data
            )
        # print("- Starting data.")
        data_and_ids = concatForComboSubset(
            all_motion_data, better_ids, del_original_data=del_original_data,
            return_indices=True
        )
        data, reorged_ids, ord_of_ids, row_vid_ids, vid_id_ranges = data_and_ids

        if labels is not None:
            # print("- Cat labels.")
            # Get 2D NDArrays from the above.
            concat_labels = np.concatenate(typing.cast(NDArray, labels))
            del labels # Delete now since it doesn't get used later anyway.

        if motion_data_keys is None:
            raise ValueError((
                "Must specify motion_data_keys order because default does not "
                "get saved anywhere!"
            ))

        # print("- Cat data.")
        # Get the "keys" as another return value so that we know the column 
        # names/order.
        _, concat_data = get2DArrayFromDataStruct(
            data, motion_data_keys, stack_axis=-1
        )
        del data # Delete now since it doesn't get used later anyway.
        
        if errs is not None:
            # print("- Cat errs.")
            _, concat_errs = get2DArrayFromDataStruct(
                errs, motion_mod_keys, stack_axis=-1
            )
            del errs # Delete now since it doesn't get used later anyway.

        # print("- Finding skip inds.\n")
        # Get the indices of each skip amount inside the concatenated 2D array
        # we created above. There might be a "smarter" way to do this given how
        # things were previously split into lists by skip amount, but whatever.
        skip_inds = []
        for i in range(1,4): # We have data for frame steps of 1, 2, and 3.
            skip_inds.append(concat_data[:, timestep_ind] == i)

        # Convert the above 3-item lists into dicts.
        skip_d = {SkipSubsetKind(i): skip_inds[i] for i in range(skip_ul)}
        skip_d[SkipSubsetKind._all] = ... # my_np_array[...] gets all elements.


        skip_counts = np.asarray([
            vid_id_ranges[s][-1] for s in range(SkipSubsetKind.max_step())]
        )
        skip_bounds = np.pad(np.cumsum(skip_counts), (1, 0))

        return DataOrganizer._SubsetConcats(
            reorged_ids, concat_data, concat_labels, concat_errs, vid_id_ranges,
            skip_bounds
        )

    def setPickAndTransform(self, columns: NDArray, transformer = None,
                            free_orig_mem: bool = False):
        # Get the data subset for the selected non-collinear columns.
        self.col_subset_whole = self.concat_whole_data[:, columns]
        
        # Our columns might be a boolean numpy array or an array of int indices.
        # We'll create `column_nums` s.t. the latter format is guaranteed.
        column_nums = columns
        if columns.dtype == bool:
            column_nums, = np.nonzero(columns)

        self.motion_data_key_subset = [
            self.motion_data_keys[i] for i in column_nums
        ]

        if transformer is not None:
            if not _isFitted(transformer):
                transformer.fit(
                    self.col_subset_whole[:self.train_val_border_ind]
                )
            # I've written _isfitted() to use hasattr with the assumption that
            # transformers/calers will not have the n_features_in_ attr until
            # fitted. If a scikit learn version changes the name or behaviour
            # of this feature, then _isFitted() will always return False. So now
            # I'm making a check for something like this:
            if not _isFitted(transformer): # SHOULD be fitted NOW!
                raise Exception("_isFitted(transformer) false after fit!")
            
            # self.untransformed_col_subset_train = self.col_subset_train
            # self.untransformed_col_subset_test = self.col_subset_test
            self.col_subset_whole = transformer.transform(self.col_subset_whole)
            

            # Make sure OneHot columns are not scaled/shifted!
            for new_i, orig_i in enumerate(column_nums):
                if isinstance(self.motion_data_keys[orig_i], OneHotMotionData):
                    self.col_subset_whole[:, new_i] = \
                        self.concat_whole_data[:, orig_i]
                    
            self._set_col_subset_slices()

        if free_orig_mem:
            del self.concat_train_data
            del self.concat_test_data
            del self.concat_validation_data
            del self.concat_whole_data
            print("Did not finish implementing deletions!")

        return
    
    # def getColumnSubsetBySeqSubset(self, subset: DataSubsetKind):
    #     ret: NDArray
    #     if subset == DataSubsetKind.TRAIN:
    #         ret = self.col_subset_train
    #     elif subset == DataSubsetKind.TEST:
    #         ret = self.col_subset_test
    #     elif subset == DataSubsetKind.VALIDATION:
    #         ret = self.col_subset_validation
    #     elif subset == DataSubsetKind.WHOLE:
    #         # TODO: The way things are currently structured, I would have to
    #         # concatenate the arrays together, which would use much more RAM.
    #         # TRY THAT FIRST, but then should modify so that the whole dataset
    #         # is loaded as a single numpy array and then the train, test, and
    #         # validation arrays are just slices into it!
    #         raise NotImplementedError("Need better handling of WHOLE option!")
    #     else:
    #         raise ValueError(
    #             "Subset kind {} is not train, test, or validation!".format(
    #                 subset
    #             )
    #         )
    #     return ret

    def getClassErrsTrain(self, pred_labels):
        return motionClassErrs(
            self.concat_train_class_errs, pred_labels,
            self.skip_bounds_structured[DataSubsetKind.TRAIN]
        )
    def getClassErrsTest(self, pred_labels):
        return motionClassErrs(
            self.concat_test_class_errs, pred_labels,
            self.skip_bounds_structured[DataSubsetKind.TEST]
        )
    
    def getClassScoresTrain(self, pred_labels):
        return motionClassScores(
            self.concat_train_class_errs, pred_labels,
            self.skip_bounds_structured[DataSubsetKind.TRAIN]
        )
    def getClassScoresTest(self, pred_labels):
        return motionClassScores(
            self.concat_test_class_errs, pred_labels,
            self.skip_bounds_structured[DataSubsetKind.TEST]
        )

    @staticmethod
    def generatedDataPath():
        DATA_PATH = "./generated_data/"
        if not os.path.exists(DATA_PATH):
            os.mkdir(DATA_PATH)
        return pathlib.Path(DATA_PATH)

    @staticmethod
    def getDumpFilenameNP(loaderClass: typing.Type[gtc.PoseLoader]):
        return DataOrganizer.generatedDataPath() / (
            "processed_columns_" + loaderClass.datasetName() + ".npz"
        )
    
    @staticmethod
    def getDumpFilenamePkl(loaderClass: typing.Type[gtc.PoseLoader]):
        return DataOrganizer.generatedDataPath() / (
            "other_processed_data_" + loaderClass.datasetName() + ".pkl"
        )
    
    def dump(self, compress: bool,
             np_file: typing.Optional[typing.Union[str, os.PathLike]] = None,
             pkl_file: typing.Optional[typing.Union[str, os.PathLike]] = None):
        if np_file is None:
            np_file = self.getDumpFilenameNP(self.LoaderClass)
        if pkl_file is None:
            pkl_file = self.getDumpFilenamePkl(self.LoaderClass)

        save_dict = {
            "concat_whole_data": self.concat_whole_data,
            "concat_whole_class_errs": self.concat_whole_class_errs,
            "concat_whole_labels": self.concat_whole_labels,
        }

        non_np = RowsAndColsHandler(
            self.single_vid_bounds, self.skip_bounds,
            self.motion_data_keys, self.motion_mod_keys, self.subset_ids,
            self.rows_per_subset
        )

        if compress:
            np.savez_compressed(np_file, **save_dict)
        else:
            np.savez(np_file, **save_dict)
        
        with open(pkl_file, "wb") as f:
            pickle.dump(non_np, f)


    @staticmethod
    def load(loader_class: typing.Type[gtc.PoseLoader],
             np_file: typing.Optional[typing.Union[str, bytes, os.PathLike]] = None,
             pkl_file: typing.Optional[typing.Union[str, bytes, os.PathLike]] = None,
             *,
             skip_filter: typing.Union[int, SkipSubsetKind] = SkipSubsetKind._all):
        if np_file is None:
            np_file = DataOrganizer.getDumpFilenameNP(loader_class)
        if pkl_file is None:
            pkl_file = DataOrganizer.getDumpFilenamePkl(loader_class)

        with open(pkl_file, "rb") as f:
            nnpl = typing.cast(RowsAndColsHandler, pickle.load(f))

        npl = np.load(np_file, allow_pickle=False)

        arrs_dict = {k: npl[k] for k in npl.files}
        sliced = nnpl.skip_selection(arrs_dict, skip_filter, delete_orig=True)
        ret = DataOrganizer(
            nnpl,
            sliced["concat_whole_data"],
            sliced["concat_whole_class_errs"],
            sliced["concat_whole_labels"],
            loader_class
        )

        npl.close()
        return ret



def joinArrays(pts_lists: typing.List[NDArray]):
    join_list = []
    dim = pts_lists[0].shape[-1]
    na_val = np.full((1, dim), np.nan)
    for idx, pts in enumerate(pts_lists):
        join_list.append(pts)
        
        # Add a separator if not the last sequence
        if idx < len(pts_lists) - 1:
            join_list.append(na_val)
    return np.concatenate(join_list, axis=0)

def getDisjointSegs(source_pts: NDArray, dest_pts: NDArray):
    source_rs = source_pts.reshape(-1, 3)
    dest_rs = dest_pts.reshape(-1, 3)
    nans = np.full_like(dest_rs, np.nan)
    return np.stack((source_rs, dest_rs, nans), axis=1).reshape(-1, 3)
