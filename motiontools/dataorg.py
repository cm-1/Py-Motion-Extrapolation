import typing
from enum import Enum, IntEnum

import numpy as np
from numpy.typing import NDArray

from motiontools.posefeatures import (
    MOTION_MODEL, MOTION_DATA, SpecifiedMotionData, ANG_OR_MAG, OTHER_DIRECTION
)
from motiontools.posefeatures import OneHotMotionData, MOTION_DATA_KEY_TYPE

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
def concatForComboSubset(data, combo_subset,
                         front_trim: int = 0, end_trim: int = 0,
                         diff_order: int = 0, del_original_data: bool = False):
    ret_val: typing.List[typing.Union[typing.Dict, NDArray]] = []
    for skip_ind in range(len(data) - 1, -1, -1):
        els_for_skip = data[skip_ind]
        # print("  - subset_via_ids creation")
        subset_via_ids = [els_for_skip[ck] for ck in combo_subset]
        concated = None
        # front_trim = 0 # May set this via param in future code.
        end = -end_trim if end_trim > 0 else None
        if len(subset_via_ids) == 0:
            return []
        elif isinstance(subset_via_ids[0], dict):
            concated = dict()
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
        else:
            if diff_order > 0:
                concated = np.concatenate([
                    np.diff(svc[front_trim:end], diff_order, axis=0)
                    for svc in subset_via_ids])
            else:
                concated = np.concatenate([
                    svc[front_trim:end] for svc in subset_via_ids
                ]) 
        ret_val.insert(0, concated)
        
        if del_original_data:
            for ck in combo_subset:
                del els_for_skip[ck]
    return ret_val


# This converts List[Dict[Any, NDArray]] items, which are lists of result 
# dicts of NDarrays indexed by frame skip, into a single 2D NDArray.
# It also returns the dictionary keys in the order that the columns appear in
# the 2D array so that we know which column is which.
def get2DArrayFromDataStruct(data: typing.List[typing.Dict[typing.Any, NDArray]], 
                            ks: typing.List[MOTION_DATA_KEY_TYPE] = None,
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
IndDict = typing.Dict[
    typing.Any, 
    typing.Union[typing.Tuple[int,int], NDArray] # NDArray holds bool indices.
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
        {k: taken_errs[inds] for k, inds in inds_dict.items()}
    return ret_dict

# Same as the above, but returns MAE per inds_dict category rather than a whole
# list of per-frame errors.
def motionClassScores(per_class_errs: NDArray, pred_labels, inds_dict = None):
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
        MOTION_DATA.ROTATION_VEC3, MOTION_DATA.ROT_ACC_VEC3
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

class DataSubsetKind(Enum):
    TRAIN = 1
    TEST = 2
    VALIDATION = 3
    WHOLE = 4

    @staticmethod
    def nonWholeValues():
        return (
            DataSubsetKind.TRAIN, DataSubsetKind.TEST, DataSubsetKind.VALIDATION
        )
    
class SkipSubsetKind(IntEnum):
    skip0 = 0
    skip1 = 1
    skip2 = 2
    _all = -1

    @property
    def display_name(self):
        return "all" if self is SkipSubsetKind._all else self.name
    
class DataOrganizer:
    class _SubsetConcats(typing.NamedTuple):
        ids: typing.Iterable
        concat_labels: NDArray
        concat_data: NDArray
        concat_class_errs: NDArray
        skip_inds_dict: typing.Dict[SkipSubsetKind, NDArray]
        
    def __init__(self, all_motion_data, min_norm_labels, err_norm_lists, 
                 train_ids, test_ids, validation_ids = None, 
                 motion_data_keys: typing.Optional[typing.List[MOTION_DATA_KEY_TYPE]]=None,
                 del_original_data: bool = False):
        
        # It is fine if this is None for now because, if it is, it will be
        # overwritten later.
        self.motion_data_keys = motion_data_keys

        if motion_data_keys is None:
            arbitrary_seq_dict = next(iter(all_motion_data[0].values()))
            self.motion_data_keys = list(arbitrary_seq_dict.keys())
        
        self._timestep_ind = self.motion_data_keys.index(MOTION_DATA.TIMESTEP)
        
        # We'll specify the column names/order manually for this one.
        self.motion_mod_keys = [
            MOTION_MODEL(i) for i in range(1, len(MOTION_MODEL) + 1)
        ] 
 
        DSK = DataSubsetKind
        self.subset_ids: typing.Dict[DataSubsetKind, NDArray] = dict()
        self.subset_skip_inds: typing.Dict[
            DataSubsetKind, typing.Dict[SkipSubsetKind, NDArray]
        ] = dict()

        # print("Starting training data:")
        # Training data:
        sd = self._splitAndConcatSubset(
            all_motion_data, min_norm_labels, err_norm_lists, train_ids,
            del_original_data
        )
        self.subset_ids[DSK.TRAIN], self.concat_train_labels = sd[:2]
        self.concat_train_data, self.concat_train_class_errs = sd[2:4]
        self.subset_skip_inds[DSK.TRAIN] = sd[4]

        # print("Starting testing data:")
        # Test data:
        sd = self._splitAndConcatSubset(
            all_motion_data, min_norm_labels, err_norm_lists, test_ids,
            del_original_data
        )
        self.subset_ids[DSK.TEST], self.concat_test_labels = sd[:2]
        self.concat_test_data, self.concat_test_class_errs = sd[2:4]
        self.subset_skip_inds[DSK.TEST] = sd[4]

        # print("Starting vals data:")
        # Validation data:
        sd = self._splitAndConcatSubset(
            all_motion_data, min_norm_labels, err_norm_lists, validation_ids,
            del_original_data
        )
        self.subset_ids[DSK.VALIDATION], self.concat_validation_labels = sd[:2]
        self.concat_validation_data, self.concat_validation_class_errs = sd[2:4]
        self.subset_skip_inds[DSK.VALIDATION] = sd[4]

        
        # Optional attributes to be set in later code.
        self.col_subset_train = np.empty(0)
        self.col_subset_test = np.empty(0)
        self.col_subset_validation = np.empty(0)
        self.motion_data_key_subset = []

        # self.untransformed_col_subset_train = empty_np
        # self.untransformed_col_subset_test = empty_np


    def _splitAndConcatSubset(self, all_motion_data, min_norm_labels,
                              err_norm_lists,
                              subset_ids: typing.Optional[typing.Iterable],
                              del_original_data: bool):
        if subset_ids is None:
            d_empty = np.empty((0, len(self.motion_data_keys)))
            lab_empty = np.empty((0, ), dtype=int)
            e_empty = np.empty((0, len(self.motion_mod_keys)))
            inds_empty = {k: ... for k in SkipSubsetKind}
            return DataOrganizer._SubsetConcats(
                [], lab_empty, d_empty, e_empty, inds_empty
            )

        
        # The below gets the data subset, but leaves them currently still 
        # separated by skip amount and by column key (if present). E.g., one can
        # quickly slap a "[0]" at the end of each line to just look at data for
        # one skip amount.
        # print("- Starting labels.")
        labels = concatForComboSubset(
            min_norm_labels, subset_ids, del_original_data=del_original_data
        )
        # print("- Starting errs.")
        errs = concatForComboSubset(
            err_norm_lists, subset_ids, del_original_data=del_original_data
        )
        # print("- Starting data.")
        data = concatForComboSubset(
            all_motion_data, subset_ids, del_original_data=del_original_data
        )

        # print("- Cat labels.")
        # Get 2D NDArrays from the above.
        concat_labels = np.concatenate(labels)
        del labels # Delete now since it doesn't get used later anyway.

        # print("- Cat data.")
        # Get the "keys" as another return value so that we know the column 
        # names/order.
        self.motion_data_keys, concat_data = get2DArrayFromDataStruct(
            data, self.motion_data_keys, stack_axis=-1
        )
        del data # Delete now since it doesn't get used later anyway.
        
        # print("- Cat errs.")
        _, concat_errs = get2DArrayFromDataStruct(
            errs, self.motion_mod_keys, stack_axis=-1
        )
        del errs # Delete now since it doesn't get used later anyway.

        # print("- Finding skip inds.\n")
        # Get the indices of each skip amount inside the concatenated 2D array
        # we created above. There might be a "smarter" way to do this given how
        # things were previously split into lists by skip amount, but whatever.
        skip_inds = []
        for i in range(1,4): # We have data for frame steps of 1, 2, and 3.
            skip_inds.append(concat_data[:, self._timestep_ind] == i)

        # Convert the above 3-item lists into dicts.
        skip_d = {SkipSubsetKind(i): skip_inds[i] for i in range(3)}
        skip_d[SkipSubsetKind._all] = ... # my_np_array[...] gets all elements.

        return DataOrganizer._SubsetConcats(
            subset_ids, concat_labels, concat_data, concat_errs, skip_d
        )

    def setPickAndTransform(self, columns: NDArray, transformer = None,
                            free_orig_mem: bool = False):
        # Get the data subset for the selected non-collinear columns.
        self.col_subset_train = self.concat_train_data[:, columns]
        self.col_subset_test = self.concat_test_data[:, columns]
        self.col_subset_validation = self.concat_validation_data[:, columns]
        
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
                transformer.fit(self.col_subset_train)
            # I've written _isfitted() to use hasattr with the assumption that
            # transformers/calers will not have the n_features_in_ attr until
            # fitted. If a scikit learn version changes the name or behaviour
            # of this feature, then _isFitted() will always return False. So now
            # I'm making a check for something like this:
            if not _isFitted(transformer): # SHOULD be fitted NOW!
                raise Exception("_isFitted(transformer) false after fit!")
            
            # self.untransformed_col_subset_train = self.col_subset_train
            # self.untransformed_col_subset_test = self.col_subset_test
            self.col_subset_train = transformer.transform(self.col_subset_train)
            self.col_subset_test = transformer.transform(self.col_subset_test)
            if len(self.col_subset_validation) > 0:
                # StandardScaler (and probably others) raise exceptions for
                # input shapes (0, ...).
                self.col_subset_validation = transformer.transform(
                    self.col_subset_validation
                )
            # Make sure OneHot columns are not scaled/shifted!
            
            for new_i, orig_i in enumerate(column_nums):
                if isinstance(self.motion_data_keys[orig_i], OneHotMotionData):
                    self.col_subset_train[:, new_i] = \
                        self.concat_train_data[:, orig_i]
                    self.col_subset_test[:, new_i] = \
                        self.concat_test_data[:, orig_i]
                    self.col_subset_validation[:, new_i] = \
                        self.concat_validation_data[:, orig_i]

        if free_orig_mem:
            del self.concat_train_data
            del self.concat_test_data
            del self.concat_validation_data
            print("Did not finish implementing deletions!")

        return
    
    def getColumnSubsetBySeqSubset(self, subset: DataSubsetKind):
        ret: NDArray
        if subset == DataSubsetKind.TRAIN:
            ret = self.col_subset_train
        elif subset == DataSubsetKind.TEST:
            ret = self.col_subset_test
        elif subset == DataSubsetKind.VALIDATION:
            ret = self.col_subset_validation
        else:
            raise ValueError(
                "Subset kind {} is not train, test, or validation!".format(
                    subset
                )
            )
        return ret

    def getClassErrsTrain(self, pred_labels):
        return motionClassErrs(
            self.concat_train_class_errs, pred_labels,
            self.subset_skip_inds[DataSubsetKind.TRAIN]
        )
    def getClassErrsTest(self, pred_labels):
        return motionClassErrs(
            self.concat_test_class_errs, pred_labels,
            self.subset_skip_inds[DataSubsetKind.TEST]
        )
    
    def getClassScoresTrain(self, pred_labels):
        return motionClassScores(
            self.concat_train_class_errs, pred_labels,
            self.subset_skip_inds[DataSubsetKind.TRAIN]
        )
    def getClassScoresTest(self, pred_labels):
        return motionClassScores(
            self.concat_test_class_errs, pred_labels,
            self.subset_skip_inds[DataSubsetKind.TEST]
        )

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