import typing

import numpy as np
from numpy.typing import NDArray

from motiontools.posefeatures import MOTION_MODEL, MOTION_DATA, MOTION_DATA_KEY_TYPE

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
def concatForComboSubset(data, combo_subset): #, front_trim: int = 0):
    ret_val: typing.List[typing.Union[typing.Dict, NDArray]] = []
    for els_for_skip in data:
        subset_via_ids = [els_for_skip[ck] for ck in combo_subset]
        concated = None
        front_trim = 0 # May set this via param in future code.
        if len(subset_via_ids) == 0:
            return []
        elif isinstance(subset_via_ids[0], dict):
            concated = dict()
            for k in subset_via_ids[0].keys():
                concated[k] = np.concatenate([
                    svc[k][front_trim:] for svc in subset_via_ids
                ])
        else:
            concated = np.concatenate([
                svc[front_trim:] for svc in subset_via_ids
            ]) 
        ret_val.append(concated)
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

# Type hint for a dict with string keys and items that are either (int, int)
# intervals or a bool numpy array of indices.
IndDict = typing.Dict[
    str, 
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
    ret_dict: typing.Dict[str, NDArray] = dict()
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
        inds_dict = {"all:": None}
    all_errs_dict = motionClassErrs(per_class_errs, pred_labels, inds_dict)
    mean_dict: typing.Dict[str, float] = {
        k: v.mean() for k, v in all_errs_dict.items()
    }
    return mean_dict

class DataOrganizer:
    def __init__(self, all_motion_data, min_norm_labels, err_norm_lists, 
                 train_ids, test_ids, motion_data_keys=None):
        
        self.train_ids = train_ids
        self.test_ids = test_ids

        # The below gets the training and test data, but leaves them currently
        # still separated by skip amount. E.g., one can quickly slap a "[0]" at
        # the end of each line to just look at data for one skip amount.
        train_data = concatForComboSubset(all_motion_data, train_ids)
        train_labels = concatForComboSubset(min_norm_labels, train_ids)
        train_errs = concatForComboSubset(err_norm_lists, train_ids)
        test_labels = concatForComboSubset(min_norm_labels, test_ids)
        test_data = concatForComboSubset(all_motion_data, test_ids)
        test_errs = concatForComboSubset(err_norm_lists, test_ids)

        
        # Get 2D NDArrays from the above.
        self.concat_train_labels = np.concatenate(train_labels)
        self.concat_test_labels = np.concatenate(test_labels)

        
        # Get the "keys" as another return value so that we know the column names/order.
        self.motion_data_keys, self.concat_train_data = \
            get2DArrayFromDataStruct(train_data, motion_data_keys, stack_axis=-1)
        
        _, self.concat_test_data = get2DArrayFromDataStruct(
            test_data, self.motion_data_keys, stack_axis=-1
        )

        # We'll specify the column names/order manually for this one.
        self.motion_mod_keys = [MOTION_MODEL(i) for i in range(1, len(MOTION_MODEL) + 1)] 
        _, self.concat_train_class_errs = get2DArrayFromDataStruct(
            train_errs, self.motion_mod_keys, stack_axis=-1
        )
        _, self.concat_test_class_errs = get2DArrayFromDataStruct(
            test_errs, self.motion_mod_keys, stack_axis=-1
        )


        # Get the indices of each skip amount inside the concatenated 2D array
        # we created above. There might be a "smarter" way to do this given how
        # things were previously split into lists by skip amount, but whatever.
        skip_inds = []       # For test data
        s_train_inds = [] # For trian data
        ts_ind = self.motion_data_keys.index(MOTION_DATA.TIMESTEP)
        for i in range(1,4): # We have data for frame steps of 1, 2, and 3.
            curr_skip_inds = self.concat_test_data[:, ts_ind] == i
            skip_inds.append(curr_skip_inds)
            s_train_inds.append(self.concat_train_data[:, ts_ind] == i)

        # Convert the above 3-item lists into dicts.
        self.skip_inds_dict = {"skip" + str(i): skip_inds[i] for i in range(3)}
        self.skip_inds_dict["all"] = ... # my_np_array[...] gets all elements.

        self.skip_train_inds_dict = \
            {"skip" + str(i): s_train_inds[i] for i in range(3)}
        self.skip_train_inds_dict["all"] = None

        # Optional attributes to be set in later code.
        empty_np = np.empty(0)
        self.untransformed_col_subset_train = empty_np
        self.untransformed_col_subset_test = empty_np
        self.col_subset_train = empty_np
        self.col_subset_test = empty_np


    def setPickAndTransform(self, columns, transformer = None):
        # Get the data subset for the selected non-collinear columns.
        self.col_subset_train = self.concat_train_data[:, columns]
        self.col_subset_test = self.concat_test_data[:, columns]

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
            
            self.untransformed_col_subset_train = self.col_subset_train
            self.untransformed_col_subset_test = self.col_subset_test
            self.col_subset_train = transformer.transform(self.col_subset_train)
            self.col_subset_test = transformer.transform(self.col_subset_test)
        return
    
    def getClassErrsTrain(self, pred_labels):
        return motionClassErrs(
            self.concat_train_class_errs, pred_labels, self.skip_train_inds_dict
        )
    def getClassErrsTest(self, pred_labels):
        return motionClassErrs(
            self.concat_test_class_errs, pred_labels, self.skip_inds_dict
        )
    
    def getClassScoresTrain(self, pred_labels):
        return motionClassScores(
            self.concat_train_class_errs, pred_labels, self.skip_train_inds_dict
        )
    def getClassScoresTest(self, pred_labels):
        return motionClassScores(
            self.concat_test_class_errs, pred_labels, self.skip_inds_dict
        )


