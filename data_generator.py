# Function for generating reusable on-disk train and test data for the vanilla
# regression network. 


#%% Imports
import typing
import pickle
import os

import numpy as np
from numpy.typing import NDArray


# Local code imports ===========================================================
# For reading the dataset into numpy arrays:
import gtCommon as gtc

# For the velocity-aligned frame data generation, we import some resuable stuff.
# These are a function for getting all "combos" (video IDs), a function to get
# velocity-aligned frame data per combo, a function for train/test combo splits,
# type hints, and an enum.
from data_by_combo_functions import getAllCombos, dataForCombosJAV, getTrainTestCombos
from data_by_combo_functions import ComboList, PerComboJAV # Type hints.
from data_by_combo_functions import JAV # Enum

# MOTION_DATA_KEY_TYPE is a class representing input feature column "names".
from posefeatures import MOTION_DATA_KEY_TYPE, CalcsForCombo

# Some consts used in calculating the input features.
OBJ_IS_STATIC_THRESH_MM = 10.0 # 10 millimeters; semi-arbitrary
STRAIGHT_LINE_ANG_THRESH_DEG = 30.0 # 30deg as arbitrary max "straight" angle.
CIRC_ERR_RADIUS_RATIO_THRESH = 0.10 # Threshold for if motion's circular.
MAX_MIN_JERK_OPT_ITERS = 33 # Max iters for min jerk optimization calcs.
MAX_SPLIT_MIN_JERK_OPT_ITERS = 33
ERR_NA_VAL = np.finfo(np.float32).max # A non-inf but inf-like value.


#%%
################################################################################
# CALCULATING THE INPUT DATA
################################################################################

print("Calculating input data. This may take a minute or two.")

combos = getAllCombos()

# From the combo 3-tuples, construct nametuple versions containing only the
# uniquely-identifying parts. Some functions expect this instead of the 3-tuple. 
nametup_combos = [gtc.Combo(*c[:2]) for c in combos]

cfc = CalcsForCombo(
    nametup_combos, obj_static_thresh_mm=OBJ_IS_STATIC_THRESH_MM, 
    straight_angle_thresh_deg=STRAIGHT_LINE_ANG_THRESH_DEG,
    err_na_val=ERR_NA_VAL, min_jerk_opt_iter_lim=MAX_MIN_JERK_OPT_ITERS,
    split_min_jerk_opt_iter_lim = MAX_SPLIT_MIN_JERK_OPT_ITERS,
    err_radius_ratio_thresh=CIRC_ERR_RADIUS_RATIO_THRESH
)
results = cfc.getAll()

# Input features like velocity, acceleration, jerk, rotation speed, etc.
all_motion_data = cfc.all_motion_data
# The above data has the following type: 
#     List[Dict[Combo, Dict[MOTION_DATA, NDArray]]]
# That is, we have a list of dictionaries which store the results per combo,
# where said "result" is another dict with "column" name enums as keys.
# 
# The combo-keyed dict's index in the top-level list corresponds to the number 
# of frames we are skipping when we read the dataset. So the [0] dict is when
# not skipping any  frames, the [1] dict for when reading every 2nd frame only, 
# etc.

#%%
# The below are functions that convert the above lists of dicts into single
# train/test sets that we can pass into our model training.

# First, we concatenate together the data for a subset of combos and get
# a list of 3 items, where each list index again corresponds to the frame 
# skip amount but results are no longer separated by combo.
# Motivation: We may want to quickly filter out a skip amount for training.
# ---
# I have other non-regression code that sometimes passes in a list of NDArrays
# instead of a list of dicts, hence the isinstabce() check.
def concatForComboSubset(data, combo_subset): #, front_trim: int = 0):
    ret_val = []
    for els_for_skip in data:
        subset_via_combos = [els_for_skip[ck[:2]] for ck in combo_subset]
        concated = None
        front_trim = 0 # May set this via param in future code.
        if isinstance(subset_via_combos[0], dict):
            concated = dict()
            for k in subset_via_combos[0].keys():
                concated[k] = np.concatenate([
                    svc[k][front_trim:] for svc in subset_via_combos
                ])
        else:
            concated = np.concatenate([
                svc[front_trim:] for svc in subset_via_combos
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


            
# def concatForKeys(data, keys):
#     return np.stack((data[k] for k in keys))

#%%

train_combos, test_combos = getTrainTestCombos(combos, test_ratio=0.2, random_seed=0)


# The below gets the training and test data, but leaves them currently still
# separated by skip amount. Here, one can quickly slap a "[0]" at the end of
# each line to just look at data for one skip amount, for example.
train_data = concatForComboSubset(all_motion_data, train_combos)
test_data = concatForComboSubset(all_motion_data, test_combos)

# Get 2D NDArrays from the above.

# Get the "keys" as another return value so that we know the column names/order.
motion_data_keys, concat_train_data = get2DArrayFromDataStruct(train_data, stack_axis=-1)
_, concat_test_data = get2DArrayFromDataStruct(test_data, motion_data_keys, stack_axis=-1)

################################################################################
# SAVING THE INPUT DATA
################################################################################

DATA_PATH = "./generated_data"
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

print("Saving the input train and test data.")
np.savez_compressed(DATA_PATH + "/large_train_data.npz", concat_train_data)
np.savez_compressed(DATA_PATH + "/large_test_data.npz", concat_test_data)

with open(DATA_PATH + "/large_data_columns.pickle", "wb") as col_file:
    pickle.dump(motion_data_keys, col_file)
    
#%%
################################################################################
# CALCULATING THE OUTPUT DATA
################################################################################

print("Calculating output data.")



# Function that combines the results of dataForCombosJav(...) into a 2D numpy
# array.
def dataForComboSplitJAV(train_combos: ComboList, test_combos: ComboList, *,
                         combos: typing.Optional[ComboList] = None, 
                         precalc_per_combo: typing.Optional[PerComboJAV] = None):   
    if combos is None and precalc_per_combo is None:
        raise ValueError(
            "Cannot have combos and precalc_per_combo both be None!"
        )
    elif combos is not None and precalc_per_combo is not None:
        raise ValueError(
            "Cannot provide values for both  combos and precalc_per_combo!"
        )
     
    all_data = precalc_per_combo
    if precalc_per_combo is None:
        all_data = dataForCombosJAV(
            combos, (JAV.JERK, JAV.ACCELERATION, JAV.VELOCITY)
        )
    

    train_res = np.concatenate(
        concatForComboSubset(all_data, train_combos), axis=0
    )
    test_res = np.concatenate(
        concatForComboSubset(all_data, test_combos), axis=0
    )
    return train_res, test_res

# Get local-frame data.
# The "bcs" in the name is an artifact from some older thing.
# Will have to eventually rename this and the other instances in other files to
# something else consistent.
bcs_per_combo = dataForCombosJAV(
    nametup_combos, (JAV.JERK, JAV.ACCELERATION, JAV.VELOCITY), False, False
)

bcs_train, bcs_test = dataForComboSplitJAV(
    train_combos, test_combos, precalc_per_combo=bcs_per_combo
)

################################################################################
# SAVING THE OUTPUT DATA
################################################################################


print("Saving the output train and test data.")

np.savez_compressed(DATA_PATH + "/jav_train_data.npz", bcs_train)
np.savez_compressed(DATA_PATH + "/jav_test_data.npz", bcs_test)

print("Data generation/saving completed!")