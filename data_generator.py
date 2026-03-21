# Function for generating reusable on-disk train and test data for the vanilla
# regression network. 


#%% Imports

# Local code imports ===========================================================
# For reading the dataset into numpy arrays:
from gtCommon import PoseLoaderBCOT


from motiontools.posefeatures import CalcsForVideo
from motiontools.dataorg import DataOrganizer

# Function for getting JAV data
import motiontools.shared_constants

# Some consts used in calculating the input features.
OBJ_IS_STATIC_THRESH_MM = 10.0 # 10 millimeters; semi-arbitrary
STRAIGHT_LINE_ANG_THRESH_DEG = 30.0 # 30deg as arbitrary max "straight" angle.
CIRC_ERR_RADIUS_RATIO_THRESH = 0.10 # Threshold for if motion's circular.
MAX_MIN_JERK_OPT_ITERS = 0 # Max iters for min jerk optimization calcs.
MAX_SPLIT_MIN_JERK_OPT_ITERS = 0


#%%
################################################################################
# CALCULATING THE INPUT DATA
################################################################################

print("Calculating input data. This may take a minute or two.")

# From the VidID 3-tuples, we get nametuple versions containing only the
# uniquely-identifying parts. Some functions expect this instead of the 3-tuple. 
# We also get a list of pose loaders, one for each BCOT video.
nametup_ids, bcot_loaders = PoseLoaderBCOT.getAllMinimalIDsAndLoaders(True)

cfc = CalcsForVideo(
    obj_static_thresh_mm=OBJ_IS_STATIC_THRESH_MM, 
    straight_angle_thresh_deg=STRAIGHT_LINE_ANG_THRESH_DEG,
    err_na_val=motiontools.shared_constants.ERR_NA_VAL,
    min_jerk_opt_iter_lim=MAX_MIN_JERK_OPT_ITERS,
    split_min_jerk_opt_iter_lim = MAX_SPLIT_MIN_JERK_OPT_ITERS,
    err_radius_ratio_thresh=CIRC_ERR_RADIUS_RATIO_THRESH
    # exclude_axis_angs=False, exclude_bidir=False, exclude_circ_data=False,
    # exclude_onehots=False, exclude_past_muls=False, exclude_timescaled=False,
    # exclude_vel_deg2=False
)
cfc.getAll(bcot_loaders)

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


train_ids, test_ids = PoseLoaderBCOT.trainTestByBody(
    test_ratio=0.2, random_seed=0
)
train_ids_c2 = [c[:2] for c in train_ids]
test_ids_c2 = [c[:2] for c in test_ids] # Get the unique part of each.

dog = DataOrganizer.FromCalcs(
    PoseLoaderBCOT, all_motion_data, cfc.min_norm_labels, cfc.err_norm_lists,
    train_ids_c2, test_ids_c2
)


################################################################################
# SAVING THE INPUT DATA
################################################################################

dog.dump(compress=True)
    
print("Data generation/saving completed!")
