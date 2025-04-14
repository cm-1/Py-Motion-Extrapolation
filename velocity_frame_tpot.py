#%%
################################################################################
# TPOT regression code!
################################################################################

import typing
import pickle
import psutil
import os
import datetime

import numpy as np
from numpy.typing import NDArray

from sklearn.preprocessing import StandardScaler
from tpot import TPOTRegressor

# Local code imports ===========================================================

# Stuff needed for calculating the input features for the non-RNN models.
# MOTION_DATA_KEY_TYPE is a class representing input feature column "names",
# while the MOTION_DATA enum is a "subset" of these.
from posefeatures import MOTION_DATA, MOTION_DATA_KEY_TYPE



import posemath as pm # Small "library" I wrote for vector operations.


################################################################################
# Loading/Filtering data!
###############################################################################
print("Loading data from disc.")
DATA_PATH = "./generated_data"

def load_data(fname):
    loaded = np.load(fname)
    ret = loaded[loaded.files[0]]
    loaded.close()
    return ret

print("Loading input data.")
concat_train_data = load_data(DATA_PATH + "/large_train_data.npz")
concat_test_data = load_data(DATA_PATH + "/large_test_data.npz")

motion_data_keys: typing.List[MOTION_DATA_KEY_TYPE] = []
with open(DATA_PATH + "/large_data_columns.pickle", "rb") as col_file:
    motion_data_keys = pickle.load(col_file)

print("Loading output data.")
# The "bcs" in the name is an artifact from some older thing.
# Will have to eventually rename this and the other instances in other files to
# something else consistent.

bcs_train = load_data(DATA_PATH + "/jav_train_data.npz")
bcs_test = load_data(DATA_PATH + "/jav_test_data.npz")


#%%
# We will start off the neural net code by constructing a regression network
# that predicts multipliers for velocity, acceleration, and jerk that we will
# use to construct the displacement from the current position to the position
# we predict for the next timestamp.


print("Removing collinear data columns.")

# A lot of the features we calculated might be collinear (especially since a lot 
# of very similar features were tried for the decision tree) we'll remove the
# collinear ones before training.
colin_thresh = 0.7 # Threshold for collinearity.

nonco_cols, co_mat = pm.non_collinear_features(concat_train_data, colin_thresh)

last_best_ind = motion_data_keys.index(MOTION_DATA.LAST_BEST_LABEL)
timestamp_ind = motion_data_keys.index(MOTION_DATA.TIMESTAMP)

nonco_cols[last_best_ind] = False # Needs one-hot encoding or similar.
nonco_cols[timestamp_ind] = False # Current frame number seems... unhelpful.

# select_cols = np.where(nonco_cols)[0][[0, 1, 2, 3, 13, 26, 27]]
# nonco_cols[:] = False
# nonco_cols[list(select_cols)] = True

# Get the data subset for the selected non-collinear columns.
nonco_train_data = concat_train_data[:, nonco_cols]
nonco_test_data = concat_test_data[:, nonco_cols]
nonco_train_data = nonco_train_data.astype(np.float32)
nonco_test_data = nonco_test_data.astype(np.float32)

################################################################################
# NORMALIZING THE DATA
################################################################################

print("Normalizing input data before training.")

# Z-scale each column to standard normal distribution.
bcs_scalar = StandardScaler()
z_nonco_train_data = bcs_scalar.fit_transform(nonco_train_data)
z_nonco_test_data = bcs_scalar.transform(nonco_test_data)
#%%
################################################################################
# TPOT Setup and Training
################################################################################
print("Configuring and training TPOT.")
tpot = TPOTRegressor(
    generations=5,
    population_size=50,
    random_state=42,
    max_time_mins=60  # Limit training time to 1 hour
)

tpot.fit(z_nonco_train_data, bcs_train[:, -3:])

print("Done training TPOT!")

################################################################################
# Saving the TPOT pipeline
################################################################################
print("Exporting the TPOT pipeline.")
pipeline_file = "tpot_pipeline_{:%Y-%m-%d_%H-%M-%S}.py".format(datetime.datetime.now())
tpot.export(pipeline_file)

################################################################################
# Evaluating the Model
################################################################################
print("Evaluating the TPOT regression model.")

# Get the indices of each skip amount inside the concatenated 2D array we
# created above. There might be a "smarter" way to do this given how things were
# previously split into lists by skip amount, but whatever.
s_inds = []       # For test data
s_train_inds = [] # For trian data
ts_ind = motion_data_keys.index(MOTION_DATA.TIMESTEP)
for i in range(1,4): # We have data for frame steps of 1, 2, and 3.
    curr_s_inds = concat_test_data[:, ts_ind] == i
    s_inds.append(curr_s_inds)
    s_train_inds.append(concat_train_data[:, ts_ind] == i)

# Convert the above 3-item lists into dicts.
s_ind_dict = {"skip" + str(i): s_inds[i] for i in range(3)}
s_ind_dict["all"] = None # Because my_np_array[None] returns all elements.

s_train_ind_dict = {"skip" + str(i): s_train_inds[i] for i in range(3)}
s_train_ind_dict["all"] = None
#%% Print scores on test data.



def assessJAVError(gt, pred, ind_dict):
    norms = np.linalg.norm(gt - pred, axis = -1)
    return {k: np.mean(norms[v]) for k, v in ind_dict.items()}

# Evaluate on test data
askPreds = tpot.predict(z_nonco_test_data)
auto_test_score = assessJAVError(bcs_test[:, -3:], askPreds, s_ind_dict)
askPreds = tpot.predict(z_nonco_train_data)
auto_train_score = assessJAVError(bcs_train[:, -3:], askPreds, s_train_ind_dict)
print("Train data score:")
print(auto_train_score)
print("Test data score:")
print(auto_test_score)


print("Done TPOT evaluation!")