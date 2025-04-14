#%%
################################################################################
# Auto-sklearn regression code!
################################################################################

import typing
import pickle
import psutil
import os
import shutil
import datetime

import numpy as np
from numpy.typing import NDArray

from sklearn.preprocessing import StandardScaler

import autosklearn.regression

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
# Memory Management and Auto-Sklearn Setup
################################################################################
#%%
# Settings that I'll just edit manually on each run for now.
auto_train_seconds = 3600*24
really_max_mem = False

avail_mem_bytes = psutil.virtual_memory().available
avail_mem_mb = avail_mem_bytes / (1024*1024)

# 90% and 0.8GB untouched
avail_mem_by_ratio = int(0.9 * avail_mem_bytes)
avail_mem_by_buffer = avail_mem_bytes - int(0.8 * (1024**3))

if really_max_mem:
    # 99% and 350MB untouched
    avail_mem_by_ratio = int(0.99 * avail_mem_bytes)
    avail_mem_by_buffer = avail_mem_bytes - 350 * (1024**2)

avail_mem_safe = max(avail_mem_by_ratio, avail_mem_by_buffer)
avail_mem_safe_mb = avail_mem_safe // (1024*1024)

if not really_max_mem:
    # Cap to 8GB for now. Part of motivation: running on multi-user computer.
    # Another possibility: I'm concerned that more RAM leads to overfitting.
    avail_mem_safe_mb = min(avail_mem_safe_mb, 8 * 1024)


# Ensure local tmp folder does not already exist, or else auto-sklearn will
# throw an exception.
tmp_dir_name = "./auto-sklearn-tmp"
if os.path.exists(tmp_dir_name) and os.path.isdir(tmp_dir_name):
    shutil.rmtree(tmp_dir_name)


# We need to limit the models on disc to avoid taking up too much disc space.
# To debug when too much disc space is used, I'll specify a local tmp folder
# and I won't let auto-sklearn handle the deletion (though I'll do it myself
# after so that there's room for the pickled model).
automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=3600,
    memory_limit=avail_mem_safe_mb,
    max_models_on_disc=7,
    tmp_folder=tmp_dir_name,
    delete_tmp_folder_after_terminate=False,
    smac_scenario_args={"n_trials": 35}
)

auto_start_time = datetime.datetime.now()
auto_start_str = "{}:{:02d}".format(
    auto_start_time.hour, auto_start_time.minute
)

print("Using {}/{} available MB.".format(
    automl.memory_limit, avail_mem_mb
))


################################################################################
# Training with Auto-Sklearn
################################################################################

print("Auto-sklearn fit started at {} and will last {} seconds!".format(
    auto_start_str, automl.time_left_for_this_task
))

automl.fit(z_nonco_train_data, bcs_train[:, -3:])

print("Done auto-sklearn fit!")

# If fitting did not run into exceptions, then the tmp folder currently exists
# and is not required for debugging said exceptions but *is* taking up space.
# So we delete it to ensure the pickling has enough disc space available.
if os.path.exists(tmp_dir_name) and os.path.isdir(tmp_dir_name):
    shutil.rmtree(tmp_dir_name)

# Create a unique filename and save before printing/saving other results, so
# that if there's an unexpected crash in those other steps, we do not lose
# hours of training/work!
auto_fname = "auto-{:%Y-%m-%d_%H-%M-%S}.pickle".format(datetime.datetime.now())
# with open('./auto-2025-04-03_20-18-32.pickle', 'rb') as f:
#     automl = pickle.load(f)
#%%

pickle.dump(automl, open(auto_fname, "wb"))

#%%
################################################################################
# Evaluating the Model
################################################################################
print("Evaluating the Auto-Sklearn regression model.")

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

askPreds = automl.predict(concat_test_data)
auto_test_score = assessJAVError(bcs_train[:, -3:], askPreds, s_ind_dict)
askPreds = automl.predict(concat_train_data)
auto_train_score = assessJAVError(bcs_test[:, -3:], askPreds, s_train_ind_dict)
print("Train data score:")
print(auto_train_score)
print("Test data score:")
print(auto_test_score)
print(automl.leaderboard())
import pprint
pprint.pp(automl.show_models())
automl.sprint_statistics()

auto_summary_str = ''
auto_summary_str += "Train data score:\n"
auto_summary_str += str(auto_train_score) + '\n'
auto_summary_str += "Test data score:\n"
auto_summary_str += str(auto_test_score) + '\n'
auto_summary_str += str(automl.leaderboard()) + '\n'
auto_summary_str += str(automl.show_models()) + '\n'
auto_summary_str += str(automl.sprint_statistics())

t_fname = 't_' + auto_fname + '.txt'
with open(t_fname, 'w') as f:
    f.write(auto_summary_str)

print("Done automl stuff!")

