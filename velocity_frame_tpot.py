#%%
################################################################################
# TPOT regression code!
################################################################################

import typing
import shutil
import os
import datetime

import pickle
import dill

import numpy as np

from sklearn.preprocessing import StandardScaler
import tpot

# Local code imports ===========================================================

# Stuff needed for calculating the input features for the non-RNN models.
# MOTION_DATA_KEY_TYPE is a class representing input feature column "names",
# while the MOTION_DATA enum is a "subset" of these.
from posefeatures import MOTION_DATA, MOTION_DATA_KEY_TYPE



import posemath as pm # Small "library" I wrote for vector operations.

if __name__ == "__main__":
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

    nonco_cols[:] = True
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

    # print("Normalizing input data before training.")

    # Z-scale each column to standard normal distribution.
    # bcs_scalar = StandardScaler()
    # z_nonco_train_data = bcs_scalar.fit_transform(nonco_train_data)
    # z_nonco_test_data = bcs_scalar.transform(nonco_test_data)
    #%%
    ################################################################################
    # TPOT Setup and Training
    ################################################################################

    auto_train_minutes = 60*16
    print("Configuring and training TPOT.")

    # Ensure local tmp folder does not already exist.
    tmp_dir_name = "./tpot-tmp"
    if os.path.exists(tmp_dir_name) and os.path.isdir(tmp_dir_name):
        shutil.rmtree(tmp_dir_name)

    # Starts with smaller sets of data for faster initial tests, then move on to
    # the whole dataset eventually, as described in TPOT's documentation:
    # https://epistasislab.github.io/tpot/latest/Tutorial/8_SH_and_cv_early_pruning/
    pop_size=33
    initial_pop_size=39
    pop_scaling = .35
    generations_until_end_population = 4

    budget_range = [.33, 1]
    generations_until_end_budget= 4
    budget_scaling = .5

    threshold_evaluation_pruning = [33, 90]
    threshold_evaluation_scaling = .33

    selection_evaluation_pruning = [.9, .3]
    selection_evaluation_scaling = .33

    # Note: When running this with `memory='auto'`, the "./auto" folder gets
    # crazy big (62GB before I killed the program). It contains a bunch of .pkl
    # files; each pkl file must be loaded not with dill or pickle, but with
    # joblib, via `x = joblib.load(open(f, 'rb'))`, otherwise you get errs like
    # "UnpicklingError: invalid load key, '\x01'."
    # When I investigated, each loaded pkl object was a 2-tuple where the 1st
    # entry was an NDArray and the second was a sklearn transformer/scaler,
    # rather than an actual model.
    tpot_model = tpot.TPOTEstimator(
        search_space='linear',
        scorers=['neg_mean_squared_error'],
        scorers_weights=[1],
        
        classification=False,
        # memory='auto', # Whether to use caching (which makes a tmp folder).
        processes=True, # Whether to use multiprocessing.
        early_stop=4,
        n_jobs=4,
        warm_start=True, # Whether to resume from last, e.g. for pickle load.
        periodic_checkpoint_folder="./tpot-tmp",
        verbose=2,
        # memory_limit=350*(1024**2), # Memory limit per worker in bytes.
        random_state=0,
        # generations=generations_until_end_budget,
        # max_time_mins = None,
        max_time_mins=auto_train_minutes,
        
        population_size=pop_size,
        initial_population_size=initial_pop_size,
        population_scaling = pop_scaling,
        generations_until_end_population = generations_until_end_population,

        budget_range = budget_range,
        generations_until_end_budget=generations_until_end_budget,
        threshold_evaluation_pruning = threshold_evaluation_pruning,
        threshold_evaluation_scaling = threshold_evaluation_scaling,
        selection_evaluation_pruning  = selection_evaluation_pruning,
        selection_evaluation_scaling = selection_evaluation_scaling,
    )


    auto_start_time = datetime.datetime.now()
    auto_start_str = "{}:{:02d}".format(
        auto_start_time.hour, auto_start_time.minute
    )
    print("TPOT fit started at {} and will last {} minutes!".format(
        auto_start_str, "N/A" #auto_train_minutes
    ))



    tpot_model.fit(nonco_train_data, bcs_train[:, -3:])

    print("Done training TPOT!")

    ################################################################################
    # Saving the TPOT pipeline
    ################################################################################
    print("Exporting the TPOT pipeline.")
    pipeline_file = "tpot_pipeline_{:%Y-%m-%d_%H-%M-%S}.pickle".format(datetime.datetime.now())

    # For pickle *reloading*, see:
    # https://github.com/EpistasisLab/tpot/issues/520#issuecomment-891277613
    with open(pipeline_file, "wb") as f:
        dill.dump(tpot_model.fitted_pipeline_, f)

    my_dict = list()

    # Get a list of more models than just the best one.
    # Inspired by similar code in a comment for this GitHub issue:
    # https://github.com/EpistasisLab/tpot/issues/703
    print("===================================================================")
    for model_name, model_info in tpot_model.evaluated_individuals.items():
        print(model_name)
        print("Model score:", model_info.get('internal_cv_score'))
        print("---")
        print(model_info)
        print("---------------------------------------------------------------")



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
    askPreds = tpot_model.predict(nonco_test_data)
    auto_test_score = assessJAVError(bcs_test[:, -3:], askPreds, s_ind_dict)
    askPreds = tpot_model.predict(nonco_train_data)
    auto_train_score = assessJAVError(bcs_train[:, -3:], askPreds, s_train_ind_dict)
    print("Train data score:")
    print(auto_train_score)
    print("Test data score:")
    print(auto_test_score)


    print("Done TPOT evaluation!")
