#%%
################################################################################
# Vanilla Regression Network Code!
################################################################################

import typing
import pickle

import numpy as np
from numpy.typing import NDArray

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import keras

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

# %% 


################################################################################
# Custom classes/functions used to create the NN model.
################################################################################


# Get the pose loss for a set of Jerk, Acceleration, & Velocity multipliers.
def poseLossJAV(y_true, y_pred):
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
    pred_disp_0 = y_true[:, 0] * y_pred[:, 0] + y_true[:, 1] * y_pred[:, 1] \
        + y_true[:, 3] * y_pred[:, 2]
    pred_disp_1 = y_true[:, 2] * y_pred[:, 1] + y_true[:, 4] * y_pred[:, 2]
    pred_disp_2 = y_true[:, 5] * y_pred[:, 2]

    pred_disp = tf.stack([pred_disp_0, pred_disp_1, pred_disp_2], axis=-1)

    true_disp = y_true[:, 6:]

    err_vec3 = true_disp - pred_disp
    return tf.norm(err_vec3, axis=-1)


# Custom importance weighting layer suggested/described "in theory" by a friend.
# Then, I had the class written by ChatGPT and manually verified.
# (But I'm not a big tensorflow expert, so maybe my verification was faulty...)
class ImportanceLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, weight_decay=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.weight_decay = weight_decay

    def build(self, input_shape):
        # Define per-feature weights with L2 regularization (weight decay)
        self.importance_weights = self.add_weight(
            shape=(self.input_dim,), 
            initializer="ones",  # Start with all weights = 1
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True
        )

    def call(self, inputs):
        return inputs * self.importance_weights  # Element-wise multiplication

#%% Creating the NN model

################################################################################
# BUILDING THE NETWORK
################################################################################
print("Building the NN.")


dropout_rate = 0.2
nodes_per_layer = 128
bcs_model = keras.Sequential([
    keras.layers.Input((nonco_train_data.shape[1],)),
    # ImportanceLayer(nonco_train_data.shape[1]),  # Custom importance layer
    keras.layers.Dense(nodes_per_layer, activation='relu'),
    keras.layers.Dropout(dropout_rate),
    keras.layers.Dense(nodes_per_layer, activation='relu'),
    keras.layers.Dropout(dropout_rate),
    keras.layers.Dense(nodes_per_layer, activation='relu'),
    keras.layers.Dense(3)
])

bcs_model.summary()
bcs_model.compile(loss=poseLossJAV, optimizer='adam')
# %%
################################################################################
# NORMALIZING THE DATA
################################################################################

print("Normalizing input data before training.")

# Convert from numpy array to tf tensor.
bcs_train = tf.convert_to_tensor(bcs_train, dtype=tf.float32)
bcs_test = tf.convert_to_tensor(bcs_test, dtype=tf.float32)

# Z-scale each column to standard normal distribution.
bcs_scalar = StandardScaler()
z_nonco_train_data = bcs_scalar.fit_transform(nonco_train_data)
#%% 
################################################################################
# TRAINING THE NETWORK
################################################################################
print("Training has begun. Loss reported is actual millimeter MAE.")
bcs_hist = bcs_model.fit(z_nonco_train_data, bcs_train, epochs=32, shuffle=True)

#%% 
################################################################################
# EVALUATING THE NETWORK
################################################################################

z_nonco_test_data = bcs_scalar.transform(nonco_test_data)
bcs_pred = bcs_model.predict(z_nonco_test_data)
bcs_test_errs = poseLossJAV(bcs_test, bcs_pred)


#%%
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

bcs_test_scores = {k: np.mean(bcs_test_errs[v]) for k, v in s_ind_dict.items()}
print("\n\nScores on the test data (MAE, millimeters):")
print(bcs_test_scores)

#%%

################################################################################
# Train loss graph.
################################################################################
import matplotlib.pyplot as plt

plt.plot(bcs_hist.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("MAE (mm)")
plt.title("Loss for agrregate-skip train")
plt.show()

#%%

################################################################################
# Column Scrambling to Assess Feature Importance
################################################################################

print("Testing feature importance. Feel free to stop code if uninterested.")

# Get the column names for each of the kept columns.
nonco_featnames = [
    k.name for i, k in enumerate(motion_data_keys) if nonco_cols[i]
]

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
    preds = model.predict(data_scramble, verbose=0, batch_size=1024)
    errs = model.loss(y_true, preds)
    return errs

# For each column and for each skip amount, scramble the column and find the new
# score.
keys_s_ind = s_ind_dict.keys()
scramble_scores = np.empty((z_nonco_test_data.shape[1], len(s_ind_dict.keys())))
print()
num_nonco_cols = z_nonco_test_data.shape[1]
for col_ind in range(num_nonco_cols):
    print(
        "Testing column index {:03d}/{}.".format(col_ind + 1, num_nonco_cols),
        end='\r', flush=True
    )
    errs = errsForColScramble(bcs_model, z_nonco_test_data, col_ind, bcs_test)
    for i, k in enumerate(keys_s_ind):
        scramble_scores[col_ind, i] = np.mean(errs[s_ind_dict[k]])
#%%
scramble_rank = np.argsort(scramble_scores, axis=0)[::-1]
head_num = 10 # How many "best" to print.
head_best = scramble_rank[:head_num]
print(
    "Most important feature indices:", 
    "(With columns representing ({}))".format(", ".join(keys_s_ind)),
    head_best, sep='\n'
)
    
print("Names:")
for hb in head_best:
    print([nonco_featnames[i] for i in hb])
