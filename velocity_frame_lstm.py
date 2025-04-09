################################################################################
# Velocity-Aligned-Frame LSTM
################################################################################
import numpy as np

import keras

# We'll use a MinMax scaler to "normalize" the data, as per a tutorial.
from sklearn.preprocessing import MinMaxScaler

# Local code imports ===========================================================
# For reading the dataset into numpy arrays:
import gtCommon as gtc
# Functions to get RNN "windows" and velocity-aligned frame data, respectively.
from data_by_combo_functions import rnnDataWindows, dataForCombosJAV
# Functions to get all combos and to split combos into train/test sets:
from data_by_combo_functions import getAllCombos, getTrainTestCombos
from data_by_combo_functions import JAV # Enum
from data_by_combo_functions import UnscaledDistanceLogger # Custom callback

WIN_SIZE = 4

# How many frames to skip when going over the pose dataset; skipping more frames
# simulates larger motions or smaller fps. Typical values are 0, 1, 2.
DATASET_SKIP_FRAMES = 2

combos = getAllCombos()

# From the combo 3-tuples, construct nametuple versions containing only the
# uniquely-identifying parts. Some functions expect this instead of the 3-tuple. 
nametup_combos = [gtc.Combo(*c[:2]) for c in combos]

print("Getting train-test split.")
train_combos, test_combos = getTrainTestCombos(combos, test_ratio=0.2, random_seed=0)

#%%
################################################################################
# READING/NORMALIZING THE DATA
################################################################################

print("Reading/normalizing data.")
# As with the "vanilla" regression network, we'll get our data in a velocity
# -aligned frame. But this time, we'll start by keeping things separated by 
# combo so that we know the starting and ending points for a video's data and,
# thus, can create our data windows without having a window erroneously overlap
# two separate videos.
all_jav = dataForCombosJAV(
    nametup_combos, (JAV.VELOCITY, JAV.ACCELERATION, JAV.JERK)
)

jav_scalers = [MinMaxScaler(feature_range=(0,1)) for _ in range(3)]
jav_scalers_3 = [MinMaxScaler(feature_range=(0,1)) for _ in range(3)]

for skip in range(3):
    # We'll *temporarily* combine the data for all combos into a single numpy
    # array, but that will just be to fit the Scaler used to normalize the data.
    all_jav_concat = np.concatenate(
        [all_jav[skip][c[:2]] for c in combos], axis=0
    )
    # Scaler for all data columns.
    jav_scalers[skip].fit(all_jav_concat)
    # Scale all 3 axes by uniform amount and centre to 0 for later simplicity
    jav_scalers[skip].scale_[-3:] = jav_scalers[skip].scale_[-1]
    jav_scalers[skip].min_[-3:] = 0.0

    # Scaler for just the last three columns, the "displacement" ones.
    jav_scalers_3[skip].fit(all_jav_concat[:, -3:])
    # Ensure this scaler matches the other one.
    jav_scalers_3[skip].scale_[-3:] = jav_scalers[skip].scale_[-1]
    jav_scalers_3[skip].min_[-3:] = 0.0

    # We no longer need all_jav_concat, and it might take up a fair bit of RAM.
    del all_jav_concat

#%% 
################################################################################
# CALCULATING THE DATA WINDOWS
################################################################################

print("Calculating data's sliding \"windows\" for the network.")

jav_lstm_skip = DATASET_SKIP_FRAMES # Renaming var.

train_jav_in, train_jav_out = rnnDataWindows(
    all_jav[jav_lstm_skip], train_combos, WIN_SIZE, jav_scalers[jav_lstm_skip], 
    0 # Always a window "skip" of 0 in this case because the frame is rotating...
)

#%%
################################################################################
# BUILDING THE NETWORK
################################################################################
print("Building the NN.")
jav_lstm_model = keras.Sequential([
    # keras.layers.LSTM(128, return_sequences=True),
    keras.layers.LSTM(50),
    keras.layers.Dense(3) #num_classes, activation='sigmoid')
])

jav_lstm_model.summary()

# Log the loss in mm, rather than just scaled coordinates.
jav_lstm_logger = UnscaledDistanceLogger(
    train_jav_in[..., -3:], train_jav_out[:, -3:], jav_scalers_3[jav_lstm_skip]
)
print("Note: The default reported \"loss\" will not be in millimeters, because training data was normalized.")
print("Actual MAE in millimeters will be printed explicitly/separately.")

jav_lstm_model.compile(optimizer='adam', loss='mse')
#%%
################################################################################
# TRAINING THE NETWORK
################################################################################

jav_lstm_hist = jav_lstm_model.fit(
    train_jav_in[..., -3:], train_jav_out[:, -3:], epochs=32,
    callbacks=[jav_lstm_logger]
)

#%%
################################################################################
# EVALUATING THE NETWORK
################################################################################

test_jav_in, test_jav_out = rnnDataWindows(
    all_jav[jav_lstm_skip], test_combos, WIN_SIZE, jav_scalers[jav_lstm_skip], 
    0 # Always a window "skip" of 0 in this case because the frame is rotating...

)

jav_lstm_test_pred = jav_lstm_model.predict(test_jav_in[..., -3:])

def getErrsLSTM(scaler, unscaled_preds, scaled_gt):
    scaled_preds = scaler.inverse_transform(unscaled_preds)
    return np.linalg.norm(scaled_preds - scaled_gt, axis=-1)

# Transform the coordinates back into non-normalized form to get the errors
# in millimeters.
jav_scaled_lstm_gt_test = jav_scalers_3[jav_lstm_skip].inverse_transform(
    test_jav_out[:, -3:]
)

# jav_scaled_lstm_gt_train = jav_scalers_3[jav_lstm_skip].inverse_transform(
#     train_jav_out[:, -3:]
# )
jav_lstm_test_errs = getErrsLSTM(
    jav_scalers_3[jav_lstm_skip], jav_lstm_test_pred, jav_scaled_lstm_gt_test
)
print("\n\nJAV LSTM score (millimeters) on test data:", jav_lstm_test_errs.mean())
print("Above score is for skip{} data.".format(jav_lstm_skip))
#%%

################################################################################
# Train loss graph.
################################################################################
import matplotlib.pyplot as plt

plt.plot(jav_lstm_logger.mean_distances)

plt.xlabel("Epoch")
plt.ylabel("MAE (mm)")
plt.title("Loss for skip{} dataset train".format(jav_lstm_skip))
plt.grid(True)
plt.show()

print("Done program.")
