import typing
import warnings

import numpy as np
from numpy.typing import NDArray

import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf



# Local code imports ===========================================================
# For reading the dataset into numpy arrays:
import gtCommon as gtc
from gtCommon import PoseLoaderBCOT

import posemath as pm # Small "library" I wrote for vector operations.

from motiontools.posefeatures import PoseLoaderList, NumpyForSkipAndID
from motiontools.posefeatures import dataForCombosJAV, JAV

from motiontools.dataorg import concatForComboSubset


# Video categories; currently not *really* used in this file, but that might 
# change in the near future if I want to analyze the categories separately.
motion_kinds = [
    "movable_handheld", "movable_suspension", "static_handheld",
    "static_suspension", "static_trans"
]
# motion_kinds_plus = motion_kinds + ["all"]


# For type hint. Seems that sklearn's "Scalers" don't have a single superclass
# that has a transform() method, unless I'm missing something. So I'm doing a
# type union instead.
ScalerType = typing.Union[
    MinMaxScaler, sklearn.preprocessing.StandardScaler, 
    sklearn.preprocessing.RobustScaler, sklearn.preprocessing.MaxAbsScaler,
    sklearn.preprocessing.PowerTransformer, sklearn.preprocessing.Normalizer,
    sklearn.preprocessing.QuantileTransformer
]


def rnnDataWindows(data, combo_subset: typing.List[typing.Tuple], 
                   window_size: int, skip: int,
                   scaler: typing.Optional[ScalerType] = None, 
                   world2local_frames = None, centre_subsets: bool = False):
    '''Function that gets the "windows" of consecutive poses from the data for a 
    subset of combos. '''
    step = skip + 1
    data_in = []
    data_out = []
    modifying_positions = ((world2local_frames is not None) or centre_subsets)
    if (scaler is not None) and modifying_positions:
        warnings.warn((
            "When creating RNN data windows in rnnDataWindows(), passing in "
            "both a scaler and parameters for modifying the positions "
            "(world2local frames or saying to centre data) is not recommended, "
            "because the data will be scaled after these other "
            "transformations, which is may yield unexpected data distributions!"
        ), RuntimeWarning)

    for combo in combo_subset:
        combo_data = data[combo[:2]][::step]
        if scaler is not None:
            scaler.transform(combo_data)
        w2l_data = None
        # There might not be frames for the first data points.
        # This should be fine, as there still should be at least one per window
        # (under expected use cases), but we'll need to handle this.
        w2l_shift = 0 
        if world2local_frames is not None:
            w2l_data = world2local_frames[combo[:2]][::step]
            w2l_shift = len(combo_data) - len(w2l_data)
        for i in range(len(combo_data) - window_size):
            curr_in = combo_data[i:(i + window_size)]
            curr_out = combo_data[i + window_size]
            if world2local_frames is not None:
                curr_frame = w2l_data[i + window_size - 1 - w2l_shift]
                curr_in = curr_in @ curr_frame.transpose()
                curr_out = curr_frame @ curr_out
            if centre_subsets:
                centre_ref = curr_in[-1]
                curr_in = curr_in[:-1] - centre_ref
                curr_out -= centre_ref
            data_in.append(curr_in)
            data_out.append(curr_out)
    data_in_np = np.array(data_in)
    data_out_np = np.array(data_out)
    return (data_in_np, data_out_np)

def scaleWindows(data: NDArray, scaler: ScalerType):
    orig_shape = data.shape
    data_rs: NDArray
    if data.ndim == 2:
        data_rs = data.reshape(-1, 1)
    elif data.ndim == 3:
        data_rs = data.reshape((-1,) + data.shape[2:])
    else:
        raise NotImplementedError("Unsupported windowed data ndim > 3!")
    
    scaled = scaler.transform(data_rs)
    return scaled.reshape(orig_shape)

# From the combo 3-tuples, construct nametuple versions containing only the
# uniquely-identifying parts. Some functions expect this instead of the 3-tuple. 
def getAllCombosAndLoadersBCOT():

    combos = PoseLoaderBCOT.getAllIDs()
    nametup_combos = [gtc.VidBCOT(*c[:2]) for c in combos]
    bcot_loaders = [
        PoseLoaderBCOT(nc.body_ind, nc.seq_ind) for nc in nametup_combos
    ]
    return nametup_combos, bcot_loaders

# Function that combines the results of dataForCombosJav(...) into a 2D numpy
# array.
def dataForComboSplitJAV(train_combos: typing.List, test_combos: typing.List, *,
                         pose_loaders: typing.Optional[PoseLoaderList] = None, 
                         precalc_per_combo: typing.Optional[NumpyForSkipAndID] = None):   
    if pose_loaders is None and precalc_per_combo is None:
        raise ValueError(
            "Cannot have combos and precalc_per_combo both be None!"
        )
    elif pose_loaders is not None and precalc_per_combo is not None:
        raise ValueError(
            "Cannot provide values for both  combos and precalc_per_combo!"
        )
     
    all_data = precalc_per_combo
    if precalc_per_combo is None:
        all_data = dataForCombosJAV(
            pose_loaders, (JAV.JERK, JAV.ACCELERATION, JAV.VELOCITY)
        )
    

    train_res = np.concatenate(
        concatForComboSubset(all_data, train_combos), axis=0
    )
    test_res = np.concatenate(
        concatForComboSubset(all_data, test_combos), axis=0
    )
    return train_res, test_res


class UnscaledDistanceLogger(tf.keras.callbacks.Callback):
    '''
    Custom callback for when LSTM models are trained on scaled data.
    Scaled data is important for the training process, and so I want to use
    scaled data as my actual loss function, but I also want to know the unscaled
    mean distance between predicted and ground-truth points each epoch.

    This callback prints/stores this at the end of each epoch.

    Class originally written by ChatGPT. Verified/modified/commented manually.
    '''
    def __init__(self, X_val_scaled, y_val_scaled, scaler, y_val_unscaled = None):
        super().__init__()
        self.X_val_scaled = X_val_scaled
        self.y_val_scaled = y_val_scaled
        self.scaler = scaler
        self.y_val_unscaled = y_val_unscaled
        if y_val_unscaled is None:
            self.y_val_unscaled = self.scaler.inverse_transform(self.y_val_scaled)
        self.mean_distances = []  # Store distances for plotting

    def on_epoch_end(self, epoch, logs=None):
        # Use a large batch size so prediction goes faster.
        y_pred_scaled = self.model.predict(
            self.X_val_scaled, verbose=0, batch_size=1024
        )
        y_pred_unscaled = self.scaler.inverse_transform(y_pred_scaled)
        
        distances = np.linalg.norm(y_pred_unscaled - self.y_val_unscaled, axis=-1)
        mean_distance = np.mean(distances)
        
        self.mean_distances.append(mean_distance)
        print() # Ensure newline.
        print(f"Epoch {epoch+1}: Unscaled MAE (millimeters) = {mean_distance:.4f}")


