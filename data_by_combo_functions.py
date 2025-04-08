import typing
from enum import Enum
import copy

import numpy as np
from numpy.typing import NDArray

import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
import matplotlib.pyplot as plt



# Local code imports ===========================================================
# For reading the dataset into numpy arrays:
from gtCommon import BCOT_Data_Calculator
import gtCommon as gtc

import posemath as pm # Small "library" I wrote for vector operations.


# Video categories; currently not *really* used in this file, but that might 
# change in the near future if I want to analyze the categories separately.
motion_kinds = [
    "movable_handheld", "movable_suspension", "static_handheld",
    "static_suspension", "static_trans"
]
# motion_kinds_plus = motion_kinds + ["all"]

def getAllCombos():
    '''
    Generate the following tuples that represent each video:
        
        (sequence_name, body_name, motion_kind)
    
    The first two tuple elements uniquely identify a video, while the third is
    redundant (it's part of each sequence name) but might be used for more
    convenient filtering of videos.

    ---
    In the BCOT dataset, videos are categorized first by the "sequence" type 
    (which is motion/lighting/background), and then by the object ("body") 
    featured in the video. Each "combo" of a sequence and body thus represents
    a distinct video.
    '''
    combos = []
    for s, s_val in enumerate(gtc.BCOT_SEQ_NAMES):
        k = ""
        # For now, using a for loop, not regex, to get motion kind from seq name.
        for k_opt in motion_kinds:
            if k_opt in s_val:
                k = k_opt
                break
        for b in range(len(gtc.BCOT_BODY_NAMES)):
            # Some sequence-body pairs do not have videos, and some have two videos
            # with identical motion but a different camera. So we first check that 
            # a video exists and has unique motion.
            if BCOT_Data_Calculator.isBodySeqPairValid(b, s, True):
                combos.append((b, s, k))
    return combos

# Filter out combos based on the 3D object ("body") subset chosen. 
def combosByBod(bods, combos):
    return [c for c in combos if c[0] in bods]

def getTrainTestCombos(all_combos, test_ratio=0.2, random_seed=0):
    '''
    We'll split our data into train/test sets where the vids for a single body
    will either all be train vids or all be test vids. This way (a) we are
    guaranteed to have every motion "class" in our train and test sets, and (b)
    we'll know how well the models generalize to new 3D objects not trained on.
    '''
    bod_arange = np.arange(len(gtc.BCOT_BODY_NAMES), dtype=int)
    train_bodies, test_bodies = train_test_split(
        bod_arange, test_size=test_ratio, random_state=random_seed
    )

    train_combos = combosByBod(train_bodies, all_combos)
    test_combos = combosByBod(test_bodies, all_combos)  

    return train_combos, test_combos

# For type hint. Seems that sklearn's "Scalers" don't have a single superclass
# that has a transform() method, unless I'm missing something. So I'm doing a
# type union instead.
ScalerType = typing.Union[
    MinMaxScaler, sklearn.preprocessing.StandardScaler, 
    sklearn.preprocessing.RobustScaler, sklearn.preprocessing.MaxAbsScaler,
    sklearn.preprocessing.PowerTransformer, sklearn.preprocessing.Normalizer,
    sklearn.preprocessing.QuantileTransformer
]


def rnnDataWindows(data: NDArray, combo_subset: typing.List[typing.Tuple], 
                   window_size: int, scaler: ScalerType, skip: int):
    '''Function that gets the "windows" of consecutive poses from the data for a 
    subset of combos. '''
    step = skip + 1
    data_in = []
    data_out = []
    for combo in combo_subset:
        combo_data = scaler.transform(data[combo[:2]][::step])
        for i in range(len(combo_data) - window_size):
            data_in.append(combo_data[i:(i + window_size)])
            data_out.append(combo_data[i + window_size])
    data_in_np = np.array(data_in)
    data_out_np = np.array(data_out)
    return (data_in_np, data_out_np)

class JAV(Enum):
    VELOCITY = 1
    ACCELERATION = 2
    JERK = 3

ComboList = typing.List[gtc.Combo]
PerComboJAV = typing.List[typing.Dict[gtc.Combo, typing.Dict[str, NDArray]]]
OrderForJAV = typing.Tuple[JAV, JAV, JAV]


def dataForCombosJAV(combos: ComboList, vec_order: OrderForJAV,
                     return_world2locals: bool = False, 
                     return_translations: bool = False):
    '''
    For each frame of video, we consider a coordinate frame where one axis is
    aligned with the object's velocity and another is aligned with the
    acceleration (or, at least, the part of it orthogonal to velocity).
    We then calculate and return the speed, acceleration, and jerk for the current
    time and the position at the next time in this frame.
    Returns a List[Dict[Combo, Dict[str, NDArray]]] that again separates things
    by frame skip amount and by combo.
    '''

    # Empty dict for each skip amount.
    all_data: PerComboJAV = [dict() for _ in range(3)]
    all_world2local_mats: typing.List[typing.Dict[gtc.Combo, NDArray]] = \
        [dict() for _ in range(3)]
    all_translations: typing.List[typing.Dict[gtc.Combo, NDArray]] = \
        [dict() for _ in range(3)]
    
    if vec_order is None:
        vec_order = (JAV.VELOCITY, JAV.ACCELERATION, JAV.JERK)
    if len(vec_order) != 3 or {v.value for v in vec_order} != {1, 2, 3}:
        raise ValueError("Vector order must be a permutation of (velocity, acceleration, jerk)!")
    
    skip_end = 3#1 if onlySkip0 else 3
    for c in combos:
        calc_obj = BCOT_Data_Calculator(c.body_ind, c.seq_ind, 0)
        curr_translations = calc_obj.getTranslationsGTNP(False)
        for skip in range(skip_end):
            step = skip + 1
            translations = curr_translations[::step]
            vels = np.diff(translations, axis=0)
            # We need a velocity for the last timestep, but not an acceleration,
            # because we need the vectors that take each current position to
            # the next when calculating the "ground truth" for displacement
            # predictions. This is the velocity vector; acceleration vectors
            # are not needed for this; we only need "current" acceleration.
            accs = np.diff(vels[:-1], axis=0)
            jerks = np.diff(accs, axis=0)

            # Here we specify which order in which we orthonormalize our
            # velocity, acceleration, and jerk vectors into orthonormal frames.
            # The first-chosen of these gets aligned exactly with an axis, while
            # the others only get orthogonal components aligned with an axis.
            
            # To only calculate as much as we need, we clip the arrays' fronts
            # off when we can.
            default_ordered = (vels[2:-1], accs[1:], jerks)
            ordered = copy.copy(default_ordered)
            if vec_order is not None:
                ordered = tuple(default_ordered[v.value - 1] for v in vec_order)

            mags0 = np.linalg.norm(ordered[0], axis=-1)
            unit_vecs0 = pm.safelyNormalizeArray(
                ordered[0], mags0[:, np.newaxis]
            )
            # Find the magnitude of the second vector that is parallel to and
            # orthogonal to the first.
            mags_p1 = pm.einsumDot(ordered[1], unit_vecs0) # Parallel magnitude
            vecs_p1 = pm.scalarsVecsMul(mags_p1, unit_vecs0) # Parallel vec3
            vecs_o1 = ordered[1] - vecs_p1 # Orthogonal vec3
            mags_o1 = np.linalg.norm(vecs_o1, axis=-1) # Orthogonal magnitude

            unit_vecs_o1 = pm.safelyNormalizeArray(
                vecs_o1, mags_o1[:, np.newaxis]
            )

            unit_vecs2 = np.cross(unit_vecs0, unit_vecs_o1)
            # We now have matrices to convert vectors in world space into
            # these local vector-aligned frames.
            mats = np.stack([unit_vecs0, unit_vecs_o1, unit_vecs2], axis=1)

            # Transform each third vector and to-next-frame displacement into
            # this frame via matmul.
            local_vecs2 = pm.einsumMatVecMul(mats, ordered[2])
            local_diffs = pm.einsumMatVecMul(mats, vels[3:])

            # We'll now return all of the data needed to convert velocity,
            # acceleration, and jerk multipliers into local vectors in these
            # new frames. To do this, we don't need to return the coordinate
            # frames themselves: we just need to know the velocity in this
            # frame (a vector [speed, 0, 0]), the acceleration in this frame
            # (i.e. [a_p, a_o, 0]), etc. And since we don't need to return 0s,
            # we can just return the following:
            c_res = (
                mags0, mags_p1, mags_o1, *(local_vecs2.T), *(local_diffs.T)
            )

            all_data[skip][c] = np.stack(c_res, axis=-1)
            if return_world2locals:
                all_world2local_mats[skip][c] = mats
            if return_translations:
                all_translations[skip][c] = translations
    if return_world2locals or return_translations:
        res = (all_data, )
        if return_world2locals:
            res += (all_world2local_mats, )
        if return_translations:
            res += (all_translations, )
        return res
    return all_data


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


