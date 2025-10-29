import typing

import numpy as np
from numpy.typing import NDArray

import tensorflow as tf

from motiontools.posefeatures import (
    HypotheticalInputsForNN, getWorldFrameDisplacements
)

def getOutputsNN(model, scaler, hyp_calcer: HypotheticalInputsForNN,
                 last_pts: NDArray, prev_pts: typing.Optional[NDArray] = None,
                 rot_mats: typing.Optional[NDArray] = None,
                 return_inputs: str = 'none'):
    '''
        For a given set of prior transformations, gets the world frame vec3
        positions predicted by a model that outputs vec12 JAV multipliers.

        Parameters:
            model: The model (e.g., a neural net) that outputs the multipliers.
                Any class that has a predict() method that takes in a 2D array
                of prediction data matching the scaler's outputs and produces
                an output array of shape (n, 12) can be used here.
            scaler: A scaler that transforms the "raw" data columns generated.
            hyp_calcer (HypotheticalInputsForNN): Used to calculate the raw
                columns, pre-scaling, for input into the model.
            last_pts (NDArray): Positions at the final timestep pre-prediction.
                If None, it is assumed that hyp_calcer is set up with this
                information already. Defaults to None.
            prev_pts (NDArray): Positions leading up to last_pts.
            rot_mats (NDArray): Rotation matrices up to and including final
                timestep pre-prediction. If None, it is assumed that hyp_calcer
                is set up with this information already. Defaults to None.
            return_inputs (str): Specifies whether to just return predicted
                vec3s ("none"), or to also return scaled ("scaled") or unscaled
                ("unscaled") columns generated as the model's input. Defaults
                to "none".

        Returns:
            NDArray or tuple: Depending on the value of return_inputs, either
                an NDArray of shape (n, 3) of world frame positions predicted
                by the model or else a 2-tuple which contains:    
                    - The above-mentioned vec3s (ndarray).
                    - The data columns generated as model input (ndarray).
        '''

    last_pts = np.atleast_2d(last_pts)
    orig_write_status = last_pts.flags.writeable

    last_pts.setflags(write=False)

    unscaled_cols: NDArray
    jav_mags: NDArray
    w2ls: NDArray
    if prev_pts is None or rot_mats is None:
        if not (prev_pts is None and rot_mats is None):
            raise ValueError("Cannot have just one of pts and mats be None!")
        unscaled_cols, jav_mags, w2ls = hyp_calcer.getHypotheticalCalcs(
            last_pts
        )
    else:
        all_pts = np.concatenate([prev_pts, [last_pts]], axis=0)
        unscaled_cols, jav_mags, w2ls = hyp_calcer.getSeparateCalcs(
            all_pts, rot_mats
        )

    scaled_inputs = scaler.transform(unscaled_cols)
    out_JAV = model.predict(scaled_inputs, batch_size=1024, verbose=0)
    displacements = getWorldFrameDisplacements(jav_mags, out_JAV, w2ls)

    final_positions = last_pts + displacements

    last_pts.setflags(write=orig_write_status)
    if return_inputs == "unscaled":
        return final_positions, unscaled_cols
    elif return_inputs == "scaled":
        return final_positions, scaled_inputs
    elif return_inputs != "none":
        raise ValueError("Invalid return_inputs value of: " + return_inputs)
    return final_positions


def tfSafeDivideElseZero(x, y):
    return tf.where(tf.equal(y, 0), tf.zeros_like(x), x / y)

def tfNormalizeAll(vecs):
    norms = tf.norm(vecs, axis=-1, keepdims=True)
    return vecs / norms

def tfEinsumDot(vecs0: tf.Tensor, vecs1: tf.Tensor) -> tf.Tensor:
    return tf.einsum('...j,...j->...', vecs0, vecs1)

def tfScalarsVecsMul(scalars: tf.Tensor, vecs: tf.Tensor) -> tf.Tensor:
    '''
    Multiply each vector in an array of vectors by the corresponding scalar
    from an array of scalars using TensorFlow operations.
    '''
    return tf.einsum('...,...i->...i', scalars, vecs)

def tfParallelAndOrthoParts(vectors: tf.Tensor, dirs: tf.Tensor, dirs_already_normalized: bool = False):
    dots = tfEinsumDot(vectors, dirs)
    
    if not dirs_already_normalized:
        dots /= tfEinsumDot(dirs, dirs)
    
    parallels = tfScalarsVecsMul(dots, dirs)
    orthos = vectors - parallels
    return (parallels, orthos)

def tfSafelyNormalizeArray(vecs, mags):
    """
    Normalize the input vectors by dividing each vector by its magnitude.
    
    Args:
        vecs (tf.Tensor): Input vectors of shape (n,) or (n, m).
        mags (tf.Tensor): Magnitudes of the vectors of shape (n,) or (n, 1).
        
    Returns:
        tf.Tensor: Normalized vectors of the same shape as `vecs`.
    """
    # Ensure that mags has the correct shape for broadcasting
    if vecs.ndim == 2 and mags.ndim == 1:
        mags = tf.expand_dims(mags, axis=1)
    
    # Avoid division by zero
    mask = tf.equal(mags, 0.0)
    mags = tf.where(mask, tf.ones_like(mags), mags)
    
    # Normalize vectors
    normalized_vecs = vecs / mags
    
    return normalized_vecs

def tfGetOrthonormalFrames(returned_mats_are_world2vecs: bool,
                         vecs0: tf.Tensor,
                         vecs1: tf.Tensor,
                         vecs0_are_unit_len: bool = False,
                         zero_thresh=DEFAULT_ZERO_ANG_THRESH) -> (tf.Tensor, tf.Tensor):
    mags0 = tf.ones_like(vecs0[..., 0]) if vecs0_are_unit_len else tf.norm(vecs0, axis=-1)
    ret_mags = (mags0,)
    
    unit_vecs0 = vecs0 if vecs0_are_unit_len else safelyNormalizeArray(vecs0, mags0)

    # Find the magnitude of the second vector that is parallel to and
    # orthogonal to the first.
    mags_p1 = einsumDot(vecs1, unit_vecs0)
    vecs_p1 = scalarsVecsMul(mags_p1, unit_vecs0)
    vecs_o1 = vecs1 - vecs_p1
    mags_o1 = tf.norm(vecs_o1, axis=-1)

    # vecs_o1[i] will be zero vectors, and can't be normalized, if any vecs0[i]
    # and vecs1[i] are parallel for some i.
    v1_is_parallel = mags_o1 < zero_thresh
    unit_vecs1 = tf.zeros_like(unit_vecs0)

    v1_not_parallel = ~v1_is_parallel
    mags_o1_div = tf.where(v1_not_parallel, mags_o1[:, tf.newaxis], tf.ones_like(mags_o1))
    unit_vecs1 = vecs_o1 / mags_o1_div
    
    ret_mags += (mags_p1, mags_o1)

    unit_vecs2: tf.Tensor
    v2_is_parallel: tf.Tensor
    v2_not_parallel: tf.Tensor
    unit_vecs2 = tf.cross(unit_vecs0, unit_vecs1)
    v2_is_parallel = v1_is_parallel
    v2_not_parallel = ~v1_is_parallel
    
    unit_vecs1 = tf.where(v1_is_parallel, 0.0, unit_vecs1)
    unit_vecs2 = tf.where(v2_is_parallel, 0.0, unit_vecs2)

    stack_ax = -2 if returned_mats_are_world2vecs else -1

    all_unit_vecs = (unit_vecs0, unit_vecs1, unit_vecs2)
    mats = tf.stack(all_unit_vecs, axis=stack_ax)

    return ret_mags, mats

# @tf.function(input_signature=[
#     tf.TensorSpec(shape=(None, None), dtype=tf.float32),
#     tf.TensorSpec(shape=(), dtype=tf.float32),
#     # Other required tensors here...
# ])
# def lambda_layer(x):
#     return hypothetical_calcs(x, **other_args)

# lambda_layer = keras.layers.Lambda(lambda_layer)
