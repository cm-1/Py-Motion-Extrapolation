# pyright: reportOperatorIssue=false
# pyright: reportIndexIssue=false
import typing

import numpy as np
from numpy.typing import NDArray

import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation
import keras

from motiontools.hypothetical_inputs_calc import HypotheticalInputsForNN
from motiontools.posefeatures import (
    getWorldFrameDisplacements, MOTION_DATA, ALL_RELATIVE_VECTORS
)

from posemath import DEFAULT_ZERO_ANG_THRESH

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

def tfNormalizeAll(vecs: tf.Tensor) -> tf.Tensor:
    norms = tf.norm(vecs, axis=-1, keepdims=True)
    return tf.math.divide_no_nan(vecs, norms)

def tfEinsumDot(vecs0: tf.Tensor, vecs1: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(vecs0 * vecs1, axis=-1)

def tfScalarsVecsMul(scalars: tf.Tensor, vecs: tf.Tensor) -> tf.Tensor:
    '''
    Multiply each vector in an array of vectors by the corresponding scalar
    from an array of scalars using TensorFlow operations.
    '''
    return tf.multiply(scalars[:, tf.newaxis], vecs)

def tfParallelAndOrthoParts(vectors: tf.Tensor, dirs: tf.Tensor, dirs_already_normalized: bool = False):
    dots = tfEinsumDot(vectors, dirs)
    
    if not dirs_already_normalized:
        dots /= tfEinsumDot(dirs, dirs)
    
    parallels = tfScalarsVecsMul(dots, dirs)
    orthos = vectors - parallels
    return (parallels, orthos)

def tfCross(x: tf.Tensor, y:tf.Tensor):
    # i,  j,  k
    # x0, x1, x2
    # y0, y1, y2
    z0 = x[..., 1] * y[..., 2] - y[..., 1] * x[..., 2]
    z1 = x[..., 2] * y[..., 0] - y[..., 2] * x[..., 0]
    z2 = x[..., 0] * y[..., 1] - y[..., 0] * x[..., 1]
    return tf.stack((z0, z1, z2), axis=-1)

def tfOrthonormalFramesFromUnitVec0s(returned_mats_are_world2vecs: bool,
                                     unit_vecs0: tf.Tensor, vecs1: tf.Tensor,
                                     zero_thresh: float):    

    # Find the magnitude of the second vector that is parallel to and
    # orthogonal to the first.
    mags_p1 = tfEinsumDot(vecs1, unit_vecs0)
    vecs_p1 = tfScalarsVecsMul(mags_p1, unit_vecs0)
    vecs_o1 = vecs1 - vecs_p1
    mags_o1 = tf.norm(vecs_o1, axis=-1)

    v1_is_parallel = mags_o1 < zero_thresh
    unit_vecs1 = tf.math.divide_no_nan(vecs_o1, mags_o1[..., tf.newaxis])
    unit_vecs1 = tf.where(
        v1_is_parallel[:, tf.newaxis], tf.zeros_like(unit_vecs1), unit_vecs1
    )


    ret_mags = (mags_p1, mags_o1)

    # Compute the third orthogonal vector
    unit_vecs2 = tfCross(unit_vecs0, unit_vecs1)
    
    stack_ax = -2 if returned_mats_are_world2vecs else -1

    all_unit_vecs = (unit_vecs0, unit_vecs1, unit_vecs2)
    mats = tf.stack(all_unit_vecs, axis=stack_ax)

    return ret_mags, mats


def compute_relative_rotations(axis_angles1: tf.Tensor, axis_angles2: tf.Tensor):
    """
    Computes the relative axis-angle rotations between two sets of axis-angle
    rotations using quaternions.

    Args:
        axis_angles1: A tensor of shape (n, 3) representing the first set
                      of axis-angle rotations.
        axis_angles2: A tensor of shape (n, 3) representing the second set
                      of axis-angle rotations.

    Returns:
        A tensor of shape (n, 3) representing the relative axis-angle rotations.
    """

    angs1 = tf.norm(axis_angles1, axis=-1, keepdims=True)
    angs2 = tf.norm(axis_angles2, axis=-1, keepdims=True)

    units1 = tf.math.divide_no_nan(axis_angles1, angs1)
    units2 = tf.math.divide_no_nan(axis_angles2, angs2)

    # Convert axis angles to quaternions
    quaternions1 = tfg_transformation.quaternion.from_axis_angle(units1, angs1)
    quaternions2 = tfg_transformation.quaternion.from_axis_angle(units2, angs2)

    # Compute the relative quaternion
    relative_quaternions = tfg_transformation.quaternion.multiply(
        quaternions2, tfg_transformation.quaternion.conjugate(quaternions1)
    )

    # Convert the relative quaternion back to axis angles
    relative_axis_angles = tfg_transformation.axis_angle.from_quaternion(
        relative_quaternions
    )

    # Convert from (axis, angles) tuple to scaled vec3 axis-angles.
    return relative_axis_angles[0] * relative_axis_angles[1]

def ang_vel_extrapolate(axis_angles1: tf.Tensor, axis_angles2: tf.Tensor):
    """
    Computes the relative axis-angle rotations between two sets of axis-angle
    rotations using quaternions.

    Args:
        axis_angles1: A tensor of shape (n, 3) representing the first set
                      of axis-angle rotations.
        axis_angles2: A tensor of shape (n, 3) representing the second set
                      of axis-angle rotations.

    Returns:
        A tensor of shape (n, 3) representing the relative axis-angle rotations.
    """

    angs1 = tf.norm(axis_angles1, axis=-1, keepdims=True)
    angs2 = tf.norm(axis_angles2, axis=-1, keepdims=True)

    units1 = tf.math.divide_no_nan(axis_angles1, angs1)
    units2 = tf.math.divide_no_nan(axis_angles2, angs2)

    # Convert axis angles to quaternions
    quaternions1 = tfg_transformation.quaternion.from_axis_angle(units1, angs1)
    quaternions2 = tfg_transformation.quaternion.from_axis_angle(units2, angs2)

    # Compute the relative quaternion
    relative_quaternions = tfg_transformation.quaternion.multiply(
        quaternions2, tfg_transformation.quaternion.conjugate(quaternions1)
    )

    new_quaternions = tfg_transformation.quaternion.multiply(
        relative_quaternions, quaternions2
    )

    # Convert the new quaternion back to axis angles
    relative_axis_angles = tfg_transformation.axis_angle.from_quaternion(
        new_quaternions
    )

    # Convert from (axis, angles) tuple to scaled vec3 axis-angles.
    return relative_axis_angles[0] * relative_axis_angles[1]


def _get_JAV_muls(step: typing.Union[int, float]):
        _jav_muls = tf.Variable(tf.zeros((4, 3), dtype=tf.float32))
        # Jerk through crackle each have 3 components we must multiply by a
        # respective power of the step assuming it's also the next "delta T".
        range_rs = tf.cast(tf.reshape(tf.range(3, 6), (3, 1)), tf.float32)
        _jav_muls[1:].assign(float(step) ** range_rs)
        # Then velocity and acceleration are special because they only have 1
        # and 2 multipliers, respectively.
        _jav_muls[0, :1].assign([step])
        _jav_muls[0, 1:3].assign(step ** 2)
        _jav_muls = tf.reshape(_jav_muls, [-1])
        return tf.constant(_jav_muls, dtype=tf.float32)

class LayerPostOutJAV12(keras.layers.Layer):
    def __init__(self, step: int, **kwargs):
        super(LayerPostOutJAV12, self).__init__(**kwargs)

        self.step = step
        
        self._jav_muls = _get_JAV_muls(step)


    def call(self, inputs):
        # Multiply the JAV outputs by the timestep powers
        # inputs should be shape (batch_size, 12) - the 12 JAV components
        return inputs * self._jav_muls[tf.newaxis, :]



class DerivativeCollectionConstTimeTF:
    def __init__(self, displacements: tf.Tensor, max_derivative_order: int,
                 time_step: typing.Union[int, float]):
                        
        self.time_step = time_step
        self.velocities = displacements / time_step

        if max_derivative_order >= 2:
            self.accelerations = self._recursiveDeriv(self.velocities, 2)
        if max_derivative_order >= 3:
            self.jerks = self._recursiveDeriv(self.accelerations, 3)
        if max_derivative_order >= 4:
            self.snaps = self._recursiveDeriv(self.jerks, 4)
        if max_derivative_order >= 5:
            self.crackles = self._recursiveDeriv(self.snaps, 5)
        if max_derivative_order >= 6:
            raise NotImplementedError("Derivative orders >= 6 not supported!")

    def _recursiveDeriv(self, prev_vals: tf.Tensor, deriv_power: int):
        """Compute higher-order derivatives recursively.
        E.g., prev_vals are the last accelerations, deriv_power is 3 
        (for jerk)."""
        ret_val = keras.ops.diff(prev_vals, 1, axis=0)
        ret_val = ret_val / self.time_step
        return ret_val


class DerivativeCollectionWithTimestampsTF:
    def __init__(self, displacements: tf.Tensor, max_derivative_order: int,
                 timestamps: tf.Tensor):
                
        self.velocities = typing.cast(tf.Tensor, tf.identity(displacements))
        
        # Reshape for broadcasting.
        coeff_shape = self.velocities.shape[:-1] + tf.TensorShape(1)
        self.unflat_timestamps = tf.reshape(timestamps, coeff_shape)

        time_deltas = keras.ops.diff(self.unflat_timestamps, 1, axis=0)
        self.velocities = displacements / time_deltas

        if max_derivative_order >= 2:
            self.accelerations = self._recursiveDeriv(self.velocities, 2)
        if max_derivative_order >= 3:
            self.jerks = self._recursiveDeriv(self.accelerations, 3)
        if max_derivative_order >= 4:
            self.snaps = self._recursiveDeriv(self.jerks, 4)
        if max_derivative_order >= 5:
            self.crackles = self._recursiveDeriv(self.snaps, 5)
        if max_derivative_order >= 6:
            raise NotImplementedError("Derivative orders >= 6 not supported!")

    def _recursiveDeriv(self, prev_vals: tf.Tensor, deriv_power: int):
        """Compute higher-order derivatives recursively.
        E.g., prev_vals are the last accelerations, deriv_power is 3 
        (for jerk)."""
        ret_val = keras.ops.diff(prev_vals, 1, axis=0)

        time_div = self.unflat_timestamps[deriv_power:] - self.unflat_timestamps[:-deriv_power]
        # Do scalar math first for efficiency, as otherwise you perform
        # a division on vec3s and then a mul on vec3s, instead of doing
        # the division on "vec1s". Could use parentheses, but this is
        # more explicit.
        scalars = deriv_power / time_div
        ret_val = scalars * ret_val
        return ret_val

@keras.saving.register_keras_serializable()
class PointsToInputsConstStep(keras.layers.Layer):
    def __init__(self, step: typing.Union[int, float],
                 scale_means: NDArray, scale_scales: NDArray,
                 to_nn_permut: NDArray,
                 zero_angle_thresh: float = DEFAULT_ZERO_ANG_THRESH, **kwargs):
        super(PointsToInputsConstStep, self).__init__(**kwargs)

        self.step = step
        self.zero_angle_thresh = zero_angle_thresh
        self.scale_means = tf.constant(scale_means.astype(np.float32))
        self.scale_scales = tf.constant(scale_scales.astype(np.float32))

        # 8: vecs, accs, jerks, snaps, crackles, rot vel, rot acc, rot jerk
        self._n_vec_kinds = 8
        self._n_other_vec_kinds = self._n_vec_kinds - 1
        
        self.to_nn_permut = tf.constant(to_nn_permut, dtype=tf.int32)


        # Pre-compute lower triangle indices in __init__ for efficiency
        # Create index arrays using NumPy, then convert to TensorFlow constants
        np_tril_rows, np_tril_cols = np.tril_indices(7)
        
        # Stack them together: shape (num_tril_elements, 2)
        tl_inds = np.column_stack([np_tril_rows, np_tril_cols]).astype(np.int32)
        
        # Store as a TensorFlow constant
        self._tril_indices = tf.constant(tl_inds)


        self._jav_muls = _get_JAV_muls(step)

        CIND = ALL_RELATIVE_VECTORS.index(MOTION_DATA.JERK_ERR_VEC3)
        # We don't divide by crackle's mag because it's not used as a relative
        # axis.
        self._non_crackles = tf.constant([
            i for i in range(1, self._n_vec_kinds) if i != CIND
        ], dtype=tf.int32)
        if len(self._non_crackles) != 6:
            raise Exception(
                "Hardcoded self._non_crackles length in non_vel_projs_flat no "
                "longer correct!"
            )

        
        # Calculate the size of non_vel_projs for fixed-size TensorArray
        # For each i in _non_crackles: we add 1 tensor, plus 1 more if i < _n_other_vec_kinds
        self._non_vel_projs_size = sum(
            2 if i < self._n_other_vec_kinds else 1
            for i in range(1, self._n_vec_kinds) if i != CIND
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "step": self.step,
            "zero_angle_thresh": self.zero_angle_thresh,
            "scale_means": self.scale_means.numpy().tolist(),
            "scale_scales": self.scale_scales.numpy().tolist(),
            "to_nn_permut": self.to_nn_permut.numpy().tolist(),
        })
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        tf.print("config:", config)
        
        # Extract parameters from the config
        step = config["step"]
        zero_angle_thresh = config["zero_angle_thresh"]
        scale_means = config["scale_means"]
        scale_scales = config["scale_scales"]
        to_nn_permut = config["to_nn_permut"]
        


        # Create a new instance of the layer
        return cls(
            step=step,
            scale_means=np.array(scale_means),
            scale_scales=np.array(scale_scales),
            to_nn_permut=np.array(to_nn_permut),
            zero_angle_thresh=zero_angle_thresh,
        )
    # The below function is basically a version of HypotheticalInputsForNN
    # but using tensorflow instead of numpy. Because I'm not very familiar with
    # tensorflow's API I experimented by letting Claude Code take a shot at it.
    # I've read through it and tested it a lot, and changed it a fair bit, with
    # much of said revision coming from advice I read in the tf docs, so
    # hopefully it's not too bad but maybe it's not as optimized as it could be.
    @tf.function
    def calculateOutputs(self, x0_through_5: tf.Tensor, aa0_through_5: tf.Tensor):

        scaled_disps = keras.ops.diff(x0_through_5, 1, axis=0)
        all_pds = DerivativeCollectionConstTimeTF(
            typing.cast(tf.Tensor, scaled_disps), 5, self.step
        )
        
        scaled_aas = aa0_through_5 #/ self.rot_scale
        angvel_aas = compute_relative_rotations(
            scaled_aas[:-1], scaled_aas[1:]
        )

        all_rds = DerivativeCollectionConstTimeTF(angvel_aas, 3, self.step)

        all_prev_vels = all_pds.velocities[:-1]
        prev_vel = all_prev_vels[-1]
        prev_acc = all_pds.accelerations[-2]
        prev_jerk = all_pds.jerks[-2]
        prev_snap = all_pds.snaps[-2]
        
        # Calculate velocities, speeds, and indices where speed is 0.
        vels = all_pds.velocities[-1]
        vel_mags: tf.Tensor = tf.norm(vels, axis=-1, keepdims=True)
        accs = all_pds.accelerations[-1]
        jerks = all_pds.jerks[-1]
        snaps = all_pds.snaps[-1]
        crackles = all_pds.crackles[-1]


        prev_vel_mags = tf.norm(all_prev_vels, axis=-1)
        # rev_prev_vel_mags = prev_vel_mags[..., ::-1]
        # last_nonzero_vels = all_prev_vels[-1]
        
        # pm.handleCondsAtStart(
        #     rev_vel_mags == 0.0, rev_vel_mags, self._replaceAtInd,
        #     arr_to_mod=last_nonzero_vels
        # )
        
        # last_nonzero_unit_vels = tfNormalizeAll(last_nonzero_vels)
        prev_speed = tf.norm(prev_vel, axis=-1, keepdims=True)
        prev_unit_vel = tf.math.divide_no_nan(prev_vel, prev_speed)

        prev_ang_vel = all_rds.velocities[-2]
        prev_ang_acc = all_rds.accelerations[-2]
        prev_ang_jerk = all_rds.jerks[-2]

        new_ang_vel = all_rds.velocities[-1]
        new_ang_acc = all_rds.accelerations[-1]
        new_ang_jerk = all_rds.jerks[-1]

        prev_ortho_mags, prev_ortho_dirs = tfOrthonormalFramesFromUnitVec0s(
            True, prev_unit_vel, prev_acc, self.zero_angle_thresh
        )

        a0_is_0 = tf.equal(prev_ortho_mags[1], 0.0)
        
        # Only compute if any acceleration orthogonal components are zero
        def update_ortho_dirs_with_jerk(ortho_dirs, jerk_vals, condition_mask):
            """Update third orthogonal direction using jerk where condition is True."""
            indices = tf.where(condition_mask)
            jerk_masked = tf.gather_nd(jerk_vals, indices)
            ortho_dirs_0_masked = tf.gather_nd(ortho_dirs[:, 0, :], indices)
            
            _, j_orth = tfParallelAndOrthoParts(
                jerk_masked, ortho_dirs_0_masked, True
            )
            j_orth_normalized = tfNormalizeAll(j_orth)
            
            # Build indices for scatter: we need to update ortho_dirs[i, 2, :] for each i in indices
            # Cast to match indices dtype (int64 from tf.where)
            # NOTE: Need to use tf.shape(...) instead of x.shape in graph mode!
            scatter_indices = tf.concat([
                indices,
                tf.cast(tf.fill([tf.shape(indices)[0], 1], 2), indices.dtype)
            ], axis=1)
            
            # Update only the masked indices in the third orthogonal direction
            return tf.tensor_scatter_nd_update(
                ortho_dirs,
                scatter_indices,
                j_orth_normalized
            )
        
        prev_ortho_dirs = tf.cond(
            tf.reduce_any(a0_is_0),
            lambda: update_ortho_dirs_with_jerk(prev_ortho_dirs, prev_jerk, a0_is_0),
            lambda: prev_ortho_dirs
        )

        # print("hyp pre:", prev_ortho_dirs)

        prev_relative_vecs = tf.stack(
            (
                prev_vel, prev_acc, prev_jerk, prev_snap,
                prev_ang_vel, prev_ang_acc, prev_ang_jerk,
                prev_ortho_dirs[..., 1, :], prev_ortho_dirs[..., 2, :]
            ), axis=0
        )

        # Because the scales of the prev_ortho_dirs are just 1, we exclude these
        # from the array, hence the "-2" instances below.
        # num_scale_types = self._n_rel_axes_excl_ortho  # 7 relative axes (vel, acc, jerk, snap, ang_vel, ang_acc, ang_jerk)

        # inner_prev_vecs_shape = tf.shape(prev_relative_vecs)[1:-1]
        # n_ins = x0_through_5.shape[1]
        # prev_scale_shape = (num_scale_types, n_ins)

        # Build prev_relative_scales using tensor_scatter_nd_update for graph compatibility
        scale_0 = tf.reshape(prev_vel_mags[-1], [1, -1])
        scale_1 = tf.reshape(tf.sqrt(prev_ortho_mags[0]**2 + prev_ortho_mags[1]**2), [1, -1])
        scales_2_onwards = tf.norm(prev_relative_vecs[2:-2], axis=-1)  # Shape: (5, n_ins)
        
        prev_relative_scales = tf.concat([scale_0, scale_1, scales_2_onwards], axis=0)

        # Safely obtain unit velocities, 
        unit_vels = tf.math.divide_no_nan(vels, vel_mags)
        vel_mag_is_0 = tf.reshape(vel_mags, [-1]) < self.zero_angle_thresh
        # unit_vels = tf.where(
        #     vel_mag_is_0, last_nonzero_unit_vels, unit_vels
        # )

        all_curr_vecs = tf.stack(
            (
                vels, accs, jerks, snaps, crackles,
                new_ang_vel, new_ang_acc, new_ang_jerk
            ), axis=0
        )

        tf.debugging.assert_equal(
            self._n_vec_kinds, all_curr_vecs.shape[0],
            "Hardcoded value for self._n_vec_kinds no longer correct!"
        )

        all_dots_with_prev = tf.einsum(
            'aik,bik->abi', all_curr_vecs, prev_relative_vecs
        )

        all_proj_with_prev = tf.math.divide_no_nan(
            all_dots_with_prev[:, :-2], prev_relative_scales 
        )

        
        (a_proj_v, a_ortho_v), curr_ortho_mats = tfOrthonormalFramesFromUnitVec0s(
            True, unit_vels, accs, self.zero_angle_thresh 
        )
        a0_is_0 = tf.equal(a_ortho_v, 0.0)
        
        curr_ortho_mats = tf.cond(
            tf.reduce_any(a0_is_0),
            lambda: update_ortho_dirs_with_jerk(curr_ortho_mats, jerks, a0_is_0),
            lambda: curr_ortho_mats
        )
        # print("hyp curr:", curr_ortho_mats)

        # For the various dot products between *current* frame values, we'll
        # keep track of them in a list of tensors so that we don't duplicate
        # dot products.
        # We have 8 non-unit vec3s (vel..crackle and rot vel..jerk), but we'll 
        # handle the unit-length orthogonal relative axes separately, so our
        # dimensions will be 7x7x...
        # NOTE: Need to use tf.shape(...) instead of x.shape in graph mode!
        n_ins = tf.shape(x0_through_5)[1]
        

        accs_2D = tf.stack((a_proj_v, a_ortho_v), axis=-1)
        
        # Rather than keep the self-dots in the "diagonal", I think it may be
        # slightly more efficient to keep them separate; we'll need to divide
        # by them later, so this prevents the need for diag indexing or copies.
        # Build non_vel_mags using concat for graph compatibility
        acc_mag = tf.norm(accs_2D, axis=-1, keepdims=True)  # Shape: (n_ins, 1)
        other_mags = tf.norm(all_curr_vecs[2:], axis=-1)    # Shape: (6, n_ins)
        non_vel_mags = tf.concat([tf.transpose(acc_mag), other_mags], axis=0)  # Shape: (7, n_ins)

        # Build tri_dots row by row using TensorArray for graph compatibility
        tri_dots_rows = tf.TensorArray(
            dtype=tf.float32,
            size=self._n_other_vec_kinds - 1,
            dynamic_size=False,
            clear_after_read=False
        )
        
        # First, the dots with velocity. Row looks like [v*a, 0, 0, ...]
        row_0_val = tf.reshape(vel_mags, [-1]) * a_proj_v  # Shape: (n_ins,)
        row_0 = tf.concat([
            row_0_val[tf.newaxis, :],  # Element [0,0]
            tf.zeros((self._n_other_vec_kinds - 1, n_ins), dtype=tf.float32)  # Rest are zeros
        ], axis=0)
        

        # Rows 1 through n_other_vec_kinds-1: compute dot products
        i = tf.constant(2)
        lt_n_vec_kinds = lambda _i, _, _2: tf.less(_i, self._n_vec_kinds)
        def loop_bod(i, all_curr_vecs, tri_dots_rows): #: _Try_Dot_Bod_Tup):
            # Compute tri_dots[i-1, :i] = dot products with all previous vectors
            dots_i = tf.einsum('jk,ijk->ij', all_curr_vecs[i], all_curr_vecs[:i])
            # Pad with zeros for the rest of the row
            row_i = tf.concat([
                dots_i,  # Shape: (i, n_ins)
                tf.zeros((self._n_other_vec_kinds - i, n_ins), dtype=tf.float32)
            ], axis=0)
            
            return i+1, all_curr_vecs, tri_dots_rows.write(i - 2, row_i)
        # I'm trying to follow a working example:
        #https://github.com/onnx/tensorflow-onnx/issues/1899
        # 
        i, all_curr_vecs, tri_dots_rows = tf.while_loop(
            lt_n_vec_kinds, loop_bod, loop_vars=[i, all_curr_vecs, tri_dots_rows]
        )
        # Stack all rows together
        tri_dot_stack = tri_dots_rows.stack()  # Shape: (n_other_vec_kinds, n_other_vec_kinds, n_ins)
        tri_dots = tf.concat((tf.reshape(row_0, [1, 7, n_ins]), tri_dot_stack), axis=0) 



        # For the projections, we don't worry about "duplicates" since there are
        # none: projecting jerk onto velocity is different from vice versa.
        # However, we have the diagonal separated out already, so we can
        # exclude that.
        acc_vel_proj = tf.where(
            vel_mag_is_0, tf.zeros_like(a_proj_v), a_proj_v
        )
        acc_vel_proj = acc_vel_proj[tf.newaxis]
        other_projs_on_vel = tf.math.divide_no_nan(
            tri_dots[1:, 0], tf.reshape(vel_mags, [-1])
        )


        # Collect non_vel_projs by flattening each piece before storing
        # Since the final result needs to be concatenated anyway, we flatten to 1D first
        non_vel_projs = tf.TensorArray(
            dtype=tf.float32, 
            size=len(self._non_crackles),
            dynamic_size=False,
            clear_after_read=False,
            infer_shape=False  # Don't infer shape from first element - allow varying lengths
        )
        
        lt_len_non_crack = lambda _i, _, _2: tf.less(_i, len(self._non_crackles))
        def nvp_loop_bod(i_ind, arr_idx, non_vel_projs):
            i = self._non_crackles[i_ind]
            prev_i = i - 1
            mag_i = non_vel_mags[prev_i]
            proj_val = tf.math.divide_no_nan(tri_dots[prev_i, :i], mag_i)
            # Flatten to 1D: shape (i, n_ins) -> (i * n_ins,)
            proj_val_flat = tf.reshape(proj_val, [-1])
        # def nvp_loop_bod2(i_ind, arr_idx, non_vel_projs2):
            # i = self._non_crackles[i_ind]
            # if i < self._n_other_vec_kinds:
            towrite = proj_val_flat
            if i < self._n_other_vec_kinds:
                proj_val_h = tf.math.divide_no_nan(tri_dots[i:, i], mag_i)
                # Flatten to 1D: shape (n_other_vec_kinds - i, n_ins) -> ((n_other_vec_kinds - i) * n_ins,)
                proj_val_h_flat = tf.reshape(proj_val_h, [-1])
                towrite = tf.concat((proj_val_flat, proj_val_h_flat), axis=0)
            non_vel_projs = non_vel_projs.write(arr_idx, towrite)
            return i_ind + 1, arr_idx + 1, non_vel_projs
        i = tf.constant(0)
        arr_idx = 0
        i, arr_idx, non_vel_projs = tf.while_loop(lt_len_non_crack, nvp_loop_bod, [i, arr_idx, non_vel_projs])
        
        non_vel_projs_flat = tf.concat((
            non_vel_projs.read(0), #non_vel_projs2.read(0),
            non_vel_projs.read(1), #non_vel_projs2.read(1),
            non_vel_projs.read(2), #non_vel_projs2.read(2),
            non_vel_projs.read(3), #non_vel_projs2.read(3),
            non_vel_projs.read(4), #non_vel_projs2.read(4),
            non_vel_projs.read(5) #, non_vel_projs2.read(5),
            # non_vel_projs.read(6) #, non_vel_projs2.read(0),
        ), axis=0)
        # Concatenate all flattened pieces into one 1D tensor, then reshape to (total_elements, n_ins)
        # non_vel_projs_flat = non_vel_projs.concat()
        # The total number of rows across all pieces
        non_vel_projs_stacked = tf.reshape(non_vel_projs_flat, [-1, n_ins])
        
        a_proj_with_curr_rel = (
            a_ortho_v, #curr_ortho_mags[4], curr_ortho_mags[5]
        )
        remaining_proj_with_curr_rel = tf.einsum(
            'ijk,jbk->ibj', all_curr_vecs[2:], curr_ortho_mats[:, 1:]
        )

        tri_dots_lower = tf.gather_nd(tri_dots, self._tril_indices)
        ret = tf.transpose(tf.concat(
            (
                tf.cast(tf.fill((1, n_ins), self.step), tf.float32),
                tf.reshape(all_dots_with_prev, [-1, n_ins]),
                tf.reshape(all_proj_with_prev, [-1, n_ins]),
                tf.reshape(vel_mags, (1, n_ins)), non_vel_mags, tri_dots_lower,
                acc_vel_proj, other_projs_on_vel, 
                non_vel_projs_stacked,
                a_proj_with_curr_rel,
                tf.reshape(remaining_proj_with_curr_rel, [-1, n_ins])
            ), axis=0
        ))
        # missing_zeros = tf.zeros((n_ins, 227))
        # ret = tf.concat((ret, missing_zeros), axis=1)
        
        # Apply column permutation to match the reference key order
        ret = tf.gather(ret, self.to_nn_permut, axis=1)            

        ret = (ret - self.scale_means) / self.scale_scales

        # 1 + 8*(7+9) + 8 + 1+2+3+4+5+6+7 + (7*6 + 7) + (8*2 - 3)

        jsc_on_vel = tf.reshape(other_projs_on_vel[:3], [3, 1, -1]) # Jerk, Snap, Crackle are first 3, then comes the rotation stuff.
        jsc_on_aj = remaining_proj_with_curr_rel[:3]
        #(shape (3, 1, n) and (3, 2, n), respectively)

        jsc_on_all = tf.concat((jsc_on_vel, jsc_on_aj), axis=1) # (3, 3, n)
        jsc_on_all_t = tf.transpose(jsc_on_all, [2, 0, 1])

        # Build the 12-component JAV vector plus 3 zeros
        zeros_3 = tf.zeros((n_ins, 3), dtype=tf.float32)
        jav_15 = tf.concat((
            tf.reshape(vel_mags, [n_ins, 1]),    # velocity magnitude
            tf.reshape(a_proj_v, [n_ins, 1]),    # acceleration parallel
            tf.reshape(a_ortho_v, [n_ins, 1]),   # acceleration orthogonal
            tf.reshape(jsc_on_all_t, [n_ins, 9]),
            zeros_3
        ), axis=1)
        
        # TODO: The way I handle non-1 timesteps in my other JAV calculations
        # right now is kinda messy. I basically pre-multiply the acceleration,
        # velocity, etc. by delta T, (delta T)^2, etc. so that the loss function
        # calculation is (slightly) faster. But of course, this doesn't quite
        # work when the timesteps are not constant, so those have to be handled
        # a bit differently. Anyway, this discrepancy makes things ripe for bugs
        # (solving one such is what motivated this comment) so I need to find a
        # better way of handling this so that whoever is using these functions
        # does not make understandable, but wrong, assumptions.
        # ---
        # Because of the above, here I need to do said JAV multiplications.
        if self.step != 1:
            # Multiply the first 12 components by the appropriate power of step
            jav_first_12 = jav_15[:, :12] * self._jav_muls #[tf.newaxis, :]
            jav_15 = tf.concat((jav_first_12, zeros_3), axis=1)

        return ret, jav_15, curr_ortho_mats 
    
    @tf.function
    def call(self, inputs):
        """Keras layer call method for computing features from raw pose sequences.
        
        Args:
            inputs: Tensor of shape (batch_size, n_frames*6) where last dimension
                   is [x, y, z, aa_x, aa_y, aa_z] for each frame.
        
        Returns:
            Tensor of shape (batch_size, n_features) containing computed features.
        """
        input_shape = tf.shape(inputs)
        inputs_rs = tf.reshape(inputs, [input_shape[0], 6, 6])
        inputs_swapax = tf.transpose(inputs_rs, [1, 0, 2])
        # Split inputs into positions and axis-angles
        positions = inputs_swapax[..., :3]    # Shape: (batch_size, n_frames, 3)
        axis_angles = inputs_swapax[..., 3:]  # Shape: (batch_size, n_frames, 3)
        
        # # Process each batch item separately using map_fn
        # def process_single_window(single_input):
        #     pos = single_input[0]   # Shape: (n_frames, 3)
        #     aa = single_input[1]    # Shape: (n_frames, 3)
            
        #     # calculateOutputs returns (features, jav_stack, curr_ortho_mats)
        #     # We only need features for the neural network input
        #     features, _, _ = self.calculateOutputs(pos, aa)
        #     return features
        
        # # Apply to all batch items
        # batch_features = tf.map_fn(
        #     process_single_window,
        #     (positions, axis_angles),
        #     dtype=tf.float32
        # )
        
        return self.calculateOutputs(positions, axis_angles)
    

'''
Some notes/references for myself:

https://www.tensorflow.org/guide/advanced_autodiff
https://www.tensorflow.org/guide/autodiff
https://www.tensorflow.org/guide/intro_to_graphs
https://stackoverflow.com/questions/42848322/what-does-my-choice-of-glfw-samples-actually-do

`Don't rely on Python side effects like object mutation or LIST APPENDS.`
tf.config.run_functions_eagerly(bool)

When tf.function does decide to trace, the tracing stage is immediately followed by the second stage, so calling the tf.function both creates and runs the tf.Graph.
    Later you will see how you can run only the tracing stage with get_concrete_function.

You can use pretty_printed_concrete_signatures() to see all of the available traces:
> print(double.pretty_printed_concrete_signatures())

https://www.tensorflow.org/guide/function#obtaining_concrete_functions
https://www.tensorflow.org/guide/saved_model
https://www.tensorflow.org/api_docs/python/tf/print

https://stackoverflow.com/questions/73267868/how-to-do-slice-and-update-operation-on-tensors-in-tensorflow-2-0
https://stackoverflow.com/questions/62092147/how-to-efficiently-assign-to-a-slice-of-a-tensor-in-tensorflow
https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update


https://stackoverflow.com/questions/56616485/equality-comparison-does-not-work-inside-tensorflow-2-0-tf-function



'''
