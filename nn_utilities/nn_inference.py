# pyright: reportOperatorIssue=false
# pyright: reportIndexIssue=false
import typing

import numpy as np
from numpy.typing import NDArray

import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation
import keras

from posemath import DEFAULT_ZERO_ANG_THRESH

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

def ang_vel_displace(axis_angles_start: tf.Tensor, axis_angles_disp: tf.Tensor):
    """
    Computes the modification of an axis angle orientation by an axis angle
    displacement rotation.

    Args:
        axis_angles_start: A tensor of shape (n, 3) representing the starting
                           orientation axis angles.
        axis_angles_disp:  A tensor of shape (n, 3) representing the
                           displacement rotation.

    Returns:
        A tensor of shape (n, 3) representing the new axis-angle orientations.
    """

    angs_start = tf.norm(axis_angles_start, axis=-1, keepdims=True)
    angs_disp = tf.norm(axis_angles_disp, axis=-1, keepdims=True)

    units_start = tf.math.divide_no_nan(axis_angles_start, angs_start)
    units_disp = tf.math.divide_no_nan(axis_angles_disp, angs_disp)

    # Convert axis angles to quaternions
    qs_start = tfg_transformation.quaternion.from_axis_angle(units_start, angs_start)
    qs_disp = tfg_transformation.quaternion.from_axis_angle(units_disp, angs_disp)

    qs_final = tfg_transformation.quaternion.multiply(
        qs_disp, qs_start
    )

    # Convert the new quaternion back to axis angles
    axis_angles_final = tfg_transformation.axis_angle.from_quaternion(
        qs_final
    )

    # Convert from (axis, angles) tuple to scaled vec3 axis-angles.
    return axis_angles_final[0] * axis_angles_final[1]

def ang_vel_extrapolate(axis_angles1: tf.Tensor, axis_angles2: tf.Tensor):
    """
    Extrapolates the relative axis-angle rotations between two sets of axis-angle
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
                
        
        # Reshape for broadcasting.
        coeff_shape = displacements.shape[:-1] + tf.TensorShape(1)
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

        # time_div = self.unflat_timestamps[deriv_power:] - self.unflat_timestamps[:-deriv_power]
        end = tf.shape(self.unflat_timestamps)[0]
        next_times = self.unflat_timestamps[deriv_power:end]
        prev_times = self.unflat_timestamps[0:(end - deriv_power)]
        time_div = next_times - prev_times

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
        self._diag_stride = self._n_vec_kinds + 1
        self._n_vec_kinds_sq = self._n_vec_kinds * self._n_vec_kinds
        
        self.to_nn_permut = tf.constant(to_nn_permut, dtype=tf.int32)


        # Pre-compute lower triangle indices in __init__ for efficiency
        # Create index arrays using NumPy, then convert to TensorFlow constants
        np_tril_rows, np_tril_cols = np.tril_indices(7)
        
        # Stack them together: shape (num_tril_elements, 2)
        tl_inds = np.column_stack([np_tril_rows, np_tril_cols]).astype(np.int32)
        
        # Store as a TensorFlow constant
        self._tril_indices = tf.constant(tl_inds)


        self._jav_muls = _get_JAV_muls(step)

        if self._n_vec_kinds != 8:
            raise Exception(
                "Hardcoded 'loop' iterations and whatnot assume "
                "self.n_vec_kinds == 8, but this is no longer true!"
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
    
    @tf.function
    def calculateOutputs(self, x0_through_5: tf.Tensor, aa0_through_5: tf.Tensor):

        # n_ins = tf.shape(x0_through_5)[1]

        # test_concat = tf.concat((x0_through_5[0], aa0_through_5[0]), axis=-1)
        # num_remaining_zeros = 227 - tf.shape(test_concat)[1]
        # remaining_zeros = tf.zeros((n_ins, num_remaining_zeros))

        # ret0 = tf.concat((test_concat, remaining_zeros), axis=-1)
        # ret1 = safe_slice_ax1(ret0, 0, 15)
        # ret2 = tf.reshape(safe_slice_ax1(ret0, 0, 9), [n_ins, 3, 3])
        # return ret0, ret1, ret2

        scaled_disps = keras.ops.diff(x0_through_5, 1, axis=0)
        all_pds = DerivativeCollectionConstTimeTF(
            typing.cast(tf.Tensor, scaled_disps), 5, self.step
        )
        
        scaled_aas = aa0_through_5 #/ self.rot_scale
        angvel_aas = compute_relative_rotations(
            scaled_aas[2:-1], scaled_aas[3:]
        )

        all_rds = DerivativeCollectionConstTimeTF(angvel_aas, 3, self.step)
        
        # Calculate velocities, speeds, and indices where speed is 0.
        vels = all_pds.velocities[-1]
        accs = all_pds.accelerations[-1]
        jerks = all_pds.jerks[-1]
        snaps = all_pds.snaps[-1]
        crackles = all_pds.crackles[-1]

        new_ang_vel = all_rds.velocities[-1]
        new_ang_acc = all_rds.accelerations[-1]
        new_ang_jerk = all_rds.jerks[-1]

        all_curr_vecs = tf.stack(
            (
                vels, accs, jerks, snaps, crackles,
                new_ang_vel, new_ang_acc, new_ang_jerk
            ), axis=1
        )
        all_dots = tf.linalg.matmul(
            all_curr_vecs, all_curr_vecs, transpose_b = True
        )
        # NOTE: Need to use tf.shape(...) instead of x.shape in graph mode!
        n_ins = tf.shape(x0_through_5)[1]
        _flatter_dots = tf.reshape(all_dots, [n_ins, self._n_vec_kinds_sq])
        self_dots = _flatter_dots[:, ::self._diag_stride]
        all_mags = tf.reshape(tf.sqrt(self_dots), [n_ins, self._n_vec_kinds, 1])

        vel_mags = all_mags[:, 0]
        unit_vels = tf.math.divide_no_nan(vels, vel_mags)

        tf.debugging.assert_equal(
            self._n_vec_kinds, all_curr_vecs.shape[1],
            "Hardcoded value for self._n_vec_kinds no longer correct!"
        )

        
        (a_proj_v, a_ortho_v), curr_ortho_mats = tfOrthonormalFramesFromUnitVec0s(
            True, unit_vels, accs, self.zero_angle_thresh 
        )
        # a0_is_0 = tf.equal(a_ortho_v, 0.0)
        
        # curr_ortho_mats = tf.cond(
        #     tf.reduce_any(a0_is_0),
        #     lambda: update_ortho_dirs_with_jerk(curr_ortho_mats, jerks, a0_is_0),
        #     lambda: curr_ortho_mats
        # )

        # all_dots: (n, 8, 8), all_mags: (n, 8, 1)
        all_proj = tf.math.divide_no_nan(all_dots, all_mags)
        all_unit_dots = tf.math.divide_no_nan(
            all_proj, tf.reshape(all_mags, [n_ins, 1, -1])
        )

        # accs_2D = tf.stack((a_proj_v, a_ortho_v), axis=-1)
        # acc_mag = tf.norm(accs_2D, axis=-1, keepdims=True)  # Shape: (n_ins, 1)
        
        remaining_proj_with_curr_rel = tf.linalg.matmul(
            all_curr_vecs[:, 2:], curr_ortho_mats[:, 1:], transpose_b=True
        ) # shape: n_ins, 6, 2

        def unitDotToAng(tensorVal):
            return tf.acos(tf.clip_by_value(tensorVal, -1.0, 1.0))

        # tri_dots_lower = tf.gather_nd(tri_dots, self._tril_indices)

        all_angs = unitDotToAng(all_unit_dots)
        angs_and_projs_to_concat = (
            all_angs[:, 1, :1], all_angs[:, 2, :2], all_angs[:, 3, :3],
            all_angs[:, 4, :4], all_angs[:, 5, :5], all_angs[:, 6, :6],
            all_angs[:, 7, :7],
            all_proj[:, 0, 1:],
            all_proj[:, 1, :1], all_proj[:, 1, 2:],
            all_proj[:, 2, :2], all_proj[:, 2, 3:],
            all_proj[:, 3, :3], all_proj[:, 3, 4:],
            all_proj[:, 4, :4], all_proj[:, 4, 5:],
            all_proj[:, 5, :5], all_proj[:, 5, 6:],
            all_proj[:, 6, :6], all_proj[:, 6, 7:],
            all_proj[:, 7, :7],
        )

        a_ang_with_curr_rel = \
            unitDotToAng(tf.math.divide_no_nan(a_ortho_v, all_mags[:, 1, 0]))

        remaining_ang_with_curr_rel = unitDotToAng(
            tf.math.divide_no_nan(
                remaining_proj_with_curr_rel, all_mags[:, 2:]
            )
        )
        ret_tup = ((
            tf.cast(tf.fill((n_ins, 1), self.step), tf.float32),
            tf.reshape(all_mags, [n_ins, -1]), 
        ) + angs_and_projs_to_concat + 
        (
            tf.reshape(a_ortho_v, [n_ins, 1]),
            tf.reshape(remaining_proj_with_curr_rel, [n_ins, -1]),
            tf.reshape(a_ang_with_curr_rel, [n_ins, 1]),
            tf.reshape(remaining_ang_with_curr_rel, [n_ins, -1])
        ))
        ret = tf.concat(
            ret_tup, axis=1
        )
        
        # Apply column permutation to match the reference key order
        ret = tf.gather(ret, self.to_nn_permut, axis=1) # Should permute elsewhere!!!            

        ret = (ret - self.scale_means) / self.scale_scales

        # 1 + 8*(7+9) + 8 + 1+2+3+4+5+6+7 + (7*6 + 7) + (8*2 - 3)

        jsc_on_vel = tf.reshape(all_proj[:, 0, 2:5], [-1, 3]) # Jerk, Snap, Crackle are first 3, then comes the rotation stuff.
        jsc_on_aj = remaining_proj_with_curr_rel[:, :3]
        #(shape (n, 3) and (n, 3, 2), respectively)

        # Build the 12-component JAV vector plus 3 zeros
        zeros_3 = tf.zeros((n_ins, 3), dtype=tf.float32)
        jav_15 = tf.concat((
            vel_mags,    # velocity magnitude
            tf.reshape(a_proj_v, [n_ins, 1]),    # acceleration parallel
            tf.reshape(a_ortho_v, [n_ins, 1]),   # acceleration orthogonal
            jsc_on_vel[:, :1], jsc_on_aj[:, 0],
            jsc_on_vel[:, 1:2], jsc_on_aj[:, 1],
            jsc_on_vel[:, 2:3], jsc_on_aj[:, 2],
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
        positions = inputs_swapax[:, :, 0:3]    # Shape: (batch_size, n_frames, 3)
        axis_angles =  inputs_swapax[:, :, 3:6]  # Shape: (batch_size, n_frames, 3)
        
        return self.calculateOutputs(positions, axis_angles)
