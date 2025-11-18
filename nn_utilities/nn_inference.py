import typing

import numpy as np
from numpy.typing import NDArray

import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation
import keras

from motiontools.posefeatures import (
    HypotheticalInputsForNN, getWorldFrameDisplacements
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

def tfNormalizeAll(vecs: tf.Tensor):
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
    unit_vecs2 = tf.linalg.cross(unit_vecs0, unit_vecs1)
    
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


class LayerPostOutJAV12(keras.layers.Layer):
    def __init__(self, step: int, **kwargs):
        super(LayerPostOutJAV12, self).__init__(**kwargs)

        self.step = step
        
        self._jav_muls = tf.Variable(tf.zeros((4, 3), dtype=tf.int32))
        # Jerk through crackle each have 3 components we must multiply by a
        # respective power of the step assuming it's also the next "delta T".
        self._jav_muls[1:].assign(self.step ** tf.reshape(tf.range(3, 6), (3, 1)))
        # Then velocity and acceleration are special because they only have 1
        # and 2 multipliers, respectively.
        self._jav_muls[0, :1].assign([self.step])
        self._jav_muls[0, 1:3].assign(self.step ** 2)
        self._jav_muls = tf.reshape(self._jav_muls, [-1])


    def call(self, inputs):
        
        return transformed_input



class DerivativeCollectionConstTimeTF:
    def __init__(self, displacements: tf.Tensor, max_derivative_order: int,
                 time_step: int):
                        
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
                
        self.velocities: tf.Tensor = tf.identity(displacements)
        
        # Reshape for broadcasting.
        coeff_shape = self.velocities.shape[:-1] + (1, )
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

class PointsToInputsConstStep(tf.keras.layers.Layer):
    def __init__(self, step: typing.Union[int, float],
                 zero_angle_thresh: float = DEFAULT_ZERO_ANG_THRESH, **kwargs):
        super(PointsToInputsConstStep, self).__init__(**kwargs)

        self.step = step
        self.zero_angle_thresh = zero_angle_thresh

        # 8: vecs, accs, jerks, snaps, crackles, rot vel, rot acc, rot jerk
        self._n_vec_kinds = 8
        self._n_other_vec_kinds = self._n_vec_kinds - 1



        # Pre-compute lower triangle indices in __init__ for efficiency
        # Create index arrays using NumPy, then convert to TensorFlow constants
        np_tril_rows, np_tril_cols = np.tril_indices(7)
        
        # Stack them together: shape (num_tril_elements, 2)
        tl_inds = np.column_stack([np_tril_rows, np_tril_cols]).astype(np.int32)
        
        # Store as a TensorFlow constant
        self._tril_indices = tf.constant(tl_inds)


        # self._rel_ax_order = tuple(
        #     k for k in ALL_RELATIVE_VECTORS if k != MOTION_DATA.VEL_DEG2_VEC3
        # )



    def updatePrecalcs(self, x0_through_5: tf.Tensor, aa0_through_5: tf.Tensor,
                       last_nonzero_unit_vels: tf.Tensor):

        all_pds = DerivativeCollectionConstTimeTF(
            keras.ops.diff(x0_through_5, 1, axis=0), 5, self.step
        )
        
        angvel_aas = compute_relative_rotations(
            aa0_through_5[:-1], aa0_through_5[1:]
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


        prev_ang_vel = all_rds.velocities[-2]
        prev_ang_acc = all_rds.accelerations[-2]
        prev_ang_jerk = all_rds.jerks[-2]

        new_ang_vel = all_rds.velocities[-1]
        new_ang_acc = all_rds.accelerations[-1]
        new_ang_jerk = all_rds.jerks[-1]

        prev_ortho_mags, prev_ortho_dirs = tfOrthonormalFramesFromUnitVec0s(
            True, last_nonzero_unit_vels, prev_acc, self.zero_angle_thresh
        )

        a0_is_0 = tf.equal(prev_ortho_mags[1], 0.0)
        if tf.reduce_any(a0_is_0):
            _, j_orth = tfParallelAndOrthoParts(
                prev_jerk[a0_is_0], prev_ortho_dirs[0][a0_is_0], True
            )
            prev_ortho_dirs[..., 2, :][a0_is_0] = tfNormalizeAll(
                j_orth[a0_is_0]
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
        num_scale_types = len(self._rel_ax_order) - 2

        inner_prev_vecs_shape = prev_relative_vecs.shape[1:-1]
        prev_scale_shape = (num_scale_types, ) + inner_prev_vecs_shape #+ (1, )

        prev_relative_scales = tf.zeros(prev_scale_shape)
        prev_relative_scales[0] = prev_vel_mags[-1] #[..., np.newaxis]
        prev_relative_scales[1] = tf.sqrt(
            prev_ortho_mags[0]**2 + prev_ortho_mags[1]**2
        )#[..., np.newaxis]
        prev_relative_scales[2:] = tf.norm(
            prev_relative_vecs[2:-2], axis=-1#, keepdims=True
        )

        # Safely obtain unit velocities, 
        unit_vels = tf.math.divide_no_nan(vels, vel_mags[-1])
        vel_mag_is_0 = tf.reshape(vel_mags[-1], [-1]) < self.zero_angle_thresh
        unit_vels = tf.where1(
            vel_mag_is_0, unit_vels, last_nonzero_unit_vels
        )
        # unit_vels[vel_mag_is_0] = last_nonzero_unit_vels[vel_mag_is_0]

        all_curr_vecs = tf.stack(
            (
                vels, accs, jerks, snaps, crackles,
                new_ang_vel, new_ang_acc, new_ang_jerk
            ), axis=0
        )

        tf.debugging.assert_equal(
            self._n_vec_kinds, len(all_curr_vecs),
            "Hardcoded value for self._n_vec_kinds no longer correct!"
        )

        all_dots_with_prev = tf.einsum(
            'aik,bik->abi', all_curr_vecs, prev_relative_vecs
        )

        all_proj_with_prev = tf.math.divide_no_nan(
            all_dots_with_prev[:, :-2], prev_relative_scales 
        )

        
        (a_proj_v, a_ortho_v), curr_ortho_mats = tfOrthonormalFramesFromUnitVec0s(
            True, unit_vels, accs, vecs0_are_unit_len=True,
            set_zeros_to_zero=True
        )

        a0_is_0 = tf.where1(a_ortho_v == 0.0)[0]
        _, j_orth = tfParallelAndOrthoParts(
            jerks[a0_is_0], curr_ortho_mats[a0_is_0, 0], True
        )
        curr_ortho_mats[a0_is_0, 2] = tfNormalizeAll(j_orth)
        # print("hyp curr:", curr_ortho_mats)

        # For the various dot products between *current* frame values, we'll
        # keep track of them in a list of tensors so that we don't duplicate
        # dot products.
        # We have 8 non-unit vec3s (vel..crackle and rot vel..jerk), but we'll 
        # handle the unit-length orthogonal relative axes separately, so our
        # dimensions will be 7x7x...
        n_ins = x0_through_5.shape[1]
        
        tri_dots = tf.zeros((
            self._n_other_vec_kinds, self._n_other_vec_kinds, n_ins
        ))

        # Rather than keep the self-dots in the "diagonal", I think it may be
        # slightly more efficient to keep them separate; we'll need to divide
        # by them later, so this prevents the need for diag indexing or copies.
        non_vel_mags = tf.zeros((self._n_other_vec_kinds, n_ins))
        non_vel_mags[0] = tf.norm(accs_2D, axis=-1) # sqrt(a*a)
        non_vel_mags[1:] = tf.norm(all_curr_vecs[2:], axis=-1)
        # We'll now copy in the magnitudes calculated during the orthonormal
        # frame calculations.
        # First, the dots with velocity:
        tri_dots[0, 0] = tf.reshape(vel_mags, [-1]) * a_proj_v # v*a
        # tri_dots[1, 0] = v_mags * curr_ortho_mags[3] # v*j


        # Then, the dots with acceleration:
        accs_2D = tf.stack((a_proj_v, a_ortho_v), axis=-1)
        # tri_dots[1, 1] = pm.einsumDot(accs_2D, np.transpose(curr_ortho_mags[3:5])) # a*j (2D)
        # The remaining dots have no precalculations to take advantage of:
        for i in range(2, self._n_vec_kinds):
            tri_dots[i-1, :i] = tf.einsum(
                'jk,ijk->ij', all_curr_vecs[i], all_curr_vecs[:i]
            )

        # For the projections, we don't worry about "duplicates" since there are
        # none: projecting jerk onto velocity is different from vice versa.
        # However, we have the diagonal separated out already, so we can
        # exclude that.
        acc_vel_proj = tf.identity(a_proj_v)
        acc_vel_proj = tf.where1(
            vel_mag_is_0, unit_vels, tf.zeros_like(acc_vel_proj)
        )
        acc_vel_proj = acc_vel_proj[tf.newaxis, ...]
        other_projs_on_vel = tf.math.divide_no_nan(tri_dots[1:, 0], vel_mags)


        CIND = 4 # Crackle's index
        # We don't divide by crackle's mag because it's not used as a relative
        # axis.
        non_crackles = [i for i in range(1, self._n_vec_kinds) if i != CIND]
        non_vel_projs = []
        for i in non_crackles:
            prev_i = i - 1
            mag_i = non_vel_mags[prev_i]
            non_vel_projs.append(tf.math.divide_no_nan(
                tri_dots[prev_i, :i], mag_i
            ))
            if i < self._n_other_vec_kinds:
                non_vel_projs.append(tf.math.divide_no_nan(
                    tri_dots[i:, i], mag_i
                ))
        
        a_proj_with_curr_rel = (
            a_ortho_v, #curr_ortho_mags[4], curr_ortho_mags[5]
        )
        remaining_proj_with_curr_rel = tf.einsum(
            'ijk,jbk->ibj', all_curr_vecs[2:], curr_ortho_mats[:, 1:]
        )


        other_frame_proj_zip = zip(
            other_projs_on_vel[:3], remaining_proj_with_curr_rel[:3]
        )
        other_frame_proj_tups = [(a, b, c) for a, (b, c) in other_frame_proj_zip]
        other_frame_projs = [a for tup in other_frame_proj_tups for a in tup]

        tri_dots_lower = tf.gather_nd(tri_dots, self._tril_indices)
        ret = tf.transpose(tf.concat(
            (
                tf.fill((1, n_ins), self.step),
                all_dots_with_prev.reshape(-1, n_ins),
                all_proj_with_prev.reshape(-1, n_ins),
                tf.reshape(vel_mags, (1, n_ins)), non_vel_mags, tri_dots_lower,
                acc_vel_proj, other_projs_on_vel, *non_vel_projs,
                a_proj_with_curr_rel,
                remaining_proj_with_curr_rel.reshape(-1, n_ins)
            ), axis=0
        ))
        # 1 + 8*(7+9) + 8 + 1+2+3+4+5+6+7 + (7*6 + 7) + (8*2 - 3)


        # Now, we'll calculate the JAV values that get used by the loss func.
        # These are NOT required in the NN's initial input layer!
        zeros_3d = tf.zeros((3, n_ins))
        jav_tup = (
            tf.reshape(vel_mags, [-1]), a_proj_v, a_ortho_v, *other_frame_projs,
            *zeros_3d
        )
        jav_stack = tf.stack(jav_tup, axis=-1)

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
            jav_stack[..., :12] *= self._jav_muls

        return ret[:, self.to_nn_permut], jav_stack, curr_ortho_mats


'''
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

'''
