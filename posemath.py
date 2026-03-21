import typing
from collections import namedtuple

import numpy as np
from numpy.typing import NDArray, ArrayLike

HALF_PI_NP = np.pi/2.0
DEFAULT_ZERO_ANG_THRESH = 0.0001

PlaneInfoType = namedtuple(
    "PlaneInfo", ['plane_axes', 'normals', 'offset_dists']
)

def replaceAtEnd(to_replace: np.ndarray, replace_with: np.ndarray, len_diff_check = None):
    ret_arr = np.empty(to_replace.shape)
    len_diff = len(to_replace) - len(replace_with)
    if len_diff_check is not None and len_diff != len_diff_check:
        e_str = "Expected difference in array len ({}) was instead {}.".format(
            len_diff_check, len_diff
        )
        raise Exception(e_str)
    ret_arr[:len_diff] = to_replace[:len_diff]
    ret_arr[len_diff:] = replace_with
    return ret_arr

def normalizeAll(vecs: np.ndarray):
    ret_val = vecs / np.linalg.norm(vecs, axis = -1, keepdims=True)
    return typing.cast(NDArray, ret_val)


def conjugateQuats(quats: np.ndarray):
    conjugate_quats = np.empty(quats.shape)
    conjugate_quats[..., 0] = quats[..., 0]
    conjugate_quats[..., 1:] = -quats[..., 1:]
    return conjugate_quats

def rotateVecsByQuats(quats: np.ndarray, vecs: np.ndarray):
    v_quats = np.insert(vecs, 0, np.zeros(vecs.shape[:-1]), axis = -1)
    conjs = conjugateQuats(quats)
    return multiplyQuatLists(quats, multiplyQuatLists(v_quats, conjs))[..., 1:]

def reflectVecsOverLines(vecs: np.ndarray, line_dirs: np.ndarray, dirs_unit_len: bool):
    proj_scalars = einsumDot(vecs, line_dirs)
    if not dirs_unit_len:
        proj_scalars /= einsumDot(line_dirs, line_dirs)
    proj = scalarsVecsMul(proj_scalars, line_dirs)
    return (proj + proj) - vecs

def loneQuatFromAxisAngle(unit_axis, angle):
    half_ang = angle / 2
    return np.array([np.cos(half_ang), np.sin(half_ang) * unit_axis])

def quatsFromAxisAngles(unit_axes, angles):
    angles_reshaped = angles
    if angles.ndim != unit_axes.ndim:
        angles_reshaped = angles[..., np.newaxis]
    half_angs = angles_reshaped / 2

    quaternions = np.hstack((np.cos(half_angs), np.sin(half_angs) * unit_axes))
    return quaternions

def _defaultReplacementVec(length):
    ret = np.zeros(length)
    ret[0] = 1.0
    return ret

def _back_prop(arr: NDArray, count: int):
    if count >= len(arr):
        raise Exception("All vecs fail condition; propagation impossible!")
    arr[:count] = arr[count]

def _single_replacer(edited_arr: NDArray, _: int, replacement: NDArray):
    edited_arr[0] = replacement[0]

def safelyNormalizeArray(array: np.ndarray,
                         norms: typing.Optional[NDArray] = None,
                         vec_for_zero_norms: typing.Optional[NDArray] = None,
                         propagate_last_nonzero_vec: bool = True,
                         propagate_back_if_first_vecs_zero: bool = False,
                         zero_norm_inds = None, propagation_axis: int = -2):
    '''
    Normalize an array while handling the case where some elements have a norm 
    of zero, which would cause division errors.

    Parameters:
        array (np.ndarray): Array to normalize.
        norms (np.ndarray): Norms, if already calculated (else None).
        vec_for_zero_norms (np.ndarray): Default vec to replace 0 vecs.
        propagate_last_nonzero_vec (bool): Self-descriptive.
        propagate_back_if_verst_vecs_zero (bool): Self-descriptive.
        zero_norm_inds: Boolean indices for which vectors have norm of 0.

    Returns:
        np.ndarray: The normalized result.

    '''
    
    if norms is None:
        norms = typing.cast(
            NDArray, np.linalg.norm(array, axis=-1, keepdims=True)
        )

    # Validate vec_for_zero_norms if provided
    if vec_for_zero_norms is not None:
        vec_norm = np.linalg.norm(vec_for_zero_norms)
        if not (np.isclose(vec_norm, 0.0) or np.isclose(vec_norm, 1.0)):
            raise ValueError(
                f"vec_for_zero_norms must be either a zero vector or a unit vector, "
                f"but has norm {vec_norm}"
            )

    if zero_norm_inds is None:
        zero_norm_inds = (norms == 0)
    zero_norm_inds = zero_norm_inds[..., 0]

    if not np.any(zero_norm_inds):
        return array / norms

    pos_prop_ax = propagation_axis % array.ndim
    
    pos_norm_inds = ~np.asarray(zero_norm_inds)
    normed = np.empty_like(array)
    normed[pos_norm_inds] = array[pos_norm_inds]/norms[pos_norm_inds]
    if propagate_back_if_first_vecs_zero and not propagate_last_nonzero_vec:
        # TODO-reusability: There's really no reason *not* to support this other
        # than the fact that I don't have time for a refactor right now.
        raise NotImplementedError(
            "Propagating back but not forwards not currently supported!"
        )
    # For zero axes, we can either use a supplied default vector, create our
    # own default, or propagate the last nonzero vector.
    if propagate_last_nonzero_vec and array.ndim > 1:
        if propagation_axis == -1:
            raise NotImplementedError((
                "Propagation axis of -1 not currently supported! "
                "Implementation assumes -1 is the per-vector axis!"
            ))
        use_single_replacement = True
        # First, we make sure that if the first vec is zero, that we replace it
        # with some default, since there'd be no previous vec to copy.
        if vec_for_zero_norms is None:
            if propagate_back_if_first_vecs_zero:
                handleCondsAtStart(
                    zero_norm_inds, normed, _back_prop, pos_prop_ax
                )
                use_single_replacement = False
            else:
                vec_for_zero_norms = _defaultReplacementVec(array.shape[-1])
        if use_single_replacement:
            if vec_for_zero_norms.ndim != 1:
                raise NotImplementedError(
                    "Only shape (k,) supported for vec_for_zero_norms so far!"
                )
            broadcast_replacer = np.broadcast_to(
                vec_for_zero_norms, zero_norm_inds.shape + (normed.shape[-1], )
            )
            handleCondsAtStart(
                zero_norm_inds, normed, _single_replacer, pos_prop_ax,
                False, True, replacement=broadcast_replacer
            )
        if propagation_axis != -2:            
            normed = normed.swapaxes(-2, pos_prop_ax)
            zero_norm_inds = zero_norm_inds.swapaxes(-1, pos_prop_ax)
        
        # To copy the last nonzero vectors, we'll use the technique proposed in 
        # a 2015-05-27 StackOverflow answer by user "jme" (1231929/jme) to a
        # 2015-05-27 question, "Fill zero values of 1d numpy array with last
        # non-zero values" (https://stackoverflow.com/q/30488961) by user "mgab"
        # (3406913/mgab). A 2016-12-16 edit to "Most efficient way to 
        # forward-fill NaN values in numpy array" by user Xukrao
        # (7306999/xukrao) shows this to be more efficient than similar
        # for-loop, numba, pandas, etc. solutions.

        seq_parent_shape = zero_norm_inds.shape[:-1]
        replacement_inds = np.tile(np.arange(len(norms)), seq_parent_shape + (1, ))
        int_zero_inds = np.nonzero(zero_norm_inds)
        replacement_inds[int_zero_inds] = 0
        replacement_inds = np.maximum.accumulate(replacement_inds, axis = -1)
        
        zn_where = np.argwhere(zero_norm_inds).transpose()
        zn_where[-1] = replacement_inds[int_zero_inds]

        normed[int_zero_inds] = normed[tuple(zn_where)]

        # We also swap axes back the way they were if necessary
        if propagation_axis != -1:
            zero_norm_inds = zero_norm_inds.swapaxes(-1, pos_prop_ax)
            normed = normed.swapaxes(-2, pos_prop_ax)
    else:
        if vec_for_zero_norms is None:
            replacement_vec = _defaultReplacementVec(array.shape[-1])
            normed[zero_norm_inds] = replacement_vec
        else:
            normed[zero_norm_inds] = vec_for_zero_norms
    return normed

def quatsFromAxisAngleVec3s(axisAngleVals):
    angles = np.linalg.norm(axisAngleVals, axis=1, keepdims=True)
    
    # Because we're returning quaternions in the next step, if the angle is 0,
    # the axis will get multiplied by zero and (0, 0, 0) will be the last values
    # of the respective quaternion, which is correct. Therefore, for
    # normalization, we can just set these axes to zero already.
    normed = safelyNormalizeArray(
        axisAngleVals, angles, vec_for_zero_norms=np.zeros(3),
        propagate_last_nonzero_vec=False
    )

    return quatsFromAxisAngles(normed, angles)

def einsumDot(vecs0: np.ndarray, vecs1: np.ndarray) -> np.ndarray:
    '''
    Here, np.einsum is used to dot-product together each vector in one array
    of vectors with the vector in the second array at the same index.
    '''
    # Dot products occur along 'j' axis; preserves existence of the 'i' axis.
    # For understanding einsum, this ref might be handy:
    # https://ajcr.net/Basic-guide-to-einsum/
    # Here's a StackOverflow post suggesting that einsum might be faster than
    # doing `(arr[1:] * arr[:-1]).sum(axis=1)`:
    # https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
    # (see answer with plots further down page)

    return np.einsum('...j,...j->...', vecs0, vecs1)

def einsumMatVecMul(mats: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    '''
    Multiply each vector in an array of vectors by the corresponding matrix
    from an array of matrices using np.einsum.
    '''
    return np.einsum('...ij,...j->...i', mats, vecs)

def einsumMatMatMul(mats0: np.ndarray, mats1: np.ndarray) -> np.ndarray:
    '''
    Multiply each matrix in an array of matrices with the corresponding matrix
    from a second such array using np.einsum.
    '''
    return np.einsum("...ij,...jk->...ik", mats0, mats1)

def scalarsVecsMul(scalars: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    '''
    Multiply each vector in an array of vectors by the corresponding scalar
    from an array of scalars using np.einsum.
    '''
    return np.einsum('...,...i->...i', scalars, vecs)

def scalarsMatsMul(scalars, mats) -> np.ndarray:
    return np.einsum('...,...ij->...ij', scalars, mats)

def parallelAndOrthoParts(vectors, dirs, dirs_already_normalized = False):
    dots = einsumDot(vectors, dirs)

    if not dirs_already_normalized:
        dots /= einsumDot(dirs, dirs)
    
    parallels = scalarsVecsMul(dots, dirs)
    orthos = vectors - parallels
    return (parallels, orthos)

def getOrthonormalFrames(returned_mats_are_world2vecs: bool, vecs0: np.ndarray,
                         vecs1: typing.Optional[np.ndarray] = None,
                         vecs2: typing.Optional[np.ndarray] = None, 
                         vecs0_are_unit_len: bool = False,
                         zero_thresh = DEFAULT_ZERO_ANG_THRESH,
                         set_zeros_to_zero: bool = False):
    vecs1_na = (vecs1 is None)
    vecs2_na = (vecs2 is None)

    assert vecs2_na or (not vecs1_na), "Shouldn't specify vecs2 but not vecs1!"

    mags0 = \
        np.asarray(1.0) if vecs0_are_unit_len \
        else np.linalg.norm(vecs0, axis=-1)
    
    ret_mags = (mags0,)
    
    unit_vecs0 = vecs0 if vecs0_are_unit_len else safelyNormalizeArray(
        vecs0, mags0 if vecs0.ndim == 1 else mags0[:, np.newaxis]
    )

    if vecs1_na:
        vecs1 = np.ones_like(vecs0)

    # Find the magnitude of the second vector that is parallel to and
    # orthogonal to the first.
    mags_p1 = einsumDot(vecs1, unit_vecs0) # Parallel magnitude
    vecs_p1 = scalarsVecsMul(mags_p1, unit_vecs0) # Parallel vec3
    vecs_o1 = vecs1 - vecs_p1 # Orthogonal vec3
    mags_o1 = np.linalg.norm(vecs_o1, axis=-1) # Orthogonal magnitude

    # vecs_o1[i] will be zero vectors, and can't be normalized, if any vecs0[i]
    # and vecs1[i] are parallel for some i.
    # First, we need to figure out where that happens.
    v1_is_parallel = np.asarray(mags_o1 < zero_thresh)
    unit_vecs1 = np.empty_like(unit_vecs0)

    v1_not_parallel = ~v1_is_parallel
    mags_o1_div = mags_o1[
        v1_not_parallel, ... if vecs_o1.ndim == 1 else np.newaxis
    ]
    unit_vecs1[v1_not_parallel] = vecs_o1[v1_not_parallel] / mags_o1_div

    if not vecs1_na:
        ret_mags += (mags_p1, mags_o1)

    unit_vecs2: np.ndarray
    v2_is_parallel: np.ndarray
    v2_not_parallel: np.ndarray
    if vecs2_na:
        unit_vecs2 = np.cross(unit_vecs0, unit_vecs1)
        v2_is_parallel = v1_is_parallel.copy()
        v2_not_parallel = v1_not_parallel.copy()
    else:
        mags_p20 = einsumDot(vecs2, unit_vecs0) # Parallel magnitude
        vecs_p20 = scalarsVecsMul(mags_p20, unit_vecs0)
        mags_p21 = einsumDot(vecs2, unit_vecs1)
        vecs_p21 = scalarsVecsMul(mags_p21, unit_vecs1)
        vecs_o2 = vecs2 - (vecs_p20 + vecs_p21)
        mags_o2 = np.linalg.norm(vecs_o2, axis=-1)

        unit_vecs2 = np.empty_like(unit_vecs0)

        v2_is_parallel = np.asarray(mags_o2 < zero_thresh)

        v2_not_parallel = ~v2_is_parallel
        mags_o2_div = mags_o2[
            v2_not_parallel, ... if vecs_o2.ndim == 1 else np.newaxis
        ]
        unit_vecs2[v2_not_parallel] = vecs_o2[v2_not_parallel] / mags_o2_div
            

        ret_mags += (mags_p20, mags_p21, mags_o2)
    

    if set_zeros_to_zero:
        unit_vecs1[v1_is_parallel] = 0.0
        unit_vecs2[v2_is_parallel] = 0.0
    else:
        # If we have access to non-parallel vec2s, we can use those to set the
        # parallel vec1s and vice versa.
        v1_p_v2_n = v1_is_parallel & v2_not_parallel
        unit_vecs1[v1_p_v2_n] = np.cross(
            unit_vecs2[v1_p_v2_n], unit_vecs0[v1_p_v2_n]
        )
        v2_p_v1_n = v2_is_parallel & v1_not_parallel
        unit_vecs2[v2_p_v1_n] = np.cross(
            unit_vecs0[v2_p_v1_n], unit_vecs1[v2_p_v1_n]
        )

        # For cases where *both* are parallel, we'll use a Householder transform
        # to craft frame.
        # TODO: Have option to propagate previous frames via RMF. 
        v1v2_is_parallel = v1_is_parallel & v2_is_parallel

        # Find out, further, when we vecs0 are parallel to [1, 0, 0].
        v0_non_x_norms = np.linalg.norm(
            unit_vecs0[v1v2_is_parallel][..., 1:], axis=-1
        )
        v0_x_parallel_subs = np.asarray(v0_non_x_norms < zero_thresh)
        v0_is_x_parallel = v1v2_is_parallel.copy()
        v0_not_x_parallel = v1v2_is_parallel.copy()
        v0_is_x_parallel[v1v2_is_parallel] &= v0_x_parallel_subs
        v0_not_x_parallel[v1v2_is_parallel] &= (~v0_x_parallel_subs)

        unit_vecs1[v0_is_x_parallel] = [0.0, 1.0, 0.0]
        unit_vecs2[v0_is_x_parallel] = [0.0, 0.0, 1.0]
        # Replacement of vecs_o1 via Householder transformation:
        unit_vecs0_not_x = unit_vecs0[v0_not_x_parallel]
        refls = np.empty_like(unit_vecs0_not_x)
        refls[..., 1:] = unit_vecs0_not_x[..., 1:]
        refls[..., 0] = unit_vecs0_not_x[..., 0] - 1.0
        refl_scale = 2 / einsumDot(refls, refls)

        unit_vecs1[v0_not_x_parallel] = -scalarsVecsMul(
            refl_scale * refls[..., 2], refls
        )
        unit_vecs1[v0_not_x_parallel, 2] += 1.0

        # Householder version also applied here:
        unit_vecs2[v0_not_x_parallel] = -scalarsVecsMul(
            refl_scale * refls[..., 1], refls
        )
        unit_vecs2[v0_not_x_parallel, 1] += 1.0


    stack_ax = -2 if returned_mats_are_world2vecs else -1

    all_unit_vecs = (unit_vecs0, unit_vecs1, unit_vecs2)
    if not set_zeros_to_zero:
        if not areAxisArraysOrthonormal(all_unit_vecs, loud=True):
            raise Exception((
                "Generated matrices were not orthonormal!"
            ))
    mats = np.stack(all_unit_vecs, axis=stack_ax)

    return ret_mags, mats

   

# def getPlaneAxes(roughAxes0, roughAxes1):
#     nax0 = normalizeAll(roughAxes0)
#     dots01 = einsumDot(nax0, roughAxes1)
#     ax1 = roughAxes1 - scalarsVecsMul(dots01, nax0)
#     ax1_norms = np.linalg.norm(ax1, axis=-1, keepdims=True)
#     # Reusability-TODO: Maybe I want non-nan behaviour for zero norms here.
#     nax1  = ax1 / ax1_norms
#     return (nax0, nax1)

def getPlaneInfo(roughAxes0, roughAxes1, ptsOnPlane):
    _, all_axes = getOrthonormalFrames(True, roughAxes0, roughAxes1)
    norm_axes = (all_axes[..., 0, :], all_axes[..., 1, :])

    normals = all_axes[..., 2, :]
    offset_dists = einsumDot(ptsOnPlane, normals)
    return PlaneInfoType(norm_axes, normals, offset_dists)

def vecsTo2D(vecs3D, nax0, nax1):
    shape = vecs3D.shape[:-1] + (2,) # Replace last shape dim, 3, with 2.
    retVal = np.empty(shape)
    retVal[..., 0] = einsumDot(vecs3D, nax0)
    retVal[..., 1] = einsumDot(vecs3D, nax1)
    return retVal

def vecsTo3D(vecs2D, plane_axes0, plane_axes1, plane_offset_vecs):
    in_plane_disp = vecs2D[:, :1] * plane_axes0 + vecs2D[:, 1:] * plane_axes1
    return in_plane_disp + plane_offset_vecs

def vecsTo3DUsingPlaneInfo(vecs2D, planeInfo: PlaneInfoType):
    axes = planeInfo.plane_axes
    offset_vecs = scalarsVecsMul(planeInfo.offset_dists, planeInfo.normals)
    return vecsTo3D(vecs2D, axes[0], axes[1], offset_vecs)


def areAxisArraysOrthonormal(axisArrays, threshold = 0.0001, loud = False):
    for i in range(len(axisArrays)):
        axis_norms_sq = einsumDot(axisArrays[i], axisArrays[i])
        if np.abs(1.0 - axis_norms_sq).max() > threshold:
            if loud:
                print("Unit length check failed!")
            return False
        for j in range(i + 1, len(axisArrays)):
            dots = einsumDot(axisArrays[i], axisArrays[j])
            if np.abs(dots).max() > threshold:
                if loud:
                    print(f"Orthogonality failed between axes {i} and {j}.")
                return False
    return True

def areVecArraysInSamePlanes(vecArrays, threshold = 0.0001):
    if len(vecArrays) <= 3:
        return True
    
    diffs0 = vecArrays[1] - vecArrays[0]
    diffs1 = vecArrays[2] - vecArrays[1]
    normals = normalizeAll(np.cross(diffs0, diffs1))
    normDots = np.array([einsumDot(va, normals) for va in vecArrays])
    dotDiffs = np.diff(normDots, 1, axis=0)
    return np.abs(dotDiffs).max() <= threshold


def integrateAngularVelocityRK(angular_velocities, starting_poses, order=1):
    """
    Numerically integrates angular velocities into final poses using Runge-Kutta methods.

    Parameters:
        angular_velocities: np.ndarray of shape (n_rigidbodies, n_timesteps, 3)
            Angular velocities for each rigid body (in xyz components).
        starting_poses: np.ndarray of shape (n_rigidbodies, 4)
            Initial quaternions (poses) for each rigid body.
        order: int
            Order of the Runge-Kutta method to use (1, 2, or 4).

    Returns:
        final_poses: np.ndarray of shape (n_rigidbodies, 4)
            Final quaternions representing poses for each rigid body.
    """
    # Note: Function was originally generated by ChatGPT but underwent manual 
    # verification and manual modification, e.g. for improved code quality
    # (e.g., moving calculations common to all if-statement branches out up front).
    n_rigidbodies = angular_velocities.shape[0]
    n_timesteps = angular_velocities.shape[1] - 1
    if order > 1:
        n_timesteps = n_timesteps // 2

    dt = 1.0 / n_timesteps  # Assume one unit of time passes in total

    # Start with the provided initial poses
    current_poses = starting_poses.copy()
    quat_list_shape = (n_rigidbodies, 4)

    for t in range(n_timesteps):
        w = angular_velocities[:, t]  # Angular velocities for this timestep (shape: n_rigidbodies, 3)
        w_mid_quat = None
        w_end_quat = None

        if order > 1:
            t_ind = t << 1
            w = angular_velocities[:, t_ind]
            t_ind += 1
            w_mid_quat = np.empty(quat_list_shape)
            w_mid_quat[:, 0] = 0.0
            w_mid_quat[:, 1:] = angular_velocities[:, t_ind]
            if order > 2:
                t_ind += 1
                w_end_quat = np.empty(quat_list_shape)
                w_end_quat[:, 0] = 0.0
                w_end_quat[:, 1:] = angular_velocities[:, t_ind]


        # Convert angular velocity to quaternion form
        w_quat = np.empty(quat_list_shape)
        w_quat[:, 0] = 0.0
        w_quat[:, 1:] = w

        if order == 1:  # RK1 (Euler method)
            dq_dt = 0.5 * multiplyQuatLists(w_quat, current_poses)
            current_poses += dq_dt * dt

        elif order == 2:  # RK2 (midpoint method)
            # Step 1: Calculate k1
            k1 = 0.5 * multiplyQuatLists(w_quat, current_poses)

            # Step 2: Estimate midpoint
            midpoint_poses = current_poses + k1 * (dt / 2)

            # Step 3: Calculate k2 at the midpoint
            k2 = 0.5 * multiplyQuatLists(w_mid_quat, midpoint_poses)

            # Final step: Combine results
            current_poses += k2 * dt

        elif order == 4:  # RK4
            # Step 1: Calculate k1
            k1 = 0.5 * multiplyQuatLists(w_quat, current_poses)

            # Step 2: Calculate k2 (midpoint)
            midpoint_poses_k2 = current_poses + k1 * (dt / 2)
            k2 = 0.5 * multiplyQuatLists(w_mid_quat, midpoint_poses_k2)

            # Step 3: Calculate k3 (another midpoint)
            midpoint_poses_k3 = current_poses + k2 * (dt / 2)
            k3 = 0.5 * multiplyQuatLists(w_mid_quat, midpoint_poses_k3)

            # Step 4: Calculate k4 (endpoint)
            endpoint_poses = current_poses + k3 * dt
            k4 = 0.5 * multiplyQuatLists(w_end_quat, endpoint_poses)

            # Final step: Combine results
            current_poses += (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

        else:
            raise ValueError("Invalid order. Supported orders are 1 (RK1), 2 (RK2), or 4 (RK4).")

        # Normalize the quaternion to maintain unit length
        current_poses /= np.linalg.norm(current_poses, axis=1, keepdims=True)

    # Return only the final poses
    return current_poses

def _safeDivideHelper(numerator: ArrayLike, denominator: NDArray,
                      out_arr_creator, denom_nz: typing.Optional[np.ndarray] = None):
    if denom_nz is None:
        denom_nz = np.asarray(denominator != 0.0)
    if np.all(denom_nz):
        return numerator / denominator, True, denom_nz
    out_arr: NDArray = out_arr_creator(denominator.shape)
    np.divide(numerator, denominator, out=out_arr, where=denom_nz)
    return out_arr, False, denom_nz

def safeDivideElseZero(numerator, denominator: np.ndarray,
                       denom_nonzero_bools: typing.Optional[np.ndarray] = None):
    '''Performs normal division where the denominator is nonzero and outputs
    0 for any indicies where the denominator is zero.'''
    
    # Because np.zeros() is way faster than np.full(), we can save a few
    # steps in this particular case. 
    return _safeDivideHelper(
        numerator, denominator, lambda s: np.zeros(s), denom_nonzero_bools
    )[0]

    
def safeDivide(numerator, denominator: np.ndarray, na_value: float,
               denom_nonzero_bools: typing.Optional[np.ndarray] = None,
               denom_zero_bools: typing.Optional[np.ndarray] = None):
    '''Note: If na_value is 0, faster to use safeDivideElseZero()!'''

    ret, all_good, denom_nz = _safeDivideHelper(
        numerator, denominator, lambda s: np.empty(s), denom_nonzero_bools
    )
    if all_good:
        return ret
    if denom_nonzero_bools is None:
        denom_zero_bools = ~denom_nz
    ret[denom_zero_bools] = na_value
    return ret


# Takes in array of 2D points and, for each 3 consecutive points, gives the 
# centre of the circle defined by them.
def circleCentres2D(pts2D_0, pts2D_1, pts2D_2, na_value = np.inf):
    # Let x_i be the points on the circle and let m_i = (x_i + x_{i+1})/2 be the
    # midpoints. Let o_0 be orthogonal to (x_1 - x_0). The circle centre is
    # located at m_0 t*o_0 for t such that m_0 + t*o_0 - m_1 is orthogonal 
    # to (x_2 - x_1).
    # That is, we find t s.t. dot(m_0 + t*o_0 - m_1, x_2 - x_1) == 0.
    # I.e., dot(x_0 + x_1 + 2t*o_0 - (x_1 + x_2), x_2 - x_1) == 0
    # I.e., t = dot(x_2 - x_0, x_2 - x_1)/2*dot(o_0, x_2 - x_1)
    # => c = 0.5 * (x_0 + x_1 + dot(x_2 - x_0, x_2 - x_1)/dot(o, x2 - x1)*o_0)
    diffs_1m0 = pts2D_1 - pts2D_0
    diffs_2m1 = pts2D_2 - pts2D_1
    ortho_dirs = np.empty(diffs_1m0.shape)
    ortho_dirs[:, 0] = diffs_1m0[:, 1]
    ortho_dirs[:, 1] = -diffs_1m0[:, 0]
    # Reusability-TODO: Need a check for when dot(ortho_dirs, x_2 - x_1) == 0,
    # as then there is no circle going through the points (only a line).

    numerator_dot = einsumDot(pts2D_2 - pts2D_0, diffs_2m1)
    denominator_dot = einsumDot(ortho_dirs, diffs_2m1)
    t_vals = safeDivide(numerator_dot, denominator_dot, na_value)
    t_scaled_orthos = scalarsVecsMul(t_vals, ortho_dirs)
    centres = (pts2D_0 + pts2D_1 + t_scaled_orthos) / 2

    return centres

def rotateBySinCos2D(vecs, cosines, sines):
    rot_mats = np.moveaxis(np.array([
        [cosines, -sines],
        [sines, cosines]
    ]), -1, 0)

    return einsumMatVecMul(rot_mats, vecs)

def getAcuteAngles(angles):
    obtuse_inds = (angles > HALF_PI_NP)
    acute_inds = np.invert(obtuse_inds)
    acute_angs = np.empty_like(angles)
    acute_angs[obtuse_inds] = np.pi - angles[obtuse_inds]
    acute_angs[acute_inds] = angles[acute_inds]
    return acute_angs


def anglesBetweenVecs(vecs0, vecs1, normalization_needed = True):
    if normalization_needed:
        vecs0 = normalizeAll(vecs0)
        vecs1 = normalizeAll(vecs1)

    vals_preds_dot = einsumDot(vecs0, vecs1)
    return np.arccos(np.clip(vals_preds_dot, -1, 1))

# The 2acos(abs(dot(q0, q1))) between quaternions is the angle (rad) between the
# two rotations (i.e., the angle of the rotation from one to another). If you
# look at the formula for the scalar component of quaternion multiplication for 
# the desired rotation `q1*q0(^-1)`, it is clearly just dot(q0, q1). So, 
# cos(angle/2) == dot(q0,q1). The abs of that is cos(angle/2) if angle/2 is in
# [-pi/2, pi/2] and is otherwise cos(+-(pi - angle/2)); picture a unit circle to
# see why. Then 2acos is either abs(angle) or abs(2pi - angle); either way, it
# will be the correct angle between 0 and pi.
def anglesBetweenQuats(quats0, quats1):
    vals_preds_dot = einsumDot(quats0, quats1)
    half_angle = np.arccos(np.clip(np.abs(vals_preds_dot), -1, 1))
    return half_angle + half_angle

def quatSlerp(quats0, quats1, t: typing.Union[int, float, NDArray],
              zeroAngleThresh: float = 0.0001):
    t_is_const = (isinstance(t, float) or isinstance(t, int))
    if t_is_const:
        # For certain values of t, like t=2, optimizations are possible.
        if t == 2:
            # The below is the result of plugging t=2 into SLERP, using double
            # angle identities, and performing cancellations.
            t_eq_2_scalars = 2*einsumDot(quats0, quats1)
            return scalarsVecsMul(t_eq_2_scalars, quats1) - quats0
        if t == 0.5:
            bisect_dir = quats0 + quats1
            bisect_lens = np.linalg.norm(bisect_dir, axis = -1, keepdims=True)
            return bisect_dir / bisect_lens
        if t == 0:
            return quats0.copy()
        if t == 1:
            return quats1.copy()
    retVal = np.empty(quats0.shape)

    # Find the angles between the quaternions *interpreted as vec4s*, *not* the
    # angles between the rotations they represent!
    dots = einsumDot(quats0, quats1)
    angles = np.arccos(np.clip(dots, -1, 1))
    
    # No abs needed for next step, as prev call guaranteed to be positive, since
    # np.acos is guaranteed to be positive.
    zero_angle_bools = np.greater(zeroAngleThresh, angles)

    # For angles of zero, should just copy one of the quaternions.
    # For the others, normal quaternion SLERP applies.
    pos_angle_bools = np.invert(zero_angle_bools)
    zero_angle_inds = np.nonzero(zero_angle_bools)
    pos_angle_inds = np.nonzero(pos_angle_bools)
    pos_angles = angles[pos_angle_inds, np.newaxis]
    t_reshape = t
    if not t_is_const:
        t_reshape = typing.cast(NDArray, t)[pos_angle_inds].reshape(
            pos_angles.shape
        )

    retVal[zero_angle_inds] = quats0[zero_angle_inds]
    
    sin_vals = np.sin(pos_angles)
    scales0 = np.sin((1 - t_reshape) * pos_angles) / sin_vals
    scales1 = np.sin(t_reshape * pos_angles) / sin_vals
    retVal[pos_angle_inds] = scales0 * quats0[pos_angle_inds] + scales1 * quats1[pos_angle_inds]
    return retVal

def quatBezier(ctrl_qs, u):
    prev_qs = ctrl_qs
    next_qs = []
    deg = len(ctrl_qs) - 1
    for d in range(deg, 0, -1):
        for i in range(d):
            next_qs.append(quatSlerp(prev_qs[i], prev_qs[i + 1], u))
        prev_qs = next_qs
        next_qs = []
    
    return prev_qs[0]


def squad(qs, u):
    doubles_a = quatSlerp(qs[:-2], qs[1:-1], 2)
    aqs = quatSlerp(doubles_a, qs[2:], 0.5)
    bqs = quatSlerp(aqs, qs[1:-1], 2)

    return quatBezier([qs[1:-1], aqs, bqs, qs[2:]], u)

def randomRotationMat():
    scaled_axis = np.random.uniform(-np.pi, np.pi, 3)
    return matFromAxisAngle(scaled_axis)

def closestAnglesAboutAxis(rotatingFrames, targetFrames, axes):
    # Let X = [x,y,z]^T be one world-to-local frame & F = [r,s,t]^T be another.
    # We want the rotation R about axis w minimizing the angle between the
    # local-to-world frames RX^T and F^T.

    # Below, let "*" represent a dot product. Because X(F^T) is a rotation:
    # x*r + y*s + z*t = Trace([x,y,z]^T[r,s,t]) = Trace(XF^T) = 1 + 2cos(angle). 
    
    # Note: Func params are local-to-world, so "extra" transposes are needed.
    X_mats = np.swapaxes(rotatingFrames, -1, -2) # X^T^T = X
    X_F_Ts = einsumMatMatMul(X_mats, targetFrames)
    traces = X_F_Ts.trace(axis1 = -2, axis2 = -1)

    # Since 1+2cos(angle) is strictly decreasing for angles in [0, pi],
    # minimizing the angle is the same as maximizing x*r + y*s + z*t.

    # We thus want to find the R about w that maximizes Rx*r + Ry*s + Rz*t.

    # For space, let cr() represent cross() and let "0" represent R's rotation
    # angle theta, since they look similar. Rodrigues' rotation formula states:
    # Rx = (x*w)w + cos0(x - (x*w)w) + sin0 cr(w,x)

    # For any vector v, v = (X^T)Xv = (v*x)x + (v*y)y + (v*z)z, which we'll use
    # to rewrite our cross products without having to use the cr() operator.

    # By the properties of the scalar triple product, we can say things like:
    # y*cr(w, x) = w*cross(x, y) = z*w

    # Therefore:
    # y*cr(w,x) = z*w,   z*cr(w,x) = -y*w,   x*cr(w,y) = -z*w,   z*cr(w,y) = x*w
    # x*cr(w,z) = y*w,   y*cr(w,z) = -x*w

    # And, of course, v*cr(w,v) = 0 for any v. This gives us equalities like:
    # cr(w,x) = (x*cr(w,x))x + (y*cr(w,x))y + (z*cr(w,x))z = (w*z)y - (w*y)z

    # Putting this together, we modify Rodrigues' rotation formula to get:
    # Rx = (x*w)w + cos0(x - (x*w)w) + sin0 ((w*z)y - (w*y)z)

    # Rx*r = (x*w)(r*w) + cos0(x*r - (x*w)(r*w)) + sin(0) ((z*w)y*r - (y*w)z*r)

    # Skipping some steps to save space and letting C represent terms unaffected
    # by theta, and Tr() be Trace(), we can see that:
    # Rx*r + Ry*s + Rz*s
    # = C + cos0(Tr(XF^T) - Xw*Fw) + sin0(Xw*[z*s-y*t, x*t-z*r, y*r-x*s]^T)
    
    Xw = einsumMatVecMul(X_mats, axes).reshape(-1, 3)
    F_mats = np.swapaxes(targetFrames, -1, -2)
    Fw = einsumMatVecMul(F_mats, axes).reshape(-1, 3)
    other_axes = np.stack([
        X_F_Ts[..., 2, 1] - X_F_Ts[..., 1, 2],
        X_F_Ts[..., 0, 2] - X_F_Ts[..., 2, 0],
        X_F_Ts[..., 1, 0] - X_F_Ts[..., 0, 1]
    ], axis = -1)
    sin_component = einsumDot(Xw, other_axes)
    Xw_dot_Fw = einsumDot(Xw, Fw)
    cos_component = traces - Xw_dot_Fw

    # If we consider the non-C part as the 2D vector [cos0, sin0] dotted with
    # another 2D vector, then it is clear that the maximal solution is:
    # theta = atan2(Xw*[...], Xw*Fw + Trace(XF^T))
    thetas = np.arctan2(sin_component, cos_component)
    
    # I'm assuming there's ways to simplify this further... E.g., that vector
    # that sin(theta)Xw is being dotted with is the axis of the rotation between
    # the two original frames (multiplied by a scalar). And Xw*Fw = w*(X^T)Fw,
    # i.e., a cos(angle) between w and a rotated w. And I'm sure there's ways
    # that the sum-of-angle identities could play out, and *also* would be
    # interested in what cancellations happen when plugging this theta solution
    # into the Rodrigues formulas for Rx, Ry, and Rz.

    return thetas

def _axisAnglesFromQuatsHelp(quatVals: np.ndarray, remove_jumps: bool,
                             zeroAngleThresh: float = DEFAULT_ZERO_ANG_THRESH):
    halfAngles = np.arccos(np.clip(quatVals[..., 0:1], -1, 1))
    
    # For zero angle quats, we'll need to propogate the last nonzero axis.
    # First, we get the indices of the last nonzero axis for each zero axis.
    zeroHalfAngleThresh =  zeroAngleThresh / 2
    zeroAngleInds = np.nonzero((halfAngles < zeroHalfAngleThresh).flatten())
    angleInds = np.arange(len(halfAngles))
    angleInds[zeroAngleInds] = 0
    angleInds = np.maximum.accumulate(angleInds, axis = -1)
    angPropInds = angleInds[zeroAngleInds]

    # We want to propagate the non unit axes before looking for flips for
    # relatively obvious reasons.
    nonUnitAxes = quatVals[..., 1:].copy()
    nonUnitAxes[zeroAngleInds] = nonUnitAxes[angPropInds]
    if remove_jumps:
        # At this step, all angles SHOULD be positive.
        angle_dots = einsumDot(
            nonUnitAxes[..., 1:, :], nonUnitAxes[..., :-1, :]
        ) 
        needs_flip = np.logical_xor.accumulate(angle_dots < 0, axis = -1)
        halfAngles[..., 1:, :][needs_flip] = -halfAngles[..., 1:, :][needs_flip]

    # We can calculate the values of sine after any angle flipping because
    # of sign cancellation stuff that preserves the final rotation.
    sinHalf = np.sin(halfAngles)
    # Propagate what we're dividing by too so that we, in essence, propagate
    # the unit axes.
    sinHalf[zeroAngleInds] = sinHalf[angPropInds]

    # If the first rotations are zero-angle, then they have no previous axis to
    # copy. So they'll still be NaNs or whatever, so we should fix those.
    numIssueAxesAtFront = 0
    while numIssueAxesAtFront < len(halfAngles):
        if halfAngles[numIssueAxesAtFront] >= zeroHalfAngleThresh:
            break
        numIssueAxesAtFront += 1
    # Sine we don't know what axis to use at the start in a real live scenario,
    # we'll choose an arbitrary axis, [1, 0, 0], and ensure we scale it by
    # 1 later so that it stays unit length.
    nonUnitAxes[:numIssueAxesAtFront] = [1, 0, 0]
    sinHalf[:numIssueAxesAtFront] = 1

    if not remove_jumps:
        tooLarge = (halfAngles > HALF_PI_NP)
        sinHalf[tooLarge] = -sinHalf[tooLarge]
        halfAngles[tooLarge] = np.pi - halfAngles[tooLarge]
        angles = halfAngles + halfAngles
        return angles, sinHalf, nonUnitAxes

    # Now comes the jump removal version.
    angles = halfAngles + halfAngles

    # For comments describing the correctness of this code, see the mat3->aa
    # function comments.
    np_tau = 2.0 * np.pi
    tau_facs = np.round(np.diff(angles) / np_tau)
    angle_corrections = np_tau * np.cumsum(tau_facs, axis = -1)
    angles[..., 1:] -= angle_corrections
    return angles, sinHalf, nonUnitAxes


# Angles returned should be in the 0-PI range unless we remove jumps.
def axisAnglesFromQuats(quatVals: np.ndarray, remove_jumps: bool,
                        zeroAngleThresh: float = DEFAULT_ZERO_ANG_THRESH):
    '''Angles returned will be in the 0-PI range if `remove_jumps` is False.
    Otherwise, axes and angles will be returned such that the angles between
    consecutive axes are not obtuse and the angle values do not jump from,
    for example, 2pi minus epsilon to epsilon, but instead go to 2pi plus
    epsilon in such a case.'''
    angles, sinHalf, nonUnitAxes = _axisAnglesFromQuatsHelp(
        quatVals, remove_jumps, zeroAngleThresh
    )
    return nonUnitAxes / sinHalf, angles

def axisAngleVec3sFromQuats(quatVals: np.ndarray, remove_jumps: bool,
                            zeroAngleThresh: float = DEFAULT_ZERO_ANG_THRESH):
    '''Angles returned will be in the 0-PI range if `remove_jumps` is False.
    Otherwise, axes and angles will be returned such that the angle values do
    not jump from, for example, 2pi minus epsilon to epsilon, but instead go to
    2pi plus epsilon in such a case.'''
    angles, sinHalf, nonUnitAxes = _axisAnglesFromQuatsHelp(
        quatVals, remove_jumps, zeroAngleThresh
    )
    scalars = angles / sinHalf
    return scalars * nonUnitAxes

def multiplyQuatLists(q0, q1):
    num_qs = len(q0) if q0.ndim > 1 else len(q1)
    e = np.empty((4, num_qs), dtype=np.float64)
    q0w, q0x, q0y, q0z = q0.transpose()
    q1w, q1x, q1y, q1z = q1.transpose()
 
    e[0] = q0w*q1w - q0x*q1x - q0y*q1y - q0z*q1z
    e[1] = q0w*q1x + q0x*q1w + q0y*q1z - q0z*q1y
    e[2] = q0w*q1y - q0x*q1z + q0y*q1w + q0z*q1x
    e[3] = q0w*q1z + q0x*q1y - q0y*q1x + q0z*q1w
    return e.transpose()

def multiplyLoneQuats(q0, q1):
    e = np.empty(4, dtype=np.float64)
    q0w, q0x, q0y, q0z = q0
    q1w, q1x, q1y, q1z = q1
 
    e[0] = q0w*q1w - q0x*q1x - q0y*q1y - q0z*q1z
    e[1] = q0w*q1x + q0x*q1w + q0y*q1z - q0z*q1y
    e[2] = q0w*q1y - q0x*q1z + q0y*q1w + q0z*q1x
    e[3] = q0w*q1z + q0x*q1y - q0y*q1x + q0z*q1w

    return e

# Makes rotation matrix from an axis (with angle being encoded in axis length).
# Uses common formula that you can google if need-be.
def matFromAxisAngle(scaledAxis):
    angle = np.linalg.norm(scaledAxis)
    if angle == 0.0:
        return np.identity(3)
    unitAxis = scaledAxis / angle
    x, y, z = unitAxis.flatten()
    skewed = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])
    return np.identity(3) + np.sin(angle) * skewed + (1.0 - np.cos(angle)) * (skewed @ skewed)

def matsFromAxisAngleArrays(angles, unitAxes):
    flat_angs = angles.flatten()
    xs, ys, zs = unitAxes.reshape(-1, 3).transpose()
    _0s = np.zeros_like(xs)
    
    skeweds = np.moveaxis(np.array([
        [_0s, -zs, ys],
        [zs, _0s, -xs],
        [-ys, xs, _0s]
    ]), -1, 0)
    idens = np.repeat([np.eye(3)], len(flat_angs), axis=0)
    sin_part = scalarsMatsMul(np.sin(flat_angs), skeweds)
    skeweds2 = einsumMatMatMul(skeweds, skeweds)
    cos_part = scalarsMatsMul(1.0 - np.cos(flat_angs), skeweds2)
    mats_list = idens + sin_part + cos_part
    return mats_list.reshape(angles.shape + (3, 3))

def matsFromScaledAxisAngleArray(scaledAxisAngles: NDArray):
    angles = np.linalg.norm(scaledAxisAngles, axis=-1)
    axes = np.zeros_like(scaledAxisAngles)
    posInds = (angles != 0)
    axes[posInds] = scaledAxisAngles[posInds] / angles[posInds][..., np.newaxis]
    return matsFromAxisAngleArrays(angles, axes)

def matsFromQuaternions(quats: np.ndarray):
    # Math source: https://www.songho.ca/opengl/gl_quaternion.html
    s, x, y, z = quats.transpose()
    _2x2 = 2*x**2
    _2y2 = 2*y**2
    _2z2 = 2*z**2
    _2xy = 2*x*y
    _2xz = 2*x*z
    _2yz = 2*y*z
    _2sx = 2*s*x
    _2sy = 2*s*y
    _2sz = 2*z*z
    return np.moveaxis(np.array([
        [1 - _2y2 - _2z2,  _2xy - _2sz,      _2xz + _2sy],
        [_2xy + _2sz,      1 - _2x2 - _2z2,  _2yz - _2sx],
        [_2xz - _2sy,      _2yz + _2sx,      1 - _2x2 - _2y2]
    ]), -1, 0)

def handleCondsAtStart(cond_bools: NDArray, ref_arr: NDArray, func_to_app,
                       cond_bools_time_axis: int = -1, use_count: bool = True,
                       update_cond_bools: bool = False, **kwargs):

    
    pos_time_axis = cond_bools_time_axis
    pos_default_axis = -1
    new_cond_bools = cond_bools
    if cond_bools_time_axis != -1:
        pos_time_axis = cond_bools_time_axis % cond_bools.ndim
        pos_default_axis = cond_bools.ndim - 1
        new_cond_bools = np.swapaxes(cond_bools, pos_default_axis, pos_time_axis)
    n = new_cond_bools.shape[-1] # Length per sequence
    flatter_conds = new_cond_bools.reshape(-1, n)
    
    is_np_d = {k: isinstance(v, np.ndarray) for k, v in kwargs.items()}
    
    array_args = {k: v for k, v in kwargs.items() if is_np_d[k]}
    array_args["ref_arr"] = ref_arr
    
    new_kwargs = dict(kwargs)
    
    for k, arr in array_args.items():
        extra_ndims = arr.ndim - cond_bools.ndim
        if extra_ndims != 0 and extra_ndims != 1:
            if extra_ndims < 0:
                raise ValueError("{}.ndim must >= cond_bools.ndim".format(k))
            else:
                # I'm not %100 sure I won't ever encounter a situation where I
                # need arr.ndim - cond_bools.ndim > 1, e.g. for arrays of
                # rotation matrices, so *maybe* I'll eventually have to change
                # this code so that I don't raise an Error here. But for now,
                # this behaviour would be unexpected and worth catching.
                raise ValueError((
                    "Currently, having {}.ndim > cond_bools.ndim + 1 is "
                    "unexpected behaviour, though maybe it isn't for your use "
                    "case; in that event, the handleCondsAtStart function "
                    "should be edited to support your use case!"
                ).format(k))
        cshape = cond_bools.shape
        ashape = arr.shape
        for cd, ad in zip(cshape, ashape[:cond_bools.ndim]):
            if cd != 1 and ad != 1 and cd != ad:
                raise ValueError(
                    f"{k} & cond_bools ({ashape} & {cshape}) not compatible!"
                )
        
        if cond_bools_time_axis != -1:
            arr = np.swapaxes(arr, pos_time_axis, pos_default_axis)
            
        flatter_shape = (-1, ) + arr.shape[(cond_bools.ndim - 1):]
        arr = arr.reshape(flatter_shape)
        
        # Make sure our relevant params are updated!
        if k == "ref_arr":
            ref_arr = arr
        else:
            new_kwargs[k] = arr

    
    seq_inds_starting_with_cond = np.where(flatter_conds[:, 0])[0]
    seqs_starting_with_cond = flatter_conds[seq_inds_starting_with_cond]
    for i, c_row in zip(seq_inds_starting_with_cond, seqs_starting_with_cond):
        # We know there's at least one because we already pre-filtered.
        conds_at_front = 1 
        if use_count:
            while conds_at_front < n and c_row[conds_at_front]:
                conds_at_front += 1
        kwargs_i = {
            k: v if not is_np_d[k] else v[i] for k, v in new_kwargs.items()
        }
        func_to_app(ref_arr[i], conds_at_front, **kwargs_i)
        if update_cond_bools:
            flatter_conds[i][:conds_at_front] = False
    return

# Input is assumed to be a numpy array with shape (n,3,3) for some n > 0.
# Return value thus has shape (n,3).
def axisAngleFromMatArray(matrixArray, zeroAngleThresh = DEFAULT_ZERO_ANG_THRESH) -> NDArray:
    # Reusability-TODO: Last I checked (2024-10-19), it's fine, but if changed
    # since, see if supporting arrays with more dims than (n,3,3) leads to any 
    # inefficiency; if so, remove, or use an "if" to switch to better "flat"
    # version, because I don't know of a practical purpose off-hand for
    # supporting more dims than that. In fact, it'd possibly hinder multicore 
    # processing, which may be the better and/or faster approach for more dims.

    # --------------------------------------------------------------------------
    # We'll start by using Shepperd's algorithm to obtain initial results:
    #   Shepperd, Stanley W. "Quaternion from rotation matrix."
    #   Journal of guidance and control 1.3 (1978): 223-224.
    #   doi:10.2514/3.55767b
    # Then we'll modify the initial results to remove discontinuous jumps
    # between consecutive axis-angle vectors, as described later.
    # --------------------------------------------------------------------------
    # In Shepperd's algorithm, we take different steps for each rotation matrix
    # depending on whether the trace or a diagonal entry is larger. We'll 
    # use numpy slices to accomplish this.

    # --------------------------------------------------------------------------
    # Shepperd's Algorithm: Initial Setup
    # --------------------------------------------------------------------------


    # We'll specify the trace axes so that we can get the trace of each rotation
    # matrix from an array of them and still get correct output.
    matrixTraceVals = matrixArray.trace(axis1 = -2, axis2 = -1)
    # np.diagonal(...) makes no copy, so this should be reasonably efficient.
    matrixDiags = np.diagonal(matrixArray, axis1 = -2, axis2 = -1)
    
    angles = np.empty(matrixArray.shape[:-2]) # Storage for resulting angles.
    nonUnitAxes = np.empty(matrixDiags.shape) # Storage for resulting axes.

    # Get largest diagonal entries' locations. We'll reuse this later on too.
    # Note: newaxis used instead of keepdims to support older numpy versions.
    whereMaxDiag = np.argmax(matrixDiags, axis = -1)[..., np.newaxis]

    # Extract the value of the largest diagonal entry.
    # Earlier numpy versions don't have `take_along_axis`; in that case, you'd
    # use something like arr[np.arange(len(...)), colIndices].
    diagMaxes = np.take_along_axis(matrixDiags, whereMaxDiag, axis = -1)[..., 0]

    # Slice indices for applying different steps to different rotation matrices.
    useTraceBool = np.greater(matrixTraceVals, diagMaxes)
    useDiagBool = np.invert(useTraceBool)
    # I think indices are oft faster than bool indexing. But needs confirming.
    useTrace = np.nonzero(useTraceBool)
    useDiag = np.nonzero(useDiagBool)

    anyUseTrace = np.sum([len(ut) for ut in useTrace]) > 0
    anyUseDiag = np.sum([len(ud) for ud in useDiag]) > 0
    

    # --------------------------------------------------------------------------
    # Shepperd's Algorithm: Case Where Trace Was Greater.
    # --------------------------------------------------------------------------

    if anyUseTrace:
        # The angle of rotation about the above axis direction.
        # Outputs of acos are constrained to [0, pi], which impacts later code.
        # Input needs to be clamped to [-1, 1] in case fp precision causes it to
        # exit that interval and, thus, the domain for acos.
        acosInput = np.clip((matrixTraceVals[useTrace] - 1.0)/2.0, -1.0, 1.0)
        angles[useTrace] = np.arccos(acosInput)

        matrixOffDiags = matrixArray[useTrace]

        nonUnitAxes[useTrace] = np.stack([
            matrixOffDiags[...,2,1] - matrixOffDiags[...,1,2],
            matrixOffDiags[...,0,2] - matrixOffDiags[...,2,0],
            matrixOffDiags[...,1,0] - matrixOffDiags[...,0,1]
        ], axis=-1) # Axis needs specifying for if input is a list of matrices.


    # --------------------------------------------------------------------------
    # Shepperd's Algorithm: Case Where a Diagonal Entry Was Greater.
    # --------------------------------------------------------------------------

    if anyUseDiag:
        i_s = whereMaxDiag[useDiag][..., 0]
        j_s = (i_s + 1) % 3
        k_s = (j_s + 1) % 3

        matsWhereDiagUsed = matrixArray[useDiag]
        # The only way I know to slice with variable last-axis-indices is to
        # pass an arange for the first axis; simply using `[:, i_s, j_s]` FAILS!
        # Maybe there's a better way I'm unaware of, though.
        arangeUseDiag = np.arange(len(i_s))
        Aij = matsWhereDiagUsed[arangeUseDiag, i_s, j_s]
        Aji = matsWhereDiagUsed[arangeUseDiag, j_s, i_s]
        Aik = matsWhereDiagUsed[arangeUseDiag, i_s, k_s]
        Aki = matsWhereDiagUsed[arangeUseDiag, k_s, i_s]
        Ajk = matsWhereDiagUsed[arangeUseDiag, j_s, k_s]
        Akj = matsWhereDiagUsed[arangeUseDiag, k_s, j_s]

        diagMaxSubset = diagMaxes[useDiag]
        
        # The below is `2sin(angle/2) * axis`
        sqrtInput = 1 + diagMaxSubset + diagMaxSubset - matrixTraceVals[useDiag]
        # Because max-diag-element >= trace, sqrt(input) >= 1; no fp concerns.
        ax_i = np.sqrt(sqrtInput)
        nonUnitAxes[useDiag + (i_s, )] = ax_i
        nonUnitAxes[useDiag + (j_s, )] = (Aij + Aji)/ax_i
        nonUnitAxes[useDiag + (k_s, )] = (Aik + Aki)/ax_i

        # Again, we need to clamp/clip in case of fp precision causing problems.
        acosInput = np.clip((Akj - Ajk)/(ax_i + ax_i), -1.0, 1.0)
        halfAngles = np.arccos(acosInput)
        angles[useDiag] = halfAngles + halfAngles # Will be between 0 and 2pi.

    # if np.any(angles < 0):
    #     raise Exception("I was wrong about all pos angles at this step!")

    # --------------------------------------------------------------------------
    # "Corrections" Proceeding Shepperd's Algorithm
    # --------------------------------------------------------------------------

    # For reasons described shortly, we may want to use an axis other than the
    # default [0, 0, 0] to represent a rotation by `2*n*pi` radians.
    # To fix this, we'll choose to propagate the last nonzero axis.
    # Note: if the FIRST axes are all [0, 0, 0], that's okay; we don't need to
    # worry about propagating BACKWARDS, just FORWARDS. We'll  use the technique
    # proposed in a 2015-05-27 StackOverflow answer by user "jme" (1231929/jme)
    # to a 2015-05-27 question, "Fill zero values of 1d numpy array with last
    # non-zero values" (https://stackoverflow.com/q/30488961) by user "mgab"
    # (3406913/mgab). A 2016-12-16 edit to "Most efficient way to forward-fill 
    # NaN values in numpy array" by user Xukrao (7306999/xukrao) shows this to
    # be more efficient than similar for-loop, numba, pandas, etc. solutions.
    # --------------------------------------------------------------------------
    # This has been modified to work with more dimensions than just a flat list
    # of angles! This works by, basically, doing:
    #  * `inds = np.tile(np.arange(angles.shape[-1]), angles.shape[:-1] + (1,))`
    #    * An array with same shape as angles, but with aranges on last axis.
    #  * Do the zero-setting and max accumulation similar to before.
    #  * For copying, could use something like `put_along_axis`, but that's
    #    copying way more items than need-be. Could instead maybe do:
    #      * `where = np.argwhere(angles < zeroThresh).transpose()`
    #      * `where[-1] = accumulated_inds[angles < zeroThresh]`
    #      * `axes[angles < zeroThresh] = axes[tuple(where)]`
    #  * Or could flatten earlier and accept weirdness if first angles are 0.
    n_per_seq = angles.shape[-1]
    seq_parent_shape = angles.shape[:-1]
    angleInds = np.tile(np.arange(n_per_seq), seq_parent_shape + (1, ))
    # At this step, all angles SHOULD be positive.
    zero_angle_mask = angles < zeroAngleThresh
    zeroAngleInds = np.nonzero(zero_angle_mask)
    angleInds[zeroAngleInds] = 0
    angleInds = np.maximum.accumulate(angleInds, axis = -1)
    za_where = np.argwhere(zero_angle_mask).transpose()
    za_where[-1] = angleInds[zeroAngleInds]
    nonUnitAxes[zeroAngleInds] = nonUnitAxes[tuple(za_where)]

    # TL;DR: Angles for the 1st case of Shepperd's algorithm, as output of acos,
    # start out in interval [0, pi]. Thus, the similar rotations
    # [pi - epsilon, 0, 0] and [pi + epsilon, 0, 0] will be represented by
    # distant vectors (pi - epsilon)[+1, 0, 0] and (pi - epsilon)[-1, 0, 0].
    # Similar *could* happen for the 2nd case too, though is "less guaranteed".
    # Anyway, we detect when this happens by observing the axes and then we
    # correct the angles (e.g., adding 2pi multiples, i.e. taus) to fix.
    # --------------------------------------------------------------------------
    # Let `norm = angle/2sin(angle)` for the 1st case of Shepperd's algorithm,
    # and let `norm = angle/2sin(angle/2)` for the 2nd. If we now just return
    # `norm * nonUnitAxes`, we would have *accurate* angle-axis results stored 
    # as vec3s. However, you can get axis-angle vec3 "jumps", like we described
    # above, over small rotation changes. So, we look at the unnormalized axes
    # for such flips, and we start our corrections by negating corresponding
    # angles. Now, you might think this either (a) generates incorrect results,
    # or (b) does nothing. You might think (a) because obviously rotations by
    # +alpha and -alpha about the same axis differ. But in a later step, because
    # we get our final axis-angle vec3s by multiplying `norm * nonUnitAxes`, and
    # because `angle/sin(angle) == (-angle)/sin(-angle)`, this would have no
    # effect on our output if we took no further steps. Which may lead one to
    # think that (b) applies. BUT, now angles are set up for correction by 
    # adding taus: in our "pi + epsilon" example, our consecutive angles become
    # "pi - epsilon" and "-(pi - epsilon)" after negation; we can correct the
    # latter by adding 2pi to get "pi + epsilon", which is the "best" way to
    # represent those consecutive rotations!
    # We'll only add taus s.t. angles become within pi distance of each other.
    # E.g., consecutive angles +epsilon and -epsilon would not be affected.
    # Unfortunately, because `sin((angle + tau)/2) = -sin(angle/2)`, an extra
    # negation gets introduced into the later normalization of the axes for the
    # 2nd case of Shepperd's algorithm. SO, we should normalize the axes AFTER
    # negating angles but BEFORE adding taus!!!
    # --------------------------------------------------------------------------
    # If you still doubt any of the above, please at least be very careful in
    # making any "corrections". I've thought about this quite thoroughly, but I 
    # don't want to take up too much space justifying it further.
    # --------------------------------------------------------------------------
    # In summary, we perform the following steps:
    #  1. Detect if axis flipped, via dot product.
    #  2. If axis was flipped, negate angle. 
    #  3. Add necessary multiples of 2pi to angles to prevent large angle jump.
    #     (May be necessary even if axis did not flip on this frame!)
    #     (E.g., corrections to previous frames could lead to lastAngle > 2pi)
    # --------------------------------------------------------------------------

    # Numpy-styled axis-flip-detection:
    # Here's a StackOverflow post suggesting that einsum might be faster than
    # doing `(arr[1:] * arr[:-1]).sum(axis=1)`:
    # https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
    # (see answer with plots further down page)
    # For the first negative dot prod, we will need to flip the corresponding
    # axis. If the next index's *original* dot product was positive, we'll need
    # to flip the next axis also, to keep it aligned with the "new" previous, 
    # and so on until we reach another *original* dot product that was negative.
    # I.e., we "accumulate" the number of flips needed, and even numbers cancel
    # out. This is why we use an accumulation of xors; it sort of functions like
    # mod 2 addition, but I'm hoping it's cheaper.
    angle_dots = einsumDot(nonUnitAxes[..., 1:, :], nonUnitAxes[..., :-1, :]) 
    needs_flip = np.logical_xor.accumulate(angle_dots < 0, axis = -1)
    angles[..., 1:][needs_flip] = -angles[..., 1:][needs_flip]

    # (!!!) PLEASE DO NOT MOVE THIS LINE WITHOUT READING EARLIER COMMENTS
    #       EXPLAINING WHY IT SHOULD BE *EXACTLY* HERE!
    # Now comes the axis normalization/flipping. We have 2 cases to consider:
    #   1. Axes that need dividing by 2sin(angle) or 2sin(angle/2). For these,
    #      as shown before, no further action is required in terms of sign flips
    #      and whatnot if we normalize NOW, before adding taus.
    #   2. Axes for angles 2pi*n, which copied the last non-zero-angle axis.
    #      For these, we'll just copy over the normalized axes.
    unitAxes = np.empty(matrixDiags.shape) # Storage for resulting axes.
    
    # First, we'll handle Shepperd's Algorithm case 1:
    sinVals = np.sin(angles[useTrace])
    sinVals += sinVals
    unitAxes[useTrace] = \
        nonUnitAxes[useTrace] / sinVals[..., np.newaxis]
    # Then Shepperd's Algorithm case 2:
    sinVals = np.sin(angles[useDiag]/2.0)
    unitAxes[useDiag] = nonUnitAxes[useDiag] / (sinVals + sinVals)[..., np.newaxis]
    # Then zero-angle:
    unitAxes[zeroAngleInds] = unitAxes[tuple(za_where)]

    def _make_firsts_0(arr: NDArray, n: int): arr[:n] = 0.0

    handleCondsAtStart(zero_angle_mask, unitAxes, _make_firsts_0)
    
    # Now we add 2pi multiples to make angles within pi of each other.
    # We want the following difference-to-correction mapping:
    # ..., (-3pi, -pi) -> tau, (-pi, pi) -> 0, (pi, 3pi) -> -tau,
    # (3pi, 5pi) -> -2tau, ... (note: interval endpoints don't matter)
    # The following code achieves this:
    np_tau = 2.0 * np.pi
    tau_facs = np.round(np.diff(angles) / np_tau)
    # We need to accumulate the correction sum because, if two angles are within
    # pi of each other, and the former gets incremented by n*tau, the latter 
    # must also be incremented by n*tau so that they stay within pi distance.
    angle_corrections = np_tau * np.cumsum(tau_facs, axis = -1)
    angles[..., 1:] -= angle_corrections

    # if np.any(np.greater(np.abs(np.diff(angles)), np.pi + 0.00001)):
    #     raise Exception("Numpification of AA code resulted in angle diff > pi!")
                
    # Now we combine the angles and unit axes into a final array of vec3s.
    return np.einsum('...i,...ij->...ij', angles, unitAxes)


# def axisAngleListFromMats(matList):
#     # differences = []
#     lastAngle = 0.0
#     lastDir = np.array([0.0, 0.0, 0.0])
#     retList = []
#     for m in matList:
#         lastAngle, val = axisAngleFromMat(m, lastAngle, lastDir)
#         lastDir = val
#         retList.append(val)
#         # recovered = matFromAxisAngle(val)
#         # differences.append(np.linalg.norm(recovered - m))
#     # print("Max difference:", np.max(np.array(differences)))
    
#     return np.array(retList)


def flipObtuseAxes(unflipped_axes: np.ndarray):
    angle_dots = einsumDot(unflipped_axes[1:], unflipped_axes[:-1])
    needs_flip = np.logical_xor.accumulate(angle_dots < 0, axis = -1)
    needs_no_flip = np.invert(needs_flip)

    ret_axes = np.empty(unflipped_axes.shape)
    ret_axes[0] = unflipped_axes[0]
    ret_axes[1:][needs_flip] = -unflipped_axes[1:][needs_flip]
    ret_axes[1:][needs_no_flip] = unflipped_axes[1:][needs_no_flip]
    return ret_axes

def quadraticFormulaScalar(a: float, b: float, c: float) -> typing.List[float]:
    a2 = a + a
    discrim = b*b - (a2 + a2)*c

    if discrim > 0.0:
        sqrt_discrim = np.sqrt(discrim)
        ret_val_0 = (-b - sqrt_discrim)/a2
        ret_val_1 = (-b + sqrt_discrim)/a2
        return [ret_val_0, ret_val_1]
    elif discrim == 0.0:
        return [-b/a2]
    return []

def quadraticFormulaNP(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    a2 = a + a
    discrim = b*b - (a2 + a2)*c

    root_inds = discrim >= 0.0
    roots = np.empty(a.shape + (2,))

    sqrts = np.sqrt(discrim[root_inds])
    a2_r = a2[root_inds]
    neg_b_r = -b[root_inds] 
    roots[root_inds, 0] = (neg_b_r - sqrts)/a2_r
    roots[root_inds, 1] = (neg_b_r + sqrts)/a2_r
    roots[~root_inds] = np.nan
    return roots

def since_calc(bool_inds: np.ndarray, num_total_input_frames: int, 
               features_to_sum: typing.List[np.ndarray],
               prev_inds_included_in_sum: int):
    # If we have n+1 total frames and n "input" frames, then we have n-1
    # velocities, n-2 accelerations, etc. so the number of bools for our events
    # is n-k for some k. But to make things a bit easier to USE (even if 
    # CREATION is harder), I think all arrays should be length n. 
    # We'll also init using 0s, not empty, in order to propagate last indices.
    last_inds = np.zeros(num_total_input_frames, dtype=np.int32)
    k = num_total_input_frames - len(bool_inds)
    # We need to shift all bool-to-int inds by k to compensate.
    int_inds = np.where(bool_inds)[0] + k
    # We'll assume the event happened for the first k frames, because otherwise
    # we must make an arbitrary nonzero choice of time-since-event for them. 
    for i in range(1, k): # last_inds[0] == 0 already.
        last_inds[i] = i
    last_inds[int_inds] = int_inds
    
    # As described better (e.g., efficiency) in other comments in my xode where
    # I do ths,we'll use the technique proposed in a 2015-05-27 StackOverflow 
    # answer by user "jme" (1231929/jme) to a 2015-05-27 question, "Fill zero 
    # values of 1d numpy array with last non-zero values" 
    # (https://stackoverflow.com/q/30488961) by user "mgab" (3406913/mgab). 
    last_inds = np.maximum.accumulate(last_inds)
    
    time_since = np.arange(num_total_input_frames) - last_inds
    
    ret_features: typing.List[np.ndarray] = []
    for feature in features_to_sum:
        feature_len_diff = num_total_input_frames - len(feature)
        # We want to take the sum since frame 0 to the "present" and subtract 
        # the sum from frame 0 until the last event.
        feature_sums = np.empty(num_total_input_frames)
        feature_sums[:feature_len_diff] = 0.0
        feature_sums[feature_len_diff:] = np.cumsum(feature)
        start_inds = last_inds - prev_inds_included_in_sum
        inds_to_subtract = np.maximum(start_inds, 0)
        feature_since = feature_sums - feature_sums[inds_to_subtract]
        ret_features.append(feature_since)
    
    return (last_inds, time_since, ret_features)


def non_collinear_features(X: np.ndarray, threshold: float = 0.99):
    """
    Detects collinear columns from a 2D NumPy array based on the given correlation threshold.
    
    Parameters:
        X : np.ndarray
            Input 2D array (samples x features).
        threshold: float
            Correlation threshold for collinearity. Columns with correlation 
            above this value will be removed.

    Returns:
        (to_keep, upper_tri): (np.ndarray, np.ndarray)
            - Boolean mask indicating which columns to keep.
            - Upper triangular of covariance matrix.
    """
    # Note: Function generated by ChatGPT. Manually commented/verified below.

    # Correlation matrix, assuming each column in X contains the values for a
    # distinct variable.
    corr_matrix = np.corrcoef(X, rowvar=False)

    # We look at the upper triangular only for high covariances so that if the
    # variables for data columns m and n, n > m, are correlated, the correlation
    # matrix column m will not contain this high correlation value in its n-th
    # row (which will be zero) but the nth column will contain the high value in
    # its mth row. So we'll remove column n only instead of both or neither.
    upper_tri = np.triu(np.abs(corr_matrix), k=1)  # k=1 -> zeros on diagonal.
    to_keep = ~(np.any(upper_tri > threshold, axis=0))
    
    return to_keep, upper_tri

def cross2D(vecs0, vecs1):
    '''
    Perform the cross-product of two arrays of 2D vectors. What this means is
    treating them as 3D vectors in the xy plane and returning the z component
    of the cross product (since the x and y would be zero).

    This could currently be done by np.cross(...), but its support of 2D inputs
    is deprecated; this function is a future-proof replacement.
    '''
    x0 = vecs0[..., 0]
    y0 = vecs0[..., 1]
    x1 = vecs1[..., 0]
    y1 = vecs1[..., 1]

    return (x0 * y1) - (x1 * y0)

# TODO: Move this to another file.
def poseLossAngle(y_true, y_pred):
    # 2acos(abs([cos||x/2|| cos||y/2|| + <x>*<y> sin||x/2|| sin||y/2||))
    ang_true = np.linalg.norm(y_true, axis=-1) + 0.00000001
    ang_pred = np.linalg.norm(y_pred, axis=-1) + 0.00000001
    half_ang_true = ang_true / 2
    half_ang_pred = ang_pred / 2
    cos_true = np.cos(half_ang_true)
    cos_pred = np.cos(half_ang_pred)
    sin_true = np.sin(half_ang_true)
    sin_pred = np.sin(half_ang_pred)
    vec3_dots = einsumDot(y_true, y_pred)
    unit_vec3_dots = vec3_dots / (ang_true * ang_pred)
    quat_dots = cos_true * cos_pred + unit_vec3_dots * sin_true * sin_pred
    quat_dots_1 = np.abs(np.clip(quat_dots, -1, 1))
    return 2 * np.arccos(quat_dots_1) #tf.convert_to_tensor(...)
