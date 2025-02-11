import numpy as np

from collections import namedtuple


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
    # if (np.isnan(ret_val).any()):
    #     raise Exception("NaN when normalizing vectors!")
    return ret_val


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

# Normalize an array while handling the case where some elements have a norm of
# zero, which would cause division errors.
def safelyNormalizeArray(array: np.ndarray, norms: np.ndarray = None,
                         vec_for_zero_norms: np.ndarray = None,
                         propagate_last_nonzero_vec: bool = True,
                         propagate_back_if_first_vecs_zero: bool = False):
    
    if norms is None:
        norms = np.linalg.norm(array, axis=-1, keepdims=True)

    zero_norm_inds = (norms == 0).flatten()

    if not np.any(zero_norm_inds):
        return array / norms
    
    pos_norm_inds = np.invert(zero_norm_inds)
    normed = np.empty_like(array)
    normed[pos_norm_inds] = array[pos_norm_inds]/norms[pos_norm_inds]
    # For zero axes, we can either use a supplied default vector, create our
    # own default, or propograte the last nonzero vector.
    if propagate_last_nonzero_vec:
        # First, we make sure that if the first vec is zero, that we replace it
        # with some default, since there'd be no previous vec to copy.
        if zero_norm_inds[0]:
            if vec_for_zero_norms is None:
                if propagate_back_if_first_vecs_zero:
                    search_ind = 1
                    while search_ind < len(zero_norm_inds):
                        if not zero_norm_inds[search_ind]:
                            normed[0] = normed[search_ind]
                            break
                        search_ind += 1
                else:
                    replacement_vec = np.zeros(array.shape[-1])
                    replacement_vec[0] = 1.0
                    normed[0] = replacement_vec
            else:
                normed[0] = vec_for_zero_norms
        # To copy the last nonzero vectors, we'll use the technique proposed in 
        # a 2015-05-27 StackOverflow answer by user "jme" (1231929/jme) to a
        # 2015-05-27 question, "Fill zero values of 1d numpy array with last
        # non-zero values" (https://stackoverflow.com/q/30488961) by user "mgab"
        # (3406913/mgab). A 2016-12-16 edit to "Most efficient way to 
        # forward-fill NaN values in numpy array" by user Xukrao
        # (7306999/xukrao) shows this to be more efficient than similar
        # for-loop, numba, pandas, etc. solutions.
        replacement_inds = np.arange(len(norms))
        int_zero_inds = np.nonzero(zero_norm_inds)
        replacement_inds[int_zero_inds] = 0
        replacement_inds = np.maximum.accumulate(replacement_inds, axis = -1)
        normed[int_zero_inds] = normed[replacement_inds[int_zero_inds]]
    else:
        if vec_for_zero_norms is None:
            replacement_vec = np.zeros(array.shape[-1])
            replacement_vec[0] = 1.0
            normed[zero_norm_inds] = replacement_vec
        else:
            normed[zero_norm_inds] = vec_for_zero_norms
    return normed

def quatsFromAxisAngleVec3s(axisAngleVals):
    angles = np.linalg.norm(axisAngleVals, axis=1, keepdims=True)
    normed = safelyNormalizeArray(
        axisAngleVals, angles, propagate_back_if_first_vecs_zero=True
    )

    return quatsFromAxisAngles(normed, angles)

def einsumDot(vecs0, vecs1):
    # Einsum is used to perform a dot product between consecutive axes.
    # Dot products occur along 'j' axis; preserves existence of the 'i' axis.
    # For understanding einsum, this ref might be handy:
    # https://ajcr.net/Basic-guide-to-einsum/
    # Here's a StackOverflow post suggesting that einsum might be faster than
    # doing `(arr[1:] * arr[:-1]).sum(axis=1)`:
    # https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
    # (see answer with plots further down page)

    return np.einsum('...ij,...ij->...i', vecs0, vecs1)

def einsumMatVecMul(mats, vecs):
    return np.einsum('bij,bj->bi', mats, vecs)

def einsumMatMatMul(mats0, mats1):
    return np.einsum("bij,bjk->bik", mats0, mats1)

def scalarsVecsMul(scalars, vecs):
    return np.einsum('b,bi->bi', scalars, vecs)

def scalarsMatsMul(scalars, mats):
    return np.einsum('b,bij->bij', scalars, mats)

def parallelAndOrthoParts(vectors, dirs, dirs_already_normalized = False):
    dots = einsumDot(vectors, dirs)

    if not dirs_already_normalized:
        dots /= einsumDot(dirs, dirs)
    
    parallels = scalarsVecsMul(dots, dirs)
    orthos = vectors - parallels
    return (parallels, orthos)

def getOrthonormalFrames(vecs0: np.ndarray, vecs1: np.ndarray = None,
                         vecs0_are_unit_len: bool = False):
    mats = np.empty(vecs0.shape[:-1] + (3, 3))
    mats[..., 0] = vecs0 if vecs0_are_unit_len else safelyNormalizeArray(vecs0)
    if vecs1 is None:
        vecs1 = np.ones_like(vecs0)
    v0v1dot = einsumDot(mats[..., 0], vecs1)
    parallels = scalarsVecsMul(v0v1dot, mats[..., 0])
    mats[..., 1] = safelyNormalizeArray(vecs1 - parallels)
    mats[..., 2] = np.cross(mats[..., 0], mats[..., 1])
    return mats

def getPlaneAxes(roughAxes0, roughAxes1):
    nax0 = normalizeAll(roughAxes0)
    dots01 = einsumDot(nax0, roughAxes1)
    ax1 = roughAxes1 - scalarsVecsMul(dots01, nax0)
    ax1_norms = np.linalg.norm(ax1, axis=-1, keepdims=True)
    # TODO: Maybe I want non-nan behaviour for zero norms here.
    nax1  = ax1 / ax1_norms
    return (nax0, nax1)

def getPlaneInfo(roughAxes0, roughAxes1, ptsOnPlane):
    norm_axes = getPlaneAxes(roughAxes0, roughAxes1)

    normals = np.cross(*norm_axes)
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


# Takes in array of 2D points and, for each 3 consecutive points, gives the 
# centre of the circle defined by them.
def circleCentres2D(pts2D_0, pts2D_1, pts2D_2):
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
    # TODO: Need a check for when dot(ortho_dirs, x_2 - x_1) == 0, as then
    # there is no circle going through the points (only a line).

    numerator_dot = einsumDot(pts2D_2 - pts2D_0, diffs_2m1)
    t_vals = numerator_dot / einsumDot(ortho_dirs, diffs_2m1)
    t_scaled_orthos = scalarsVecsMul(t_vals, ortho_dirs)
    centres = (pts2D_0 + pts2D_1 + t_scaled_orthos) / 2

    return centres

def rotateBySinCos2D(vecs, cosines, sines):
    rot_mats = np.moveaxis(np.array([
        [cosines, -sines],
        [sines, cosines]
    ]), -1, 0)

    return einsumMatVecMul(rot_mats, vecs)

def anglesBetweenVecs(vecs0, vecs1, normalization_needed = True):
    if normalization_needed:
        vecs0 = normalizeAll(vecs0)
        vecs1 = normalizeAll(vecs1)

    vals_preds_dot = einsumDot(vecs0, vecs1)
    return np.arccos(np.clip(np.abs(vals_preds_dot), -1, 1))

# The 2acos(abs(dot(q0, q1))) between quaternions is the angle (rad) between the
# two rotations (i.e., the angle of the rotation from one to another). If you
# look at the formula for the scalar component of quaternion multiplication for 
# the desired rotation `q1*q0(^-1)`, it is clearly just dot(q0, q1). So, 
# cos(angle/2) == dot(q0,q1). The abs of that is cos(angle/2) if angle/2 is in
# [-pi/2, pi/2] and is otherwise cos(+-(pi - angle/2)); picture a unit circle to
# see why. Then 2acos is either abs(angle) or abs(2pi - angle); either way, it
# will be the correct angle between 0 and pi.
def anglesBetweenQuats(quats0, quats1):
    # Start by finding the angle between quats when they're interpreted as vec4s
    half_angle = anglesBetweenVecs(quats0, quats1, False)
    return half_angle + half_angle

def quatSlerp(quats0, quats1, t, zeroAngleThresh: float = 0.0001):
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
        t_reshape = t[pos_angle_inds].reshape(pos_angles.shape)

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

# Angles returned should be in the 0-PI range.
def axisAnglesFromQuats(quatVals):
    # By the same logic as in the function where we find the angles between
    # rotations represented by pairs of quaternions, we will take the abs of
    # the cos of the halfangle to extract the one in the 0-pi range.
    halfAngles = np.arccos(np.clip(np.abs(quatVals[..., 0:1]), -1, 1))
    angles = halfAngles + halfAngles
    axes = safelyNormalizeArray(quatVals[..., 1:], np.sin(halfAngles))
    return axes, angles

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
    xs, ys, zs = unitAxes.reshape(-1, 3).transpose()
    _0s = np.zeros(len(angles))
    
    skeweds = np.moveaxis(np.array([
        [_0s, -zs, ys],
        [zs, _0s, -xs],
        [-ys, xs, _0s]
    ]), -1, 0)
    idens = np.repeat([np.eye(3)], len(angles), axis=0)
    sin_part = scalarsMatsMul(np.sin(angles), skeweds)
    skeweds2 = einsumMatMatMul(skeweds, skeweds)
    cos_part = scalarsMatsMul(1.0 - np.cos(angles), skeweds2)
    return idens + sin_part + cos_part

def matsFromScaledAxisAngleArray(scaledAxisAngles):
    angles = np.linalg.norm(scaledAxisAngles, axis = -1)
    axes = np.zeros((len(angles), 3))
    posInds = (angles != 0)
    axes[posInds] = scaledAxisAngles[posInds] / angles[posInds][..., np.newaxis]
    return matsFromAxisAngleArrays(angles, axes)


# Input is assumed to be a numpy array with shape (n,3,3) for some n > 0.
# Return value thus has shape (n,3).
def axisAngleFromMatArray(matrixArray, zeroAngleThresh = 0.0001):
    # TODO: I think the only parts of the code below that do not yet support
    # more than 3 dimensions are the handling of axes for angles of zero.
    # There may not yet be a *benefit* to full support, but noting just in case.
    if matrixArray.ndim > 3:
        raise Exception("Input array of matrices must have (n,3,3) shape!")

    # TODO: Replace instances of np.stack(...), np.concat(...), and similar with
    # np.empty(...) followed by assigning to slices. I'm guessing it'd be more
    # efficient? Less allocating/freeing of memory, right?
    # TODO: Last I checked (2024-10-19), it's fine, but if changed since, see if
    # supporting a arrays with more dims than shape (n,3,3) leads to any 
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
    whereMaxDiag = np.argmax(matrixDiags, axis = -1, keepdims=True)
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
    

    # --------------------------------------------------------------------------
    # Shepperd's Algorithm: Case Where Trace Was Greater.
    # --------------------------------------------------------------------------

    # The angle of rotation about the above axis direction.
    # Outputs of acos are constrained to [0, pi], which will impact later code.
    # Input needs to be clamped to [-1, 1] in case fp precision causes it to
    # exit that interval and, thus, the domain for acos.
    acosInput = np.clip((matrixTraceVals[useTrace] - 1.0)/2.0, -1.0, 1.0)
    angles[useTrace] = np.arccos(acosInput)

    matrixOffDiags = matrixArray[useTrace]


    # Vec3 representing the direction of our rotation axis. Not yet unit length.
    nonUnitAxes[useTrace] = np.stack([
        matrixOffDiags[...,2,1] - matrixOffDiags[...,1,2],
        matrixOffDiags[...,0,2] - matrixOffDiags[...,2,0],
        matrixOffDiags[...,1,0] - matrixOffDiags[...,0,1]
    ], axis=-1) # Axis specification needed for input being a list of matrices.


    # --------------------------------------------------------------------------
    # Shepperd's Algorithm: Case Where a Diagonal Entry Was Greater.
    # --------------------------------------------------------------------------

    i_s = whereMaxDiag[useDiag][:, 0]
    j_s = (i_s + 1) % 3
    k_s = (j_s + 1) % 3

    matsWhereDiagUsed = matrixArray[useDiag]
    # The only way I know of to slice with variable last-axis-indices is to
    # pass in an arange for the first axis; simply using `[:, i_s, j_s]` FAILS!
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
    # Because max-diag-element >= trace, the sqrt input >= 1; no fp concerns.
    ax_i = np.sqrt(sqrtInput)
    nonUnitAxes[useDiag, i_s] = ax_i
    nonUnitAxes[useDiag, j_s] = (Aij + Aji)/ax_i
    nonUnitAxes[useDiag, k_s] = (Aik + Aki)/ax_i

    # Again, we need to clamp/clip in case of fp precision causing problems.
    acosInput = np.clip((Akj - Ajk)/(ax_i + ax_i), -1.0, 1.0)
    halfAngles = np.arccos(acosInput)
    angles[useDiag] = halfAngles + halfAngles # Will be between 0 and 2pi.

    # TODO: Remove this test:
    if np.any(angles < 0):
        raise Exception("I was wrong about all pos angles at this step!")

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
    # The below only works for a flat list of angles! *Might* be neat to
    # consider more dimensions (but efficiency?). I think this could work by:
    #  * `inds = np.tile(np.arange(angles.shape[-1]), angles.shape[:-1] + (1,))`
    #    * An array with same shape as angles, but with aranges on last axis.
    #  * Do the zero-setting and max accumulation similar to before.
    #  * For copying, could use something like `put_along_axis`, but that's
    #    copying way more items than need-be. Could instead maybe do:
    #      * `where = np.argwhere(angles < zeroThresh).transpose()`
    #      * `where[-1] = accumulated_inds[angles < zeroThresh]`
    #      * `axes[angles < zeroThresh] = axes[tuple(where)]`
    #  * Or could flatten earlier and accept weirdness if first angles are 0.
    angleInds = np.arange(len(angles))
    # At this step, all angles SHOULD be positive.
    zeroAngleInds = np.nonzero(angles < zeroAngleThresh)
    angleInds[zeroAngleInds] = 0
    angleInds = np.maximum.accumulate(angleInds, axis = -1)
    nonUnitAxes[zeroAngleInds] = nonUnitAxes[angleInds[zeroAngleInds]]

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
    unitAxes[zeroAngleInds] = unitAxes[angleInds[zeroAngleInds]]
    # If the first rotations are zero-angle, then they have no previous axis to
    # copy. So they'll still be NaNs or whatever, so we should fix those.
    numIssueAxesAtFront = 0
    while numIssueAxesAtFront < len(angles):
        if angles[numIssueAxesAtFront] >= zeroAngleThresh:
            break
        numIssueAxesAtFront += 1
    unitAxes[:numIssueAxesAtFront] = 0.0

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

    # TODO: Remove this after sufficient testing!
    if np.any(np.greater(np.abs(np.diff(angles)), np.pi + 0.00001)):
        raise Exception("Numpification of AA code resulted in angle diff > pi!")
        
    print("Reminder to remove Exception checks and look @ other TODOs.")
        
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

