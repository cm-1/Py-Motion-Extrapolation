from collections import namedtuple

import numpy as np

import posemath as pm


CirclesInfo = namedtuple(
    "CirclesInfo", ['predictions', 'angle_pairs', 'sq_radii', 'discarded_inds', 'circle_plane_info']
)

def circle_preds(translations: np.ndarray, translation_diffs: np.ndarray = None,
                 backup_preds: np.ndarray = None, ):
    rough_circ_axes_0 = translation_diffs[1:-1]
    rough_circ_axes_1 = -translation_diffs[:-2]
    circle_plane_info = pm.getPlaneInfo(
        rough_circ_axes_0, rough_circ_axes_1, translations[:-3]
    )

    circle_axes = circle_plane_info.plane_axes

    circle_pts_2D = []
    for j in range(3):
        circle_pts_2D.append(pm.vecsTo2D(
            translations[j:(-3 + j)], *circle_axes
        ))
        
    circle_centres = pm.circleCentres2D(*circle_pts_2D)

    diffs_from_centres = []
    for j in range(3):
        diffs_from_centres.append(circle_pts_2D[j] - circle_centres)
   
    sq_radii = pm.einsumDot(diffs_from_centres[0], diffs_from_centres[0])
    # A good max for forearm length is 30cm, and 20cm for hand length.
    # So a decent circle radius max is 50cm = 500mm.
    radii_too_long = sq_radii > 250000 

    c_angles = []
    c_cosines = []
    for j in range(2):
        c_diff_dot = pm.einsumDot(
            diffs_from_centres[j], diffs_from_centres[j + 1]
        )
        c_cosines.append(
            c_diff_dot/sq_radii
        )
        c_angles.append(np.arccos(c_cosines[-1]))

        # Numpy cross products of 2D vecs treat them like 3D vecs and return 
        # only the z-coordinate of the result (since the rest are 0). We'll look
        # at the sign to determine rotation direction about the circle axis.
        c_crosses = np.cross(diffs_from_centres[j], diffs_from_centres[j + 1])
        c_crosses_flip = (c_crosses < 0.0)
        c_angles[-1][c_crosses_flip] = -c_angles[-1][c_crosses_flip]

    MAX_CIRC_ANGLE = np.pi/2.0
    prev_angle_sum = c_angles[0] + c_angles[1]
    prev_angles_too_big = np.abs(prev_angle_sum) > MAX_CIRC_ANGLE

    invalid_circ_indices = np.logical_or(radii_too_long, prev_angles_too_big)

    c_pred_angles_base = 1.5 * c_angles[1] - 0.5 * c_angles[0]

    c_pred_angles = np.clip(
        c_pred_angles_base, a_min=-MAX_CIRC_ANGLE, a_max=MAX_CIRC_ANGLE
    )

    c_cosines_pred = np.cos(c_pred_angles)
    c_sines_pred = np.sin(c_pred_angles)

    c_trans_preds_2D = pm.rotateBySinCos2D(
        diffs_from_centres[2], c_cosines_pred, c_sines_pred
    )

    c_trans_preds = None
    c_only_preds = pm.vecsTo3DUsingPlaneInfo(
        circle_centres + c_trans_preds_2D, circle_plane_info
    )
    if backup_preds is None:
        c_trans_preds = c_only_preds
    else:
        pred_len_diff = len(backup_preds) - len(c_only_preds)
        c_only_preds[invalid_circ_indices] = backup_preds[pred_len_diff:][invalid_circ_indices]
        c_trans_preds = pm.replaceAtEnd(c_only_preds, backup_preds, pred_len_diff)

    return CirclesInfo(c_trans_preds, c_angles, sq_radii, invalid_circ_indices, circle_plane_info)

def camObjConstAngularVelPreds(known_rotations_qs: np.ndarray, backup_predictions = None):

    num_preds = len(known_rotations_qs) - 2 # How many predictions we'll create.
    C_quats = np.empty((num_preds, 4))
    
    # We're going to assume two axes of constant angular velocity operating in
    # different spaces. One specific example of this, which we'll refer to from
    # now on, is the camera and object both rotating in body space with constant
    # angular velocity.
    # Let M_k and C_k be the world orientations of the object and camera,
    # respectively, at frame k. Let B_k = C_k^T M_k, and let C and M denote the
    # per-frame body-space rotations s.t. C_{k+1} = C_k C and M_{k+1} = M_k M.
    # We can observe that B_{k-n} = C^n B_k (M^T)^n for n >= 0. This means that
    # M^T = B_k^T C^T B_{k-1}, meaning B_{k-2} = C B_{k-1} B_k^T C^T B_{k-1}.
    # Rearranged, B_{k-2} B_{k-1}^T = C B_{k-1} B_k^T C^T

    # Now, how do we find C? An exact solution might not exist; how do we find a
    # close one? Well, R Q R^T for any rotation matrices Q and R yields a new
    # rotation which rotates Q's axis by R. So, we need to find the C that
    # rotates the rotation axes appropriately.
    # There are many such rotations; to narrow things down, if there are more
    # than three previous orientations, we can observe that
    # B_{k-3} = C B_{k-2} B_k^T C^T B_{k-1}. So we also want C to take the axis
    # of B_{k-2}B_k^T to B_{k-3}B_{k-1}^T. We'll prioritize the alignment of the
    # B_{k-1}B_k axis to the B_{k-2}B_{k-1}^T axis, and then we'll use this 
    # other goal to narrow down the equally-good options. I don't have time to
    # type up the full justification right now, but essentially, we want the
    # rotation that both takes the B_{k-1}B_k axis to the B_{k-2}B_{k-1}^T axis
    # and would rotate the B_{k-2}B_k^T axis into the same plane spanned by
    # the B_{k-2}B_{k-1}^T axis and the B_{k-3}B_{k-1}^T axis. Which means the
    # cross of the axes to be rotated will be rotated into the cross of the
    # target axes. If we know that Rx = y and Rv = w for four vecs v, w, x, y,
    # then the rotation axis is parallel to cross(x-y, v-w). 
    rev_rotations_qs = pm.conjugateQuats(known_rotations_qs)
    rotation_q_diffs = pm.multiplyQuatLists(
        known_rotations_qs[1:], rev_rotations_qs[:-1]
    )
    rotation_q_step2_diffs = pm.multiplyQuatLists(
        rotation_q_diffs[1:], rotation_q_diffs[:-1]
    )

    step1_diff_aas = pm.axisAnglesFromQuats(rotation_q_diffs)
    step1_diff_unit_axes, step1_diff_angles = step1_diff_aas
    step2_diff_unit_axes, _ = pm.axisAnglesFromQuats(rotation_q_step2_diffs)
    unnormed_plane_axes = np.cross(
        step1_diff_unit_axes[1:], step2_diff_unit_axes
    )
    
    # So, as a small update to the above: if the axes have an angle of over 90
    # degrees, I'm going to instead rotate to the negative axes. Because if, for
    # example, your two rotations were [0.1, 0, 0] and [-0.2, 0, 0], it'd be
    # ridiculous to assume you have some angular velocity of 180deg/s in play.
    # But because this would in some cases yield extremely large differences
    # between the scaled axes, I'll have to do a thresholding after.
    aligned_step1_diff_axes = pm.flipObtuseAxes(step1_diff_unit_axes)
    plane_ax_norms = np.linalg.norm(unnormed_plane_axes, axis=-1, keepdims=True)
    zero_plane_ax_inds = (plane_ax_norms <= 0.0).flatten()
    plane_axes = np.empty(unnormed_plane_axes.shape)
    pos_plane_ax_inds = np.invert(zero_plane_ax_inds)
    plane_axes[pos_plane_ax_inds] = unnormed_plane_axes[pos_plane_ax_inds] / plane_ax_norms[pos_plane_ax_inds]
    plane_axes[zero_plane_ax_inds] = unnormed_plane_axes[zero_plane_ax_inds]
    aligned_plane_axes = pm.flipObtuseAxes(plane_axes)

    step1_disps = np.diff(aligned_step1_diff_axes[1:], 1, axis=0)
    plane_disps = np.diff(aligned_plane_axes, 1, axis=0)
    plane_disps[zero_plane_ax_inds[1:]] = 0.0
    plane_disps[zero_plane_ax_inds[:-1]] = 0.0
    
    C_axes = np.empty((num_preds, 3))
    C_angles = np.empty(num_preds)
    C_axes[0] = 0.0 # Placeholder, since we'll use a different approach for [0].
    C_axes[1:] = np.cross(step1_disps, plane_disps)
    # Now, if any of our cross products at any step are zero, the above won't
    # work. In that case, and also when we only have three prior frames to look 
    # at, we'll choose the smallest rotation taking the B_{k-1}B_k axis to the
    # B_{k-2}B_{k-1}^T axis.
    sq_C_ax_lens = pm.einsumDot(C_axes, C_axes)
    cross_0_inds = (sq_C_ax_lens <= 0.00001)
    valid_inds = np.invert(cross_0_inds)
    
    C_axes[valid_inds] = pm.normalizeAll(C_axes[valid_inds])

    _, prev_ortho_unnormed= pm.parallelAndOrthoParts(
        aligned_step1_diff_axes[:-1][valid_inds], C_axes[valid_inds], True
    )
    _, next_ortho_unnormed = pm.parallelAndOrthoParts(
        aligned_step1_diff_axes[1:][valid_inds], C_axes[valid_inds], True
    )
    prev_ortho_unit = pm.normalizeAll(prev_ortho_unnormed)
    next_ortho_unit = pm.normalizeAll(next_ortho_unnormed)

    ortho_dots = pm.einsumDot(prev_ortho_unit, next_ortho_unit)
    C_angles[valid_inds] = np.arccos(np.clip(ortho_dots, -1.0, 1.0))

    km1_km2_axes = aligned_step1_diff_axes[:-1]
    k_km1_axes = aligned_step1_diff_axes[1:]
    step1_crosses = np.cross(k_km1_axes, km1_km2_axes)
    step1_cross_norms = np.linalg.norm(
        step1_crosses[cross_0_inds], axis=-1, keepdims=True
    )
    can_norm_inds = np.full(cross_0_inds.shape, False)
    zero_step1_norm_ind_subset = (step1_cross_norms >= 0.00001).flatten()
    can_norm_inds[cross_0_inds] = zero_step1_norm_ind_subset
    
    inds_to_div = np.logical_and(cross_0_inds, can_norm_inds)
    step1_crosses[inds_to_div] /= step1_cross_norms[zero_step1_norm_ind_subset]
    C_axes[cross_0_inds] = step1_crosses[cross_0_inds]
    C_angles[cross_0_inds] = np.arccos(np.clip(pm.einsumDot(
        k_km1_axes[cross_0_inds], km1_km2_axes[cross_0_inds]
    ), -1.0, 1.0))

    obtuse_inds = np.full(valid_inds.shape, False)
    obtuse_inds[valid_inds] = np.signbit(pm.einsumDot(
        step1_crosses[valid_inds], C_axes[valid_inds]
    ))

    flipped_ax_inds = np.logical_and(valid_inds, obtuse_inds)
    C_angles[flipped_ax_inds] = -C_angles[flipped_ax_inds]

    C_quats = pm.quatsFromAxisAngles(C_axes, C_angles)



    # Our next rotation will be C^T B_k M
    Ct_quats = pm.conjugateQuats(C_quats)
    B_k_quats = known_rotations_qs[2:]
    B_km1_t_quats = rev_rotations_qs[1:-1]
    M_quats = pm.multiplyQuatLists(
        B_km1_t_quats, pm.multiplyQuatLists(C_quats, B_k_quats)
    )
    retVal = pm.multiplyQuatLists(
        Ct_quats, pm.multiplyQuatLists(B_k_quats, M_quats)
    )

    # Below is used for debugging if I'm passing in an input that should have
    # a "perfect" solution available:
    # C_test = pm.rotateVecsByQuats(C_quats[1:], step2_diff_unit_axes[1:])
    # if np.abs(C_test - step2_diff_unit_axes[:-1]).max() > 0.0001:
    #     print("OK, messed something up! Num valid inds:", valid_inds.sum())

    if backup_predictions is not None:
        C_eval_pt1 = pm.multiplyQuatLists(C_quats, rotation_q_diffs[1:])
        C_eval_qs = pm.multiplyQuatLists(C_eval_pt1, Ct_quats)
        C_eval_errs = pm.anglesBetweenQuats(C_eval_qs, rotation_q_diffs[:-1])
        errs_too_big = (C_eval_errs > np.pi)
        ind_pad = len(backup_predictions) - num_preds
        retVal[errs_too_big] = backup_predictions[ind_pad:][errs_too_big]


    return Ct_quats, M_quats, retVal
