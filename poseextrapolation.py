import typing

import numpy as np

import posemath as pm
# A good max for forearm length is 30cm, and 20cm for hand length.
# So a decent circle radius max is 50cm = 500mm.
DEFAULT_MAX_CIRC_SQ_RADIUS = 250000 
DEFAULT_MAX_CIRC_ANGLE = np.pi/2.0

class CircularMotionAnalysis:
    def __init__(self, translations: np.ndarray,
                 translation_diffs: np.ndarray = None,
                 backup_preds: np.ndarray = None,
                 max_circ_angle: float = DEFAULT_MAX_CIRC_ANGLE,
                 max_sq_radii: float = DEFAULT_MAX_CIRC_SQ_RADIUS):
        rough_circ_axes_0 = translation_diffs[1:-1]
        rough_circ_axes_1 = -translation_diffs[:-2]
        self.circle_plane_info = pm.getPlaneInfo(
            rough_circ_axes_0, rough_circ_axes_1, translations[:-3]
        )

        circle_axes = self.circle_plane_info.plane_axes

        circle_pts_2D = []
        for j in range(3):
            circle_pts_2D.append(pm.vecsTo2D(
                translations[j:(-3 + j)], *circle_axes
            ))
            
        self.c_centres_2D = pm.circleCentres2D(*circle_pts_2D)

        diffs_from_centres = []
        for j in range(3):
            diffs_from_centres.append(circle_pts_2D[j] - self.c_centres_2D)
    
        self._sq_radii = pm.einsumDot(diffs_from_centres[0], diffs_from_centres[0])
        radii_too_long = self._sq_radii > max_sq_radii 

        self._angle_arr_pair: typing.List[np.ndarray] = []
        c_cosines = []
        for j in range(2):
            c_diff_dot = pm.einsumDot(
                diffs_from_centres[j], diffs_from_centres[j + 1]
            )
            c_cosines.append(
                c_diff_dot/self._sq_radii
            )
            curr_angs = np.arccos(c_cosines[-1])

            # Numpy cross products of 2D vecs treat them like 3D vecs and return 
            # only the z-coordinate of the result (since the rest are 0). We'll look
            # at the sign to determine rotation direction about the circle axis.
            # TODO: Because np.cross of 2D vecs is deprecated, do the cross
            # product myself.
            c_crosses = np.cross(diffs_from_centres[j], diffs_from_centres[j + 1])
            flips = (c_crosses < 0.0)
            curr_angs[flips] = -curr_angs[flips]
            self._angle_arr_pair.append(curr_angs)

        self.first_angles = self._angle_arr_pair[0]
        self.second_angles = self._angle_arr_pair[1]

        prev_angle_sum = self.first_angles + self.second_angles
        prev_angles_too_big = np.abs(prev_angle_sum) > max_circ_angle

        self.invalid_circ_indices = np.logical_or(
            radii_too_long, prev_angles_too_big
        )

        c_pred_angles_base = 1.5 * self.second_angles - 0.5 * self.first_angles

        self.c_pred_angles = np.clip(
            c_pred_angles_base, a_min=-max_circ_angle, a_max=max_circ_angle
        )

        c_cosines_pred = np.cos(self.c_pred_angles)
        c_sines_pred = np.sin(self.c_pred_angles)

        c_trans_preds_2D = pm.rotateBySinCos2D(
            diffs_from_centres[2], c_cosines_pred, c_sines_pred
        )

        c_only_preds = pm.vecsTo3DUsingPlaneInfo(
            self.c_centres_2D + c_trans_preds_2D, self.circle_plane_info
        )
        self.c_trans_preds: np.ndarray = c_only_preds

        if backup_preds is not None:
            pred_len_diff = len(backup_preds) - len(c_only_preds)
            if pred_len_diff < 0:
                raise NotImplementedError(
                    "Handling shorter backup preds not yet supported!"
                )
            c_only_preds[self.invalid_circ_indices] = \
                backup_preds[pred_len_diff:][self.invalid_circ_indices]
            self.c_trans_preds = pm.replaceAtEnd(
                backup_preds, c_only_preds, pred_len_diff
            )

        # Only create if need-be later.
        self._c_centres_3D: np.ndarray = None 
        self._radii: np.ndarray = None 
    
    def getRadii(self):
        if self._radii is None:
            self._radii = np.sqrt(self._sq_radii)
        return self._radii

    def getCentres3D(self):
        if self._c_centres_3D is None:
            self._c_centres_3D = pm.vecsTo3DUsingPlaneInfo(
                self.c_centres_2D, self.circle_plane_info
            )
        return self._c_centres_3D

    # TODO: I still need to make sure this doesn't have any bugs in it.
    def isMotionStillCircular(self, next_translations: np.ndarray,
                              err_radius_ratio_thresh: float):

        self.getCentres3D() # Makes sure they are created and not None.

        vecs_from_c_centres = next_translations - self._c_centres_3D[:-1]
        _, inplane_vecs_from_c_centres = pm.parallelAndOrthoParts(
            vecs_from_c_centres, self.circle_plane_info.normals[:-1], True
        )

        prev_radii = self.getRadii()[:-1]
        closest_circle_pts = self._c_centres_3D[:-1] + pm.scalarsVecsMul(
            prev_radii, pm.normalizeAll(inplane_vecs_from_c_centres)
        )
        vecs_from_circ = next_translations - closest_circle_pts
        dist_from_circ = np.linalg.norm(vecs_from_circ, axis=-1)

        ratio_from_circ = dist_from_circ/prev_radii
        print("TODO: prop circ params from prev when successful if new is too dissimilar?")


        c_over_thresh = ratio_from_circ > err_radius_ratio_thresh
        c_invalid = self.invalid_circ_indices[1:]
        last_c_invalid = self.invalid_circ_indices[:-1]
        # centre_big_move = c_centre_diff/prev_radii > 0.1 # Arbitrary threshold
        # radii_big_change = radii_diffs / prev_radii > 0.2 # Arbitrary threshold

        c_bool_list = [c_over_thresh, c_invalid, last_c_invalid]
        non_circ_bool_inds = c_bool_list[0]
        for cbools in c_bool_list[1:]:
            non_circ_bool_inds = np.logical_or(non_circ_bool_inds, cbools)
        
        return non_circ_bool_inds

    def vec6CircleDists(self):
        scaled_circle_normals = pm.scalarsVecsMul(
            self.getRadii(), self.circle_plane_info.normals
        )
        circle_vec6_info = np.concatenate([
            scaled_circle_normals, self.getCentres3D()
        ], axis=1)
        return np.linalg.norm(np.diff(circle_vec6_info, 1, axis=0), axis=-1)
# End of class

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
