import typing

import numpy as np
from numpy.typing import NDArray

import posemath as pm
# A good max for forearm length is 30cm, and 20cm for hand length.
# So a decent circle radius max is 50cm = 500mm.
DEFAULT_MAX_CIRC_SQ_RADIUS = 250000 
DEFAULT_MAX_CIRC_ANGLE = np.pi/2.0

class CircularMotionAnalysis:
    class Closenesses(typing.NamedTuple):
        non_circ_bool_inds: NDArray
        dists: NDArray[np.float64]
        dist_radius_ratios: NDArray[np.float64]

    def __init__(self, translations: np.ndarray,
                 translation_diffs: np.ndarray = None,
                 backup_preds: np.ndarray = None,
                 max_circ_angle: float = DEFAULT_MAX_CIRC_ANGLE,
                 max_sq_radii: float = DEFAULT_MAX_CIRC_SQ_RADIUS):
        
        self.backup_preds = backup_preds
        self.max_circ_angle = max_circ_angle

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

        self.diffs_from_centres: typing.List[np.ndarray] = []
        for j in range(3):
            self.diffs_from_centres.append(circle_pts_2D[j] - self.c_centres_2D)
    
        self._sq_radii = pm.einsumDot(
            self.diffs_from_centres[0], self.diffs_from_centres[0]
        )
        radii_too_long = self._sq_radii > max_sq_radii 

        self._angle_arr_pair: typing.List[np.ndarray] = []
        c_cosines = []
        for j in range(2):
            c_diff_dot = pm.einsumDot(
                self.diffs_from_centres[j], self.diffs_from_centres[j + 1]
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
            c_crosses = np.cross(
                self.diffs_from_centres[j], self.diffs_from_centres[j + 1]
            )
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

        pred_angles_vel_d1 = self.second_angles

        # x2 + 1.5 (x2 - x1) - 0.5 (x1 - x0)
        # x2 + 1.5x2 - 2 x1 + 0.5x0
        # x2 + (x2 - x1) + 0.5 (x2 - 2x1 + x0)
        pred_angles_vel_d2 = 1.5 * self.second_angles - 0.5 * self.first_angles

        # x2 + 2 (x2 - x1) - (x1 - x0)
        # x2 + 2x2 - 3x1 + x0
        pred_angles_acc = 2 * self.second_angles - self.first_angles

        self.disp_angles_vel_deg1 = self.clipAngsToMax(pred_angles_vel_d1)
        self.disp_angles_vel_deg2 = self.clipAngsToMax(pred_angles_vel_d2)
        self.disp_angles_acc = self.clipAngsToMax(pred_angles_acc)

        self.vel_deg1_preds_2D = self.anglesToPreds2D(self.disp_angles_vel_deg1)
        self.vel_deg2_preds_2D = self.anglesToPreds2D(self.disp_angles_vel_deg2)
        self.acc_preds_2D = self.anglesToPreds2D(self.disp_angles_acc)

        self.vel_deg1_preds_3D = self.toPreds3D(self.vel_deg1_preds_2D)
        self.vel_deg2_preds_3D = self.toPreds3D(self.vel_deg2_preds_2D)
        self.acc_preds_3D = self.toPreds3D(self.acc_preds_2D)

        # Only create if need-be later.
        self._c_centres_3D: np.ndarray = None 
        self._radii: np.ndarray = None 
    
    def clipAngsToMax(self, angs):
        return np.clip(
            angs, a_min=-self.max_circ_angle, a_max=self.max_circ_angle
        )

    def anglesToPreds2D(self, new_ang_displacement):

        c_cosines_pred = np.cos(new_ang_displacement)
        c_sines_pred = np.sin(new_ang_displacement)

        c_trans_preds_2D = pm.rotateBySinCos2D(
            self.diffs_from_centres[2], c_cosines_pred, c_sines_pred
        )

        return c_trans_preds_2D
    
    def toPreds3D(self, preds_2D):

        c_only_preds = pm.vecsTo3DUsingPlaneInfo(
            self.c_centres_2D + preds_2D, self.circle_plane_info
        )
        ret_trans_preds: np.ndarray = c_only_preds

        if self.backup_preds is not None:
            pred_len_diff = len(self.backup_preds) - len(c_only_preds)
            if pred_len_diff < 0:
                raise NotImplementedError(
                    "Handling shorter backup preds not yet supported!"
                )
            c_only_preds[self.invalid_circ_indices] = \
                self.backup_preds[pred_len_diff:][self.invalid_circ_indices]
            ret_trans_preds = pm.replaceAtEnd(
                self.backup_preds, c_only_preds, pred_len_diff
            )

        return ret_trans_preds
    
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
                              err_radius_ratio_thresh: float,
                              dist_when_not_circ = None):

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
        # print("TODO: prop circ params from prev when successful if new is too dissimilar?")


        c_over_thresh = ratio_from_circ > err_radius_ratio_thresh
        c_invalid = self.invalid_circ_indices[1:]
        last_c_invalid = self.invalid_circ_indices[:-1]
        # centre_big_move = c_centre_diff/prev_radii > 0.1 # Arbitrary threshold
        # radii_big_change = radii_diffs / prev_radii > 0.2 # Arbitrary threshold

        c_bool_list = [c_over_thresh, c_invalid, last_c_invalid]
        non_circ_bool_inds = c_bool_list[0]
        for cbools in c_bool_list[1:]:
            non_circ_bool_inds = np.logical_or(non_circ_bool_inds, cbools)

        if dist_when_not_circ is not None:
            dist_from_circ[non_circ_bool_inds] = dist_when_not_circ
            ratio_from_circ[non_circ_bool_inds] = dist_when_not_circ
        
        return CircularMotionAnalysis.Closenesses(
            non_circ_bool_inds, dist_from_circ, ratio_from_circ
        )

    def vec6CircleDists(self):
        scaled_circle_normals = pm.scalarsVecsMul(
            self.getRadii(), self.circle_plane_info.normals
        )
        circle_vec6_info = np.concatenate([
            scaled_circle_normals, self.getCentres3D()
        ], axis=1)
        return np.linalg.norm(np.diff(circle_vec6_info, 1, axis=0), axis=-1)
# End of class
