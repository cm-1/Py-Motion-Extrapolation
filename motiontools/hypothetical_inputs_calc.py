import typing

import numpy as np
from numpy.typing import NDArray

import posemath as pm

from motiontools.derivative_collection import DerivativeCollection

from motiontools.key_and_vec_specs import (
    MOTION_DATA, SpecifiedMotionData, ANG_OR_MAG, OTHER_DIRECTION,
    ALL_RELATIVE_VECTORS,
    MOTION_DATA_KEY_TYPE
)


class HypotheticalInputsForNN:
    def __init__(self, x0_through_4:NDArray, rmats0_through_5: NDArray,
                 step: int, ref_keys: typing.List[MOTION_DATA_KEY_TYPE]):
        # Attributes we set now.
        self.step = step
        self.x0_through_4 = x0_through_4
        self.ref_keys = ref_keys

        # Some calculations we'll reuse that we can precalculate here.
        # ---
        # We *sorta* have "4" sets of "xyz" multipliers.
        self._jav_muls = np.empty((4, 3))
        # Jerk through crackle each have 3 components we must multiply by a
        # respective power of the step assuming it's also the next "delta T".
        self._jav_muls[1:] = self.step ** np.arange(3, 6).reshape(3, 1)
        # Then velocity and acceleration are special because they only have 1
        # and 2 multipliers, respectively.
        self._jav_muls = self._jav_muls.flatten()
        self._jav_muls[0] = self.step
        self._jav_muls[1:3] = self.step ** 2


        # Attributes we'll set in the following function calls.
        self.prev_pds: DerivativeCollection
        self.all_rds: DerivativeCollection
        self.rel_ax_order: typing.Tuple[MOTION_DATA, ...]
        self._orig_key_order: typing.Sequence[MOTION_DATA_KEY_TYPE]
        self.to_nn_permut: NDArray
 
        # Function calls to continue setting up attributes.
        self._keyPermutation()
        self.updatePrecalcs(x0_through_4, rmats0_through_5)
    
    @staticmethod
    def _generated_key_order():
        MD = MOTION_DATA
        AM = ANG_OR_MAG
        kp_t = [MD.TIMESTEP]
        v3s = (
            MD.VEL_DEG1_VEC3, MD.ACC_VEC3, MD.JERK_VEC3, MD.JERK_ERR_VEC3,
            MD.CRACKLE_VEC3, MD.ROTATION_VEC3, MD.ROT_ACC_VEC3, MD.ROT_JERK_VEC3
        )
        rel_v3s = ( # All v3s except crackle
            MD.VEL_DEG1_VEC3, MD.ACC_VEC3, MD.JERK_VEC3, MD.JERK_ERR_VEC3,
            MD.ROTATION_VEC3, MD.ROT_ACC_VEC3, MD.ROT_JERK_VEC3
        )
        rel_axes = [
            ra for ra in ALL_RELATIVE_VECTORS if ra != MD.VEL_DEG2_VEC3
        ]
        kp_pd = []
        for bc in v3s:
            for ax in rel_axes:
                a = AM.MAG_DOT if isinstance(ax, MOTION_DATA) else AM.MAG_PROJ
                kp_pd.append(SpecifiedMotionData(bc, ax, a, False, True))
        kp_pp = [
            SpecifiedMotionData(bc, ax, AM.MAG_PROJ, False, True)
            for bc in v3s for ax in rel_v3s
        ]
        kp_m = [
            SpecifiedMotionData(bc, bc, AM.MAG_PROJ, False, False) for bc in v3s
        ]
        kp_nd = []
        for i in range(1, len(v3s)):
            for j in range(i):
                base_vec = v3s[i]
                rel_vec = v3s[j]
                if rel_vec not in ALL_RELATIVE_VECTORS:
                    base_vec = v3s[j]
                    rel_vec = v3s[i]
                kp_nd.append(SpecifiedMotionData(
                    base_vec, rel_vec, AM.MAG_DOT, False, False
                ))
        kp_np = [
            SpecifiedMotionData(bc, ra, AM.MAG_PROJ, False, False)
            for ra in rel_v3s for bc in v3s if ra != bc
        ]

        kp_np2 = []
        kp_np2.append(SpecifiedMotionData(
            MD.ACC_VEC3, OTHER_DIRECTION.ACC_ORTHO_DEG1,
            AM.MAG_PROJ, False, False
        ))
        kp_np2 += [
            SpecifiedMotionData(bc, ax, AM.MAG_PROJ, False, False)
            for bc in v3s[2:] for ax in OTHER_DIRECTION
        ]

        kp = kp_t + kp_pd + kp_pp + kp_m + kp_nd + kp_np + kp_np2
        return kp
        
    def _keyPermutation(self):
        kp = self._generated_key_order()
        self._orig_key_order = kp
        self.to_nn_permut = np.asarray([kp.index(k) for k in self.ref_keys])
        return
    
    @staticmethod
    def _replaceAtInd(source_arr: NDArray, amt: int, arr_to_mod: NDArray):
        arr_to_mod[0, :] = source_arr[amt]

    def updatePrecalcs(self, x0_through_4:NDArray, rmats0_through_5: NDArray):
        assert len(x0_through_4) == 5, "Must give exactly 5 fixed points!"
        assert len(rmats0_through_5) == 6, "Must give exactly 6 axis angles!"
        assert max(x0_through_4.ndim, rmats0_through_5.ndim - 1) <= 3, \
        (
            "Can only have shape (5, 3) or (5, n, 3) for x0_through_x4 vecs, "
            "(6, 3, 3) or (6, n, 3, 3) for rmats0_through_5 mats!"
        )
        # Yes, technically I don't check for all possible shape issues and the
        # above assert won't catch *all* things that violate the conditions
        # stated in its str message, but for efficiency's sake I only really
        # want to test mistakes I think are *likely* to happen.
        self.x0_through_4 = x0_through_4

        self.prev_pds = DerivativeCollection(
            np.diff(x0_through_4, 1, axis=0), 5, self.step
        )
        
        rev_mats = np.swapaxes(rmats0_through_5[:-1], -2, -1)
        angvel_mats = pm.einsumMatMatMul(rmats0_through_5[1:], rev_mats)
        angvel_mats_am = np.swapaxes(angvel_mats, 0, -3)
        angvel_aas_am = pm.axisAngleFromMatArray(angvel_mats_am)
        angvel_aas = np.swapaxes(angvel_aas_am, 0, -2)

        self.all_rds = DerivativeCollection(angvel_aas, 3, self.step)

        all_vels = self.prev_pds.velocities
        prev_vel = all_vels[-1]
        prev_acc = self.prev_pds.accelerations[-1]
        prev_jerk = self.prev_pds.jerks[-1]
        prev_snap = self.prev_pds.snaps[-1]
        self.prev_vel = prev_vel
        self.prev_acc = prev_acc
        self.prev_jerk = prev_jerk
        self.prev_snap = prev_snap

        vel_mags = np.linalg.norm(all_vels, axis=-1)
        last_nonzero_vels = all_vels[-1:]
        pm.handleCondsAtStart(
            vel_mags[::-1] == 0.0, all_vels[::-1], self._replaceAtInd,
            cond_bools_time_axis=0, arr_to_mod=last_nonzero_vels
        )
        
        self.last_nonzero_unit_vels = pm.normalizeAll(last_nonzero_vels[0])


        prev_ang_vel = self.all_rds.velocities[-2]
        prev_ang_acc = self.all_rds.accelerations[-2]
        prev_ang_jerk = self.all_rds.jerks[-2]

        self.new_ang_vel = self.all_rds.velocities[-1]
        self.new_ang_acc = self.all_rds.accelerations[-1]
        self.new_ang_jerk = self.all_rds.jerks[-1]

        self.rel_ax_order = tuple(
            k for k in ALL_RELATIVE_VECTORS if k != MOTION_DATA.VEL_DEG2_VEC3
        )

        prev_ortho_mags, prev_ortho_dirs = pm.getOrthonormalFrames(
            True, self.last_nonzero_unit_vels, prev_acc,
            vecs0_are_unit_len=True, set_zeros_to_zero=True
        )
        a0_is_0 = np.asarray(prev_ortho_mags[2] == 0.0)
        if np.any(a0_is_0):
            _, j_orth = pm.parallelAndOrthoParts(
                prev_jerk[a0_is_0], prev_ortho_dirs[0][a0_is_0], True
            )
            prev_ortho_dirs[..., 2, :][a0_is_0] = pm.normalizeAll(j_orth[a0_is_0])

        # print("hyp pre:", prev_ortho_dirs)

        self.prev_relative_vecs = np.stack(
            (
                prev_vel, prev_acc, prev_jerk, prev_snap,
                prev_ang_vel, prev_ang_acc, prev_ang_jerk,
                prev_ortho_dirs[..., 1, :], prev_ortho_dirs[..., 2, :]
            ), axis=0
        )

        # Because the scales of the prev_ortho_dirs are just 1, we exclude these
        # from the array, hence the "-2" instances below.
        num_scale_types = len(self.rel_ax_order) - 2

        inner_prev_vecs_shape = self.prev_relative_vecs.shape[1:-1]
        prev_scale_shape = (num_scale_types, ) + inner_prev_vecs_shape #+ (1, )

        self.prev_relative_scales = np.empty(prev_scale_shape)
        self.prev_relative_scales[0] = np.asarray(vel_mags[-1])#[..., np.newaxis]
        self.prev_relative_scales[1] = np.sqrt(
            prev_ortho_mags[1]**2 + prev_ortho_mags[2]**2
        )#[..., np.newaxis]
        self.prev_relative_scales[2:] = np.linalg.norm(
            self.prev_relative_vecs[2:-2], axis=-1#, keepdims=True
        )
        if self.prev_relative_scales.ndim == 1:
            self.prev_relative_scales = self.prev_relative_scales[..., np.newaxis]
        self.prev_relative_scales_nonzero = (self.prev_relative_scales != 0.0)

    def _stepDiv(self, vecs: NDArray):
        return vecs if self.step == 1 else (vecs / self.step)

    def getHypotheticalCalcs(self, all_x5_choices: NDArray):
        assert all_x5_choices.ndim == 2 and all_x5_choices.shape[1] == 3, \
        "The input of x5 choices must have shape (n, 3)!"

        # Calculate velocities, speeds, and indices where speed is 0.
        vels: NDArray = self._stepDiv(all_x5_choices - self.x0_through_4[-1])
        vel_mags: NDArray = np.linalg.norm(vels, axis=-1, keepdims=True)
        vel_mag_is_0 = np.where((vel_mags == 0.0).flatten())[0]
        
        # Safely obtain unit velocities, 
        safediv_vel_mags = np.copy(vel_mags)
        safediv_vel_mags[vel_mag_is_0] = 1.0 # Avoid 0 div; overwritten later.
        unit_vels = vels / safediv_vel_mags
        last_nz_uvels = self.last_nonzero_unit_vels
        if last_nz_uvels.ndim > 1:
            last_nz_uvels = last_nz_uvels[vel_mag_is_0]
        unit_vels[vel_mag_is_0] = last_nz_uvels
        
        accs = self._stepDiv(vels - self.prev_vel)
        jerks = self._stepDiv(accs - self.prev_acc)
        snaps = self._stepDiv(jerks - self.prev_jerk)
        crackles = self._stepDiv(snaps - self.prev_snap)

        full_shape = vels.shape
        all_curr_vecs = np.stack(
            (
                vels, accs, jerks, snaps, crackles,
                np.broadcast_to(self.new_ang_vel, full_shape),
                np.broadcast_to(self.new_ang_acc, full_shape),
                np.broadcast_to(self.new_ang_jerk, full_shape),
            ), axis=0
        )

        pr_str = 'bk' if self.prev_relative_vecs.ndim == 2 else 'bik'
        all_dots_with_prev = typing.cast(NDArray, np.einsum(
            'aik,' + pr_str + '->abi', all_curr_vecs, self.prev_relative_vecs
        ))

        all_proj_with_prev = pm.safeDivideElseZero(
            all_dots_with_prev[:, :-2], self.prev_relative_scales,
            self.prev_relative_scales_nonzero
        )

        
        (_, a_proj_v, a_ortho_v), curr_ortho_mats = pm.getOrthonormalFrames(
            True, unit_vels, accs, vecs0_are_unit_len=True,
            set_zeros_to_zero=True
        )

        a0_is_0 = np.where(a_ortho_v == 0.0)[0]
        _, j_orth = pm.parallelAndOrthoParts(
            jerks[a0_is_0], curr_ortho_mats[a0_is_0, 0], True
        )
        curr_ortho_mats[a0_is_0, 2] = pm.normalizeAll(j_orth)
        # print("hyp curr:", curr_ortho_mats)

        # For the various dot products between *current* frame values, we'll
        # keep track of them in a triangular matrix so that we don't duplicate
        # dot products.
        # We have 8 non-unit vec3s (vel..crackle and rot vel..jerk), but we'll 
        # handle the unit-length orthogonal relative axes separately, so our
        # dimensions will be 7x7x...
        n_vec_kinds = len(all_curr_vecs)
        n_other_vec_kinds = n_vec_kinds - 1
        n_ins = len(all_x5_choices)
        tri_dots = np.empty((n_other_vec_kinds, n_other_vec_kinds, n_ins))
        # Rather than keep the self-dots in the diagonal, I think it may be
        # slightly more efficient to keep them separate; we'll need to divide
        # by them later, so this prevents the need for diag indexing or copies.
        non_vel_mags = np.empty((n_other_vec_kinds, n_ins))
        # We'll now copy in the magnitudes calculated during the orthonormal
        # frame calculations.
        # First, the dots with velocity:
        tri_dots[0, 0] = vel_mags.flatten() * a_proj_v # v*a
        # tri_dots[1, 0] = v_mags * curr_ortho_mags[3] # v*j
        # Then, the dots with acceleration:
        accs_2D = np.stack((a_proj_v, a_ortho_v), axis=-1)
        non_vel_mags[0] = np.linalg.norm(accs_2D, axis=-1) # sqrt(a*a)
        # tri_dots[1, 1] = pm.einsumDot(accs_2D, np.transpose(curr_ortho_mags[3:5])) # a*j (2D)
        # The remaining dots have no precalculations to take advantage of:
        non_vel_mags[1:] = np.linalg.norm(all_curr_vecs[2:], axis=-1)
        for i in range(2, n_vec_kinds):
            tri_dots[i-1, :i] = np.einsum(
                'jk,ijk->ij', all_curr_vecs[i], all_curr_vecs[:i]
            )

        # For the projections, we don't worry about "duplicates" since there are
        # none: projecting jerk onto velocity is different from vice versa.
        # However, we have the diagonal separated out already, so we can
        # exclude that.
        acc_vel_proj = a_proj_v.copy()
        acc_vel_proj[vel_mag_is_0] = 0.0
        vel_projs = [[acc_vel_proj]] #, curr_ortho_mags[3]]]
        other_projs_on_vel = tri_dots[1:, 0] / safediv_vel_mags.flatten()
        other_projs_on_vel[:, vel_mag_is_0] = 0.0

        vel_projs.append(other_projs_on_vel)

        CIND = 4 # Crackle's index
        # We don't divide by crackle's mag because it's not used as a relative
        # axis.
        non_crackles = [i for i in range(1, n_vec_kinds) if i != CIND]
        non_vel_mags_nonzero = (non_vel_mags != 0.0)
        non_vel_projs = []
        for i in non_crackles:
            prev_i = i - 1
            mag_i = non_vel_mags[prev_i]
            mag_i_nonzero = non_vel_mags_nonzero[prev_i]
            non_vel_projs.append(pm.safeDivideElseZero(
                tri_dots[prev_i, :i], mag_i, mag_i_nonzero
            ))
            if i < n_other_vec_kinds:
                non_vel_projs.append(pm.safeDivideElseZero(
                    tri_dots[i:, i], mag_i, mag_i_nonzero
                ))
        
        a_proj_with_curr_rel = (
            a_ortho_v, #curr_ortho_mags[4], curr_ortho_mags[5]
        )
        remaining_proj_with_curr_rel = np.einsum(
            'ijk,jbk->ibj', all_curr_vecs[2:], curr_ortho_mats[:, 1:]
        )


        other_frame_proj_zip = zip(
            other_projs_on_vel[:3], remaining_proj_with_curr_rel[:3]
        )
        other_frame_proj_tups = [(a, b, c) for a, (b, c) in other_frame_proj_zip]
        other_frame_projs = [a for tup in other_frame_proj_tups for a in tup]
        zeros_3d = np.zeros((3, n_ins))

        tri_inds = np.tril_indices(n_other_vec_kinds)
        ret = np.concatenate(
            (
                np.full((1, n_ins), self.step),
                all_dots_with_prev.reshape(-1, n_ins),
                all_proj_with_prev.reshape(-1, n_ins),
                vel_mags.reshape(1, n_ins), non_vel_mags, tri_dots[tri_inds],
                *vel_projs, *non_vel_projs, a_proj_with_curr_rel,
                remaining_proj_with_curr_rel.reshape(-1, n_ins)
            ), axis=0
        ).transpose()
        # 1 + 8*(7+9) + 8 + 1+2+3+4+5+6+7 + (7*6 + 7) + (8*2 - 3)


        # Now, we'll calculate the JAV values that get used by the loss func.
        # These are NOT required in the NN's initial input layer!
        jav_tup = (
            vel_mags.flatten(), a_proj_v, a_ortho_v, *other_frame_projs,
            *zeros_3d
        )
        jav_stack = typing.cast(NDArray, np.stack(jav_tup, axis=-1))

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

    def getSeparateCalcs(self, all_xs: NDArray, all_rmats: NDArray):
        xshape = all_xs.shape
        rshape = all_rmats.shape
        assert (len(xshape) == 3 and xshape[0] == 6 and xshape[-1] == 3), \
        "Translations must have shape (6, n, 3), is instead {}".format(xshape)
        
        req_rshape = xshape + (3,)
        assert rshape == req_rshape, \
        "Rotations require shape {}, not {}, based on translations.".format(
            req_rshape, rshape
        )

        self.updatePrecalcs(all_xs[:5], all_rmats)
        return self.getHypotheticalCalcs(all_xs[5])

    @staticmethod
    def getInputGridOfVec3s(resolution: int, extent: float, generating_vec3_pt: NDArray,):
        deltas_1D = (np.arange(resolution) / resolution) * 2 - 1
        deltas_1D *= extent

        deltaYs, deltaXs = np.meshgrid(deltas_1D, deltas_1D)
        hyp_deltas = np.dstack((deltaXs, deltaYs, np.zeros_like(deltaXs)))
        hyp_pts_grid = hyp_deltas + generating_vec3_pt
        return hyp_pts_grid.reshape(-1, 3)        

    @staticmethod
    def getMostlyStaticPrevPts(position_noise: float = 0.0, rotation_noise: float = 0.0):
        # We need 7 input points to get crackle calculations because my code currently
        # assumes the last one is ground truth for which it shouldn't generate any
        # predictions, and we need 6 input points to calculate nonzero crackle.
        pts_shape = (7, 3)
        rand_pts = np.zeros(pts_shape)
        rand_pts[-3] = (15, 0, 0)
        default_aas = np.ones_like(rand_pts)
    
        if position_noise > 0.0:    
            rand_pts += np.random.normal(0.0, position_noise, pts_shape)
        if rotation_noise > 0.0:
            default_aas += np.random.normal(0.0, rotation_noise, pts_shape)
        return rand_pts, default_aas

    def getConstAccPreds(self, all_x5_choices: NDArray):
        x4 = self.x0_through_4[4]
        x3 = self.x0_through_4[3]
        return (3 * all_x5_choices) - (3 * x4) + x3
