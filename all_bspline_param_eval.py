#%%
from dataclasses import dataclass
import typing

import numpy as np
import matplotlib.pyplot as plt

from spline_approximation import BSplineFitCalculator, SplinePredictionMode
import gtCommon as gtc
import posemath as pm

from gtCommon import PoseLoaderBCOT

TRANSLATION_THRESH_5 = 50.0#50.0
TRANSLATION_THRESH_2 = 20.0

ROTATION_THRESH_RAD_5 = np.deg2rad(5.0)#5.0)
ROTATION_THRESH_RAD_2 = np.deg2rad(2.0)

#%%

@dataclass
class PoseData:
    translations: np.ndarray
    rotations_aa: np.ndarray
    translations_gt: np.ndarray
    rotations_gt_aa: np.ndarray
    rotations_gt_quats: np.ndarray
    translations_quad_preds: np.ndarray
    rotations_preds: np.ndarray

skipAmount = 2

combos = []
for b in range(len(gtc.BCOT_BODY_NAMES)):
    for s in range(len(gtc.BCOT_SEQ_NAMES)):
        if PoseLoaderBCOT.isBodySeqPairValid(b, s):
            combos.append((b,s))

poseDataDict: typing.Dict[typing.Tuple[int,int], PoseData] = dict()

for combo in combos:
    calculator = PoseLoaderBCOT(combo[0], combo[1], skipAmount)
    translations_gt = calculator.getTranslationsGTNP(True)
    rotations_gt_aa = calculator.getRotationsGTNP(True)
    #rotations_gt_quats = pm.quatsFromAxisAngleVec3s(rotations_gt_aa)
    rotations_gt_quats = pm.quatsFromAxisAngleVec3s(rotations_gt_aa)

    translations = translations_gt #+ np.random.uniform(-4, 4, translations_gt.shape)
    rotations = rotations_gt_aa # TODO: Apply quat error to these.

    translation_diffs = np.diff(translations, axis=0)
    rev_rotations_quats = np.empty(rotations_gt_quats.shape)
    rev_rotations_quats[:, 0] = rotations_gt_quats[:, 0]
    rev_rotations_quats[:, 1:] = -rotations_gt_quats[:, 1:]

    rotation_quat_diffs = pm.multiplyQuatLists(
        rotations_gt_quats[1:], rev_rotations_quats[:-1]
    )

    t_vel_preds = translations[1:-1] + translation_diffs[:-1]
    r_vel_preds = pm.multiplyQuatLists(
        rotation_quat_diffs[:-1], rotations_gt_quats[1:-1]
    )
    r_slerp_preds = pm.quatSlerp(rotations_gt_quats[1:-1], r_vel_preds, 1.0)#0.75)

    
    # Converts quaternions to axis-angle, then corrects jumps.
    # TODO: Document better, maybe find way to combine with mat->AA code?
    unitAxes, angles = pm.axisAnglesFromQuats(r_slerp_preds)
    angles = angles.flatten()

    angle_dots = pm.einsumDot(unitAxes[1:], unitAxes[:-1]) 
    needs_flip = np.logical_xor.accumulate(angle_dots < 0, axis = -1)
    angles[1:][needs_flip] = -angles[1:][needs_flip]
    unitAxes[1:][needs_flip] = -unitAxes[1:][needs_flip]
    np_tau = 2.0 * np.pi
    tau_facs = np.round(np.diff(angles) / np_tau)
    angle_corrections = np_tau * np.cumsum(tau_facs, axis = -1)
    angles[1:] -= angle_corrections
    r_slerp_preds_aa = pm.scalarsVecsMul(angles, unitAxes)
    
    r_slerp_preds_aa = np.vstack((rotations[:1], r_slerp_preds_aa))
    
    t_quad_preds = 3 * translations[2:-1] - 3 * translations[1:-2] + translations[:-3]
    t_quad_preds = np.vstack((translations[:1], t_vel_preds[:1], t_quad_preds))


    poseDataDict[combo] = PoseData(
        translations, rotations, translations_gt, rotations_gt_aa,
        rotations_gt_quats, t_quad_preds, r_slerp_preds_aa
    )
    

deg_range_inclusive = (1, 5)
max_ctrl_pts = 10
max_input_pts = 20

deg_range_len = deg_range_inclusive[1] + 1 - deg_range_inclusive[0]
ctrl_pt_range_len = max_ctrl_pts - deg_range_inclusive[0]
input_range_len = max_input_pts - deg_range_inclusive[0]

out_shape = (deg_range_len, ctrl_pt_range_len, input_range_len, len(combos))

#%%
results_2cm = np.zeros(out_shape)
results_2deg = np.zeros(out_shape)
results_5cm = np.zeros(out_shape)
results_5deg = np.zeros(out_shape)
results_mean_dist = np.zeros(out_shape)
results_mean_angle = np.zeros(out_shape)

mode = SplinePredictionMode.EXTRAPOLATE

smooth_and_accel = (mode == SplinePredictionMode.SMOOTH_AND_ACCEL)
for deg in range(deg_range_inclusive[0], deg_range_inclusive[1] + 1):
    for num_ctrl_pts in range(deg + 1, max_ctrl_pts + 1):
        for num_input_pts in range(num_ctrl_pts, max_input_pts + 1):
            spline_pred_calculator = BSplineFitCalculator(
                deg, num_ctrl_pts, num_input_pts, [mode]
            )
            for combo_ind, combo in enumerate(combos):
                pd = poseDataDict[combo]

                all_spline_preds = None
                if mode == SplinePredictionMode.EXTRAPOLATE:
                    all_spline_preds = spline_pred_calculator.fitAllData(
                        np.hstack((pd.translations, pd.rotations_aa))
                    )
                elif mode == SplinePredictionMode.SMOOTH:
                    all_spline_preds = spline_pred_calculator.smoothAllData(
                        np.hstack((pd.translations, pd.rotations_aa)),
                        np.hstack((pd.translations_quad_preds, pd.rotations_preds))
                    )
                else:
                    all_spline_preds = spline_pred_calculator.constantAccelPreds(
                        np.hstack((pd.translations, pd.rotations_aa)),
                        smooth_and_accel
                    )


                t_spline_preds = all_spline_preds[:, :3]
                r_aa_spline_preds = all_spline_preds[:, 3:]

                # TODO: Remove this; for now, it's just to verify results
                # against code that (as far as I can tell) works.
                if mode == SplinePredictionMode.EXTRAPOLATE and (deg == 2):
                    t_diff = pd.translations[deg - 1] - pd.translations[deg - 2]
                    t_vel_pred = pd.translations[deg - 1] + t_diff
                    
                    t_spline_preds = np.vstack(([pd.translations[0]], [t_vel_pred], t_spline_preds))
                
                    r_aa_spline_preds = np.vstack((pd.rotations_aa[:2], r_aa_spline_preds))

                t_errs_start = len(pd.translations_gt) - len(t_spline_preds)
                r_errs_start = len(pd.rotations_gt_aa) - len(r_aa_spline_preds)
                t_errs = pd.translations_gt[t_errs_start:] - t_spline_preds
                t_err_norms = np.linalg.norm(t_errs, axis = -1)
                r_aa_spline_pred_quats = pm.quatsFromAxisAngleVec3s(
                    r_aa_spline_preds
                )

                r_aa_errs = pm.anglesBetweenQuats(
                    r_aa_spline_pred_quats, pd.rotations_gt_quats[r_errs_start:]
                )

                # Calculate indices within output array
                deg_ind = deg - deg_range_inclusive[0]
                num_ctrl_ind = num_ctrl_pts - (deg_range_inclusive[0] + 1)
                num_ins_ind = num_input_pts - (deg_range_inclusive[0] + 1)
                # And then combo_ind comes from enumerate() already.

                res_2cm = (t_err_norms < TRANSLATION_THRESH_2).mean()
                results_2cm[
                    deg_ind, num_ctrl_ind, num_ins_ind, combo_ind
                ] = res_2cm

                res_2deg = (r_aa_errs < ROTATION_THRESH_RAD_2).mean()
                results_2deg[
                    deg_ind, num_ctrl_ind, num_ins_ind, combo_ind
                ] = res_2deg
                
                res_5cm = (t_err_norms < TRANSLATION_THRESH_5).mean()
                results_5cm[
                    deg_ind, num_ctrl_ind, num_ins_ind, combo_ind
                ] = res_5cm
                
                res_5deg = (r_aa_errs < ROTATION_THRESH_RAD_5).mean()
                results_5deg[
                    deg_ind, num_ctrl_ind, num_ins_ind, combo_ind
                ] = res_5deg
                
                t_d = (t_err_norms).mean()
                results_mean_dist[
                    deg_ind, num_ctrl_ind, num_ins_ind, combo_ind
                ] = t_d
                
                r_a = (r_aa_errs).mean()
                results_mean_angle[
                    deg_ind, num_ctrl_ind, num_ins_ind, combo_ind
                ] = r_a
            
            # Print out current progress, as this can take a long time.
            progress_str = "\rdeg {}, ctrl {}, num_in {}".format(
                deg, num_ctrl_pts, num_input_pts
            )
            print(progress_str, end = '', flush=True)
print()

#%%

np.savez_compressed(
    "./bspline_param_evals_latest.npz", res_2cm = results_2cm, res_2deg = results_2deg,
     res_5cm = results_5cm, res_5deg = results_5deg,
     res_mean_dist = results_mean_dist,
     res_mean_angle = results_mean_angle
)                        

#%% 
load_target = "./default_filename.npz"
if mode == SplinePredictionMode.EXTRAPOLATE:
    load_target = "./results/bspline_param_evals2_c.npz"
elif mode == SplinePredictionMode.SMOOTH:
    load_target = "./results/bspline_param_evals_quad_c.npz"
elif mode == SplinePredictionMode.SMOOTH_AND_ACCEL:
    load_target = "./results/bspline_param_evals_smoothderiv_c.npz"
elif mode == SplinePredictionMode.CONST_ACCEL:
    load_target = "./results/bspline_param_evals_deriv_c.npz"
load_result = np.load(load_target)
# load_result.close()

#%%


# Mean over all body/seq combos.
data_2cm = load_result['res_2cm'].mean(axis = -1)

#%% 
lr2 = np.load("./bspline_param_evals_latest.npz")
# data_latest = lr2['res_2cm'].mean(axis = -1)
# ds = data_2cm[:data_latest.shape[0], :data_latest.shape[1], :data_latest.shape[2]]
data = lr2['res_mean_dist'].mean(axis = -1)
for k in load_result.files:
    print("Max load comparison diff for key", k, ":", np.abs(load_result[k] - lr2[k]).max())

#%%
# Find out which indices do not contain data because they'd have too few ctrl
# or input pts for B-Spline fitting.
data_na = np.full(data.shape, False)
for deg_ind in range(data.shape[0]):
    data_na[deg_ind, :deg_ind] = True
    for nc_ind in range(deg_ind, data.shape[1]):
        data_na[deg_ind, nc_ind, :nc_ind] = True

if np.any(data[data_na] > 0):
    raise Exception("Made a mistake!")

# Setting values to nan will skip them during plotting.
data[data_na] = np.nan
data_min = np.nanmin(data)

data_2cm = load_result['res_2cm'].mean(axis = -1)
data_2deg_full = load_result['res_2deg']
data_2deg = data_2deg_full.mean(axis = -1)

#%%
# Create the figure
fig = plt.figure(0)
fig.clear()

data[data > 30 * data_min] = np.nan

ax = fig.add_subplot(111, projection='3d')


# Loop through each 2D slice in the 3D array and plot it as a surface
for level in range(data.shape[0]):
    # Get the 2D slice at the current level    
    Z = data[level]
    
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    min_order = 1 + deg_range_inclusive[0]
    # Plot the surface (with a label for the corresponding k val)
    order = level + min_order
    surf = ax.plot_surface(
        X + min_order, Y + min_order, Z, alpha=0.7, label=f'k={order}'
    )
    

ax.set_title('3D Surface Plot for Each Order')
ax.set_xlabel('Num input pts')
ax.set_ylabel('Num ctrl pts (m + 1)')
ax.set_zlabel('Mean distance')
ax.legend()

plt.show(block=True)

