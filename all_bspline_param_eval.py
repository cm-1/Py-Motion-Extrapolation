from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from bspline_approximation import BSplineFitCalculator
import gtCommon as gtc

from gtCommon import BCOT_Data_Calculator

TRANSLATION_THRESH_5 = 50.0#50.0
TRANSLATION_THRESH_2 = 20.0

ROTATION_THRESH_RAD_5 = np.deg2rad(5.0)#5.0)
ROTATION_THRESH_RAD_2 = np.deg2rad(2.0)



@dataclass
class PoseData:
    translations: np.ndarray
    rotations_aa: np.ndarray
    translations_gt: np.ndarray
    rotations_gt_aa: np.ndarray
    rotations_gt_quats: np.ndarray


skipAmount = 2

combos = []
for b in range(len(gtc.BCOT_BODY_NAMES)):
    for s in range(len(gtc.BCOT_SEQ_NAMES)):
        if BCOT_Data_Calculator.isBodySeqPairValid(b, s):
            combos.append((b,s))

poseDataDict = dict()

for combo in combos:
    calculator = BCOT_Data_Calculator(combo[0], combo[1], skipAmount)
    translations_gt = calculator.getTranslationsGTNP(True)
    rotations_gt_aa = calculator.getRotationsGTNP(True)
    #rotations_gt_quats = gtc.quatsFromAxisAngles(rotations_gt_aa)
    rotations_gt_quats = gtc.quatsFromAxisAngles(rotations_gt_aa)

    translations = translations_gt #+ np.random.uniform(-4, 4, translations_gt.shape)
    rotations = rotations_gt_aa # TODO: Apply quat error to these.

    poseDataDict[combo] = PoseData(
        translations, rotations, translations_gt, rotations_gt_aa,
        rotations_gt_quats
    )


deg_range_inclusive = (1, 5)
max_ctrl_pts = 10
max_input_pts = 20

deg_range_len = deg_range_inclusive[1] + 1 - deg_range_inclusive[0]
ctrl_pt_range_len = max_ctrl_pts - deg_range_inclusive[0]
input_range_len = max_input_pts - deg_range_inclusive[0]

out_shape = (deg_range_len, ctrl_pt_range_len, input_range_len, len(combos))

results_2cm = np.zeros(out_shape)
results_2deg = np.zeros(out_shape)
results_5cm = np.zeros(out_shape)
results_5deg = np.zeros(out_shape)
results_mean_sq_dist = np.zeros(out_shape)
results_mean_sq_angle = np.zeros(out_shape)

'''
for deg in range(deg_range_inclusive[0], deg_range_inclusive[1] + 1):
    for num_ctrl_pts in range(deg + 1, max_ctrl_pts + 1):
        for num_input_pts in range(num_ctrl_pts, max_input_pts + 1):
            for combo_ind, combo in enumerate(combos):
                pd = poseDataDict[combo]
                spline_pred_calculator = BSplineFitCalculator(
                    deg, num_ctrl_pts, num_input_pts
                )

                all_spline_preds = spline_pred_calculator.fitAllData(np.hstack((
                    pd.translations, pd.rotations_aa
                )))

                t_spline_preds = all_spline_preds[:, :3]
                r_aa_spline_preds = all_spline_preds[:, 3:]

                # TODO: Remove this; for now, it's just to verify results
                # against code that (as far as I can tell) works.
                if (deg == 2):
                    t_diff = pd.translations[deg - 1] - pd.translations[deg - 2]
                    t_vel_pred = pd.translations[deg - 1] + t_diff
                    
                    t_spline_preds = np.vstack(([pd.translations[0]], [t_vel_pred], t_spline_preds))
                
                    r_aa_spline_preds = np.vstack((pd.rotations_aa[:2], r_aa_spline_preds))

                t_errs_start = len(pd.translations_gt) - len(t_spline_preds)
                r_errs_start = len(pd.rotations_gt_aa) - len(r_aa_spline_preds)
                t_errs = pd.translations_gt[t_errs_start:] - t_spline_preds
                t_err_norms = np.linalg.norm(t_errs, axis = -1)
                r_aa_spline_pred_quats = gtc.quatsFromAxisAngles(
                    r_aa_spline_preds
                )

                r_aa_errs = gtc.anglesBetweenQuats(
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
                
                t_d_sq = (t_err_norms**2).mean()
                results_mean_sq_dist[
                    deg_ind, num_ctrl_ind, num_ins_ind, combo_ind
                ] = t_d_sq
                
                r_a_sq = (r_aa_errs**2).mean()
                results_mean_sq_angle[
                    deg_ind, num_ctrl_ind, num_ins_ind, combo_ind
                ] = r_a_sq
            
            # Print out current progress, as this can take a long time.
            progress_str = "\rdeg {}, ctrl {}, num_in {}".format(
                deg, num_ctrl_pts, num_input_pts
            )
            print(progress_str, end = '', flush=True)
print()

#%%

np.savez(
    "./bspline_param_evals2.npz", res_2cm = results_2cm, res_2deg = results_2deg,
     res_5cm = results_5cm, res_5deg = results_5deg,
     res_mean_sq_dist = results_mean_sq_dist,
     res_mean_sq_angle = results_mean_sq_angle
)                        
'''
#%% 
load_result = np.load("../../bspline_param_evals2.npz")
# load_result.close()

#%%

# Create the figure
fig = plt.figure(0)
fig.clear()

# Mean over all body/seq combos.
data = load_result['res_mean_sq_dist'].mean(axis = -1) 

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
data[data > 30 * data_min] = np.nan

ax = fig.add_subplot(111, projection='3d')


# Loop through each 2D slice in the 3D array and plot it as a surface
for level in range(data.shape[0]):
    # Get the 2D slice at the current level    
    Z = data[level]
    
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    
    # Plot the surface (with a label for the corresponding k val)
    order = level + 1 + deg_range_inclusive[0]
    surf = ax.plot_surface(X, Y, Z, alpha=0.7, label=f'k={order}')
    

ax.set_title('3D Surface Plot for Each Order')
ax.set_xlabel('Num input pts')
ax.set_ylabel('Num ctrl pts (m + 1)')
ax.set_zlabel('Mean squared distance')
ax.legend()

plt.show(block=True)

