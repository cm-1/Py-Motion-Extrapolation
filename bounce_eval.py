import numpy as np

from gtCommon import PoseLoaderBCOT
import gtCommon as gtc
import posemath as pm

import matplotlib.pyplot as plt


_, combos_with_mk = PoseLoaderBCOT.trainTestByBody(0.2)
combos = [c[:2] for c in combos_with_mk]

combo_translations = dict()
for combo in combos:
    calculator = PoseLoaderBCOT(combo[0], combo[1])
    translations = calculator.getTranslationsGTNP()[::1]
    combo_translations[combo] = translations
#%%
def getBounceScore(thresh: float, skip_amt: int):
    step = skip_amt + 1
    bounce_err_sum = 0.0
    vel_err_sum = 0.0
    quad_err_sum = 0.0
    optimal_err_sum = 0.0
    errs_count = 0
    for combo in combos:
        translations = combo_translations[combo][::step]
        vels_deg1 = np.diff(translations[:-1], 1, axis=0)
        t_vel_preds = translations[1:-1] + vels_deg1
        t_quad_preds = np.empty_like(translations[1:])
        t_quad_preds[0] = translations[0]
        t_quad_preds[1] = t_vel_preds[0]
        t_quad_preds[2:] = 3 * vels_deg1[1:] + translations[:-3]

        num_errs_to_keep = len(translations) - 4 # Same as "CUT_FOR_JERK" in motionExperiments.py
        
        t_poly_preds = t_quad_preds.copy()
        unit_vels = pm.safelyNormalizeArray(vels_deg1)
        unit_vel_dots = pm.einsumDot(unit_vels[1:], unit_vels[:-1])    
        vel_bounce = (unit_vel_dots < thresh) #
        t_poly_preds[2:][vel_bounce] = t_vel_preds[1:][vel_bounce]
        t_errs = translations[1:] - t_poly_preds
        t_err_norms = np.linalg.norm(t_errs, axis = -1)
        bounce_err_sum += t_err_norms[-num_errs_to_keep:].sum()
        just_vel_err_norms = np.linalg.norm(
            (translations[2:] - t_vel_preds), axis = -1
        )

        t_quad_err_norms = np.linalg.norm(
            translations[1:] - t_quad_preds, axis = -1
        )
        vel_better_inds = just_vel_err_norms < t_quad_err_norms[1:]
        t_opt_vel_preds = t_quad_preds.copy()
        t_opt_vel_preds[1:][vel_better_inds] = t_vel_preds[vel_better_inds]
        t_opt_vel_err_norms = np.linalg.norm(
            translations[1:] - t_opt_vel_preds, axis = -1
        )
        vel_err_sum += just_vel_err_norms[-num_errs_to_keep:].sum()
        quad_err_sum += t_quad_err_norms[-num_errs_to_keep:].sum()
        optimal_err_sum += t_opt_vel_err_norms[-num_errs_to_keep:].sum()
        errs_count += num_errs_to_keep

    # Negating so that higher values are better.
    return -np.array([
        vel_err_sum, quad_err_sum, bounce_err_sum, optimal_err_sum
    ]) / errs_count

#%%

PLOTTING_SKIP = 2
thresholds = np.linspace(-1, 1, 100)
scores = np.array([getBounceScore(th, PLOTTING_SKIP) for th in thresholds])[::-1]

angs = (1.0 - thresholds[::-1]) * np.pi/2

plt.plot(angs, -scores[:, 2], label="using bounce")
plt.plot(angs[[0, -1]], -scores[:, 1][[0, -1]], label="just acc")
plt.plot(angs[[0, -1]], -scores[:, 3][[0, -1]], label="opt vel")
plt.legend()
plt.show()
#%%