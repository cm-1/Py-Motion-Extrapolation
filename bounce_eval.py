import numpy as np
from bayes_opt import BayesianOptimization

from gtCommon import PoseLoaderBCOT
import gtCommon as gtc
import posemath as pm

import matplotlib.pyplot as plt


combos = []
for b in range(len(gtc.BCOT_BODY_NAMES)):
    for s in range(len(gtc.BCOT_SEQ_NAMES)):
        if PoseLoaderBCOT.isBodySeqPairValid(b, s):
            combos.append((b,s))

combo_translations = dict()
for combo in combos:
    calculator = PoseLoaderBCOT(combo[0], combo[1], 0)
    translations = calculator.getTranslationsGTNP(True)
    combo_translations[combo] = translations
#%%
def getBounceScore(thresh: float, skip_amt: int):
    step = skip_amt + 1
    all_err_means = []
    all_vel_err_means = []
    all_opt_vel_err_means = []
    for combo in combos:
        translations = combo_translations[combo][::step]
        vels_deg1 = np.diff(translations[:-1], 1, axis=0)
        t_vel_preds = translations[1:-1] + vels_deg1
        t_quad_preds = np.empty_like(translations[1:])
        t_quad_preds[0] = translations[0]
        t_quad_preds[1] = t_vel_preds[0]
        t_quad_preds[2:] = 3 * vels_deg1[1:] + translations[:-3]
        
        t_poly_preds = t_quad_preds.copy()
        unit_vels = pm.safelyNormalizeArray(vels_deg1)
        unit_vel_dots = pm.einsumDot(unit_vels[1:], unit_vels[:-1])    
        vel_bounce = (unit_vel_dots < thresh) #
        t_poly_preds[2:][vel_bounce] = t_vel_preds[1:][vel_bounce]
        t_errs = translations[1:] - t_poly_preds
        t_err_norms = np.linalg.norm(t_errs, axis = -1)
        mean_t_err_norm = t_err_norms.mean()
        all_err_means.append(mean_t_err_norm)
        just_vel_err_norms = np.linalg.norm(
            (translations[2:] - t_vel_preds), axis = -1
        )
        all_vel_err_means.append(just_vel_err_norms.mean())

        t_quad_err_norms = np.linalg.norm(
            translations[1:] - t_quad_preds, axis = -1
        )
        vel_better_inds = just_vel_err_norms < t_quad_err_norms[1:]
        t_opt_vel_preds = t_quad_preds.copy()
        t_opt_vel_preds[1:][vel_better_inds] = t_vel_preds[vel_better_inds]
        t_opt_vel_err_norms = np.linalg.norm(
            translations[1:] - t_opt_vel_preds, axis = -1
        ) 
        all_opt_vel_err_means.append(t_opt_vel_err_norms.mean())

    # Negating so that higher values are better.
    return (-np.mean(all_err_means), -np.mean(all_vel_err_means), -np.mean(all_opt_vel_err_means))

#%%

PLOTTING_SKIP = 2
thresholds = np.linspace(-1, 1, 100)
scores = np.array([getBounceScore(th, PLOTTING_SKIP) for th in thresholds])

plt.plot(thresholds, scores[:, 0], label="using bounce")
plt.plot(thresholds, scores[:, 1], label="just vel")
plt.plot(thresholds, scores[:, 2], label="opt vel")
plt.legend()
plt.show()
