import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import cinpact

import gtCommon
from gtCommon import BCOT_Data_Calculator, quatsFromAxisAngles




# For predictions that require multiple prior points, they may lack a prediction
# for the 2nd point, 3rd point, etc. In this case, the CV algorithm would just
# assume no motion for those frames, so we'll reflect that here as well.
def prependMissingPredictions(values, predictions):
    numMissing = (len(values) - 1) - len(predictions)
    return np.vstack((values[0:numMissing], predictions))

# Note: for rotations, returns errors in radians!
def getPredictionError(values, predictions, treatAsRotations: bool):
    full_predictions = prependMissingPredictions(values, predictions)
    err_dists = None
    if treatAsRotations:
        # First, we create quaternions. Then the acos of the abs of the dot
        # product between two quaternions is the radian distance between the
        # two rotations (i.e., the angle of the rotation from one to another).
        # TODO: Cite/explain why that is the case!
        # (The short version is: look at the scalar component of the output of 
        # quaternion multiplication; it can be expressed as the dot product of
        # two quaternions.)
        v_qs = quatsFromAxisAngles(values[1:])
        p_qs = quatsFromAxisAngles(predictions)
        v_qs_p_qs_dot = np.einsum('ij,ij->i', v_qs, p_qs)
        err_dists = np.arccos(np.clip(np.abs(v_qs_p_qs_dot), -1, 1))
    else:
        pred_diffs = values[1:] - full_predictions
        err_dists = np.linalg.norm(pred_diffs, axis=1) # Norm for each vec3
    return err_dists

def getTranslationScore(translations, preds):
    errs = getPredictionError(translations, preds, False)
    return (errs <= 50.0).mean()



def CompileTranslationScores(static, vel, acc, accLerp, translations):
    t_perfect_errors = getPredictionError(translations, translations[1:], False)
    perfect_score = (t_perfect_errors <= 0.0001).mean()

    return np.array(
        [perfect_score] + [getTranslationScore(translations, p) for p in [
            static, vel, acc, accLerp
        ]]
    )

combos = []
for b in range(len(gtCommon.BCOT_BODY_NAMES)):
    for s in range(len(gtCommon.BCOT_SEQ_NAMES)):
        if BCOT_Data_Calculator.isBodySeqPairValid(b, s):
            combos.append((b,s))

t_scores = np.empty((len(combos), 5))
r_scores = np.empty((len(combos), 3))
skipAmount = 2
for i, combo in enumerate(combos):
    calculator = BCOT_Data_Calculator(combo[0], combo[1], skipAmount)

    translations = calculator.getTranslationsGTNP(True)
    rotations = calculator.getRotationsGTNP(True)

    numTimestamps = len(translations)
    timestamps = np.arange(numTimestamps)

    translations_vel = np.diff(translations[:-1], axis=0)
    rotations_vel = np.diff(rotations[:-1], axis=0)

    translations_acc = np.diff(translations_vel, axis=0)

    t_vel_preds = translations[1:-1] + translations_vel
    r_vel_pred = rotations[1:-1] + rotations_vel

    t_acc_delta = translations_vel[1:] + (0.5 * translations_acc)
    t_acc_preds = translations[2:-1] + t_acc_delta
    t_accLERP_preds = translations[2:-1] + 0.25 * t_acc_delta
    t_acc_preds = np.vstack((t_vel_preds[:1], t_acc_preds))
    t_accLERP_preds = np.vstack((t_vel_preds[:1], t_accLERP_preds))

    tx_coords = np.stack((timestamps, translations[:, 0]), axis = -1)
    ptx_coords = np.stack((timestamps[2:], t_vel_preds[:, 0]), axis = -1)

    v_line_coords = np.hstack((tx_coords[:-2], ptx_coords)).reshape((len(ptx_coords), 2, 2))

    

    r_static_errors = getPredictionError(rotations, rotations[:-1], True)
    r_static_score = (r_static_errors <= np.deg2rad(5)).mean()
    r_perfect_rrors = getPredictionError(rotations, rotations[1:], True)
    r_perfect_score = (r_perfect_rrors <= 0.00001).mean()

    # Dumb comment.
    t_scores[i] = CompileTranslationScores(
        translations[:-1], t_vel_preds, t_acc_preds, t_accLERP_preds,
        translations
    )
    r_scores[i] = np.array([r_perfect_score, 0, r_static_score])
print("T Scores:", t_scores.mean(axis=0))
print("R Scores:", r_scores.mean(axis=0))


#%%
fig = plt.figure(0) # Arg of "0" means same figure reused if cell ran again.
fig.clear() # Good to do for iPython running, if running a plot cell again.
ax = fig.subplots()

vlc = LineCollection(v_line_coords)

ax.add_collection(vlc)
ax.autoscale()
ax.margins(0.1)

plt.show()#block=True) # block=True used for separate-window iPython plotting.

