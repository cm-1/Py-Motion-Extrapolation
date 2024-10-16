import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import cinpact

from gtCommon import BCOT_Data_Calculator, quatsFromAxisAngles

bodIndex = 1
seqIndex = 11
skipAmount = 2
calculator = BCOT_Data_Calculator(bodIndex, seqIndex, skipAmount)

translations = calculator.getTranslationsGTNP(True)
rotations = calculator.getRotationsGTNP(True)

numTimestamps = len(translations)
timestamps = np.arange(numTimestamps)

translations_vel = np.diff(translations[:-1], axis=0)
rotations_vel = np.diff(rotations[:-1], axis=0)

t_vel_pred = translations[1:-1] + translations_vel
r_vel_pred = rotations[1:-1] + rotations_vel

tx_coords = np.stack((timestamps, translations[:, 0]), axis = -1)
ptx_coords = np.stack((timestamps[2:], t_vel_pred[:, 0]), axis = -1)

v_line_coords = np.hstack((tx_coords[:-2], ptx_coords)).reshape((len(ptx_coords), 2, 2))



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

t_static_errors = getPredictionError(translations, translations[:-1], False)
t_vel_errors = getPredictionError(translations, t_vel_pred, False)
t_perfect_errors = getPredictionError(translations, translations[1:], False)

t_static_score = (t_static_errors <= 50.0).mean()
t_vel_score = (t_vel_errors <= 50.0).mean()
t_perfect_score = (t_perfect_errors <= 0.0001).mean()

r_static_errors = getPredictionError(rotations, rotations[:-1], True)
r_static_score = (r_static_errors <= np.deg2rad(5)).mean()
r_perfect_rrors = getPredictionError(rotations, rotations[1:], True)
r_perfect_score = (r_perfect_rrors <= 0.00001).mean()

print("Scores:", t_perfect_score, t_static_score, t_vel_score, r_perfect_score, r_static_score)


vlc = LineCollection(v_line_coords)


fig, ax = plt.subplots()
ax.add_collection(vlc)
ax.autoscale()
ax.margins(0.1)
plt.show()

