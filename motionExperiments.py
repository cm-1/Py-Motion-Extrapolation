import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import bspline

import gtCommon
from gtCommon import BCOT_Data_Calculator, quatsFromAxisAngles

SPLINE_DEGREE = 2
PTS_USED_TO_CALC_LAST = 6 # Must be at least spline's degree + 1.
DESIRED_CTRL_PTS = 5
if (PTS_USED_TO_CALC_LAST < SPLINE_DEGREE + 1):
    raise Exception("Need at least order=k=(degree + 1) input pts for calc!")
if DESIRED_CTRL_PTS > PTS_USED_TO_CALC_LAST:
    raise Exception("Need at least as many input points as control points!")
if DESIRED_CTRL_PTS < SPLINE_DEGREE + 1:
    raise Exception("Need at least order=k=(degree + 1) control points!")


# For predictions that require multiple prior points, they may lack a prediction
# for the 2nd point, 3rd point, etc. In this case, the CV algorithm would just
# assume no motion for those frames, so we'll reflect that here as well.
def prependMissingPredictions(values, predictions):
    numMissing = (len(values) - 1) - len(predictions)
    return np.vstack((values[0:numMissing], predictions))

# Note: returns errors in radians!
def getQuatError(values, predictions):
    full_predictions = prependMissingPredictions(values, predictions)
    # The acos of the abs of the dot product between two quaternions is the
    # radian distance between the two rotations 
    # (i.e., the angle of the rotation from one to another).
    # TODO: Cite/explain why that is the case! (The short version is: look at
    # the scalar component of the output of quaternion multiplication; it can be
    # expressed as the dot product of two quaternions.)
    vals_preds_dot = np.einsum('ij,ij->i', values[1:], full_predictions)
    return np.arccos(np.clip(np.abs(vals_preds_dot), -1, 1))

# Note: returns errors in radians!
def getAxisAngleError(values, predictions): 
    v_qs = quatsFromAxisAngles(values)
    p_qs = quatsFromAxisAngles(predictions)
    return getQuatError(v_qs, p_qs)

def getTranslationError(values, predictions):
    full_predictions = prependMissingPredictions(values, predictions)

    pred_diffs = values[1:] - full_predictions
    return np.linalg.norm(pred_diffs, axis=1) # Norm for each vec3

def getTranslationScore(translations, preds):
    errs = getTranslationError(translations, preds)
    return (errs <= 50.0).mean()

def getAxisAngleScore(rotations, preds):
    r_errors = getAxisAngleError(rotations, preds)
    return (r_errors <= np.deg2rad(5)).mean()

def getQuaternionScore(rotations, preds):
    r_errors = getQuatError(rotations, preds)
    return (r_errors <= np.deg2rad(5)).mean()

def CompileTranslationScores(static, vel, acc, accLerp, spline, translations):
    t_perfect_errors = getTranslationError(translations, translations[1:])
    perfect_score = (t_perfect_errors <= 0.0001).mean()

    return np.array(
        [perfect_score] + [getTranslationScore(translations, p) for p in [
            static, vel, acc, accLerp, spline
        ]]
    )

combos = []
for b in range(len(gtCommon.BCOT_BODY_NAMES)):
    for s in range(len(gtCommon.BCOT_SEQ_NAMES)):
        if BCOT_Data_Calculator.isBodySeqPairValid(b, s):
            combos.append((b,s))

t_scores = np.empty((len(combos), 6))
r_scores = np.empty((len(combos), 3))
skipAmount = 2

splineMats = []
for i in range(SPLINE_DEGREE + 1, PTS_USED_TO_CALC_LAST):
    # startInd = max(0, i - PTS_USED_TO_CALC_LAST + 1)
    uInterval = (SPLINE_DEGREE, i + 1)# - startInd)
    ctrlPtCount = min(i + 1, DESIRED_CTRL_PTS)
    numTotal = i + 2 #1 - startInd + 1
    knotList = np.arange(ctrlPtCount + SPLINE_DEGREE + 1)


    uVals = np.linspace(uInterval[0], uInterval[1], numTotal)

    splineMats.append(bspline.bSplineFittingMat(
        ctrlPtCount, SPLINE_DEGREE + 1, i + 1, uVals, knotList
    ))

for i, combo in enumerate(combos):
    calculator = BCOT_Data_Calculator(combo[0], combo[1], skipAmount)

    translations = calculator.getTranslationsGTNP(True)
    rotations = calculator.getRotationsGTNP(True)

    numTimestamps = len(translations)
    timestamps = np.arange(numTimestamps)

    translations_vel = np.diff(translations[:-1], axis=0)
    # rotations_vel = np.diff(rotations[:-1], axis=0)

    rotations_quats = gtCommon.quatsFromAxisAngles(rotations)
    
    rev_rotations_quats = np.empty(rotations_quats.shape)
    rev_rotations_quats[:, 0] = rotations_quats[:, 0]
    rev_rotations_quats[:, 1:] = -rotations_quats[:, 1:]

    testQuats = gtCommon.multiplyQuatLists(rotations_quats, rev_rotations_quats)
    expectedTestQuats = np.zeros(testQuats.shape)
    expectedTestQuats[:, 0] = 1
    testQuatDiff = expectedTestQuats - testQuats
    maxTestDiff = testQuatDiff.max()
    print(maxTestDiff)

    rotations_vel = gtCommon.multiplyQuatLists(rotations_quats[1:-1], rev_rotations_quats[:-2])


    translations_acc = np.diff(translations_vel, axis=0)

    t_vel_preds = translations[1:-1] + translations_vel
    # r_vel_pred = rotations[1:-1] + rotations_vel
    r_vel_preds = gtCommon.multiplyQuatLists(rotations_quats[1:-1], rotations_vel)

    t_acc_delta = translations_vel[1:] + (0.5 * translations_acc)
    t_acc_preds = translations[2:-1] + t_acc_delta
    t_accLERP_preds = translations[2:-1] + 0.25 * t_acc_delta
    t_acc_preds = np.vstack((t_vel_preds[:1], t_acc_preds))
    t_accLERP_preds = np.vstack((t_vel_preds[:1], t_accLERP_preds))

    tx_coords = np.stack((timestamps, translations[:, 0]), axis = -1)
    ptx_coords = np.stack((timestamps[2:], t_vel_preds[:, 0]), axis = -1)

    v_line_coords = np.hstack((tx_coords[:-2], ptx_coords)).reshape((len(ptx_coords), 2, 2))

    num_spline_preds = (len(translations) - 1) - (SPLINE_DEGREE + 1) 
    t_spline_preds = np.empty((num_spline_preds, 3))
    for j in range(SPLINE_DEGREE + 1, len(translations) - 1):
        startInd = max(0, j - PTS_USED_TO_CALC_LAST + 1)
        uInterval = (SPLINE_DEGREE, j + 1 - startInd)
        ctrlPtCount = min(j + 1, DESIRED_CTRL_PTS)
        numTotal = j + 1 - startInd + 1
        knotList = np.arange(ctrlPtCount + SPLINE_DEGREE + 1)
    
    
        uVals = np.linspace(uInterval[0], uInterval[1], numTotal)
    
        ptsToFit_j = translations[startInd:(j+1)]
        ctrlPts = np.empty((ctrlPtCount, 0)) 
        mat = splineMats[min(j - (SPLINE_DEGREE + 1), len(splineMats) - 1)]

        for k in range(3):
            
            ptsToFit1D = ptsToFit_j[:, k]

            ctrlPts1D = np.linalg.lstsq(mat, ptsToFit1D, rcond = None)[0]
            ctrlPts = np.hstack((ctrlPts, ctrlPts1D.reshape((-1, 1))))

        next_spline_pt = bspline.bSplineInner(
            uVals[-1], SPLINE_DEGREE + 1, ctrlPtCount - 1, ctrlPts, knotList
        )
        t_spline_preds[j - (SPLINE_DEGREE + 1)] = next_spline_pt
    r_static_score = getAxisAngleScore(rotations, rotations[:-1])
    r_perfect_rrors = getAxisAngleError(rotations, rotations[1:])
    r_perfect_score = (r_perfect_rrors <= 0.00001).mean()
    r_vel_score = getQuaternionScore(rotations_quats, r_vel_preds)

    # Dumb comment.
    t_scores[i] = CompileTranslationScores(
        translations[:-1], t_vel_preds, t_acc_preds, t_accLERP_preds, t_spline_preds,
        translations
    )
    r_scores[i] = np.array([r_perfect_score, r_vel_score, r_static_score])
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

