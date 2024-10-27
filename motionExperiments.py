import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import bspline

import gtCommon
from gtCommon import BCOT_Data_Calculator, quatsFromAxisAngles

TRANSLATION_THRESH = 50.0

ROTATION_THRESH_RAD = np.deg2rad(5)

SPLINE_DEGREE = 2
PTS_USED_TO_CALC_LAST = 6 # Must be at least spline's degree + 1.
DESIRED_CTRL_PTS = 5
if (PTS_USED_TO_CALC_LAST < SPLINE_DEGREE + 1):
    raise Exception("Need at least order=k=(degree + 1) input pts for calc!")
if DESIRED_CTRL_PTS > PTS_USED_TO_CALC_LAST:
    raise Exception("Need at least as many input points as control points!")
if DESIRED_CTRL_PTS < SPLINE_DEGREE + 1:
    raise Exception("Need at least order=k=(degree + 1) control points!")

def getRandomQuatError(shape, max_err_rads):
    axes = np.random.uniform(-1.0, 1.0, shape[:-1] + (3,))
    unit_axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)
    half_max = max_err_rads / 2.0
    half_angles = np.random.uniform(
        -half_max, half_max, shape[:-1] + (1,)
    )

    error_quats = np.empty(shape)
    error_quats[..., 0:1] = np.cos(half_angles)
    error_quats[..., 1:] = np.sin(half_angles) * unit_axes
    return error_quats

# For predictions that require multiple prior points, they may lack a prediction
# for the 2nd point, 3rd point, etc. In this case, the CV algorithm would just
# assume no motion for those frames, so we'll reflect that here as well.
def prependMissingPredictions(values, predictions):
    numMissing = (len(values) - 1) - len(predictions)
    return np.vstack((values[0:numMissing], predictions))

# Note: returns errors in radians!
def getQuatError(values, predictions):
    full_predictions = prependMissingPredictions(values, predictions)
    return gtCommon.anglesBetweenQuats(values[1:], full_predictions)

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
    return (errs <= TRANSLATION_THRESH).mean()

def getAxisAngleScore(rotations, preds):
    r_errors = getAxisAngleError(rotations, preds)
    return (r_errors <= ROTATION_THRESH_RAD).mean()

def getQuaternionScore(rotations, preds):
    r_errors = getQuatError(rotations, preds)
    return (r_errors <= ROTATION_THRESH_RAD).mean()

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
r_scores = np.empty((len(combos), 8))
skipAmount = 2

splineMats = []
for i in range(SPLINE_DEGREE, PTS_USED_TO_CALC_LAST):
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

    translations_gt = calculator.getTranslationsGTNP(True)
    rotations_gt = calculator.getRotationsGTNP(True)
    rotations_gt_quats = gtCommon.quatsFromAxisAngles(rotations_gt)


    translations = translations_gt + np.random.uniform(-4, 4, translations_gt.shape)
    rotations_quats = gtCommon.multiplyQuatLists(
        getRandomQuatError(rotations_gt_quats.shape, ROTATION_THRESH_RAD), 
        rotations_gt_quats
    )
    rotations = rotations_gt # TODO: Apply quat error to these.

    numTimestamps = len(translations_gt)
    timestamps = np.arange(numTimestamps)

    translations_vel = np.diff(translations[:-1], axis=0)
    rotations_aa_vel = np.diff(rotations[:-1], axis=0)

    
    
    rev_rotations_quats = np.empty(rotations_quats.shape)
    rev_rotations_quats[:, 0] = rotations_quats[:, 0]
    rev_rotations_quats[:, 1:] = -rotations_quats[:, 1:]



    rotations_vel = gtCommon.multiplyQuatLists(rotations_quats[1:-1], rev_rotations_quats[:-2])


    testQuats = gtCommon.multiplyQuatLists(rotations_vel, rotations_quats[:-2])
    testQuatDiff = testQuats - rotations_quats[1:-1]
    maxTestDiff = testQuatDiff.max()
    if (maxTestDiff) > 0.0001:
        raise Exception("Error!")
    #print(maxTestDiff)

    translations_acc = np.diff(translations_vel, axis=0)
    rotations_aa_acc = np.diff(rotations_aa_vel, axis=0)

    t_vel_preds = translations[1:-1] + translations_vel
    # r_vel_pred = rotations[1:-1] + rotations_vel
    r_vel_preds = gtCommon.multiplyQuatLists(rotations_vel, rotations_quats[1:-1])
    r_aa_vel_preds = rotations[1:-1] + rotations_aa_vel

    r_slerp_preds = gtCommon.quatSlerp(rotations_quats[1:-1], r_vel_preds, 0.75)

    t_acc_delta = translations_vel[1:] + (0.5 * translations_acc)
    t_acc_preds = translations[2:-1] + t_acc_delta
    t_accLERP_preds = translations[2:-1] + 0.75 * t_acc_delta
    t_acc_preds = np.vstack((t_vel_preds[:1], t_acc_preds))
    t_accLERP_preds = np.vstack((t_vel_preds[:1], t_accLERP_preds))

    r_aa_acc_delta = rotations_aa_vel[1:] + (0.5 * rotations_aa_acc)
    r_aa_acc_preds = rotations[2:-1] + r_aa_acc_delta
    r_aa_accLERP_preds = rotations[2:-1] + 0.5 * r_aa_acc_delta
    r_aa_acc_preds = np.vstack((r_aa_vel_preds[:1], r_aa_acc_preds))
    r_aa_accLERP_preds = np.vstack((r_aa_vel_preds[:1], r_aa_accLERP_preds))



    num_spline_preds = (len(translations) - 1) - (SPLINE_DEGREE) 
    t_spline_preds = np.empty((num_spline_preds, 3))
    r_aa_spline_preds = np.empty((num_spline_preds, 3))
    for j in range(SPLINE_DEGREE, len(translations) - 1):
        startInd = max(0, j - PTS_USED_TO_CALC_LAST + 1)
        uInterval = (SPLINE_DEGREE, j + 1 - startInd)
        ctrlPtCount = min(j + 1, DESIRED_CTRL_PTS)
        numTotal = j + 1 - startInd + 1
        knotList = np.arange(ctrlPtCount + SPLINE_DEGREE + 1)
    
    
        uVals = np.linspace(uInterval[0], uInterval[1], numTotal)

        ptsToFit_j = np.empty((j + 1 - startInd, 6))
        ptsToFit_j[:, :3] = translations[startInd:(j+1)]
        ptsToFit_j[:, 3:] = rotations[startInd:(j+1)]
        ctrlPts = np.empty((ctrlPtCount, 0)) 
        mat = splineMats[min(j - (SPLINE_DEGREE), len(splineMats) - 1)]

        for k in range(6):
            
            ptsToFit1D = ptsToFit_j[:, k]

            ctrlPts1D = np.linalg.lstsq(mat, ptsToFit1D, rcond = None)[0]
            ctrlPts = np.hstack((ctrlPts, ctrlPts1D.reshape((-1, 1))))

        next_spline_pt = bspline.bSplineInner(
            uVals[-1], SPLINE_DEGREE + 1, ctrlPtCount - 1, ctrlPts, knotList
        )
        t_spline_preds[j - (SPLINE_DEGREE)] = next_spline_pt[:3]
        r_aa_spline_preds[j - (SPLINE_DEGREE)] = next_spline_pt[3:]
    t_spline_preds = np.vstack((t_vel_preds[:1], t_spline_preds))

    r_static_score = getAxisAngleScore(rotations_gt, rotations[:-1])
    r_perfect_rrors = getAxisAngleError(rotations_gt, rotations[1:])
    r_perfect_score = (r_perfect_rrors <= 0.00001).mean()
    r_vel_score = getQuaternionScore(rotations_gt_quats, r_vel_preds)
    r_slerp_score = getQuaternionScore(rotations_gt_quats, r_slerp_preds)

    r_aa_vel_score = getAxisAngleScore(rotations_gt, r_aa_vel_preds)
    r_aa_acc_score = getAxisAngleScore(rotations_gt, r_aa_acc_preds)
    r_aa_accLERP_score = getAxisAngleScore(rotations_gt, r_aa_accLERP_preds)
    r_aa_spline_score = getAxisAngleScore(rotations_gt, r_aa_spline_preds)

    # Dumb comment.
    t_scores[i] = CompileTranslationScores(
        translations[:-1], t_vel_preds, t_acc_preds, t_accLERP_preds, t_spline_preds,
        translations_gt
    )
    r_scores[i] = np.array([
        r_perfect_score, r_static_score, r_vel_score, r_slerp_score, 
        r_aa_vel_score, r_aa_acc_score, r_aa_accLERP_score, r_aa_spline_score
    ])
print("T Scores:", t_scores.mean(axis=0))
print("R Scores:", r_scores.mean(axis=0))

tx_coords = np.stack((timestamps, translations[:, 2]), axis = -1)
ptx_coords = np.stack((timestamps[2:], t_vel_preds[:, 2]), axis = -1)

v_line_coords = np.hstack((tx_coords[:-2], ptx_coords)).reshape((len(ptx_coords), 2, 2))
#%%
fig = plt.figure(0) # Arg of "0" means same figure reused if cell ran again.
fig.clear() # Good to do for iPython running, if running a plot cell again.
ax = fig.subplots()

vlc = LineCollection(v_line_coords)

ax.add_collection(vlc)
ax.autoscale()
ax.margins(0.1)

plt.show()#block=True) # block=True used for separate-window iPython plotting.

