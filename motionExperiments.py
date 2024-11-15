import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from dataclasses import dataclass, field
import typing

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


@dataclass
class PredictionResult:
    name: str
    # predictions: np.ndarray
    errors: typing.List[np.ndarray]
    scores: typing.List[float]

class ConsolidatedResults:
    def __init__(self): #, numScenarios: int):
        self.translation_results = dict() # Name -> PredictionResult
        self.rotation_results = dict() # Name -> PredictionResult

        # Order in which to display results.
        self._ordered_translation_result_names = []
        self._ordered_rotation_result_names = []

        # self.body_seq_to_row = dict() # (bodyID: int , seqID: int) -> int

        self._translations_gt = None
        self._axisangles_gt = None
        self._quaternions_gt = None

    # Predictions that require multiple prior points may lack a prediction for
    # the 2nd point, 3rd point, etc. In this case, the CV version of the code
    # would use a different prediction method (including assuming no motion).
    def prependMissingPredictions(backup_predictions, predictions):
        numMissing = (len(backup_predictions) - 1) - len(predictions)
        return np.vstack((backup_predictions[0:numMissing], predictions))

    # Note: returns errors in radians!
    def getQuatError(values, predictions):
        full_predictions = ConsolidatedResults.prependMissingPredictions(
            values, predictions
        )
        return gtCommon.anglesBetweenQuats(values[1:], full_predictions)

    # Note: returns errors in radians!
    def getAxisAngleError(values, predictions): 
        v_qs = quatsFromAxisAngles(values)
        p_qs = quatsFromAxisAngles(predictions)
        return ConsolidatedResults.getQuatError(v_qs, p_qs)

    def getTranslationError(values, predictions):
        full_predictions = ConsolidatedResults.prependMissingPredictions(
            values, predictions
        )

        return values[1:] - full_predictions
    
    def updateGroundTruth(self, translations, axisangles, quats): # bod_ID, seq_ID)
        # self.current_ID = (bod_ID, seq_ID)
        self._translations_gt = translations
        self._axisangles_gt = axisangles
        self._quaternions_gt = quats
        t_perfect_errs = ConsolidatedResults.getTranslationError(
            translations, translations[1:]
        )
        t_perfect_score = (t_perfect_errs <= 0.00001).mean()
        self._addPredictionResult(
            self.translation_results, self._ordered_translation_result_names, 
            "Perfect", t_perfect_errs, t_perfect_score
        )

        r_perfect_errs = ConsolidatedResults.getAxisAngleError(
            axisangles, axisangles[1:]
        )
        r_perfect_score = (r_perfect_errs <= 0.00001).mean()
        self._addPredictionResult(
            self.rotation_results, self._ordered_rotation_result_names,
            "Perfect", r_perfect_errs, r_perfect_score
        )

        # self.body_seq_to_row[self.current_ID] = self.curr_row_count
        # self.curr_row_count += 1

    def addTranslationResult(self, name, predictions):
        errs = ConsolidatedResults.getTranslationError(
            self._translations_gt, predictions
        )
        # Get the norm of each vec3, and find the ratio under the threshold.
        score = (np.linalg.norm(errs, axis = -1) <= TRANSLATION_THRESH).mean()
        self._addPredictionResult(
            self.translation_results, self._ordered_translation_result_names,
            name, errs, score
        )

    def addAxisAngleResult(self, name, predictions):
        errs = ConsolidatedResults.getAxisAngleError(
            self._axisangles_gt, predictions
        )
        score = (errs <= ROTATION_THRESH_RAD).mean()
        self._addPredictionResult(
            self.rotation_results, self._ordered_rotation_result_names,
            name, errs, score)

    def addQuaternionResult(self, name, predictions):
        errs = ConsolidatedResults.getQuatError(
            self._quaternions_gt, predictions
        )
        score = (errs <= ROTATION_THRESH_RAD).mean()
        self._addPredictionResult(
            self.rotation_results, self._ordered_rotation_result_names,
            name, errs, score
        )

    def _addPredictionResult(self, all_results_dict, name_order_list, name, errs, score):
        if name in all_results_dict.keys():
            all_results_dict[name].errors.append(errs)
            all_results_dict[name].scores.append(score)
        else:
            all_results_dict[name] = PredictionResult(name, [errs], [score])
            name_order_list.append(name)

    def _printTable(names, results_for_names):
        name_lens = []
        for name in names:
            # We want each printed column to be at least 10 digits plus 2 spaces
            # because the default numpy precision in printing is 8, which means
            # that to match it, we want at least 10 chars for the number in the
            # next row, plus a space.
            print("{:>10} ".format(name), end = "")
            name_lens.append(max(10, len(name)))
        print() # Newline after row.
        for i, name in enumerate(names):
            scores = results_for_names[name].scores
            # Calculate mean and round it to Numpy's default float print digits.
            mean_score = round(np.array(scores).mean(), 8)

            print("{val:>{width}} ".format(val=str(mean_score), width=name_lens[i]), end = "")
        print() # Newline after row.


    def printResults(self):
        print("Translation Results:")
        ConsolidatedResults._printTable(
            self._ordered_translation_result_names, self.translation_results
        )
        print("\nRotation Results:")
        ConsolidatedResults._printTable(
            self._ordered_rotation_result_names, self.rotation_results
        )

combos = []
for b in range(len(gtCommon.BCOT_BODY_NAMES)):
    for s in range(len(gtCommon.BCOT_SEQ_NAMES)):
        if BCOT_Data_Calculator.isBodySeqPairValid(b, s):
            combos.append((b,s))

skipAmount = 2

splineMats = []
for i in range(SPLINE_DEGREE, PTS_USED_TO_CALC_LAST):
    # startInd = max(0, i - PTS_USED_TO_CALC_LAST + 1)
    ctrlPtCount = min(i + 1, DESIRED_CTRL_PTS)
    numTotal = i + 2 #1 - startInd + 1
    knotList = np.arange(ctrlPtCount + SPLINE_DEGREE + 1)

    uInterval = (SPLINE_DEGREE, ctrlPtCount)# - startInd)
    uVals = np.linspace(uInterval[0], uInterval[1], numTotal)

    matA = bspline.bSplineFittingMat(
        ctrlPtCount, SPLINE_DEGREE + 1, i + 1, uVals, knotList
    )
    # print(matA)
    pseudoInv = np.linalg.inv(matA.transpose() @ matA) @ matA.transpose()

    splineMats.append(pseudoInv)

consolidatedResultObj = ConsolidatedResults()
for i, combo in enumerate(combos):
    calculator = BCOT_Data_Calculator(combo[0], combo[1], skipAmount)

    translations_gt = calculator.getTranslationsGTNP(True)
    rotations_aa_gt = calculator.getRotationsGTNP(True)
    rotations_gt_quats = gtCommon.quatsFromAxisAngles(rotations_aa_gt)


    translations = translations_gt + np.random.uniform(-4, 4, translations_gt.shape)
    rotations_quats = gtCommon.multiplyQuatLists(
        getRandomQuatError(rotations_gt_quats.shape, ROTATION_THRESH_RAD), 
        rotations_gt_quats
    )
    rotations = rotations_aa_gt # TODO: Apply quat error to these.

    consolidatedResultObj.updateGroundTruth(
        translations_gt, rotations_aa_gt, rotations_gt_quats
    )

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
        ctrlPtCount = min(j + 1, DESIRED_CTRL_PTS)
        uInterval = (SPLINE_DEGREE, ctrlPtCount)#j + 1 - startInd)
        numTotal = j + 1 - startInd + 1
        knotList = np.arange(ctrlPtCount + SPLINE_DEGREE + 1)
    
    
        uVals = np.linspace(uInterval[0], uInterval[1], numTotal)

        ptsToFit_j = np.empty((j + 1 - startInd, 6))
        ptsToFit_j[:, :3] = translations[startInd:(j+1)]
        ptsToFit_j[:, 3:] = rotations[startInd:(j+1)]
        ctrlPts = np.empty((ctrlPtCount, 0)) 
        # Note: Use of pseudoinverse is much faster than calling linalg.lstsqr
        # each iteration and gives results that, so far, seem identical.
        mat = splineMats[min(j - (SPLINE_DEGREE), len(splineMats) - 1)]

        ctrlPts = (mat @ ptsToFit_j)

        next_spline_pt = bspline.bSplineInner(
            uVals[-1], SPLINE_DEGREE + 1, ctrlPtCount - 1, ctrlPts, knotList
        )
        t_spline_preds[j - (SPLINE_DEGREE)] = next_spline_pt[:3]
        r_aa_spline_preds[j - (SPLINE_DEGREE)] = next_spline_pt[3:]
    t_spline_preds = np.vstack((t_vel_preds[:1], t_spline_preds))

    consolidatedResultObj.addAxisAngleResult("Static", rotations[:-1])
    consolidatedResultObj.addQuaternionResult("QuatVel", r_vel_preds)
    consolidatedResultObj.addQuaternionResult("QuatVelSLERP", r_slerp_preds)

    consolidatedResultObj.addAxisAngleResult("AA_Vel", r_aa_vel_preds)
    consolidatedResultObj.addAxisAngleResult("AA_Acc", r_aa_acc_preds)
    consolidatedResultObj.addAxisAngleResult("AA_AccLERP", r_aa_accLERP_preds)
    consolidatedResultObj.addAxisAngleResult("AA_Spline", r_aa_spline_preds)


    # Dumb comment.
    consolidatedResultObj.addTranslationResult("Static", translations[:-1])
    consolidatedResultObj.addTranslationResult("Vel", t_vel_preds)
    consolidatedResultObj.addTranslationResult("Acc", t_acc_preds)
    consolidatedResultObj.addTranslationResult("AccLERP", t_accLERP_preds)
    consolidatedResultObj.addTranslationResult("Spline", t_spline_preds)


consolidatedResultObj.printResults()

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

