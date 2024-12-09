import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from dataclasses import dataclass, field
import typing

from bspline_approximation import bSplineFittingMat, BSplineFitCalculator
import bspline

import gtCommon as gtc
from gtCommon import BCOT_Data_Calculator, quatsFromAxisAngles

TRANSLATION_THRESH = 20.0#50.0

ROTATION_THRESH_RAD = np.deg2rad(2.0)#5.0)

SPLINE_DEGREE = 1
PTS_USED_TO_CALC_LAST = 5 # Must be at least spline's degree + 1.
DESIRED_CTRL_PTS = 4
if (PTS_USED_TO_CALC_LAST < SPLINE_DEGREE + 1):
    raise Exception("Need at least order=k=(degree + 1) input pts for calc!")
if DESIRED_CTRL_PTS > PTS_USED_TO_CALC_LAST:
    raise Exception("Need at least as many input points as control points!")
if DESIRED_CTRL_PTS < SPLINE_DEGREE + 1:
    raise Exception("Need at least order=k=(degree + 1) control points!")

def wahba(rough_mats):
    U, S, Vh = np.linalg.svd(rough_mats)
    dets_U = np.linalg.det(U)
    dets_V = np.linalg.det(Vh)
    last_diags = dets_U * dets_V
    Vh[:,-1,:] *= last_diags[..., np.newaxis]
    return gtc.einsumMatMatMul(U, Vh)

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
        return gtc.anglesBetweenQuats(values[1:], full_predictions)

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

    def applyBestTranslationResult(self, names, agg_name, use_shift = False):
        ConsolidatedResults._applyBestResult(
            self.translation_results, self._ordered_translation_result_names,
            names, agg_name, TRANSLATION_THRESH, False, use_shift
        )

    def applyBestRotationResult(self, names, agg_name, use_shift = False):
        ConsolidatedResults._applyBestResult(
            self.rotation_results, self._ordered_rotation_result_names,
            names, agg_name, ROTATION_THRESH_RAD, True, use_shift
        )

    def _applyBestResult(results_dict, name_order_list, names, agg_name, thresh, errs_are_1D, use_shift):
        num_combos = len(results_dict[names[0]].errors)
        errs = []
        scores = []
        for i in range(num_combos):
            stacked_errs = np.stack([
                results_dict[n].errors[i] for n in names
            ], axis = 0)
            stacked_norms = stacked_errs
            if not errs_are_1D:
                stacked_norms = np.linalg.norm(stacked_errs, axis = -1)
            min_inds_1D = np.argmin(stacked_norms, axis = 0, keepdims=True)
            if use_shift:
                min_inds_1D = min_inds_1D[:, :-1]
                stacked_errs = stacked_errs[:, 1:]
                stacked_norms = stacked_norms[:, 1:]
            min_inds_e = min_inds_1D
            if not errs_are_1D:
                min_inds_e = min_inds_1D.reshape(1,-1,1)
            es = np.take_along_axis(stacked_errs, min_inds_e, axis = 0)
            min_norms = np.take_along_axis(stacked_norms, min_inds_1D, axis = 0)
            if use_shift:
                default_err = results_dict["Static"].errors[i][0]
                default_norm = default_err
                if not errs_are_1D:
                    default_norm = np.linalg.norm(default_err)
                errs.append(np.insert(es, 0, default_err, axis = 0))
                min_norms = np.insert(min_norms, 0, default_norm, axis = 0)

            scores.append((min_norms <= thresh).mean())
        results_dict[agg_name] = PredictionResult(agg_name, errs, scores)
        name_order_list.append(agg_name)

    def _addPredictionResult(self, all_results_dict, name_order_list, name, errs, score):
        if name in all_results_dict.keys():
            all_results_dict[name].errors.append(errs)
            all_results_dict[name].scores.append(score)
        else:
            all_results_dict[name] = PredictionResult(name, [errs], [score])
            name_order_list.append(name)

    def printTable(names, results_for_names):
        name_lens = []
        for name in names:
            # We want each printed column to be at least 10 digits plus 2 spaces
            # because the default numpy precision in printing is 8, which means
            # that to match it, we want at least 10 chars for the number in the
            # next row, plus a space.
            print("{:>10} ".format(name), end = "")
            name_lens.append(max(10, len(name)))
        print() # Newline after row.
        for i in range(len(names)):
            # Calculate mean and round it to Numpy's default float print digits.
            mean_score = round(results_for_names[i], 8)

            print("{val:>{width}} ".format(val=str(mean_score), width=name_lens[i]), end = "")
        print() # Newline after row.


    def printResults(self):
        ordered_t_score_means = [
            np.array(self.translation_results[n].scores).mean()
            for n in self._ordered_translation_result_names
        ]
        ordered_r_score_means = [
            np.array(self.rotation_results[n].scores).mean()
            for n in self._ordered_rotation_result_names
        ]

        print("Translation Results:")
        ConsolidatedResults.printTable(
            self._ordered_translation_result_names, ordered_t_score_means
        )
        print("\nRotation Results:")
        ConsolidatedResults.printTable(
            self._ordered_rotation_result_names, ordered_r_score_means
        )

combos = []
for b in range(len(gtc.BCOT_BODY_NAMES)):
    for s in range(len(gtc.BCOT_SEQ_NAMES)):
        if BCOT_Data_Calculator.isBodySeqPairValid(b, s):
            combos.append((b,s))

skipAmount = 2


spline_pred_calculator = BSplineFitCalculator(
    SPLINE_DEGREE, DESIRED_CTRL_PTS, PTS_USED_TO_CALC_LAST
)

allResultsObj = ConsolidatedResults()#len(combos))

egFound = False
sampleAccErrs = None
sampleDeltas = None

# A vector that will hold a subset of translations between frames, using these
# to figure out the best fraction to take for the acceleration deltas.
gt_for_accel_deltas = np.empty((0,3))
all_accel_deltas = np.empty((0,3))

all_vel_ratios = np.zeros((0,))
all_vel_graphing_ratios = np.zeros((0,))
all_acc_delta_ratios = np.zeros((0,))
all_acc_ortho_ratios = np.zeros((0,))

other_spline_errs = []

all_speeds = np.zeros((0,))
all_acc_mags = np.zeros((0,))
all_vel_angles = np.zeros((0,))
all_acc_angles = np.zeros((0,))
all_acc_ortho_mags = np.zeros((0,))
all_vel_acc_2D_solves = np.zeros((0, 2))
maxTimestamps = 0
max_angle = 0
for i, combo in enumerate(combos):
    calculator = BCOT_Data_Calculator(combo[0], combo[1], skipAmount)

    translations_gt = calculator.getTranslationsGTNP(True)
    rotations_aa_gt = calculator.getRotationsGTNP(True)
    rotations_gt_quats = gtc.quatsFromAxisAngles(rotations_aa_gt)


    translations = translations_gt #+ np.random.uniform(-4, 4, translations_gt.shape)
    rotations_quats = rotations_gt_quats
    #  = gtc.multiplyQuatLists(
    #     getRandomQuatError(rotations_gt_quats.shape, ROTATION_THRESH_RAD), 
    #     rotations_gt_quats
    # )
    rotations = rotations_aa_gt # TODO: Apply quat error to these.

    allResultsObj.updateGroundTruth(
        translations_gt, rotations_aa_gt, rotations_gt_quats
    )


    translation_diffs = np.diff(translations, axis=0)
    rotation_aa_diffs = np.diff(rotations, axis=0)

    
    
    rev_rotations_quats = gtc.conjugateQuats(rotations_quats)



    rotation_quat_diffs = gtc.multiplyQuatLists(
        rotations_quats[1:], rev_rotations_quats[:-1]
    )

    # At this time, logging the finite difference acceleration at the last
    # timestep isn't required, as there's no further prediction to make with it.
    translations_acc = np.diff(translation_diffs[:-1], axis=0)
    rotations_aa_acc = np.diff(rotation_aa_diffs[:-1], axis=0)

    t_vel_preds = translations[1:-1] + translation_diffs[:-1]
    t_screw_preds = translations[1:-1] + gtc.rotateVecsByQuats(rotation_quat_diffs[:-1], translation_diffs[:-1])
    # t_velLERP_preds = translations[1:-1] + 0.91 * translation_diffs[:-1]
    r_vel_preds = gtc.multiplyQuatLists(
        rotation_quat_diffs[:-1], rotations_quats[1:-1]
    )
    # r_vel_preds = gtc.quatSlerp(rotations_quats[:-2], rotations_quats[1:-1], 2)
    r_aa_vel_preds = rotations[1:-1] + rotation_aa_diffs[:-1]

    # r_slerp_preds = gtc.quatSlerp(rotations_quats[1:-1], r_vel_preds, 0.75)

    t_acc_delta = translation_diffs[1:-1] + (0.5 * translations_acc)
    t_acc_preds = translations[2:-1] + t_acc_delta
    # B-Spline investigation yielded the following result, which is a line of
    # best fit for the last two points and a quadratic-fit third point.
    # The last row in its pseudoinverse is [-1/6 1/3 5/6], meaning that the last
    # point on the line (the 2nd control point) would be:
    # 5/6(3x_2 - 3x_1 + x_0) + 1/3(x_2) - 1/6(x_1)
    # This can be rearranged to x_2 + (x_2 - x_1) + 5/6(x_2 - 2x_1 + x_0),
    # i.e. x_2 + velocity + (5/6) acceleration. 
    t_spline2_preds = translations[2:-1] + translation_diffs[1:-1] + (5/6) * translations_acc
    
    t_accLERP_preds = translations[2:-1] + 0.9 * t_acc_delta
    t_quadratic_preds = 3*translations[2:-1] - 3*translations[1:-2] + translations[:-3]
    t_acc_preds = np.vstack((t_vel_preds[:1], t_acc_preds))
    t_spline2_preds = np.vstack((t_vel_preds[:1], t_spline2_preds))
    t_accLERP_preds = np.vstack((t_vel_preds[:1], t_accLERP_preds))
    t_quadratic_preds = np.vstack((t_vel_preds[:1], t_quadratic_preds))

    # t_deg4_preds = t_quadratic_preds.copy()
    # t_deg4_preds[3:] = (17/24) * translations[:-5] - (11/3) * translations[1:-4] + (31/4) * translations[2:-3] - (25/3)*translations[3:-2] + (109/24)*translations[4:-1]
    t_jerk_preds = t_quadratic_preds.copy()
    t_jerk_preds[2:] = 4 * translations[3:-1] - 6 * translations[2:-2] + 4 * translations[1:-3] - translations[:-4]

    complete_vel_sq_lens = gtc.einsumDot(
        translation_diffs, translation_diffs
    ) # (1-0)^2, (2-1)^2, ...
    prev_vel_sq_lens = complete_vel_sq_lens[:-1]
    speeds = np.sqrt(prev_vel_sq_lens)
    unit_vels = translation_diffs[:-1] / speeds[..., np.newaxis]
    unit_vel_dots = gtc.einsumDot(unit_vels[1:], unit_vels[:-1])
    vel_angles = np.arccos(unit_vel_dots)
    vel_crosses = np.cross(unit_vels[:-1], unit_vels[1:])
    vel_crosses /= np.linalg.norm(vel_crosses, axis=-1, keepdims=True)
    min_vel_rot_aas = gtc.scalarsVecsMul(vel_angles, vel_crosses)
    min_vel_rot_qs = gtc.quatsFromAxisAngles(min_vel_rot_aas)
    rev_min_vel_qs = gtc.conjugateQuats(min_vel_rot_qs)

    # I guess I'm thinking that F2 = V(0>1)*F1*R1
    # Which means R1 = F1^T V(0>1)^T F2
    local_rots_vcase = gtc.multiplyQuatLists(
        rev_rotations_quats[1:-2], gtc.multiplyQuatLists(
            rev_min_vel_qs, rotations_quats[2:-1]
    ))

    r_mvel_plus_local_preds = np.empty(r_vel_preds.shape)
    r_mvel_plus_local_preds[0] = r_vel_preds[0]
    r_mvel_plus_local_preds[1:] = gtc.multiplyQuatLists(
        min_vel_rot_qs, gtc.multiplyQuatLists(
            rotations_quats[2:-1], local_rots_vcase
        )
    )


    # unit_vel_test = gtc.rotateVecsByQuats(min_vel_rot_qs, unit_vels[:-1])
    # if np.max(np.abs(unit_vel_test - unit_vels[1:])) > 0.0001:
    #     raise Exception("Made a mistake!")

    t_poly_preds = t_quadratic_preds.copy()
    vel_bounce = (unit_vel_dots < -0.99)
    t_poly_preds[1:][vel_bounce] = t_vel_preds[1:][vel_bounce]

    r_aa_acc_delta = rotation_aa_diffs[1:-1] + (0.5 * rotations_aa_acc)
    r_aa_acc_preds = rotations[2:-1] + r_aa_acc_delta
    # r_aa_accLERP_preds = rotations[2:-1] + 0.5 * r_aa_acc_delta
    r_aa_acc_preds = np.vstack((r_aa_vel_preds[:1], r_aa_acc_preds))
    # r_aa_accLERP_preds = np.vstack((r_aa_vel_preds[:1], r_aa_accLERP_preds))


    r_squad_preds = gtc.squad(rotations_quats, 2)[:-1]
    r_squad_preds = np.vstack((r_vel_preds[:1], r_squad_preds))

    # num_spline_preds = (len(translations) - 1) - (SPLINE_DEGREE) 

    r_mats = calculator.getRotationMatsGTNP(True)
    fixed_axes = rotation_quat_diffs[:, 1:] / np.linalg.norm(rotation_quat_diffs[:, 1:], axis=-1, keepdims=True)# np.sin(angles/2)[..., np.newaxis]
    r_fixed_axis_mats = np.empty((len(rotations) - 2, 3, 3))
    for j in range(len(rotations) - 2):
        r_fixed_axis_mats[j] = gtc.closestFrameAboutAxis(r_mats[j+1], r_mats[j+2], fixed_axes[j])
    r_fixed_axis_bcs = gtc.axisAngleFromMatArray(r_fixed_axis_mats)

    # Convert mats into "separated" lists of their columns.
    # wahba_points_local = np.moveaxis(r_mats, -1, 0)
    wahba_inputs = 2 * r_mats[1:-1] - r_mats[:-2]
    wahba_col_norms = np.linalg.norm(wahba_inputs, axis=1)
    for j in range(3):
        wahba_inputs[..., j] /= wahba_col_norms[..., j:j+1]
    wahba_outputs = wahba(wahba_inputs)

    wahba_pred = gtc.axisAngleFromMatArray(wahba_outputs)

    angles = gtc.anglesBetweenQuats(rotations_quats[1:], rotations_quats[:-1]).flatten()
    max_angle = max(max_angle, angles.max())

    angle_diffs = np.diff(angles, 1, axis=0)
    axis_dots = gtc.einsumDot(fixed_axes[1:], fixed_axes[:-1])
    angle_diffs[axis_dots < np.cos(np.deg2rad(1))] = 0
    angle_ratios = 2 + (angle_diffs[:-1]/angles[1:-1])

    r_fixed_axis_preds = np.empty((len(rotations) - 2, 4))
    r_fixed_axis_preds[1:] = gtc.quatSlerp(rotations_quats[1:-2], rotations_quats[2:-1], angle_ratios)
    r_fixed_axis_preds[0] = r_vel_preds[0]




    # slerp_diffs = gtc.multiplyQuatLists(r_slerp_preds, rev_rotations_quats[1:-1])
    # slerp_axes = slerp_diffs[:, 1:] / np.linalg.norm(slerp_diffs[:, 1:], axis=-1, keepdims=True)
    # axisComp = np.abs(gtc.einsumDot(slerp_axes, fixed_axes[:-1]))
    # if np.any(axisComp < 0.999):
    #     raise Exception("Axes don't match!")
    

    vel_dots = gtc.einsumDot(translation_diffs[1:], translation_diffs[:-1])
    # (1-0)*(2-1), (2-1)*(3-2), ...
    vel_proj_scalars = vel_dots / prev_vel_sq_lens
    # p(2-1)onto(1-0), p(3-2)onto(2-1), ...
    all_vel_ratios = np.concatenate((all_vel_ratios, vel_proj_scalars))
    all_vel_graphing_ratios = np.concatenate((
        all_vel_graphing_ratios, vel_proj_scalars[1:]))
    # p(3-2)onto(2-1), p(4-3)onto(3-2), ...

    best_vel_preds = translations[1:-1] + np.einsum("i,ij->ij", vel_proj_scalars, translation_diffs[:-1])

    acc_delta_sq_lens = gtc.einsumDot(t_acc_delta, t_acc_delta)
    # (2-0)^2, (3-1)^2, ...
    acc_delta_dots = gtc.einsumDot(t_acc_delta, translation_diffs[2:])
    # (2-0)*(3-2), (3-1)*(4-3), ...
    acc_delta_proj_scalars = acc_delta_dots / acc_delta_sq_lens
    # p(3-2)onto(2-0), p(4-3)onto(3-1), ...
    all_acc_delta_ratios = np.concatenate((
        all_acc_delta_ratios, acc_delta_proj_scalars
    ))

    acc_vel_dots = gtc.einsumDot(translations_acc, translation_diffs[1:-1])
    # (2-0)*(2-1), (3-1)*(3-2), ...
    acc_parallel_scalars = acc_vel_dots / prev_vel_sq_lens[1:]
    # p(2-0)onto(2-1), p(3-1)onto(3-2), ...
    acc_ortho_vecs = translations_acc - (acc_parallel_scalars[..., np.newaxis] * translation_diffs[1:-1])
    # test_ortho = gtc.einsumDot(acc_ortho_vecs, translation_diffs[1:-1])
    # print(test_ortho.max())
    acc_ortho_sq_lens = gtc.einsumDot(acc_ortho_vecs, acc_ortho_vecs)
    acc_ortho_dots = gtc.einsumDot(acc_ortho_vecs, translation_diffs[2:])
    acc_ortho_ratios = 2 * acc_ortho_dots / acc_ortho_sq_lens
    all_acc_ortho_ratios = np.concatenate((
        all_acc_ortho_ratios, acc_ortho_ratios
    ))

    all_speeds = np.concatenate((all_speeds, speeds[1:]))
    all_acc_mags = np.concatenate((all_acc_mags, np.linalg.norm(translations_acc, axis = -1)))
    all_acc_ortho_mags = np.concatenate((all_acc_ortho_mags, np.sqrt(acc_ortho_sq_lens)))
    
    all_vel_angles = np.concatenate((all_vel_angles, vel_dots[:-1]/(speeds[1:] * speeds[:-1])))
    # Dots are: (1-0)*(2-1), (2-1)*(3-2), ...
    
    acc_angle_ratios = 0.63837237 * acc_parallel_scalars + 0.83709242
    acc_angle_preds = translations[2:-1] + np.einsum('i,ij->ij', acc_angle_ratios, translation_diffs[1:-1])
    acc_angle_preds = np.vstack((t_vel_preds[:1], acc_angle_preds))
     
    # p(2-0)onto(2-1), p(3-1)onto(3-2), ...

    vel_lens = np.sqrt(prev_vel_sq_lens[1:]) # |2-1|, |3-2|, ...
    # vel 2D vector will be [speed, 0]
    acc_parallel_comp = acc_vel_dots / vel_lens
    acc_ortho_comp = np.sqrt(acc_ortho_sq_lens)
    inv_mats = np.empty((len(vel_lens), 2, 2))
    # acc 2D vector will be [acc_parallel, acc_ortho]
    # matrix is [[speed, acc_parallel], [0, acc_ortho]]
    # det is speed * acc_ortho
    # inverse is (1/speed*acc_ortho) * [[acc_ortho, -acc_parallel],[0,speed]]
    inv_mats[:, 0, 0] = 1.0 / vel_lens
    inv_mats[:, 0, 1] = -acc_parallel_comp / (vel_lens * acc_ortho_comp)
    inv_mats[:, 1, 0] = 0
    inv_mats[:, 1, 1] = 1.0 / acc_ortho_comp
    
    # Then multiply it by [next_vel_parallel, next_vel_ortho]
    next_vel_parallel = vel_dots[1:] / vel_lens
    next_vel_ortho_dots = gtc.einsumDot(translation_diffs[2:], acc_ortho_vecs)
    next_vel_vecs = np.empty((len(vel_lens), 2))
    next_vel_vecs[:, 0] = next_vel_parallel
    next_vel_vecs[:, 1] = next_vel_ortho_dots / acc_ortho_comp

    vel_acc_2D_solve = gtc.einsumMatVecMul(inv_mats, next_vel_vecs)
    all_vel_acc_2D_solves = np.concatenate((
        all_vel_acc_2D_solves, vel_acc_2D_solve
    ))

    complete_vel_sq_lens = gtc.einsumDot(
        translation_diffs, translation_diffs
    ) # (1-0)^2, (2-1)^2, ...
    prev_vel_sq_lens = complete_vel_sq_lens[:-1]

    vel_proj_scalars = vel_dots / prev_vel_sq_lens
    # p(2-1)onto(1-0), p(3-2)onto(2-1), ...
    all_vel_ratios = np.concatenate((all_vel_ratios, vel_proj_scalars))

    acc_vel_dots = gtc.einsumDot(translations_acc, translation_diffs[1:-1])
    # (2-0)*(2-1), (3-1)*(3-2), ...
    acc_parallel_scalars = acc_vel_dots / prev_vel_sq_lens[1:]

    all_acc_angles = np.concatenate((all_acc_angles, acc_parallel_scalars))

    best_2d = t_acc_preds.copy()
    best_2d_vel = vel_acc_2D_solve[:, :1] * translation_diffs[1:-1]
    best_2d_acc = vel_acc_2D_solve[:, 1:] * translations_acc
    best_2d[1:] = translations[2:-1] + best_2d_vel + best_2d_acc

    # all_lens = [len(arr) for arr in [
    #     all_speeds, all_acc_mags, all_acc_ortho_mags, all_vel_angles, 
    #     all_acc_angles, all_acc_delta_ratios, all_acc_ortho_ratios,
    #     all_vel_graphing_ratios
    # ]]
    # if np.any(np.diff(all_lens) != 0):
    #     raise Exception ("Array slice len error!")



    # t_spline_preds = np.empty((num_spline_preds, 3))
    # r_aa_spline_preds = np.empty((num_spline_preds, 3))
    # for j in range(SPLINE_DEGREE, len(translations) - 1):
    #     startInd = max(0, j - PTS_USED_TO_CALC_LAST + 1)
    #     ctrlPtCount = min(j + 1, DESIRED_CTRL_PTS)
    #     uInterval = (SPLINE_DEGREE, ctrlPtCount)#j + 1 - startInd)
    #     numTotal = j + 1 - startInd + 1
    #     knotList = np.arange(ctrlPtCount + SPLINE_DEGREE + 1)
    
    
    #     uVals = np.linspace(uInterval[0], uInterval[1], numTotal)

    #     ptsToFit_j = np.empty((j + 1 - startInd, 6))
    #     ptsToFit_j[:, :3] = translations[startInd:(j+1)]
    #     ptsToFit_j[:, 3:] = rotations[startInd:(j+1)]
    #     ctrlPts = np.empty((ctrlPtCount, 0)) 
    #     # Note: Use of pseudoinverse is much faster than calling linalg.lstsqr
    #     # each iteration and gives results that, so far, seem identical.
    #     mat = splineMats[min(j - (SPLINE_DEGREE), len(splineMats) - 1)]

    #     ctrlPts = (mat @ ptsToFit_j)

    #     next_spline_pt = bspline.bSplineInner(
    #         uVals[-1], SPLINE_DEGREE + 1, ctrlPtCount - 1, ctrlPts, knotList
    #     )
    #     t_spline_preds[j - (SPLINE_DEGREE)] = next_spline_pt[:3]
    #     r_aa_spline_preds[j - (SPLINE_DEGREE)] = next_spline_pt[3:]

    #---------------------------------------------------------------
    # all_spline_preds = spline_pred_calculator.fitAllData(np.hstack((
    #     translations, rotations
    # )))
    # t_spline_preds = all_spline_preds[:, :3]
    # r_aa_spline_preds = all_spline_preds[:, 3:]
    t_spline_preds = spline_pred_calculator.fitAllData(translations)

    # t_spline_preds = np.vstack((t_vel_preds[:1], t_spline_preds))
    
    other_spline_err = t_spline_preds - translations[SPLINE_DEGREE + 1:]
    other_spline_err_norms = np.linalg.norm(other_spline_err, axis = -1)
    other_spline_errs.append((other_spline_err_norms < TRANSLATION_THRESH).mean())

    allResultsObj.addAxisAngleResult("Static", rotations[:-1])
    allResultsObj.addQuaternionResult("QuatVel", r_vel_preds)
    # allResultsObj.addQuaternionResult("QuatVelSLERP", r_slerp_preds)

    allResultsObj.addAxisAngleResult("AA_Vel", r_aa_vel_preds)
    allResultsObj.addAxisAngleResult("AA_Acc", r_aa_acc_preds)
    # allResultsObj.addAxisAngleResult("AA_AccLERP", r_aa_accLERP_preds)
    # allResultsObj.addAxisAngleResult("AA_Spline", r_aa_spline_preds)
    allResultsObj.addAxisAngleResult("Fixed axis bcs", r_fixed_axis_bcs)
    allResultsObj.addQuaternionResult("Fixed axis acc", r_fixed_axis_preds)
    allResultsObj.addAxisAngleResult("Wahba", wahba_pred)

    # allResultsObj.addQuaternionResult("SQUAD", r_squad_preds)
    allResultsObj.addQuaternionResult("Min vel-align", r_mvel_plus_local_preds)

    # Dumb comment.
    allResultsObj.addTranslationResult("Static", translations[:-1])
    allResultsObj.addTranslationResult("Vel", t_vel_preds)
    # allResultsObj.addTranslationResult("VelLERP", t_velLERP_preds)
    allResultsObj.addTranslationResult("Vel (bcs)", best_vel_preds)
    allResultsObj.addTranslationResult("Vel (ang)", acc_angle_preds)
    allResultsObj.addTranslationResult("Acc", t_acc_preds)
    allResultsObj.addTranslationResult("AccLERP", t_accLERP_preds)
    allResultsObj.addTranslationResult("2D (bcs)", best_2d)
    allResultsObj.addTranslationResult("Quadratic", t_quadratic_preds)
    # allResultsObj.addTranslationResult("deg4", t_deg4_preds)
    allResultsObj.addTranslationResult("Jerk", t_jerk_preds)
    allResultsObj.addTranslationResult("Spline", t_spline_preds)
    allResultsObj.addTranslationResult("Spline2", t_spline2_preds)
    allResultsObj.addTranslationResult("Screw", t_screw_preds)
    # allResultsObj.addTranslationResult("StatVelAcc", t_poly_preds)
    
    maxTimestamps = max(
        maxTimestamps, len(calculator.getTranslationsGTNP(False))
    )
    if (not egFound):
        lastLerpScore = allResultsObj.translation_results["AccLERP"].scores[-1]
        accResult = allResultsObj.translation_results["Acc"]
        lastAccScore = accResult.scores[-1]
        if lastLerpScore > lastAccScore:
            egFound = True

            sampleAccErrs = accResult.errors[-1]
            t_deltas_n = np.linalg.norm(t_acc_delta, axis=-1, keepdims=True)
            sampleDeltas = t_acc_delta / t_deltas_n

allResultsObj.applyBestRotationResult(["QuatVel", "Fixed axis acc", "Static"], "agg", True)
allResultsObj.applyBestRotationResult(["Wahba", "Static"], "aggw", True)
allResultsObj.applyBestRotationResult(["QuatVel", "Min vel-align"], "aggv", True)
# allResultsObj.applyBestTranslationResult(["Static", "Vel", "Quadratic", "Screw"], "agg", True)
# allResultsObj.applyBestTranslationResult(["Static", "Vel", "Quadratic", "Jerk"], "jagg", True)


print("Max angle:", max_angle)

allResultsObj.printResults()
stat_headers = ["Mean", "Min", "Max", "Median", "std"]
def stat_results(vals):
    return [ vals.mean(), vals.min(), vals.max(), np.median(vals), np.std(vals)]

vel_stat_results = stat_results(all_vel_ratios)
acc_delta_stat_results = stat_results(all_acc_delta_ratios)
acc_ortho_stat_results = stat_results(all_acc_ortho_ratios)

print("All vel ratios stats:")
ConsolidatedResults.printTable(stat_headers, vel_stat_results)
print()

print("All acc delta ratios stats:")
ConsolidatedResults.printTable(stat_headers, acc_delta_stat_results)
print()

print("All acc ortho ratios stats:")
ConsolidatedResults.printTable(stat_headers, acc_ortho_stat_results)
print()

fig = plt.figure(0) # Arg of "0" means same figure reused if cell ran again.
fig.clear() # Good to do for iPython running, if running a plot cell again.
ax = fig.subplots()
ax.hist(all_vel_ratios, bins = 25,  range=(-2,3))
plt.show()

#%%
fig = plt.figure(0) # Arg of "0" means same figure reused if cell ran again.
fig.clear() # Good to do for iPython running, if running a plot cell again.
ax = fig.subplots()
vel_acc_start = 10 * 42
vel_acc_end = vel_acc_start + 20
ax.plot(all_vel_graphing_ratios[vel_acc_start:vel_acc_end], label = "v")
ax.plot(all_acc_delta_ratios[vel_acc_start:vel_acc_end], label = "v + a/2")
ax.plot(all_acc_ortho_ratios[vel_acc_start:vel_acc_end], label = "a ortho")

ax.plot(all_vel_acc_2D_solves[vel_acc_start:vel_acc_end, 0], label = "v*")
ax.plot(all_vel_acc_2D_solves[vel_acc_start:vel_acc_end, 1], label = "a*")

ax.plot(0.01 * all_speeds[vel_acc_start:vel_acc_end], label = "speed")
ax.plot(0.01 * all_acc_mags[vel_acc_start:vel_acc_end], label = "acc mag")

acc_ang_thing = 0.63837 * all_acc_angles[vel_acc_start:vel_acc_end] + 0.837 
ax.plot(all_vel_angles[vel_acc_start:vel_acc_end] - 1, label = "vel ang")
ax.plot(all_acc_angles[vel_acc_start:vel_acc_end] - 1, label = "acc-vel angle")
ax.plot(acc_ang_thing, label = "^ transformed")


ax.plot(-0.01 * all_acc_ortho_mags[vel_acc_start:vel_acc_end] - 3, label = "acc ortho mags")
ax.set_ylim(-5, 5)
ax.legend()
plt.show()
#%%
fig = plt.figure(0) # Arg of "0" means same figure reused if cell ran again.
fig.clear() # Good to do for iPython running, if running a plot cell again.
ax = fig.subplots()
# acc_ang_subset = all_acc_angles[vel_acc_start:vel_acc_end]
ax.scatter(all_acc_angles, all_vel_graphing_ratios)
plt.show()

#%%

plt.show()



allResultsObj.printResults()


numTimestamps = len(translations_gt)
timestamps = np.arange(numTimestamps)
tx_coords = np.stack((timestamps, translations[:, 2]), axis = -1)
ptx_coords = np.stack((timestamps[2:], t_vel_preds[:, 2]), axis = -1)

v_line_coords = np.hstack((tx_coords[:-2], ptx_coords)).reshape((len(ptx_coords), 2, 2))

#%%
acc_err_x = np.einsum('ij,ij->i', sampleDeltas, sampleAccErrs[2:])
acc_err_sqr_len = np.einsum('ij,ij->i', sampleAccErrs[2:], sampleAccErrs[2:])
acc_err_y = np.sqrt(acc_err_sqr_len - (acc_err_x**2))


# Randomly negate about half of the y values.
negate_mask = np.random.rand(len(acc_err_y)) < 0.5
acc_err_y[negate_mask] *= -1

# Create line segments from (0, 0) to each (x, y).
points = np.column_stack((acc_err_x, acc_err_y))
origin = np.array([0, 0])
segments = np.array([[origin, point] for point in points])

# Create the LineCollection with a colormap that encodes the frame numbers.
colors = (skipAmount + 1) * np.arange(len(acc_err_x))
lc = LineCollection(segments, cmap='viridis', array=colors, linewidths=2)
fig = plt.figure(0, figsize=(8, 8))
fig.clear()
ax = fig.subplots()
ax.add_collection(lc)
cbar = plt.colorbar(lc, ax=ax, orientation='vertical')
cbar.set_label('Frame number')

# Set the viewport to show all lines.
ax.set_xlim(min(acc_err_x) - 1, max(acc_err_x) + 1)
ax.set_ylim(min(acc_err_y) - 1, max(acc_err_y) + 1)
ax.set_aspect('equal', 'box')



plt.title('Errors relative to extrapolation vector')
plt.xlabel('Parallel to extrapolation')
plt.ylabel('Orthogonal to extrapolation')
plt.grid(True)
plt.show()

#%%
fig = plt.figure(0) # Arg of "0" means same figure reused if cell ran again.
fig.clear() # Good to do for iPython running, if running a plot cell again.
ax = fig.subplots()

vlc = LineCollection(v_line_coords)

ax.add_collection(vlc)
ax.autoscale()
ax.margins(0.1)

plt.show()#block=True) # block=True used for separate-window iPython plotting.

#%%
fig = plt.figure(0) # Arg of "0" means same figure reused if cell ran again.
fig.clear() # Good to do for iPython running, if running a plot cell again.
ax = fig.subplots()

from matplotlib.colors import to_rgba

vid_ind = 9
keys = ["Vel", "Acc"]#, "VelLERP", "Acc", "AccLERP"]
key_colours = ["blue", "orange", "green", "red"]
key_rgbas = [to_rgba(kc) for kc in key_colours]

num_timesteps = len(allResultsObj.translation_results[keys[0]].errors[vid_ind])
x_vals = np.arange(1, num_timesteps + 1)
all_err_norms = []
for i, key in enumerate(keys):
    errs = allResultsObj.translation_results[key].errors[vid_ind]
    err_norms = np.linalg.norm(errs, axis = -1)
    ax.plot(x_vals, err_norms, label=key,
            color=key_colours[i])
    all_err_norms.append(err_norms)
double_xs = np.repeat(np.arange(num_timesteps + 1), np.full(num_timesteps + 1, 2))
best_line_xs = double_xs[1:-1].reshape((-1, 2, 1))
best_line_cos = np.dstack((best_line_xs, np.zeros_like(best_line_xs)))
all_err_norms_np = np.array(all_err_norms)
best_inds = np.argmin(all_err_norms_np, axis=0)
best_colors = [key_rgbas[i] for i in best_inds]
blc = LineCollection(best_line_cos, colors=best_colors)

ax.add_collection(blc)

ax.set_ylim(bottom = -5)
ax.legend()
plt.show()

