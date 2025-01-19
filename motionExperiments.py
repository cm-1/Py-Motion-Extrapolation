# from dataclasses import dataclass, field
import typing
from enum import Enum

import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider


from spline_approximation import SplinePredictionMode, BSplineFitCalculator
from cinpact import CinpactAccelExtrapolater

import gtCommon as gtc
from gtCommon import BCOT_Data_Calculator

import posemath as pm
import poseextrapolation as pex

ADD_NUMERICAL_TESTS = False

SPLINE_DEGREE = 1
PTS_USED_TO_CALC_LAST = 5 # Must be at least spline's degree + 1.
DESIRED_CTRL_PTS = 4
if (PTS_USED_TO_CALC_LAST < SPLINE_DEGREE + 1):
    raise Exception("Need at least order=k=(degree + 1) input pts for calc!")
if DESIRED_CTRL_PTS > PTS_USED_TO_CALC_LAST:
    raise Exception("Need at least as many input points as control points!")
if DESIRED_CTRL_PTS < SPLINE_DEGREE + 1:
    raise Exception("Need at least order=k=(degree + 1) control points!")

# Function that replicates velocity-of-pts-on-local-sphere Wahba model by
# finding the right angle to rotate by, since axis can be shown to be the same
# as in const-angular-velocity model.
def angWahbaFunc(x: np.ndarray):
    numerator = 2.0 * np.sin(x)
    denominator = 2.0 * np.cos(x) - 1.0
    return np.arctan2(numerator, denominator) - x

def generalizedLogistic(x: np.ndarray, alpha: float, beta: float):
    return (1 + np.exp(-beta*x))**(-alpha) - 0.5

def gudermannian(x: np.ndarray):
    return 2.0 * np.arctan(np.tanh(x / 2.0))

def algebraicSigmoid(x: np.ndarray, k: float, scale: float):
    denom = (1.0 + np.abs(x/scale)**k)**(1.0/k)
    return x / denom

# rough_mats are the approximate frame axes.
def wahba(rough_mats):
    U, S, Vh = np.linalg.svd(rough_mats)
    dets_U = np.linalg.det(U)
    dets_V = np.linalg.det(Vh)
    last_diags = dets_U * dets_V
    Vh[:,-1,:] *= last_diags[..., np.newaxis]
    return pm.einsumMatMatMul(U, Vh)

# Take in approximate axes in a different frame than the one being solved for.
def wahbaMoreGeneral(rough_mats, reference_mat):
    sum_mat = np.zeros(rough_mats.shape)
    for i in range(3):
        sum_mat += np.einsum('bi,j->bij', rough_mats[:, :, i], reference_mat[:, i])
    return wahba(sum_mat)

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

# Numerical TODO items:
# -  (skip1 test is all that's left) Godot thing
# -  RK4: dot-mag
# - Only use RK4 at all when fa angle decreasing.
# - Only use (RK4, fa-acc) when acc is within some sort of time-based limit.
# -   dot-mag fixed-axis.
def numericalIdeas(angles, fixed_axes, angle_diffs, bcfas, next_bcfas, rotations_quats):
    # Stuff that's needed for all of the predictions below:
    ang_vel_vecs = pm.scalarsVecsMul(angles[:-1], fixed_axes[:-1])
    ang_acc_vecs = np.diff(ang_vel_vecs, 1, axis=0)
    # ang_vel_vecs[1:] += 0.5 * ang_acc_vecs   
    # ang_vel_vecs[0] = ang_vel_vecs[1] - ang_acc_vecs[0]

    # Stuff that's used for at least two of the predictions below:
    extrap_ang_vel_vecs = ang_vel_vecs[1:] + ang_acc_vecs
    num_start_vecs = len(ang_vel_vecs[1:])
    interp_ang_vels = np.linspace(ang_vel_vecs[1:], extrap_ang_vel_vecs, 33, axis=1)


    constspeed_extrap_vecs = pm.scalarsVecsMul(
        angles[1:-1], pm.normalizeAll(extrap_ang_vel_vecs)
    )

    interp_ang_vels2 = np.linspace(ang_vel_vecs[1:], constspeed_extrap_vecs, 33, axis=1)

    
    num_substeps_slerp = 33
    constspeed_slerp_ang_vels = np.empty((
        num_start_vecs, num_substeps_slerp + 1, 3))
    constspeed_slerp_ang_vels[:, 0, :] = ang_vel_vecs[1:]
    constspeed_slerp_ang_vels[:, -1, :] = constspeed_extrap_vecs
    for i in range(1, num_substeps_slerp):
        # The explicit float conversion here isn't needed in Python, but it
        # serves as a reminder/error-proofing in case this code is ever moved
        # over to another language like C++ where it does matter.
        constspeed_slerp_ang_vels[:, i, ] = pm.quatSlerp(
            ang_vel_vecs[1:], constspeed_extrap_vecs, float(i)/num_substeps_slerp
        )
    
    # |ax + (1-a)y|^2 = a^2|x|^2 + (1-a)^2|y|^2 + a(1-a)*dot(x,y)
    sq_ang_vel_lens_ALT = pm.einsumDot(ang_vel_vecs, ang_vel_vecs)
    sq_ang_vel_lens = angles[:-1]**2
    sq_extrap_ang_vel_lens = pm.einsumDot(extrap_ang_vel_vecs, extrap_ang_vel_vecs)
    ang_vel_interddots = pm.einsumDot(ang_vel_vecs[1:], extrap_ang_vel_vecs)
    alphas = np.linspace(0.0, 1.0, interp_ang_vels.shape[1])
    betas = 1.0 - alphas
    interp_sq_lens = np.outer(sq_ang_vel_lens, alphas**2)
    interp_sq_lens += np.outer(ang_vel_interddots, alphas*betas)
    interp_sq_lens += np.outer(sq_extrap_ang_vel_lens, betas**2)

    interp_sq_lens_alt = pm.einsumDot(interp_ang_vels, interp_ang_vels)

    


    proj_ang_vel_scalars = np.einsum('abi,ai->ab', interp_ang_vels, ang_vel_vecs[1:])
    proj_ang_vel_scalars /= interp_sq_lens
    proj_interp_ang_vels = interp_ang_vels * proj_ang_vel_scalars[..., np.newaxis]

    rk_preds = pm.integrateAngularVelocityRK(interp_ang_vels, rotations_quats[2:-1], 4)

    rk_slerp_preds = pm.integrateAngularVelocityRK(constspeed_slerp_ang_vels, rotations_quats[2:-1], 4)
    rkcslin_preds = pm.integrateAngularVelocityRK(interp_ang_vels2, rotations_quats[2:-1], 4)

    rk_proj_preds = pm.integrateAngularVelocityRK(proj_interp_ang_vels, rotations_quats[2:-1], 4)


    # Godot thing
    godot_vels = interp_ang_vels / (interp_ang_vels.shape[1] - 1)    
    godot_preds = rotations_quats[2:-1].copy()
    for i in range(interp_ang_vels.shape[1] - 1):
        godot_diffs = pm.quatsFromAxisAngleVec3s(godot_vels[:, i])
        godot_preds = pm.multiplyQuatLists(godot_diffs, godot_preds)

    # Non-numerical fixed-axis dot thing.
    parallel_ang_acc_scalars = pm.einsumDot(ang_acc_vecs, fixed_axes[1:-1])
    parallel_angs = angles[1:-1] + parallel_ang_acc_scalars**2
    accproj_diffs = pm.quatsFromAxisAngles(fixed_axes[1:-1], parallel_angs)
    accproj_preds = pm.multiplyQuatLists(accproj_diffs, rotations_quats[2:-1])

    if np.abs(sq_ang_vel_lens_ALT - sq_ang_vel_lens).max() > 0.0001:
        raise Exception("Made a mistake!")
    else:
        raise Exception("No mistake, but duplicate code should del now!")

    if np.abs(interp_sq_lens - interp_sq_lens_alt).max() > 0.0001:
        raise Exception("Made a mistake!")
    else:
        raise Exception("No mistake, but duplicate code should del now!")

    return {
        "RK": rk_preds, "RKCSLIN": rkcslin_preds, "Godot": godot_preds,
        "RKSlerp": rk_slerp_preds, "RKproj": rk_proj_preds, 
        "accproj": accproj_preds}


class DisplayGrouping(Enum):
    TOTAL_ONLY = 1
    BY_OBJECT = 2
    BY_SEQUENCE = 3

#%%
class PredictionResult:
    def __init__(self, name: str,
                 errors: typing.Dict[typing.Tuple[int, int], np.ndarray],
                 scores: typing.Dict[typing.Tuple[int, int], float]
                 ):
        self.name: str = name
        # predictions: np.ndarray
        self.errors: typing.Dict[typing.Tuple[int, int], np.ndarray] = errors
        self.scores: typing.Dict[typing.Tuple[int, int], float] = scores
        self.scores2D: np.ndarray = np.full(
            (len(gtc.BCOT_BODY_NAMES), len(gtc.BCOT_SEQ_NAMES)), np.nan
        )
        for k, v in scores.items():
            self.scores2D[k[0], k[1]] = v
        
    def addScore(self, bodSeqIDTuple: typing.Tuple[int, int], score: float):
        self.scores[bodSeqIDTuple] = score
        self.scores2D[bodSeqIDTuple[0], bodSeqIDTuple[1]] = score

class ConsolidatedResults:
    def __init__(self): #, numScenarios: int):

         # Name -> PredictionResult
        self.translation_results: typing.Dict[str, PredictionResult] = dict()
        self.rotation_results: typing.Dict[str, PredictionResult] = dict()

        # Order in which to display results.
        self._ordered_translation_result_names = []
        self._ordered_rotation_result_names = []

        # self.body_seq_to_row = dict() # (bodyID: int , seqID: int) -> int

        self._translations_gt = None
        self._axisangles_gt = None
        self._quaternions_gt = None

        self._currentKey = None # Current (body, sequence) key.
        self._allBodSeqKeys = set()

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
        return pm.anglesBetweenQuats(values[1:], full_predictions)

    # Note: returns errors in radians!
    def getAxisAngleError(values, predictions): 
        v_qs = pm.quatsFromAxisAngleVec3s(values)
        p_qs = pm.quatsFromAxisAngleVec3s(predictions)
        return ConsolidatedResults.getQuatError(v_qs, p_qs)

    def getTranslationError(values, predictions):
        full_predictions = ConsolidatedResults.prependMissingPredictions(
            values, predictions
        )

        return values[1:] - full_predictions
    
    def updateGroundTruth(self, translations, axisangles, quats, bod_ID, seq_ID):
        self._currentKey = (bod_ID, seq_ID)
        self._allBodSeqKeys.add(self._currentKey)
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
        score = ConsolidatedResults.applyThreshold(
            np.linalg.norm(errs, axis = -1), TRANSLATION_THRESH
        )
        self._addPredictionResult(
            self.translation_results, self._ordered_translation_result_names,
            name, errs, score
        )

    def addAxisAngleResult(self, name, predictions):
        errs = ConsolidatedResults.getAxisAngleError(
            self._axisangles_gt, predictions
        )
        score = ConsolidatedResults.applyThreshold(errs, ROTATION_THRESH_RAD)
        self._addPredictionResult(
            self.rotation_results, self._ordered_rotation_result_names,
            name, errs, score)

    def addQuaternionResult(self, name, predictions):
        errs = ConsolidatedResults.getQuatError(
            self._quaternions_gt, predictions
        )
        score = ConsolidatedResults.applyThreshold(errs, ROTATION_THRESH_RAD)
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

    def applyThreshold(errs, thresh):
        score = None
        if thresh is None:
            score = -errs.mean() # Negating so that larger is still better.
        else:
            score = (errs <= thresh).mean()
        return score

    def _applyBestResult(results_dict, name_order_list, names, agg_name, thresh, errs_are_1D, use_shift):
        bodSeqKeys = results_dict[names[0]].errors.keys()
        errs = dict()
        scores = dict()
        for k in bodSeqKeys:
            stacked_errs = np.stack([
                results_dict[n].errors[k] for n in names
            ], axis = 0)
            stacked_norms = stacked_errs
            if not errs_are_1D:
                stacked_norms = np.linalg.norm(stacked_errs, axis = -1)
            min_inds_1D = np.argmin(stacked_norms, axis = 0, keepdims=True)
            if use_shift:
                min_inds_1D = np.roll(min_inds_1D, 1, axis=-1)
            min_inds_e = min_inds_1D
            if not errs_are_1D:
                min_inds_e = min_inds_1D.reshape(1,-1,1)
            es = np.take_along_axis(stacked_errs, min_inds_e, axis = 0)
            min_norms = np.take_along_axis(stacked_norms, min_inds_1D, axis = 0)
            min_norms = min_norms.flatten()
            if use_shift:
                default_err = results_dict["Static"].errors[k][0]
                default_norm = default_err
                if not errs_are_1D:
                    default_norm = np.linalg.norm(default_err)
                es[0] = default_err
                min_norms[0] = default_norm
            errs[k] = es

            scores[k] = ConsolidatedResults.applyThreshold(min_norms, thresh)
        results_dict[agg_name] = PredictionResult(agg_name, errs, scores)
        name_order_list.append(agg_name)

    def _addPredictionResult(self, all_results_dict, name_order_list, name, errs, score):
        if name in all_results_dict.keys():
            all_results_dict[name].errors[self._currentKey] = errs
            all_results_dict[name].addScore(self._currentKey, score)
        else:
            errs_dict = {self._currentKey: errs}
            score_dict = {self._currentKey: score}
            all_results_dict[name] = PredictionResult(name, errs_dict, score_dict)
            name_order_list.append(name)

    def printTable(results_for_names, col_names, row_names = None, annotations = None):
        name_lens = []
        row_name_col_width = 0
        no_row_names_given = (row_names is None) or (len(row_names) == 0)
        if no_row_names_given:
            row_names = [""]
        else:
            row_name_col_width = np.max([len(rn) for rn in row_names]) + 1
        print(" " * row_name_col_width, end = "")

        for name in col_names:
            # We want each printed column to be at least 10 digits plus 2 spaces
            # because the default numpy precision in printing is 8, which means
            # that to match it, we want at least 10 chars for the number in the
            # next row, plus a space.
            # For negative numbers and numbers >= 10, I'll just decrease the
            # printed precision for now.
            print("{:>10} ".format(name), end = "")
            name_lens.append(max(10, len(name)))
        print() # Newline after row.

        for r_ind, r_name in enumerate(row_names):
            print("{val:>{width}}".format(
                val=r_name, width=row_name_col_width
            ), end = "")
            for c_ind in range(len(results_for_names)):
                width = name_lens[c_ind]
                col_vals = results_for_names[c_ind]
                val = col_vals if no_row_names_given else col_vals[r_ind]
                val_str = "" # Will update soon.

                # If 0 <= number <= 1, use Numpy's default float print digits (8).
                # Otherwise, decrease precision so printing uses 10 chars total.
                prec_to_remove = 1 if val < 0 else 0 # Case of "-" sign.
                if np.abs(val) > 10:
                    prec_to_remove += int(np.log10(np.abs(val)))

                if not (annotations is None):
                    val_str = annotations[c_ind, r_ind]
                    prec_to_remove += len(val_str)

                val_str += str(np.round(val, 8 - prec_to_remove))
                # Print value v with padding to make width w.
                print("{v:>{w}} ".format(v=val_str, w=width), end = "")
            print() # Newline after row.


    def printResults(self, group_mode: DisplayGrouping, thresh_name_t: str = "",
                     thresh_name_r: str = ""):

        mean_ax = None
        row_names = []
        if group_mode != DisplayGrouping.TOTAL_ONLY:
            if group_mode == DisplayGrouping.BY_OBJECT:
                mean_ax = 1
                row_names = [
                    gtc.shortBodyNameBCOT(n, 7) for n in gtc.BCOT_BODY_NAMES
                ]
            elif group_mode == DisplayGrouping.BY_SEQUENCE:
                mean_ax = 0
                row_names = [
                    gtc.shortSeqNameBCOT(n) for n in gtc.BCOT_SEQ_NAMES
                    if "cam2" not in n
                ]
            row_names.append("Avg")

        table_titles = ["Translation", "\nRotation"]
        ordered_names_lists = [
            self._ordered_translation_result_names,
            self._ordered_rotation_result_names
        ]
        threshes = [thresh_name_t, thresh_name_r]
        results_dicts = [self.translation_results, self.rotation_results]

        tr_zip = zip(table_titles, ordered_names_lists, threshes, results_dicts)
        for title, ordered_names, thresh, results in tr_zip:
            ordered_score_means = [
                np.nanmean(results[n].scores2D, axis=mean_ax)
                for n in ordered_names
            ]
            ordered_score_means = [
                ms[np.invert(np.isnan(ms))] for ms in ordered_score_means
            ]

            score_means = ordered_score_means
            col_names = ordered_names
            excluded_cols = []
            annotations = None
            if len(thresh) > 0:
                thresh_ind = ordered_names.index(thresh)
                thresh_means = ordered_score_means[thresh_ind]
                selected_score_means = []
                col_names = []
                for means, name in zip(ordered_score_means, ordered_names):
                    if np.any(means >= thresh_means):
                        if group_mode != DisplayGrouping.TOTAL_ONLY:
                            overall_avg = np.fromiter(
                                results[name].scores.values(), dtype=float
                            ).mean()
                            means = np.append(means, overall_avg)
                        selected_score_means.append(means)
                        col_names.append(name)
                    else:
                        excluded_cols.append(name)
                score_means = np.stack(selected_score_means, axis=0)

                # The code below for creating annotations relies on the fact
                # that "Perfect" results are in the first column in order to
                # ignore them in the result sorting in a convenient way.
                if ordered_names[0] != "Perfect":
                    raise Exception("Code written under now-false assumption \
                                    that \"Perfect\" is the first column!")
                sort_inds = np.argsort(score_means[1:], axis=0)[::-1]
                
                # The general method for converting the argsort indices into
                # ordinal rankings comes from the code for scipy's rankdata().

                arrange_1d = np.arange(len(col_names) - 1)
                arrange = np.broadcast_to(
                    arrange_1d, sort_inds.shape[::-1]
                ).transpose()
                orderings = np.empty_like(sort_inds)
                np.put_along_axis(orderings, sort_inds, arrange, axis=0)
                # np.take_along_axis(arrange, sort_inds, axis= 

                # Annotations will start out as "", but we'll specify "<U3" as
                # the dtype to indicate that the lens can be up to 3 chars.
                annotations = np.full(score_means.shape, "", dtype="<U3")
                place_limits = orderings[col_names.index(thresh) - 1]
                meets_thresh = orderings <= place_limits
                annots_to_replace = orderings[meets_thresh].astype(str) + "|"
                annotations[1:][meets_thresh] = annots_to_replace

               
            print("{} Results:".format(title))
            ConsolidatedResults.printTable(
                score_means, col_names, row_names, annotations
            )
            if len(thresh) > 0:
                print("Column to compare against:", thresh)
                excluded_str = "[]"
                if len(excluded_cols) > 0:
                    excluded_str = ", ".join(excluded_cols)
                print("Excluded columns:", excluded_str)
        return
#%%
combos = []
for b in range(len(gtc.BCOT_BODY_NAMES)):
    for s in range(len(gtc.BCOT_SEQ_NAMES)):
        if BCOT_Data_Calculator.isBodySeqPairValid(b, s, True):
            combos.append((b,s))

skipAmount = 2

fit_modes = [SplinePredictionMode.EXTRAPOLATE]
spline_pred_calculator = BSplineFitCalculator(
    SPLINE_DEGREE, DESIRED_CTRL_PTS, PTS_USED_TO_CALC_LAST, fit_modes
)
cinpact_extrapolator = CinpactAccelExtrapolater(3, 3)

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

all_rot_angles = []
all_bcsfa_angles = []
all_next_bcsfa_angles = []
# all_wahba_angles = np.zeros((0,))
maxTimestamps = 0
maxTimestampsWhenSkipped = 0
max_angle = 0
for i, combo in enumerate(combos):
    calculator = BCOT_Data_Calculator(combo[0], combo[1], skipAmount)

    translations_gt = calculator.getTranslationsGTNP(True)
    rotations_aa_gt = calculator.getRotationsGTNP(True)
    rotations_gt_quats = pm.quatsFromAxisAngleVec3s(rotations_aa_gt)


    translations = translations_gt #+ np.random.uniform(-4, 4, translations_gt.shape)
    rotations_quats = rotations_gt_quats
    #  = pm.multiplyQuatLists(
    #     getRandomQuatError(rotations_gt_quats.shape, ROTATION_THRESH_RAD), 
    #     rotations_gt_quats
    # )
    rotations = rotations_aa_gt # TODO: Apply quat error to these.

    allResultsObj.updateGroundTruth(
        translations_gt, rotations_aa_gt, rotations_gt_quats, combo[0], combo[1]
    )
    maxTimestampsWhenSkipped = max(maxTimestampsWhenSkipped, len(translations))


    translation_diffs = np.diff(translations, axis=0)
    rotation_aa_diffs = np.diff(rotations, axis=0)

    
    
    rev_rotations_quats = pm.conjugateQuats(rotations_quats)



    rotation_quat_diffs = pm.multiplyQuatLists(
        rotations_quats[1:], rev_rotations_quats[:-1]
    )

    # At this time, logging the finite difference acceleration at the last
    # timestep isn't required, as there's no further prediction to make with it.
    translations_acc = np.diff(translation_diffs[:-1], axis=0)
    rotations_aa_acc = np.diff(rotation_aa_diffs[:-1], axis=0)

    t_vel_preds = translations[1:-1] + translation_diffs[:-1]
    t_screw_preds = translations[1:-1] + pm.rotateVecsByQuats(rotation_quat_diffs[:-1], translation_diffs[:-1])
    # t_velLERP_preds = translations[1:-1] + 0.91 * translation_diffs[:-1]
    r_vel_preds = pm.multiplyQuatLists(
        rotation_quat_diffs[:-1], rotations_quats[1:-1]
    )
    # r_vel_preds = pm.quatSlerp(rotations_quats[:-2], rotations_quats[1:-1], 2)
    r_aa_vel_preds = rotations[1:-1] + rotation_aa_diffs[:-1]

    r_slerp_preds = pm.quatSlerp(rotations_quats[1:-1], r_vel_preds, 0.8525)

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

    t_deg4_preds = t_quadratic_preds.copy()
    t_deg4_preds[3:] = (17/24) * translations[:-5] - (11/3) * translations[1:-4] + (31/4) * translations[2:-3] - (25/3)*translations[3:-2] + (109/24)*translations[4:-1]
    t_jerk_preds = t_quadratic_preds.copy()
    t_jerk_preds[2:] = 4 * translations[3:-1] - 6 * translations[2:-2] + 4 * translations[1:-3] - translations[:-4]

    complete_vel_sq_lens = pm.einsumDot(
        translation_diffs, translation_diffs
    ) # (1-0)^2, (2-1)^2, ...
    prev_vel_sq_lens = complete_vel_sq_lens[:-1]
    speeds = np.sqrt(prev_vel_sq_lens)
    unit_vels = translation_diffs[:-1] / speeds[..., np.newaxis]
    unit_vel_dots = pm.einsumDot(unit_vels[1:], unit_vels[:-1])
    vel_angles = np.arccos(unit_vel_dots)
    vel_crosses = np.cross(unit_vels[:-1], unit_vels[1:])
    vel_crosses /= np.linalg.norm(vel_crosses, axis=-1, keepdims=True)
    min_vel_rot_aas = pm.scalarsVecsMul(vel_angles, vel_crosses)
    min_vel_rot_qs = pm.quatsFromAxisAngleVec3s(min_vel_rot_aas)

    rough_circ_axes_0 = translation_diffs[1:-1]
    rough_circ_axes_1 = -translation_diffs[:-2]
    circle_plane_info = pm.getPlaneInfo(
        rough_circ_axes_0, rough_circ_axes_1, translations[:-3]
    )

    circle_axes = circle_plane_info.plane_axes

    '''
    plane_normals = pm.normalizeAll(np.cross(
        rough_circ_axes_0, rough_circ_axes_1
    ))
    if not pm.areAxisArraysOrthonormal([
        circle_axes[0], circle_axes[1], plane_normals
    ], loud=True):
        raise Exception("Orthonormality issue!")
    if np.abs(pm.einsumDot(plane_normals, rough_circ_axes_1)).max() > 0.001:
        raise Exception("Plane normal issue!")
    '''
    circle_pts_2D = []
    for j in range(3):
        circle_pts_2D.append(pm.vecsTo2D(
            translations[j:(-3 + j)], *circle_axes
        ))

    # circ_pt_norms = np.linalg.norm(circle_pts_2D[0] - circle_pts_2D[1], axis=-1)
    # orig_nroms = np.linalg.norm(translation_diffs[:-2], axis=-1)
    # if np.abs(circ_pt_norms - orig_nroms).max() > 0.0001:
    #     raise Exception("Norm issue!")
        
    circle_centres = pm.circleCentres2D(*circle_pts_2D)

    diffs_from_centres = []
    for j in range(3):
        diffs_from_centres.append(circle_pts_2D[j] - circle_centres)
   
    sq_radii = pm.einsumDot(diffs_from_centres[0], diffs_from_centres[0])
    # A good max for forearm length is 30cm, and 20cm for hand length.
    # So a decent circle radius max is 50cm = 500mm.
    radii_too_long = sq_radii > 250000 

    c_angles = []
    c_cosines = []
    for j in range(2):
        c_diff_dot = pm.einsumDot(
            diffs_from_centres[j], diffs_from_centres[j + 1]
        )
        c_cosines.append(
            c_diff_dot/sq_radii
        )
        c_angles.append(np.arccos(c_cosines[-1]))

        # Numpy cross products of 2D vecs treat them like 3D vecs and return 
        # only the z-coordinate of the result (since the rest are 0). We'll look
        # at the sign to determine rotation direction about the circle axis.
        c_crosses = np.cross(diffs_from_centres[j], diffs_from_centres[j + 1])
        c_crosses_flip = (c_crosses < 0.0)
        c_angles[-1][c_crosses_flip] = -c_angles[-1][c_crosses_flip]

    MAX_CIRC_ANGLE = np.pi/2.0
    prev_angle_sum = c_angles[0] + c_angles[1]
    prev_angles_too_big = np.abs(prev_angle_sum) > MAX_CIRC_ANGLE

    invalid_circ_indices = np.logical_or(radii_too_long, prev_angles_too_big)

    c_pred_angles_base = 1.5 * c_angles[1] - 0.5 * c_angles[0]

    c_pred_angles = np.clip(
        c_pred_angles_base, a_min=-MAX_CIRC_ANGLE, a_max=MAX_CIRC_ANGLE
    )

    c_cosines_pred = np.cos(c_pred_angles)
    c_sines_pred = np.sin(c_pred_angles)

    c_trans_preds_2D = pm.rotateBySinCos2D(
        diffs_from_centres[2], c_cosines_pred, c_sines_pred
    )

    c_trans_preds = np.empty(t_vel_preds.shape)
    c_trans_preds[1:] = pm.vecsTo3DUsingPlaneInfo(
        circle_centres + c_trans_preds_2D, circle_plane_info
    )
    c_trans_preds[0] = t_vel_preds[0]
    c_trans_preds[1:][invalid_circ_indices] = t_quadratic_preds[1:][invalid_circ_indices]

    # c_centres3D = pm.vecsTo3DUsingPlaneInfo(circle_centres, circle_plane_info)
    # if not pm.areVecArraysInSamePlanes([
    #     translations[:-3], translations[1:-2], translations[2:-1],
    #     c_trans_preds, c_centres3D
    # ]):
    #     raise Exception("Vecs not in same plane as expected!") 
    # radii_comparisons = []
    # for j in range(3):
    #     c_diffs3D = translations[j:(-3 + j)] - c_centres3D
    #     radii_comparisons.append(pm.einsumDot(c_diffs3D, c_diffs3D))       
    # pred_from_c3D = c_trans_preds - c_centres3D
    # radii_comparisons.append(pm.einsumDot(pred_from_c3D, pred_from_c3D))
    # if np.abs(np.diff(radii_comparisons, 1, axis=0)).max() > 0.0001:
    # #     raise Exception("3D radii difference that wasn't expected!")
    # c_cosines_2 = pm.einsumDot(c_trans_preds_2D, diffs_from_centres[2])/sq_radii
    # if np.abs(c_cosines_1 - c_cosines_2).max() > 0.0001:
    #     raise Exception("Made an angle mistake!")

    circle_rot_quats = np.empty(circle_plane_info.normals.shape[:-1] + (4,))
    half_c_pred_angles = c_pred_angles / 2.0
    circle_rot_quats[:, 0] = np.cos(half_c_pred_angles)
    circle_rot_quats[:, 1:] = pm.scalarsVecsMul(
        np.sin(half_c_pred_angles), circle_plane_info.normals
    )


    # I guess I'm thinking that F2 = V(0>1)*F1*R1
    # Which means R1 = F1^T V(0>1)^T F2
    def getArmRotPreds(arm_component_quats):
        rev_arm_qs = pm.conjugateQuats(min_vel_rot_qs)

        local_rots = pm.multiplyQuatLists(
            rev_rotations_quats[1:-2], pm.multiplyQuatLists(
                rev_arm_qs, rotations_quats[2:-1]
        ))

        r_arm_preds = np.empty(r_vel_preds.shape)
        r_arm_preds[0] = r_vel_preds[0]
        r_arm_preds[1:] = pm.multiplyQuatLists(
            arm_component_quats, pm.multiplyQuatLists(
                rotations_quats[2:-1], local_rots
            )
        )
        return r_arm_preds

    r_v_arm_preds = getArmRotPreds(min_vel_rot_qs)
    r_c_arm_preds = getArmRotPreds(circle_rot_quats)
    r_c_arm_preds[1:][invalid_circ_indices] = r_vel_preds[1:][invalid_circ_indices]

    # unit_vel_test = pm.rotateVecsByQuats(min_vel_rot_qs, unit_vels[:-1])
    # if np.max(np.abs(unit_vel_test - unit_vels[1:])) > 0.0001:
    #     raise Exception("Made a mistake!")

    t_poly_preds = t_quadratic_preds.copy()
    vel_bounce = (unit_vel_dots < -0.99)
    t_poly_preds[1:][vel_bounce] = t_vel_preds[1:][vel_bounce]

    r_aa_acc_delta = rotation_aa_diffs[1:-1] + (0.5 * rotations_aa_acc)
    r_aa_acc_preds = rotations[2:-1] + r_aa_acc_delta
    r_aa_accLERP_preds = rotations[2:-1] + 0.5 * r_aa_acc_delta
    r_aa_acc_preds = np.vstack((r_aa_vel_preds[:1], r_aa_acc_preds))
    r_aa_accLERP_preds = np.vstack((r_aa_vel_preds[:1], r_aa_accLERP_preds))


    r_squad_preds = pm.squad(rotations_quats, 2)[:-1]
    r_squad_preds = np.vstack((r_vel_preds[:1], r_squad_preds))

    # num_spline_preds = (len(translations) - 1) - (SPLINE_DEGREE) 

    r_mats = calculator.getRotationMatsGTNP(True)
    fixed_axes, angles = pm.axisAnglesFromQuats(rotation_quat_diffs)
    angles = angles.flatten()
    #rotation_quat_diffs[:, 1:] / np.linalg.norm(rotation_quat_diffs[:, 1:], axis=-1, keepdims=True)# np.sin(angles/2)[..., np.newaxis]
    r_fixed_axis_closest_angs = pm.closestAnglesAboutAxis(r_mats[1:-1], r_mats[2:], fixed_axes[:-1])
    r_next_fixed_axis_closest_angs = pm.closestAnglesAboutAxis(r_mats[:-2], r_mats[1:-1], fixed_axes[1:])
    r_fixed_axis_closest_rots = pm.matsFromAxisAngleArrays(
        r_fixed_axis_closest_angs, fixed_axes[:-1]
    ) 
    r_fixed_axis_bcs_mats = pm.einsumMatMatMul(
        r_fixed_axis_closest_rots, r_mats[1:-1]
    )
    r_fixed_axis_bcs = pm.axisAngleFromMatArray(r_fixed_axis_bcs_mats)

    # Convert mats into "separated" lists of their columns.
    # wahba_points_local = np.moveaxis(r_mats, -1, 0)
    

    
    # wahba_inputs = 2 * r_mats[1:-1] - r_mats[:-2]
    # wahba_outputs = wahba(wahba_inputs)
    # wahba_pred = pm.axisAngleFromMatArray(wahba_outputs) 



    # angles = pm.anglesBetweenQuats(rotations_quats[1:], rotations_quats[:-1]).flatten()
    max_angle = max(max_angle, angles.max())


    angle_diffs = np.diff(angles, 1, axis=0)
    angle_diffs_cond = np.zeros_like(angle_diffs)
    axis_dots = pm.einsumDot(fixed_axes[1:], fixed_axes[:-1])
    axes_almost_eq = axis_dots < np.cos(np.deg2rad(1))
    angle_diffs_cond[axes_almost_eq] = angle_diffs[axes_almost_eq]
    angle_ratios = 2 + (angle_diffs_cond[:-1]/angles[1:-1])

    next_axis_angs = pm.closestAnglesAboutAxis(r_mats[:-3], r_mats[1:-2], fixed_axes[1:-1])
    angle_diffs2 = angles[1:-1] - next_axis_angs
    angle_ratios2 = 2 + (np.clip(angle_diffs2, a_min = None, a_max = 0)/angles[1:-1])

    r_fixed_axis_preds = np.empty((len(rotations) - 2, 4))
    r_fixed_axis_preds[1:] = pm.quatSlerp(rotations_quats[1:-2], rotations_quats[2:-1], angle_ratios)
    r_fixed_axis_preds[0] = r_vel_preds[0]

    r_fixed_axis_preds2 = np.empty((len(rotations) - 2, 4))
    r_fixed_axis_preds2[1:] = pm.quatSlerp(rotations_quats[1:-2], rotations_quats[2:-1], angle_ratios2)
    r_fixed_axis_preds2[0] = r_vel_preds[0]

    # wahba_angles = pm.replaceAtEnd(angles[:-1], angWahbaFunc(angles[:-1]))
    sigmoid_angles = algebraicSigmoid(angles[:-1], 0.75, 2 * np.pi)
    sigmoid_q_diffs = pm.quatsFromAxisAngles(fixed_axes[:-1], sigmoid_angles)
    sigmoid_pred = pm.multiplyQuatLists(sigmoid_q_diffs, rotations_quats[1:-1])
    # wahba_qs = pm.quatsFromAxisAngleVec3s(wahba_pred)
    # wahba_q_diffs = pm.multiplyQuatLists(wahba_qs, rev_rotations_quats[1:-1])
    # wahba_axes, wahba_angles = pm.axisAnglesFromQuats(wahba_q_diffs)
    all_rot_angles.append(angles[:-1])   
    # all_wahba_angles = np.concatenate((all_wahba_angles, wahba_angles.flatten()))
    all_bcsfa_angles.append(r_fixed_axis_closest_angs.flatten())
    all_next_bcsfa_angles.append(r_next_fixed_axis_closest_angs.flatten())
    
    camobj_preds = np.empty(r_vel_preds.shape)
    camobj_preds[0] = r_vel_preds[0]
    camobj_preds[1:] = pex.camObjConstAngularVelPreds(rotations_quats[:-1], r_vel_preds)[-1]

    r_rotqdiff_diffs = pm.quatSlerp(rotation_quat_diffs[:-2], rotation_quat_diffs[1:-1], 2)
    r_rotqdiff_preds = np.empty(r_vel_preds.shape)
    r_rotqdiff_preds[0] = r_vel_preds[0]
    r_rotqdiff_preds[1:] = pm.multiplyQuatLists(r_rotqdiff_diffs, rotations_quats[2:-1])

    r_movaxis_axes = pm.reflectVecsOverLines(fixed_axes[:-2], fixed_axes[1:-1], True)
    r_movaxis_diffs = pm.quatsFromAxisAngles(r_movaxis_axes, angles[1:-1] + angle_diffs[:-1])
    r_movaxis_preds = np.empty(r_vel_preds.shape)
    r_movaxis_preds[0] = r_vel_preds[0]
    r_movaxis_preds[1:] = pm.multiplyQuatLists(r_movaxis_diffs, rotations_quats[2:-1])

    scaled_axes = pm.scalarsVecsMul(angles[:-1], fixed_axes[:-1])
    # test_quats = pm.quatsFromAxisAngleVec3s(scaled_axes)
    # max_scaledax_diff = np.abs(test_quats - rotation_quat_diffs[:-1]).max()
    ang_accels = np.diff(scaled_axes, 1, axis=0)
    naive_lin_axes = pm.replaceAtEnd(scaled_axes, scaled_axes[1:] + ang_accels, 1)
    naive_lin_q_diffs = pm.quatsFromAxisAngleVec3s(naive_lin_axes)
    r_naive_lin_preds = pm.multiplyQuatLists(
        naive_lin_q_diffs, rotations_quats[1:-1]
    )


    

    # slerp_diffs = pm.multiplyQuatLists(r_slerp_preds, rev_rotations_quats[1:-1])
    # slerp_axes = slerp_diffs[:, 1:] / np.linalg.norm(slerp_diffs[:, 1:], axis=-1, keepdims=True)
    # axisComp = np.abs(pm.einsumDot(slerp_axes, fixed_axes[:-1]))
    # if np.any(axisComp < 0.999):
    #     raise Exception("Axes don't match!")
    

    vel_dots = pm.einsumDot(translation_diffs[1:], translation_diffs[:-1])
    # (1-0)*(2-1), (2-1)*(3-2), ...
    vel_proj_scalars = vel_dots / prev_vel_sq_lens
    # p(2-1)onto(1-0), p(3-2)onto(2-1), ...
    all_vel_ratios = np.concatenate((all_vel_ratios, vel_proj_scalars))
    all_vel_graphing_ratios = np.concatenate((
        all_vel_graphing_ratios, vel_proj_scalars[1:]))
    # p(3-2)onto(2-1), p(4-3)onto(3-2), ...

    best_vel_preds = translations[1:-1] + pm.scalarsVecsMul(vel_proj_scalars, translation_diffs[:-1])

    acc_delta_sq_lens = pm.einsumDot(t_acc_delta, t_acc_delta)
    # (2-0)^2, (3-1)^2, ...
    acc_delta_dots = pm.einsumDot(t_acc_delta, translation_diffs[2:])
    # (2-0)*(3-2), (3-1)*(4-3), ...
    acc_delta_proj_scalars = acc_delta_dots / acc_delta_sq_lens
    # p(3-2)onto(2-0), p(4-3)onto(3-1), ...
    all_acc_delta_ratios = np.concatenate((
        all_acc_delta_ratios, acc_delta_proj_scalars
    ))

    acc_vel_dots = pm.einsumDot(translations_acc, translation_diffs[1:-1])
    # (2-0)*(2-1), (3-1)*(3-2), ...
    acc_parallel_scalars = acc_vel_dots / prev_vel_sq_lens[1:]
    # p(2-0)onto(2-1), p(3-1)onto(3-2), ...
    acc_ortho_vecs = translations_acc - (acc_parallel_scalars[..., np.newaxis] * translation_diffs[1:-1])
    # test_ortho = pm.einsumDot(acc_ortho_vecs, translation_diffs[1:-1])
    # print(test_ortho.max())
    acc_ortho_sq_lens = pm.einsumDot(acc_ortho_vecs, acc_ortho_vecs)
    acc_ortho_dots = pm.einsumDot(acc_ortho_vecs, translation_diffs[2:])
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
    acc_angle_preds = translations[2:-1] + pm.scalarsVecsMul(acc_angle_ratios, translation_diffs[1:-1])
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
    next_vel_ortho_dots = pm.einsumDot(translation_diffs[2:], acc_ortho_vecs)
    next_vel_vecs = np.empty((len(vel_lens), 2))
    next_vel_vecs[:, 0] = next_vel_parallel
    next_vel_vecs[:, 1] = next_vel_ortho_dots / acc_ortho_comp

    vel_acc_2D_solve = pm.einsumMatVecMul(inv_mats, next_vel_vecs)
    all_vel_acc_2D_solves = np.concatenate((
        all_vel_acc_2D_solves, vel_acc_2D_solve
    ))

    complete_vel_sq_lens = pm.einsumDot(
        translation_diffs, translation_diffs
    ) # (1-0)^2, (2-1)^2, ...
    prev_vel_sq_lens = complete_vel_sq_lens[:-1]

    vel_proj_scalars = vel_dots / prev_vel_sq_lens
    # p(2-1)onto(1-0), p(3-2)onto(2-1), ...
    all_vel_ratios = np.concatenate((all_vel_ratios, vel_proj_scalars))

    acc_vel_dots = pm.einsumDot(translations_acc, translation_diffs[1:-1])
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


    #---------------------------------------------------------------
    all_spline_preds = spline_pred_calculator.fitAllData(np.hstack((
        translations, rotations
    )))
    t_spline_preds = all_spline_preds[:, :3]
    r_aa_spline_preds = all_spline_preds[:, 3:]
    # t_spline_preds = spline_pred_calculator.fitAllData(translations)

    # t_spline_preds = np.vstack((t_vel_preds[:1], t_spline_preds))
    
    other_spline_err = t_spline_preds - translations[SPLINE_DEGREE + 1:]
    other_spline_err_norms = np.linalg.norm(other_spline_err, axis = -1)
    other_spline_errs.append(ConsolidatedResults.applyThreshold(
        other_spline_err_norms, TRANSLATION_THRESH
    ))

    allResultsObj.addAxisAngleResult("Static", rotations[:-1])
    allResultsObj.addQuaternionResult("QuatVel", r_vel_preds)
    allResultsObj.addQuaternionResult("QuatVelSLERP", r_slerp_preds)

    allResultsObj.addAxisAngleResult("AA_Vel", r_aa_vel_preds)
    allResultsObj.addAxisAngleResult("AA_Acc", r_aa_acc_preds)
    allResultsObj.addAxisAngleResult("AA_AccLERP", r_aa_accLERP_preds)
    allResultsObj.addAxisAngleResult("AA_Spline", r_aa_spline_preds)
    allResultsObj.addAxisAngleResult("Fixed axis bcs", r_fixed_axis_bcs)
    allResultsObj.addQuaternionResult("Fixed axis acc", r_fixed_axis_preds)
    allResultsObj.addQuaternionResult("Fixed axis acc2", r_fixed_axis_preds2)
    # allResultsObj.addAxisAngleResult("Wahba", wahba_pred)
    allResultsObj.addQuaternionResult("Sigmoid", sigmoid_pred)

    if ADD_NUMERICAL_TESTS:
        numerical_res_dict = numericalIdeas(
            angles, fixed_axes, angle_diffs, r_fixed_axis_closest_angs,
            r_next_fixed_axis_closest_angs, rotations_quats
        )
        for numerical_k, numerical_r in numerical_res_dict.items():
            if numerical_r.shape[1] != 4:
                raise Exception("Expected a quaternion result here!")
            numerical_r_full = pm.replaceAtEnd(r_vel_preds, numerical_r, 1)

            allResultsObj.addQuaternionResult(numerical_k, numerical_r_full)

        rk_dec_only = r_vel_preds.copy()
        ang_dec_inds = angle_diffs[:-1] < 0.0
        rk_dec_only[1:][ang_dec_inds] = numerical_res_dict["RK"][ang_dec_inds]



        allResultsObj.addQuaternionResult("RKdeconly", rk_dec_only)

    allResultsObj.addQuaternionResult("SQUAD", r_squad_preds)
    allResultsObj.addQuaternionResult("Arm v", r_v_arm_preds)
    allResultsObj.addQuaternionResult("Arm c", r_c_arm_preds)
    allResultsObj.addQuaternionResult("camobj", camobj_preds)
    allResultsObj.addQuaternionResult("naiveLin", r_naive_lin_preds)

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
    allResultsObj.addTranslationResult("deg4", t_deg4_preds)
    allResultsObj.addTranslationResult("Jerk", t_jerk_preds)
    allResultsObj.addTranslationResult("Spline", t_spline_preds)
    allResultsObj.addTranslationResult("Spline2", t_spline2_preds)
    allResultsObj.addTranslationResult("Circ", c_trans_preds)
    allResultsObj.addTranslationResult("Screw", t_screw_preds)
    # allResultsObj.addTranslationResult("CINPACT", cinpact_extrapolator.apply(
    #     translations[:-1]
    # ))
    allResultsObj.addTranslationResult("StatVelAcc", t_poly_preds)
    
    maxTimestamps = max(
        maxTimestamps, len(calculator.getTranslationsGTNP(False))
    )
    if (not egFound):
        allLastLerpScores = allResultsObj.translation_results["AccLERP"].scores
        last_key = list(allLastLerpScores.keys())[-1]
        lastLerpScore = allLastLerpScores[last_key]
        accResult = allResultsObj.translation_results["Acc"]
        lastAccScore = accResult.scores[last_key]
        if lastLerpScore > lastAccScore:
            egFound = True

            sampleAccErrs = accResult.errors[last_key]
            t_deltas_n = np.linalg.norm(t_acc_delta, axis=-1, keepdims=True)
            sampleDeltas = t_acc_delta / t_deltas_n

allResultsObj.applyBestRotationResult(["QuatVel", "Fixed axis acc", "Static"], "agg", True)
# allResultsObj.applyBestRotationResult(["Wahba", "Static"], "aggw", True)
allResultsObj.applyBestRotationResult(["Fixed axis acc2", "Arm v"], "aggv", True)
# allResultsObj.applyBestTranslationResult(["Static", "Vel", "Quadratic", "Screw"], "agg", True)
# allResultsObj.applyBestTranslationResult(["Static", "Vel", "Quadratic", "Jerk"], "jagg", True)


print("Max angle:", max_angle)
print("maxTimestampsWhenSkipped:", maxTimestampsWhenSkipped)

allResultsObj.printResults(DisplayGrouping.BY_SEQUENCE, "Quadratic", "QuatVel")
allResultsObj.printResults(DisplayGrouping.BY_OBJECT, "Quadratic", "QuatVel")
stat_headers = ["Mean", "Min", "Max", "Median", "std"]
def stat_results(vals):
    return [ vals.mean(), vals.min(), vals.max(), np.median(vals), np.std(vals)]

vel_stat_results = stat_results(all_vel_ratios)
acc_delta_stat_results = stat_results(all_acc_delta_ratios)
acc_ortho_stat_results = stat_results(all_acc_ortho_ratios)

print("All vel ratios stats:")
ConsolidatedResults.printTable(vel_stat_results, stat_headers)
print()

print("All acc delta ratios stats:")
ConsolidatedResults.printTable(acc_delta_stat_results, stat_headers)
print()

print("All acc ortho ratios stats:")
ConsolidatedResults.printTable(acc_ortho_stat_results, stat_headers)
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

# ang_inds_to_dec = all_wahba_angles > np.pi
# np_tau = 2.0 * np.pi
# all_wahba_angles[ang_inds_to_dec] = np_tau - all_wahba_angles[ang_inds_to_dec]

all_prev_angles = np.concatenate(all_rot_angles)
concatenated_bcsfa_angles = np.concatenate(all_bcsfa_angles)
plt.scatter(all_prev_angles, concatenated_bcsfa_angles, label="bcs angles")
half_pi = np.pi / 2.0
plt.title("Wahba-based vs. const-angular-vel angles")
plt.xlabel("Const-angular-vel angles")
plt.ylabel("Wahba-based angles")
plt_x_vals = np.linspace(0, np.pi / 2.0, 100)
plt.plot([0,half_pi],[0,half_pi], label="y=x line")
plt.plot(plt_x_vals, angWahbaFunc(plt_x_vals), label="atan(...) - x")
plt.plot(plt_x_vals, generalizedLogistic(plt_x_vals, 1, 4), label="logistic")
plt.legend()
plt.show()


#%%
# TODO:
# - Angle and/or dot between last axes.
# - np.diff of various angle combinations.

# all_prev1_angles = np.concatenate([angs[1:] for angs in all_rot_angles])
# all_prev2_angles = np.concatenate([angs[:-1] for angs in all_rot_angles])
all_prev2_angles = np.concatenate([angs[:-1] for angs in all_next_bcsfa_angles])
all_prev1_angles = np.concatenate([bcsfas[:-1] for bcsfas in all_bcsfa_angles])
concatenated_bcsfa_angles2 = np.concatenate([
    bcsfas[1:] for bcsfas in all_bcsfa_angles
])

p2_p1_bcs_stack = np.stack([
    all_prev2_angles, all_prev1_angles, concatenated_bcsfa_angles2
], axis=-1)

def getBestFitForInds(inds):
    x = all_prev1_angles[inds]
    y = concatenated_bcsfa_angles2[inds]
    poly = Polynomial.fit(x, y, 1)

    cxs = np.array([0, half_pi])
    line_coords = [cxs, poly(cxs)]
    print(line_coords)
    return np.stack(line_coords)

prev2_angle_step = 0.1

prev2_angle_buf = 0.05

fig = plt.figure(figsize=plt.figaspect(1/3)) 
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

curr_prev2_val = 0.0
max_prev2_angle = all_prev2_angles.max()
dashwidth = 0.5
while curr_prev2_val + prev2_angle_step < max_prev2_angle:
    curr_prev2_max = curr_prev2_val + prev2_angle_step
    prev2_label = "{:0.2f}-{:0.2f}".format(curr_prev2_val, curr_prev2_max)
    curr_prev2_inds = np.logical_and(
        all_prev2_angles > curr_prev2_val, all_prev2_angles < curr_prev2_max
    )
    filtered_p2_p1_bcs = p2_p1_bcs_stack[curr_prev2_inds].transpose()
    ax.scatter(*filtered_p2_p1_bcs, label=prev2_label)
    ax2.scatter(*filtered_p2_p1_bcs[1:], label=prev2_label)
    curr_prev2_val += prev2_angle_step
    fit_is_valid = False
    if np.any(curr_prev2_inds):
        best_fit_coords = getBestFitForInds(curr_prev2_inds)
        best_fit_deltas = np.diff(best_fit_coords, axis=-1)
        best_fit_slope = best_fit_deltas[1] / best_fit_deltas[0]
        if abs(best_fit_slope) < 2:
            ax3.plot(*best_fit_coords, label=prev2_label, ls=(0, (dashwidth, 1)))
            dashwidth += 0.5
            fit_is_valid = True
        if not fit_is_valid:
            ax3.plot([0, 0], [1, 0], label=prev2_label)
    

planeYZ = np.meshgrid([-half_pi, half_pi], [-half_pi, half_pi])
planeX = np.zeros_like(planeYZ[0])
plane_surf0 = ax.plot_surface(
    planeX - prev2_angle_buf, *planeYZ, alpha=0.34, color='r', shade=False
)
plane_surf1 = ax.plot_surface(
    planeX + prev2_angle_buf, *planeYZ, alpha=0.34, color='r', shade=False
)

prev2_plane_init_inds = all_prev2_angles < prev2_angle_buf
prev2_init_data = p2_p1_bcs_stack[prev2_plane_init_inds].transpose()[1:]
prev2_plot_2D, = ax2.plot(
    *prev2_init_data, marker='.', ls='', color="yellow", label="highlighted"
)
best_fit_coords = getBestFitForInds(prev2_plane_init_inds)
best_fit_plot, = ax2.plot(*best_fit_coords)
p2_p1_bcs_lims = np.stack([
    np.min(p2_p1_bcs_stack, axis=0), np.max(p2_p1_bcs_stack, axis=0)
], axis=-1)
ax2.axis([-0.5, 2, -0.5, 2])

ax.set_xlim(*p2_p1_bcs_lims[0])
ax.set_ylim(*p2_p1_bcs_lims[1])
ax.legend()
ax2.legend(loc='upper right')
ax3.legend()

def move_plane(val):
    global plane_surf0
    global plane_surf1
    global prev2_plot_2D
    global best_fit_plot
    plane_surf0.remove()
    plane_surf0 = ax.plot_surface(
        planeX + val - prev2_angle_buf, *planeYZ, alpha=0.34, color='r',
        shade=False
    )
    plane_surf1.remove()
    plane_surf1 = ax.plot_surface(
        planeX + val + prev2_angle_buf, *planeYZ, alpha=0.34, color='r',
        shade=False
    )
    slider_plane_inds = (np.abs(val - all_prev2_angles) < prev2_angle_buf)
    slider_plane_data = p2_p1_bcs_stack[slider_plane_inds].transpose()[1:]
    prev2_plot_2D.set_data(*slider_plane_data)
    best_fit_coordvals = getBestFitForInds(slider_plane_inds)
    best_fit_plot.set_data(*best_fit_coordvals)
    fig.canvas.draw_idle()

slider_ax = plt.axes([0.6, 0.15, 0.35, 0.03])#, facecolor='lightgoldenrodyellow')
slider = Slider(slider_ax, "prev2", 0.0, half_pi, valinit=0.0)
slider.on_changed(move_plane)

plt.show()
#%%



allResultsObj.printResults(DisplayGrouping.BY_OBJECT, "Quadratic", "QuatVel")


numTimestamps = len(translations_gt)
timestamps = np.arange(numTimestamps)
tx_coords = np.stack((timestamps, translations[:, 2]), axis = -1)
ptx_coords = np.stack((timestamps[2:], t_vel_preds[:, 2]), axis = -1)

v_line_coords = np.hstack((tx_coords[:-2], ptx_coords)).reshape((len(ptx_coords), 2, 2))

#%%
acc_err_x = pm.einsumDot(sampleDeltas, sampleAccErrs[2:])
acc_err_sqr_len = pm.einsumDot(sampleAccErrs[2:], sampleAccErrs[2:])
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

vid_key = list(allResultsObj._allBodSeqKeys)[9]
keys = ["Vel", "Acc", "Quadratic", "Jerk", "Circ"]# ,"deg4"]#, "VelLERP", "Acc", "AccLERP"]
key_colours = ["blue", "orange", "green", "red", "grey", "brown"]
key_rgbas = [to_rgba(kc) for kc in key_colours]

num_timesteps = len(allResultsObj.translation_results[keys[0]].errors[vid_key])
x_vals = np.arange(1, num_timesteps + 1)
all_err_norms = []
for i, key in enumerate(keys):
    errs = allResultsObj.translation_results[key].errors[vid_key]
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
blc = LineCollection(best_line_cos, colors=best_colors, linewidths=5)

ax.add_collection(blc)

ax.set_ylim(bottom = -5)
ax.legend()
plt.show()

