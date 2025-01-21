import typing
from collections import namedtuple

import numpy as np
from scipy.stats import t as sp_t

import posemath as pm
from gtCommon import BCOT_Data_Calculator
import gtCommon as gtc

ErrStats = namedtuple(
    "ErrStats",
    [
        'mean', 'mean_conf', 'std_dev', 'N', 'eigvals', 'eigvecs', 'det',
        'avg_dist_from_mean', 'avg_sq_dist_from_mean'
    ]
)

motion_kinds = [
    "movable_handheld", "movable_suspension", "static_handheld",
    "static_suspension", "static_trans"
]

def getMinPrecisionForVec(vec: np.ndarray):
    return int(np.max(np.ceil(-np.log10(np.abs(vec)))))

def formatVec(vec: np.ndarray, prec: int = None):
        if prec is None:
            min_prec_req = getMinPrecisionForVec(vec)
            prec = max(min_prec_req, 2)
        items = ["{:.{}f}".format(v, prec) for v in vec]
        return "[{}]".format(', '.join(items))

combos = []
for s, s_val in enumerate(gtc.BCOT_SEQ_NAMES):
    k = ""
    for k_opt in motion_kinds:
        if k_opt in s_val:
            k = k_opt
            break
    for b in range(len(gtc.BCOT_BODY_NAMES)):
        if BCOT_Data_Calculator.isBodySeqPairValid(b, s, True):
            combos.append((b, s, k))


all_rotations_T: typing.Dict[typing.Tuple[int, int, int], np.ndarray] = dict()
all_translations: typing.Dict[typing.Tuple[int, int, int], np.ndarray] = dict()

for skip_amt in range(3):
    for combo in combos:
        calculator = BCOT_Data_Calculator(combo[0], combo[1], skip_amt)
        rotations = calculator.getRotationMatsGTNP(True)
        ck = (skip_amt, ) + combo[:2]
        all_rotations_T[ck] = np.swapaxes(rotations, -1, -2) # transposes
        all_translations[ck] = calculator.getTranslationsGTNP(True)

#%%



prediction_kinds = ["static", "vel-deg1", "vel-deg2", "acc-deg2"]
def get_start_data_struct():
    return [
        {pk: {mk: [] for mk in motion_kinds} for pk in prediction_kinds}
        for _ in range(3)
    ]

world_err_lists = get_start_data_struct()
local_err_lists = get_start_data_struct()
vel_deg1_err_lists = get_start_data_struct()
vel_deg2_err_lists = get_start_data_struct()

default_acc_dir = np.array([1.0, 0.0, 0.0])
for skip_amt in range(3):
    for combo in combos:
        ck = (skip_amt, ) + combo[:2]

        rt_mats = all_rotations_T[ck]
        translations = all_translations[ck]
        n_next_translations = len(translations) - 1

        prev_translations = translations[:-1]
        deg1_vels = np.diff(prev_translations, 1, axis=0)
        deg2_acc = np.diff(deg1_vels, 1, axis=0)
        half_deg2_acc = deg2_acc / 2
        
        deg2_vels = deg1_vels[1:] + half_deg2_acc

        deg2_acc_lpad = np.empty_like(deg1_vels)
        deg2_acc_lpad[1:] = deg2_acc
        deg2_acc_lpad[0] = default_acc_dir

        deg1_vel_frames = pm.getOrthonormalFrames(deg1_vels, deg2_acc_lpad)
        deg2_vel_frames = pm.getOrthonormalFrames(deg2_vels, deg2_acc)

        # Transposes
        deg1_vel_frames = np.swapaxes(deg1_vel_frames, -1, -2)
        deg2_vel_frames = np.swapaxes(deg2_vel_frames, -1, -2)

        # def colList(mats):
        #     return (mats[..., 0], mats[..., 1], mats[..., 2])

        # if not pm.areAxisArraysOrthonormal(colList(deg1_vel_frames), loud=True):
        #     raise Exception("deg1 orthonormal issue!")
        # if not pm.areAxisArraysOrthonormal(colList(deg2_vel_frames), loud=True):
        #     raise Exception("deg2 orthonormal issue!")

        temp_preds = dict()
        temp_preds["static"] = prev_translations
        temp_preds["vel-deg1"] = translations[1:-1] + deg1_vels
        temp_preds["vel-deg2"] = translations[2:-1] + deg2_vels
        temp_preds["acc-deg2"] = temp_preds["vel-deg2"] + half_deg2_acc

        # unit_deg1_vels = pm.normalizeAll(deg1_vels)
        next_deg1_vel_frames = deg1_vel_frames[1:]
        # unit_deg2_vels = pm.normalizeAll(deg2_vels)

        for pk, preds in temp_preds.items():
            len_diff = n_next_translations - len(preds)
            errs = translations[(len_diff + 1):] - preds

            world_err_lists[skip_amt][pk][combo[-1]].append(errs)

            local_err_lists[skip_amt][pk][combo[-1]].append(pm.einsumMatVecMul(
                rt_mats[len_diff:-1], errs
            ))

            deg1_vel_frame_subset = deg1_vel_frames
            if len_diff == 2:
                deg1_vel_frame_subset = next_deg1_vel_frames

            deg1_len_diff = max(1 - len_diff, 0)
            vel_deg1_err_lists[skip_amt][pk][combo[-1]].append(
                pm.einsumMatVecMul(deg1_vel_frame_subset, errs[deg1_len_diff:]
            ))

            vel_deg2_err_lists[skip_amt][pk][combo[-1]].append(
                pm.einsumMatVecMul(deg2_vel_frames, errs[2 - len_diff:]
            ))

def concatErrs(err_list_data_struct):
    ret_val = []
    for els_for_skip in err_list_data_struct:
        ret_subdict = dict()
        for pk, els in els_for_skip.items():
            ret_subdict[pk] = {mk: np.concatenate(es) for mk, es in els.items()}
            rs_vals = ret_subdict[pk].values()
            rs_t = tuple(rs_vals)
            ret_subdict[pk]["all"] = np.concatenate(rs_t)
        ret_val.append(ret_subdict)
    return ret_val

world_errs = concatErrs(world_err_lists)
local_errs = concatErrs(local_err_lists)
vel_deg1_errs = concatErrs(vel_deg1_err_lists)
vel_deg2_errs = concatErrs(vel_deg2_err_lists)

#%%

# Math comes from multiple sources, but one good one is:
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm
def meanConfInterval(mean, std_dev, N, significance = 0.05):
    which_percentile = 1.0 - (significance / 2.0)
    t_val = sp_t.ppf(which_percentile, N - 1)
    mean_rad = t_val * (std_dev / np.sqrt(N))
    return (mean - mean_rad, mean + mean_rad)

def getStats(errs, skip_amt: int, motion_kind: str, pred_kind: str):
    errs_subset = errs[skip_amt][pred_kind][motion_kind]
    mean = np.mean(errs_subset, axis=0, keepdims=True)
    diffs_from_mean = errs_subset - mean
    avg_dist = np.linalg.norm(diffs_from_mean, axis=-1).mean()
    avg_sq_dist = pm.einsumDot(diffs_from_mean, diffs_from_mean).mean()
    std_dev = np.std(errs_subset, axis=0, mean=mean, ddof=1)
    mean = mean.flatten()
    if len(mean) == 1:
        mean = mean[0]
    mc = meanConfInterval(mean, std_dev, len(errs_subset))
    cov = np.cov(errs_subset.transpose())
    eigVals, eigVecs = np.linalg.eig(cov)
    det = np.linalg.det(cov)
    # if np.abs(det - np.prod(eigVals)) > 0.0001:
    #     raise Exception("Determinant not as expected!")

    return ErrStats(
        mean, mc, std_dev, len(errs_subset), eigVals, eigVecs, det,
        avg_dist, avg_sq_dist
    )

motion_kinds_plus = motion_kinds + ["all"]
def getStatsStruct(errs):
    return [
        {
            pk: {mk: getStats(errs, skip, mk, pk) for mk in motion_kinds_plus}
            for pk in prediction_kinds
        }
        for skip in range(3)
    ]

world_stats_struct = getStatsStruct(world_errs)
local_stats_struct = getStatsStruct(local_errs)
vel_deg1_stats_struct = getStatsStruct(vel_deg1_errs)
vel_deg2_stats_struct = getStatsStruct(vel_deg2_errs)

#%%

print("Note: Default acc dir used for first deg1-vel frame is:", list(default_acc_dir))
print() # newline

for skip_amt in range(3):
    print("\nSkip {}:".format(skip_amt))

    # Finding the min or max lower or upper bound for the mean confidence
    # intervals depending on whether motion type is static or not, to ensure
    # that the statement it's always positive for static predictions and always
    # negative for others is valid. For now, I only care about reference frames
    # aligned with motion.
    ref_frame_names = ("world", "local", "v1", "v2")
    min_static_mclbs = {rfn: 1000.0 for rfn in ref_frame_names[-2:]}
    max_nonstatic_mcubs = {rfn: -1000.0 for rfn in ref_frame_names[-2:]}
    for pk in prediction_kinds:
        print("  " + pk + ":")
        for mk in motion_kinds_plus:
            print("    " + mk + ":")
            world_stats = world_stats_struct[skip_amt][pk][mk]
            local_stats = local_stats_struct[skip_amt][pk][mk]
            vel_deg1_stats = vel_deg1_stats_struct[skip_amt][pk][mk]
            vel_deg2_stats = vel_deg2_stats_struct[skip_amt][pk][mk]
            all_stats = (
                world_stats, local_stats, vel_deg1_stats, vel_deg2_stats
            )
            for name, stat in zip(ref_frame_names, all_stats):
                if "v" in name:
                    if "static" in pk:
                        curr_mclb = np.asarray(stat.mean_conf[0][0])
                        min_static_mclbs[name] = np.min(np.append(
                            curr_mclb, min_static_mclbs[name]
                        ))
                    else:
                        curr_mcub = np.asarray(stat.mean_conf[1][0])
                        max_nonstatic_mcubs[name] = np.max(np.append(
                            curr_mcub, max_nonstatic_mcubs[name]
                        ))
                print("      {} ({} samples):".format(name, stat.N))
                print("        - Mean: {}, 0.95 conf: ({}, {})".format(
                    formatVec(stat.mean),
                    formatVec(stat.mean_conf[0]), formatVec(stat.mean_conf[1])
                ))
                s = "        - Avg Distance to Mean: {}, Squared: {}".format(
                    stat.avg_dist_from_mean, stat.avg_sq_dist_from_mean
                )
                print(s)
                print("        - Det:", stat.det, end=", ")
                eig_order = np.argsort(stat.eigvals)[::-1]
                ordered_eigs = stat.eigvals[eig_order]
                print("Eigenvalues:", formatVec(ordered_eigs), end=", ")
                print("s:", formatVec(stat.std_dev))
                print("        - Eigenvectors:")
                for eig_ind in eig_order:
                    print("         ", formatVec(stat.eigvecs[eig_ind]))
            print()
        print("  " + ("-" * 78))
    print("Min mean confidence interval lower bound for 'static' motion:", min_static_mclbs)
    print("Max mean confidence interval upper bound for other motion:", max_nonstatic_mcubs)
#%%

skip_key = 0
pred_key = "vel-deg1"
motion_key = "all"
print(local_stats_struct[skip_key][pred_key][motion_key])
print(vel_deg1_stats_struct[skip_key][pred_key][motion_key])
print(vel_deg2_stats_struct[skip_key][pred_key][motion_key])
