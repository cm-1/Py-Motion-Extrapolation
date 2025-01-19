import typing
from collections import namedtuple

import numpy as np
from scipy.stats import t as sp_t

import posemath as pm
from gtCommon import BCOT_Data_Calculator
import gtCommon as gtc

ErrStats = namedtuple(
    "ErrStats", ['mean', 'mean_conf', 'std_dev', 'N']
)

motion_kinds = [
    "movable_handheld", "movable_suspension", "static_handheld",
    "static_suspension", "static_trans"
]


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
        all_rotations_T[ck] = np.moveaxis(rotations, -2, -1) # transposes
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

        temp_preds = dict()
        temp_preds["static"] = prev_translations
        temp_preds["vel-deg1"] = translations[1:-1] + deg1_vels
        temp_preds["vel-deg2"] = translations[2:-1] + deg2_vels
        temp_preds["acc-deg2"] = temp_preds["vel-deg2"] + half_deg2_acc

        unit_deg1_vels = pm.normalizeAll(deg1_vels)
        next_unit_deg1_vels = unit_deg1_vels[1:]
        unit_deg2_vels = pm.normalizeAll(deg2_vels)

        for pk, preds in temp_preds.items():
            len_diff = n_next_translations - len(preds)
            errs = translations[(len_diff + 1):] - preds

            world_err_lists[skip_amt][pk][combo[-1]].append(errs)

            local_err_lists[skip_amt][pk][combo[-1]].append(pm.einsumMatVecMul(
                rt_mats[len_diff:-1], errs
            ))

            fitted_unit_deg1_vels = unit_deg1_vels
            if len_diff == 2:
                fitted_unit_deg1_vels = next_unit_deg1_vels

            deg1_len_diff = max(1 - len_diff, 0)
            vel_deg1_err_lists[skip_amt][pk][combo[-1]].append(pm.einsumDot(
                errs[deg1_len_diff:], fitted_unit_deg1_vels
            ))

            vel_deg2_err_lists[skip_amt][pk][combo[-1]].append(pm.einsumDot(
                errs[2 - len_diff:], unit_deg2_vels
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
    std_dev = np.std(errs_subset, axis=0, mean=mean, ddof=1)
    mean = mean.flatten()
    if len(mean) == 1:
        mean = mean[0]
    mc = meanConfInterval(mean, std_dev, len(errs_subset))
    return ErrStats(mean, mc, std_dev, len(errs_subset))

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

for skip_amt in range(3):
    print("\nSkip {}:".format(skip_amt))
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
            for name, stat in zip(("world", "local", "v1", "v2"), all_stats):
                if stat.mean.ndim == 0:
                    ps = "      {}: m={} ({}, {}), s={}".format(
                        name, stat.mean, stat.mean_conf[0], stat.mean_conf[1],
                        stat.std_dev
                    )
                    print(ps)
                else:
                    ps = "      {}: m={}, s={}".format(
                        name, stat.mean, stat.std_dev
                    )
                    print(ps)
                    print("        mc =", stat.mean_conf)
            print()
        print("  " + ("-" * 78))
#%%

skip_key = 0
pred_key = "vel-deg1"
motion_key = "all"
print(local_stats_struct[skip_key][pred_key][motion_key])
print(vel_deg1_stats_struct[skip_key][pred_key][motion_key])
print(vel_deg2_stats_struct[skip_key][pred_key][motion_key])
