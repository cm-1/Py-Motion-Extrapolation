import typing
import json

import numpy as np

import errorstats as es
from gtCommon import BCOT_Data_Calculator
import gtCommon as gtc


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

        prev_translations = translations[:-1]
        deg1_vels = np.diff(prev_translations, 1, axis=0)
        deg2_acc = np.diff(deg1_vels, 1, axis=0)
        half_deg2_acc = deg2_acc / 2
        
        deg2_vels = deg1_vels[1:] + half_deg2_acc


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

        reframed_errs = es.localizeErrsInFrames(
            temp_preds, translations, rt_mats, default_acc_dir=default_acc_dir,
            deg1_vels=deg1_vels, deg2_vels=deg2_vels, deg2_acc=deg2_acc, 
        )
        
        for pk, err_coll in reframed_errs.items():
            world_err_lists[skip_amt][pk][combo[-1]].append(err_coll.wrt_world)
            local_err_lists[skip_amt][pk][combo[-1]].append(err_coll.wrt_local)
            vel_deg1_err_lists[skip_amt][pk][combo[-1]].append(
                err_coll.wrt_vel_deg1
            )
            vel_deg2_err_lists[skip_amt][pk][combo[-1]].append(
                err_coll.wrt_vel_deg2
            )
            
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

def getStats(errs, skip_amt: int, motion_kind: str, pred_kind: str):
    errs_subset = errs[skip_amt][pred_kind][motion_kind]
    return es.getStats(errs_subset)

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
            world_stats = world_stats_struct[skip_amt][pk][mk]
            print("    {} (score {:0.4f}):".format(mk, world_stats.mean_mag))
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
                print(es.formattedErrStats(stat, name, 6, world_stats))
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

#%%
# Commenting this out because I already viewed the plot and it showed that
# there is no direct linear, quadratic, etc. relationship between them.
# import matplotlib.pyplot as plt

# plt.scatter(all_dets, all_dists)
# plt.show()

class StatsJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, es.ErrStats):
            ...
            # TODO: handle namedtuple
        return super().default(obj)
    
# print(json.dumps(local_stats_struct, cls=StatsJSONEncoder, indent=2))
    