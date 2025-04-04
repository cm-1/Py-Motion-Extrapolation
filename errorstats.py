from collections import namedtuple
import typing

import numpy as np
from numpy.typing import NDArray

from scipy.stats import t as sp_t

import posemath as pm

# Classes/Structs

class ErrStats(typing.NamedTuple):
    mean: NDArray
    mean_conf: typing.Tuple[NDArray, NDArray]
    std_dev: NDArray
    N: int
    eigvals: NDArray
    eigvecs: NDArray
    det: float
    avg_dist_from_mean: float
    avg_sq_dist_from_mean: float
    cov_mat: NDArray
    mean_mag: float

class LocalizedErrsCollection(typing.NamedTuple):
    wrt_world: NDArray
    wrt_local: NDArray
    wrt_vel_deg1: NDArray
    wrt_vel_deg2: NDArray



# Math Functions:

def localizeErrsInFrames(global_preds_dict: typing.Dict[typing.Any, NDArray],
                         translations: NDArray,
                         world_to_local_rotation_mats: NDArray, *,
                         deg1_vels: typing.Optional[NDArray] = None,
                         deg2_vels: typing.Optional[NDArray] = None,
                         deg2_acc: typing.Optional[NDArray] = None,
                         default_acc_dir: NDArray = None):

    if deg1_vels is None:
        deg1_vels = np.diff(translations[:-1], 1, axis=0)
    else:
        assert len(deg1_vels) == len(translations) - 2, (
            "Must have a degree-1 velocity for every \"previous\" translation "
            "pair but no others!"
        )

    if deg2_acc is None:
        deg2_acc = np.diff(deg1_vels, 1, axis=0)
    else:
        assert len(deg2_acc) == len(translations) - 3, \
        "Must have 3 fewer accelerations than translations!"

    if deg2_vels is None:
        deg2_vels = deg1_vels[1:] + (deg2_acc / 2.0)
    else:
        assert len(deg2_vels) == len(translations) - 3, \
        "Must have 3 fewer degree-2 velocities than translations!"


    if default_acc_dir is None:
        default_acc_dir = np.array([1.0, 0.0, 0.0])

    deg2_acc_lpad = np.empty_like(deg1_vels)
    deg2_acc_lpad[1:] = deg2_acc
    deg2_acc_lpad[0] = default_acc_dir

    deg1_vel_frames = pm.getOrthonormalFrames(deg1_vels, deg2_acc_lpad)
    deg2_vel_frames = pm.getOrthonormalFrames(deg2_vels, deg2_acc)

    # Transposes
    deg1_vel_frames = np.swapaxes(deg1_vel_frames, -1, -2)
    deg2_vel_frames = np.swapaxes(deg2_vel_frames, -1, -2)
    next_deg1_vel_frames = deg1_vel_frames[1:]
    # unit_deg2_vels = pm.normalizeAll(deg2_vels)
    n_next_translations = len(translations) - 1

    res_dict: typing.Dict[typing.Any, LocalizedErrsCollection] = dict()
    for pk, preds in global_preds_dict.items():
        len_diff = n_next_translations - len(preds)
        errs = translations[(len_diff + 1):] - preds

        local_errs = pm.einsumMatVecMul(
            world_to_local_rotation_mats[len_diff:-1], errs
        )

        deg1_vel_frame_subset = deg1_vel_frames
        if len_diff == 2:
            deg1_vel_frame_subset = next_deg1_vel_frames

        deg1_len_diff = max(1 - len_diff, 0)
        vel_deg1_errs = pm.einsumMatVecMul(
            deg1_vel_frame_subset, errs[deg1_len_diff:]
        )

        vel_deg2_errs = pm.einsumMatVecMul(deg2_vel_frames, errs[2 - len_diff:])

        res_dict[pk] = LocalizedErrsCollection(
            errs, local_errs, vel_deg1_errs, vel_deg2_errs
        )
    return res_dict

# Math comes from multiple sources, but one good one is:
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm
def meanConfInterval(mean, std_dev, N, significance = 0.05):
    which_percentile = 1.0 - (significance / 2.0)
    t_val = sp_t.ppf(which_percentile, N - 1)
    mean_rad = t_val * (std_dev / np.sqrt(N))
    return (mean - mean_rad, mean + mean_rad)

def getStats(errs: NDArray):
    mean = np.mean(errs, axis=0, keepdims=True)
    diffs_from_mean = errs - mean
    avg_dist = np.linalg.norm(diffs_from_mean, axis=-1).mean()
    avg_sq_dist = pm.einsumDot(diffs_from_mean, diffs_from_mean).mean()
    std_dev = np.std(errs, axis=0, mean=mean, ddof=1)
    mean = mean.flatten()
    if len(mean) == 1:
        mean = mean[0]
    mc = meanConfInterval(mean, std_dev, len(errs))
    cov = np.cov(errs.transpose())
    eigVals, eigVecs = np.linalg.eig(cov)
    det = np.linalg.det(cov)
    # if np.abs(det - np.prod(eigVals)) > 0.0001:
    #     raise Exception("Determinant not as expected!")
    mean_mag = np.linalg.norm(errs, axis=-1).mean()

    return ErrStats(
        mean, mc, std_dev, len(errs), eigVals, eigVecs, det,
        avg_dist, avg_sq_dist, cov, mean_mag
    )


# https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
# Not super useful, but I was sort of curious and it was fast to implement.
def getKLDivergence(mean0: np.ndarray, mean1: np.ndarray,
                  cov_mat0: np.ndarray, cov_mat1: np.ndarray,
                  det0: float = None, det1: float = None):
    k = len(mean0)
    assert k == len(mean1), \
        "Dimensions ({}, {}) don't match!".format(k, len(mean1))
    
    if det0 is None:
        det0 = np.linalg.det(cov_mat0)
    if det1 is None:
        det1 = np.linalg.det(cov_mat1)

    cov_mat1_inv = np.linalg.inv(cov_mat1)
    trace_input = cov_mat1_inv @ cov_mat0
    tr = np.trace(trace_input)

    mean_diff = (mean1 - mean0).reshape(-1, 1)
    mean_pt = mean_diff.transpose() @ cov_mat1_inv @ mean_diff
    ln_pt = np.log(det1 / det0)
    return (tr - k + mean_pt + ln_pt) /  2.0


# Printing/Formatting Functions:

def getMinPrecisionForVec(vec: np.ndarray):
    return int(np.max(np.ceil(-np.log10(np.abs(vec)))))

def formatVec(vec: np.ndarray, prec: int = None):
        if prec is None:
            min_prec_req = getMinPrecisionForVec(vec)
            prec = max(min_prec_req, 2)
        items = ["{:.{}f}".format(v, prec) for v in vec]
        return "[{}]".format(', '.join(items))

def formattedErrStats(stats: ErrStats, name: str, indent_spaces: int,
            ref_stats: typing.Optional[ErrStats] = None):
    tab = " " * indent_spaces
    fs = "" # Full String

    fs += tab + "{} ({} samples):\n".format(name, stats.N)
    fs += tab + "  - Mean: {}, 0.95 conf: ({}, {})\n".format(
        formatVec(stats.mean),
        formatVec(stats.mean_conf[0]), formatVec(stats.mean_conf[1])
    )

    fs += tab + "  - Avg Distance to Mean: {}, Squared: {}\n".format(
        stats.avg_dist_from_mean, stats.avg_sq_dist_from_mean
    )
    fs += tab + "  - Det: {}, ".format(stats.det)
    eig_order = np.argsort(stats.eigvals)[::-1]
    ordered_eigs = stats.eigvals[eig_order]
    fs += "Eigenvalues: " + formatVec(ordered_eigs) + ", "
    fs += "s: " + formatVec(stats.std_dev) + ", "
    if ref_stats is not None:
        kl_val = getKLDivergence(
            stats.mean, ref_stats.mean,
            stats.cov_mat, ref_stats.cov_mat, stats.det, ref_stats.det
        )
        fs += "KL: {}\n".format(kl_val)
    else:
        fs += "\n"
    fs += tab + "  - Eigenvectors:\n"
    for eig_ind in eig_order:
        fs += tab + "    " + formatVec(stats.eigvecs[eig_ind]) + "\n"
    
    return fs
