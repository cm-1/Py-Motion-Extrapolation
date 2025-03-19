import typing

import numpy as np
from numpy.typing import NDArray 
from scipy.optimize import least_squares

import posemath as pm

def min_jerk(t_f: typing.Union[float, NDArray],
             t: typing.Union[int, NDArray], x_0: NDArray, x_f: NDArray):
    tau = t / t_f
    tau2 = tau**2
    tau3 = tau * tau2
    return x_0 + (15 * tau - 6 * tau2 - 10) * tau3 * (x_0 - x_f)



def min_jerk_sq_sum_to_opt(t_f: float, x_f: NDArray, t_vals: NDArray, 
                           x_vals: NDArray, forget_denoms: NDArray):
    min_jerk_vals = min_jerk(t_f, t_vals, x_vals[0], x_f)
    min_jerk_err_vecs = x_vals - min_jerk_vals
    ret_div = min_jerk_err_vecs / forget_denoms
    # Full result: sum(||(y_i - f(...))/a^(n-i)||^2)
    return ret_div.flatten()


def min_jerk_init_guess(x0: NDArray, x1: NDArray, x2: NDArray):
    ndim = x0.ndim
    if ndim == 1:
        x0 = x0.reshape(1,-1)
        x1 = x1.reshape(1,-1)
        x2 = x2.reshape(1,-1)
    # As an ad-hoc initial guess, I'll take a line starting at x0
    # and going through the average of x1 and x2, project all points
    # onto this line, and then calculate the min jerk solution for
    # those points, if it exists.
    avg_x1_x2 = (x1 + x2)/2.0
    dir = avg_x1_x2 - x0
    dir /= np.linalg.norm(dir, axis=-1, keepdims=True)

    # Below, let's assume any "x" is 1D, not 3D.
    # Let P_1 and P_2 be our two polynomials in terms of tau.
    # Let xd = x0 - xf

    # (x1 - x0) = P_1(1/tf) * xd
    # (x2 - x0) = P_2(1/tf) * xd

    # Interpret LHS and RHS as parallel vec2s with xd as const scale.
    # Then we know that LHS and rotated RHS are orthogonal. Get:
    # 0 = P_1(1/tf) * (x2 - x0) + P_2(1/tf) * (x0 - x1)
    # When you factor out a (1/tf)**3, get a quadratic polynomial.
    x1_1d = pm.einsumDot(x1 - x0, dir)
    x2_1d = pm.einsumDot(x2 - x0, dir)
    # (x0_1d is just 0.0)
    a = -6 * x2_1d + 192 * x1_1d # 192 = 6*2^2*2^3
    b = 15 * x2_1d - 240 * x1_1d # 240 = 15*2*2^3
    c = -10 * x2_1d + 80 * x1_1d # 80 = 10 * 2^3

    quad_solns = pm.quadraticFormulaNP(a, b, c)

    guess_len = 1 + x0.shape[1]
    guesses = np.empty((x0.shape[0], guess_len))
    quad_solns_invalid = np.isnan(quad_solns[:, 0]) | (quad_solns[:, 1] <= 0.0)
    quad_solns_valid = ~quad_solns_invalid
    valid_quad_solns = quad_solns[quad_solns_valid]
    # We want the larger value of t_f. If both tau options are > 0, that
    # means picking the first/smaller one. Otherwise, we pick the positive.
    taus = np.empty(valid_quad_solns.shape[0])
    pick_first = valid_quad_solns[:, 0] > 0.0
    not_pick_first = ~pick_first
    taus[pick_first] = valid_quad_solns[pick_first, 0]
    taus[not_pick_first] = valid_quad_solns[not_pick_first, 1]
    taus2 = taus*taus
    taus3 = taus*taus2
    x_fs_1d = -x1_1d[quad_solns_valid] / (taus3 * (15 * taus - 6 * taus2 - 10))
    guesses[quad_solns_valid, 0] = 1.0 / taus
    guesses[quad_solns_valid, 1:] = x0[quad_solns_valid] + pm.scalarsVecsMul(
        x_fs_1d, dir[quad_solns_valid]
    )

    # What if there's no solution to the quadratic? For now I'll just set x2 as 
    # the initial guess and t_f = 2.
    guesses[quad_solns_invalid, 0] = 2.0
    guesses[quad_solns_invalid, 1:] = x2[quad_solns_invalid]

    if ndim == 1:
        return guesses[0]
    return guesses

def min_jerk_lsq(prev_translations: NDArray, is_static_bools: NDArray,
                 is_ang_big_bools: NDArray, forget_fac: float = 0.91,
                 guess_xtol = 0.001, max_opt_iters: int = 33, 
                 max_pos_mag: float = 10000.0, max_jerk_duration: int = 3600,
                 vels: NDArray = None, accs: NDArray = None,
                 jerks: NDArray = None):
    
    n_input_frames = prev_translations.shape[0]
    all_ts = np.arange(n_input_frames)

    assert is_static_bools.shape[0] == (n_input_frames - 1), \
    "The number of \"is static\" bools must be 1 less than previous translations!"

    non_arm_motion_bools = np.concatenate(([True, False], is_ang_big_bools))

    # If the object *just* started moving, we'll assume there was no big angle.
    # Negation: it is currently static or was moving last frame. 
    non_arm_motion_bools[2:] &= is_static_bools[1:] | (~is_static_bools[:-1])
    
    # Finally, there's no arm motion if the object is currently static.
    non_arm_motion_bools[1:] |= is_static_bools

    pops: NDArray = None
    if vels is None:
        vels = np.diff(prev_translations, axis=0)
    if accs is None:
        accs = np.diff(vels, axis=0)
    if jerks is None:
        pops = np.diff(accs, n=4, axis=0)
    else:
        pops = np.diff(jerks, n=3, axis=0)

    

    pop_mags = np.linalg.norm(pops, axis=-1)
    lc = -len(pops)
    if lc < 0:
        acc_mags = np.linalg.norm(accs[lc:], axis=-1)
        non_arm_motion_bools[lc:] |= (
            (pop_mags > 0.1) & (pop_mags > (1.25 * acc_mags))
        )

    _, time_arm, _ = pm.since_calc(
        non_arm_motion_bools, n_input_frames, [], 0
    )

    # va_dots = pm.einsumDot(vels[1:], accs)
    # va_dot_signs = np.signbit(va_dots)
    # va_dot_sign_change = (va_dot_signs[1:] != va_dot_signs[:-1])
    # va_ind_diff = len(time_arm) - len(va_dot_sign_change)
    # sign_change_count = 0
    # ta_sub = 0
    # for i, ta in enumerate(time_arm):
    #     if ta == 1:
    #         ta_sub = 0
    #         sign_change_count = 0
    #     elif ta >= 2:
    #         if va_dot_sign_change[i - va_ind_diff]:
    #             sign_change_count += 1
    #         if sign_change_count > 1:
    #             ta_sub = ta
    #             sign_change_count = 0
    #         time_arm[i] -= ta_sub

    # The below is a small example that shows that trying to accomplish this
    # index "combination" is a lot harder with int inds than with bool inds;
    # if you look at the "max" row and try to get it to match the
    # "last non-arm ind" row, you should see why.
    # ---
    # Time since static == 1 -> arm movement assumed
    # Time since static == 0 -> no arm movement assumed
    # Time since static  > 1 -> arm movement iff time since big ang > 1
    # ---
    # last static ind:   0, 1, 1, 1, 4, 4, 4, 4, 4, 9, 10, 11, 11, 11, 11, 15,
    # last bigang ind:   0, 1, 2, 2, 2, 2, 2, 7, 8, 8, 8,  8,  8,  8,  14, 14
    # max:               0, 1, 2, 2, 4, 4, 4, 7, 8, 9, 10, 11, 11, 11, 14, 15
    # last no-narm ind:  0, 1, 1, 1, 4, 4, 4, 7, 8, 9, 10, 11, 11, 11, 14, 15
    # ---
    # time since static: 0, 0, 1, 2, 0, 1, 2, 3, 4, 0, 0,  0,  1,  2,  3,  0,
    # time since bigang: 0, 0, 0, 1, 2, 3, 4, 0, 0, 1, 2,  3,  4,  5,  0,  1,
    # time arm:          0, 0, 1, 2, 0, 1, 2, 0, 0, 0, 0,  0,  1,  2,  0,  0, 


    # If we are only 1 second in (meaning there are only 2 previous poses), then
    # there are infinitely many min jerk solutions. So we must. filter further.
    # We also cut off the front two indices because (a) we know they will be
    # invalid anyway and (b) it will make getting x0, x1, and x2 later a bit
    # simpler.
    use_min_jerk = time_arm[2:] >= 2
    # When we're first starting to use min jerk poses "again", we need to pick
    # a new initial guess; otherwise, we just use the last optimization result.
    make_init_guess = time_arm[2:] == 2

    x2_ig = prev_translations[2:][make_init_guess]
    x1_ig = prev_translations[1:-1][make_init_guess]
    x0_ig = prev_translations[:-2][make_init_guess]
    init_guesses = min_jerk_init_guess(x0_ig, x1_ig, x2_ig)


    dim = prev_translations.shape[1]
    guess_len = dim + 1
    guess = None
    ig_ind = 0

    # Filter out "valid" times via numpy so we have fewer for-loop iterations.
    min_jerk_times = time_arm[2:][use_min_jerk]
    use_min_jerk_int_inds = np.where(use_min_jerk)[0] + 2 # Main array inds.
    num_min_jerk_calcs = min_jerk_times.shape[0]
    if num_min_jerk_calcs == 0:
        return np.full_like(prev_translations, np.nan)

    # When we calculate our final min jerk predictions, we'll need an array with
    # an x0 for each calculation. So we'll need to repeat the ones we selected.
    # To find out how long we need to repeat each one, we need to find out how
    # many calculations each x0 will be used for. This means seeing when each
    # "interval" ends, which requires finding out when we go from assuming
    # min jerk motion to not doing so. 
    min_jerk_end_inds = non_arm_motion_bools[2:] & (~non_arm_motion_bools[1:-1])
    x0_repeat_lens = time_arm[1:-1][min_jerk_end_inds] - 1
    # If we had "min jerk" motion for only 2 poses (1 timestep), then we'll
    # have repeat lens of 0 that we do not want to include.
    x0_repeat_lens = x0_repeat_lens[x0_repeat_lens > 0]
    # Finally, if our last frame is min jerk motion, it wouldn't be detected via
    # the above boolean check, so we'll add it in here.
    if x0_repeat_lens.shape[0] < x0_ig.shape[0]:
        x0_repeat_lens = np.append(x0_repeat_lens, time_arm[-1] - 1)
    
    x0s = np.repeat(x0_ig, x0_repeat_lens, axis=0)

    
    # TODO: The below does get repeated each call... is there a good Pythonic
    # way to remove the redundancy? Probably put this in a class and store this
    # as a variable upon init I guess.
    max_t_vals_len = np.max(x0_repeat_lens) + 2
    t_vals_super = all_ts[:max_t_vals_len].reshape(-1, 1)
    forget_denoms_rev = forget_fac**(t_vals_super[::-1])
    
    # Bounds on the values of [t_f, x_f] for the optimization.
    # We'll start by filling out the bounds for x_f.
    bounds_0 = np.full(dim + 1, -max_pos_mag)
    bounds_1 = np.full(dim + 1, max_pos_mag)
    # The bounds for t_f are at index[0].
    bounds_0[0] = 2 # All min jerk calculations require this t_f lower bound.
    bounds_1[0] = max_jerk_duration
    bounds = (bounds_0, bounds_1)

    t_fs = np.empty(min_jerk_times.shape)
    x_fs = np.empty_like(x0s)
    # We need a while loop because we might want to skip i ahead for efficiency
    # in cases where min jerk predictions stop being valid.
    i = 0
    while i < num_min_jerk_calcs:
        curr_time_arm = min_jerk_times[i]
        main_ind = use_min_jerk_int_inds[i]
        arm_translations = prev_translations[(main_ind - curr_time_arm):(main_ind + 1)]

        num_curr_poses = curr_time_arm + 1
        t_val_subset = t_vals_super[:num_curr_poses]
        forget_denom_subset = forget_denoms_rev[-num_curr_poses:]
        # inputs here are [tf, xf]
        to_opt = lambda inputs: min_jerk_sq_sum_to_opt(
            inputs[0], inputs[1:guess_len], t_val_subset, arm_translations,
            forget_denom_subset
        )

        if curr_time_arm == 2:
            guess = init_guesses[ig_ind]
            ig_ind += 1

        # Obviously, t_f must be at least the current time.
        bounds[0][0] = curr_time_arm
        if guess[0] < curr_time_arm:
            guess[0] = curr_time_arm # Guess must be within bounds!
        # This part is the real bottleneck.
        ls_res = least_squares(
            to_opt, guess, max_nfev=max_opt_iters, xtol=guess_xtol,
            bounds=bounds
        )
        guess = ls_res.x
        t_fs[i] = guess[0]
        x_fs[i] = guess[1:guess_len]
        i += 1

    next_min_jerk_times = min_jerk_times + 1
    not_needing_clip = next_min_jerk_times < t_fs

    t_fs_subset = t_fs[not_needing_clip].reshape(-1, 1)
    mjerk_times_subset = next_min_jerk_times[not_needing_clip].reshape(-1, 1)
    x0s_subset = x0s[not_needing_clip]
    x_fs_subset = x_fs[not_needing_clip]
    min_jerk_calcs = np.empty_like(x_fs)
    min_jerk_calcs[not_needing_clip] = min_jerk(
        t_fs_subset, mjerk_times_subset, x0s_subset, x_fs_subset
    )
    needing_clip = ~not_needing_clip
    x_fs_at_ends = x_fs[needing_clip]
    min_jerk_calcs[needing_clip] = x_fs_at_ends
    min_jerk_preds = np.empty_like(prev_translations)
    min_jerk_preds[2:][use_min_jerk] = min_jerk_calcs
    min_jerk_preds[2:][~use_min_jerk] = np.nan
    min_jerk_preds[:2] = np.nan
    return min_jerk_preds #x0s, t_fs, x_fs
