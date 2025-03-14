#%%
import random

import numpy as np
import matplotlib.pyplot as plt

import posemath as pm
import minjerk as mj

x_0 = np.array([1.35])
x_f = np.array([7.5])

t_f = 37.5

ts = np.arange(t_f).reshape(-1, 1)
ys = mj.min_jerk(t_f, ts, x_0, x_f)
y_subintervals = [ys]

last_t = ts[-1][0]
last_y = x_f
switch_ind = 2
switch_time = 17
for i in range(5):
    wait_time = random.randint(1, 33)
    if i == (switch_ind + 1):
        wait_time = 0
    y_subintervals.append(np.repeat(last_y, wait_time).reshape(-1, 1))
    last_t += wait_time
    next_x_f = last_y + 45 * (2.0 * random.random() - 1.0)
    min_min_jerk_time = 33 if i == switch_ind else 4
    jerk_time = random.randint(min_min_jerk_time, 35)
    jerk_ts = np.arange(jerk_time).reshape(-1, 1)
    if i == switch_ind:
        jerk_ts = jerk_ts[:switch_time]
    next_ys = mj.min_jerk(jerk_time, jerk_ts, last_y, next_x_f)
    y_subintervals.append(next_ys)
    last_y = next_ys[-1] if i == switch_ind else next_x_f
    t_inc = jerk_time if i != switch_ind else switch_time
    last_t += t_inc
    print("wait time, jerk time, next x_f", wait_time, jerk_time, next_x_f)
all_ts = np.arange(last_t + 1)

y_concat = np.concatenate(y_subintervals, axis=0)


#%%

x0, x1, x2 = ys[:3]
guess = mj.min_jerk_init_guess(ys[0], ys[1], ys[2])
guess_tf = guess[0] * (ts[1] - ts[0])[0]
guess_xf = guess[1:]
print("guess:", guess_tf, guess_xf)
print("actual:", t_f, x_f)

#%%

is_static = np.full(len(ys) - 2, False)
is_ang_big = np.full(len(ys) - 3, False)
first_preds = mj.min_jerk_lsq(ys[:-1], is_static, is_ang_big)#, forget_fac: float = 0.91):

#%%
non_nan_first_preds = first_preds[2:]
pred_diff = non_nan_first_preds - ys[-len(non_nan_first_preds):]
print("max first preds diff:", np.max(np.abs(pred_diff)))

# plt.plot(ts, ys)
# plt.plot(ts[-len(non_nan_first_preds):], non_nan_first_preds)
# plt.show()
#%%
prev_translations = y_concat[:-1]
translation_diffs = np.diff(y_concat, 1, axis=0)

deg1_vels = translation_diffs[:-1]

deg1_speeds_full = np.linalg.norm(deg1_vels, axis=-1, keepdims=True)
unit_vels_deg1 = pm.safelyNormalizeArray(deg1_vels, deg1_speeds_full)

t_diff_angs = pm.anglesBetweenVecs(
    unit_vels_deg1[:-1], unit_vels_deg1[1:], False
)
d_under_thresh = deg1_speeds_full < 0.00001
a_over_thresh = t_diff_angs > np.pi / 2


preds = mj.min_jerk_lsq(
    prev_translations, d_under_thresh.flatten(), a_over_thresh.flatten()
)#, forget_fac: float = 0.91):
#%%
plt.plot(all_ts, y_concat)
plt.plot(all_ts[-len(preds):], preds)
plt.show()
