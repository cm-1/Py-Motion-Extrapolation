#%%
from dataclasses import dataclass
import typing

import numpy as np
from bayes_opt import BayesianOptimization

import matplotlib.pyplot as plt

import posemath as pm
from gtCommon import BCOT_Data_Calculator
import gtCommon as gtc

@dataclass
class RotationInfo:
    gt_quats: np.ndarray
    fixed_axes: np.ndarray
    angles: np.ndarray

SKIP_AMT = 2

def algebraicSigmoid(x: np.ndarray, k: float, scale: float):
    denom = (1.0 + np.abs(x/scale)**k)**(1.0/k)
    return x / denom

def generalizedLogistic(x: np.ndarray, alpha: float, beta: float):
    return (1 + np.exp(-beta*x))**(-alpha) - 0.5

def angWahbaFunc(x: np.ndarray, scale: float):
    numerator = 2.0 * np.sin(x/scale)
    denominator = 2.0 * np.cos(x/scale) - 1.0
    return scale*np.arctan2(numerator, denominator) - x

combos = []
for b in range(len(gtc.BCOT_BODY_NAMES)):
    for s in range(len(gtc.BCOT_SEQ_NAMES)):
        if BCOT_Data_Calculator.isBodySeqPairValid(b, s):
            combos.append((b,s))



combo_rotations: typing.Dict[typing.Tuple[int, int], RotationInfo] = dict()
for combo in combos:
    calculator = BCOT_Data_Calculator(combo[0], combo[1], SKIP_AMT)
    aa_rotations = calculator.getRotationsGTNP(True)
    qs = pm.quatsFromAxisAngleVec3s(aa_rotations)
    quatDiffs = pm.multiplyQuatLists(qs[1:], pm.conjugateQuats(qs[:-1]))
    axes, angles = pm.axisAnglesFromQuats(quatDiffs)
    combo_rotations[combo] = RotationInfo(qs, axes, angles)



def getSigmoidScore(scale: float, k: float, use_median: bool = False):    
    all_err_means = []
    for combo in combos:
        rotInfo = combo_rotations[combo]
        
        pred_angs = algebraicSigmoid(rotInfo.angles[:-1], k, scale)
        pred_diffs = pm.quatsFromAxisAngles(rotInfo.fixed_axes[:-1], pred_angs)
        preds = pm.multiplyQuatLists(pred_diffs, rotInfo.gt_quats[1:-1])

        r_err_angs = pm.anglesBetweenQuats(rotInfo.gt_quats[2:], preds)

        if use_median:
            all_err_means.append(r_err_angs)
        else:
            r_err_sum = r_err_angs.sum()

            # I'm guessing the fastest way to update the score to match the one
            # in my other files, which includes the prediction for the first 
            # non-init pose as just copying the init pose, would be to avoid 
            # numpy appends, slices, etc. and just do this:
            r_err_sum += rotInfo.angles[0]
            mean_r_err_norm = r_err_sum / (len(r_err_angs) + 1)

            all_err_means.append(mean_r_err_norm)

    # Negating so that higher values are better.
    ret_val = 0
    if not use_median:
        ret_val = -np.mean(all_err_means)
    else:
        ret_val = -np.median(np.concat(all_err_means))
    return ret_val

def getLinScore(scale: float, use_median: bool = False):    
    all_err_means = []
    for combo in combos:
        rotInfo = combo_rotations[combo]
        
        pred_angs = scale * rotInfo.angles[:-1]
        pred_diffs = pm.quatsFromAxisAngles(rotInfo.fixed_axes[:-1], pred_angs)
        preds = pm.multiplyQuatLists(pred_diffs, rotInfo.gt_quats[1:-1])

        r_err_angs = pm.anglesBetweenQuats(rotInfo.gt_quats[2:], preds)

        if use_median:
            all_err_means.append(r_err_angs)
        else:
            r_err_sum = r_err_angs.sum()

            # I'm guessing the fastest way to update the score to match the one
            # in my other files, which includes the prediction for the first 
            # non-init pose as just copying the init pose, would be to avoid 
            # numpy appends, slices, etc. and just do this:
            r_err_sum += rotInfo.angles[0]
            mean_r_err_norm = r_err_sum / (len(r_err_angs) + 1)

            all_err_means.append(mean_r_err_norm)
    # Negating so that higher values are better.
    ret_val = 0
    if not use_median:
        ret_val = -np.mean(all_err_means)
    else:
        concat = np.concat(all_err_means)
        ret_val = -np.median(concat.flatten())
    return ret_val

def getWahbaScore(scale: float, use_median: bool = False):    
    all_err_means = []
    for combo in combos:
        rotInfo = combo_rotations[combo]
        
        pred_angs = angWahbaFunc(rotInfo.angles[:-1], scale)
        pred_diffs = pm.quatsFromAxisAngles(rotInfo.fixed_axes[:-1], pred_angs)
        preds = pm.multiplyQuatLists(pred_diffs, rotInfo.gt_quats[1:-1])

        r_err_angs = pm.anglesBetweenQuats(rotInfo.gt_quats[2:], preds)

        if use_median:
            all_err_means.append(r_err_angs)
        else:
            r_err_sum = r_err_angs.sum()

            # I'm guessing the fastest way to update the score to match the one
            # in my other files, which includes the prediction for the first 
            # non-init pose as just copying the init pose, would be to avoid 
            # numpy appends, slices, etc. and just do this:
            r_err_sum += rotInfo.angles[0]
            mean_r_err_norm = r_err_sum / (len(r_err_angs) + 1)

            all_err_means.append(mean_r_err_norm)

    # Negating so that higher values are better.
    ret_val = 0
    if not use_median:
        ret_val = -np.mean(all_err_means)
    else:
        concat = np.concat(all_err_means)
        ret_val = -np.median(concat.flatten())
    return ret_val

#%%
best_lin = getLinScore(0.8525)
print("Best lin score:", best_lin)
#%% Graph of some sample values
pbounds = {'scale': (0.4, np.pi), 'k': (0.5, 5)}


graphScores = np.empty((33, 35))
graph_srs = np.linspace(*pbounds['scale'], len(graphScores))
graph_ks = np.linspace(*pbounds['k'], graphScores.shape[1])
for i, sr in enumerate(graph_srs):
    for j, kv in enumerate(graph_ks):
        graphScores[i, j] = getSigmoidScore(sr, kv)
#%%
graphX, graphY = np.meshgrid(graph_srs, graph_ks)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(graphX, graphY, graphScores.transpose())

ax.set_xlabel('scale')
ax.set_ylabel('k')
ax.set_zlabel('Score')

gmax_ind = np.unravel_index(np.argmax(graphScores), graphScores.shape)
gmax = graphScores[gmax_ind[0], gmax_ind[1]]
gmax_scale = graph_srs[gmax_ind[0]]
gmax_k = graph_ks[gmax_ind[1]]
ax.scatter(gmax_scale, gmax_k, gmax, color='red')
print("gmax={}, k={}, scale={}".format(gmax, gmax_k, gmax_scale))
plt.show()

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
pilin = np.linspace(-np.pi, np.pi, 500)
ax.plot(pilin, algebraicSigmoid(pilin, 3, 3))
plt.show()

#%%
slopes = np.linspace(0, 1, 500)
slope_scores = [getLinScore(slope, True) for slope in slopes]
#%%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(slopes, slope_scores)
plt.show()

#%%
wscales = np.linspace(0.0001, 5, 100)
wscale_scores = [getWahbaScore(wscale) for wscale in wscales]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(wscales, wscale_scores)
plt.show()

#%% The actual Bayesian Optimization
pbounds = {'scale': (0.4, 1.5), 'k': (0.5, 50)}

optimizer = BayesianOptimization(
    f=getSigmoidScore,
    pbounds=pbounds,
    random_state=1,
)

# print(getCinpactScore(10, 10))

optimizer.maximize(
    init_points=33,
    n_iter=100,
)

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print("-----------------------------------------------------------------------")
print(optimizer.max)
