import numpy as np
from bayes_opt import BayesianOptimization

from gtCommon import PoseLoaderBCOT
from cinpact import CinpactAccelExtrapolater
import gtCommon as gtc

import matplotlib.pyplot as plt

SKIP_AMT = 2

combos = []
for b in range(len(gtc.BCOT_BODY_NAMES)):
    for s in range(len(gtc.BCOT_SEQ_NAMES)):
        if PoseLoaderBCOT.isBodySeqPairValid(b, s):
            combos.append((b,s))

combo_translations = dict()
for combo in combos:
    calculator = PoseLoaderBCOT(combo[0], combo[1])
    translations = calculator.getTranslationsGTNP()[::(SKIP_AMT + 1)]
    combo_translations[combo] = translations

def getCinpactScore(supportRad: float, k: float):
    extrapolater = CinpactAccelExtrapolater(supportRad, k)
    
    all_err_means = []
    for combo in combos:
        translations = combo_translations[combo]
        t_spline_preds = extrapolater.apply(translations[:-1])
        t_errs = translations[1:] - t_spline_preds
        t_err_norms = np.linalg.norm(t_errs, axis = -1)
        mean_t_err_norm = t_err_norms.mean()
        all_err_means.append(mean_t_err_norm)

    # Negating so that higher values are better.
    return -np.mean(all_err_means)

pbounds = {'supportRad': (2.0001, 3.50), 'k': (0.0001, 5.00)}

#%% Graph of some sample values
graphScores = np.empty((10, 10))
graph_srs = np.linspace(*pbounds['supportRad'], len(graphScores))
graph_ks = np.linspace(*pbounds['k'], graphScores.shape[1])
for i, sr in enumerate(graph_srs):
    for j, kv in enumerate(graph_ks):
        graphScores[i, j] = getCinpactScore(sr, kv)
graphX, graphY = np.meshgrid(graph_srs, graph_ks)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(graphX, graphY, graphScores)

ax.set_xlabel('c')
ax.set_ylabel('k')
ax.set_zlabel('Score')

plt.show()

#%% The actual Bayesian Optimization

optimizer = BayesianOptimization(
    f=getCinpactScore,
    pbounds=pbounds,
    random_state=1,
)

# print(getCinpactScore(10, 10))

optimizer.maximize(
    init_points=10,
    n_iter=38,
)

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print("-----------------------------------------------------------------------")
print(optimizer.max)
