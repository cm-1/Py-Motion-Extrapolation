# %%
import numpy as np

from gtCommon import *
import posemath as pm

pl = PoseLoaderTUDL(False, 3, -1)

ts = pl.getTranslationsGTNP()

rs = pl.getRotationsGTNP()
qs = pm.quatsFromAxisAngleVec3s(rs)

qds = pm.anglesBetweenQuats(qs[1:], qs[:-1])
tds = np.linalg.norm(np.diff(ts, axis=0), axis=-1)

qm = np.mean(qds)
tm = np.mean(tds)

qv = np.std(qds)
tv = np.std(tds)

qbig = np.abs(qds - qm) > (5 * qv)
tbig = np.abs(tds - tm) > (2 * tv)

big = np.logical_or(tbig, qbig)
big_ws = [int(i + 1) for i in np.where(big)[0]]

z = zip(
    big_ws,
    [float(np.round(f, 2)) for f in tds[big]],
    [float(np.round(f,2)) for f in qds[big]]
)

print("ind\tt\tang")
for tup in z:
    print(*tup, sep='\t')
print("Num skips:", len(big_ws))
print()
print(tuple(big_ws))

print()
print("Seconds:", np.round(np.asarray(big_ws) / 30.0, 2))