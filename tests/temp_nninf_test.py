raise Exception("This test is currently broken. Need to update imports.")
import typing

import numpy as np
from numpy.typing import NDArray

import tensorflow as tf
import posemath as pm
from nn_utilities.nn_inference import PointsToInputsConstStep
from motiontools.posefeatures import HypotheticalInputsForNN

n1 = np.asarray([[0, 1, 2], [0, 0, 0], [0.33, 0.77, -100], [1, 1, 1], [1, 1, 1], [7, 8, 9]])
n2p = np.random.uniform(-1, 1, (7, ) + n1.shape)
pts = np.stack([n1, *n2p], axis=1)
aas = typing.cast(NDArray, np.random.uniform(-1, 1, pts.shape))

rmats = pm.matsFromScaledAxisAngleArray(aas)
print("rmats shape:", rmats.shape)
h = HypotheticalInputsForNN(pts[:5], rmats, 1, HypotheticalInputsForNN._generated_key_order())

print(h.last_nonzero_unit_vels)

vels = np.diff(pts, 1, axis=0)
unit_vels = pm.safelyNormalizeArray(
    vels, vec_for_zero_norms=np.zeros(3), propagation_axis=0
)


t_pts = tf.constant(pts.astype(np.float32))
t_aas = tf.constant(aas.astype(np.float32))
t_unit_vels = tf.constant(unit_vels.astype(np.float32))
print("pts shape:", t_pts.shape)
print("aas shape:", t_aas.shape)
print("unit_vels shape:", t_unit_vels.shape)

p = PointsToInputsConstStep(3)
res = p.updatePrecalcs(t_pts, t_aas)#, t_unit_vels[-2])
