# Testing to make sure changes to JAV calculations didn't introduce bugs. 
import numpy as np

from gtCommon import SyntheticPoseLoader
from motiontools.posefeatures import JAV, dataForCombosJAV

# Non-tensorflow version of loss function used in other files but that that
# relies on importing tensorflow.
def np_poseLossJAV(y_true, y_pred):

    pred_disp_0 = y_true[:, 0] * y_pred[:, 0] + y_true[:, 1] * y_pred[:, 1] \
        + y_true[:, 3] * y_pred[:, 3] + y_true[:, 6] * y_pred[:, 6] \
        + y_true[:, 9] * y_pred[:, 9]
    pred_disp_1 = y_true[:, 2] * y_pred[:, 2] + y_true[:, 4] * y_pred[:, 4] \
        + y_true[:, 7] * y_pred[:, 7] + y_true[:, 10] * y_pred[:, 10]
    pred_disp_2 = y_true[:, 5] * y_pred[:, 5] + y_true[:, 8] * y_pred[:, 8] \
        + y_true[:, 11] * y_pred[:, 11]

    pred_disp = np.stack([pred_disp_0, pred_disp_1, pred_disp_2], axis=-1)
    
    true_disp = y_true[:, 12:15] #6:9]

    err_vec3 = true_disp - pred_disp
    return np.linalg.norm(err_vec3, axis=-1)

pl = SyntheticPoseLoader(35, False, False, 3)
res = dataForCombosJAV([pl], (JAV.VELOCITY, JAV.ACCELERATION, JAV.JERK))
print([np.round(x[:33], 2) for x in res[0].values()])

skip = 0
jav_truth = tuple(res[skip].values())[0]

# Constant-jerk if velocity "already" took more-than-2-pt-Lagrange-interpolation
# into account, likewise for acceleration, etc.:
muls = np.array([[1, 1/2, 1/2, 1/6, 1/6, 1/6, 0, 0, 0, 0, 0, 0]])
# Constant jerk if only last 2 points were used to calculate velocity, last 3
# for acceleration, etc.:
muls = np.array([[1, 1, 1, 1, 1, 1., 0, 0, 0, 0, 0, 0]])
muls = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) # Constant-velocity.
# Replicating constant-velocity if all derivatives were calculated using 
# "6-point" Lagrange interpolation:
muls = np.array([[1.0, -0.5, -0.5, +1/6, +1/6, +1/6, -1/24, -1/24, -1/24, 1/120, 1/120, 1/120]])
# Constant acceleration if only last 2 points were used to calculate velocity:
muls = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# Replicating constant-acceleration of all derivatives were calculated using
# "6-point" Lagrange interpolation:
muls = np.array([[1.0, +0.5, +0.5, -5/6, -5/6, -5/6, +13/24, 13/24, 13/24, -29/120, -29/120, -29/120]])

loss = np_poseLossJAV(jav_truth, muls)
print(np.round(100 * loss, 2))

