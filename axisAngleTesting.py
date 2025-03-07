import numpy as np
from posemath import matFromAxisAngle, axisAngleFromMatArray
from posemath import matsFromScaledAxisAngleArray, matsFromAxisAngleArrays
import posemath as pm
import poseextrapolation as pex


#%%
numAAs = 4096
axisAngles = np.ones((numAAs, 3))
axisAngles[0] = np.ones(3)
distorts = np.random.sample((numAAs - 1, 3)) - 0.5
axisAngles[1:] = np.cumsum(distorts, axis=0)

#%%
mats = matsFromScaledAxisAngleArray(axisAngles)
    
newAAs = axisAngleFromMatArray(mats)

newMats = matsFromScaledAxisAngleArray(newAAs)


diffMats = newMats - mats
diffAAs = newAAs - axisAngles

matDiffMax = np.max(np.abs(diffMats))
aaDiffMax = np.max(np.linalg.norm(diffAAs, axis=-1))

print("Maxes:", matDiffMax, aaDiffMax)


#%%

axisAngles2 = 2 * np.pi * np.random.sample((numAAs, 3))

mats2 = np.ones((numAAs, 3,3))

diffMuls = np.empty((numAAs - 1, 3, 3))

for i in range(numAAs):
    mats2[i] = matFromAxisAngle(axisAngles2[i])
    if i > 0:
        diffMuls[i - 1] = mats2[i] @ mats2[i - 1].transpose()

axisNorms = np.linalg.norm(np.diff(mats2, axis=0).reshape((-1, 9)), axis=-1)

quats2 = pm.quatsFromAxisAngleVec3s(axisAngles2)

angles2 = pm.anglesBetweenQuats(quats2[:-1], quats2[1:])

reproducedNorms = 4 - 4*np.cos(angles2)

# Sum of the dot products of the axes of two frames should be 2*cos(angle) + 1.
dotSums = np.zeros(angles2.shape)
for i in range(3):
    dotSums += pm.einsumDot(mats2[:-1, :, i], mats2[1:, :, i])

diffFromExpected = dotSums - (2*np.cos(angles2) + 1)

#%%
fixed_axes = np.random.uniform(-1, 1, (numAAs, 3))
fixed_axes /= np.linalg.norm(fixed_axes, axis=-1, keepdims=True)

numNormsToTest = 100
possibleNorms = np.linspace(0, 2*np.pi, numNormsToTest)
investigation_data = np.empty((numAAs - 2, 3))
for i in range(numAAs - 2):
    rot_diff = mats[i+1] @ mats[i].transpose()
    # vel_axis = pm.axisAngleFromMatArray(rot_diff.reshape((-1,3,3)))
    # vel_axis /= np.linalg.norm(vel_axis)
    # vel_axis = vel_axis.flatten()
    prev = mats[i+1]
    target = mats[i+2]
    near_angle = pm.closestAnglesAboutAxis([prev], [target], fixed_axes[i:i+1])[0]#vel_axis)
    near = matFromAxisAngle(near_angle * fixed_axes[i]) @ prev
    between = near @ (target).transpose()
    vel_pred = rot_diff @ prev
    angle = np.arccos((np.trace(between) - 1)/2)
    orig_angle = np.arccos((np.trace(prev @ target.transpose()) - 1)/2)
    other_limit = np.arccos(np.dot(fixed_axes[i], mats2[i, :, 2]))
    rep_axes = np.repeat(fixed_axes[i:i+1], numNormsToTest, axis=0)
    rot_attempts = matsFromAxisAngleArrays(possibleNorms, rep_axes)
    mats_from_attempts = np.einsum('bij,jk->bik', rot_attempts, prev)
    diffs_from_attempts = np.einsum(
        'bij,jk->bik', mats_from_attempts, target.transpose()
    )
    attempt_trs = np.trace(diffs_from_attempts, axis1=-2, axis2=-1)
    norm_results = np.arccos((attempt_trs - 1.0)/2.0)
    
    investigation_data[i] = [angle, orig_angle, norm_results.min()] #vel_limit]

print("Any unexpecteds:", np.any(investigation_data[:, 0] > investigation_data[:, 1]))
print("Any unexpecteds(2):", np.any(investigation_data[:, 0] > investigation_data[:, 2]))
#%%

#%%

vecs = np.random.sample((numAAs, 3))
quats = pm.quatsFromAxisAngleVec3s(axisAngles)
quatRes = pm.rotateVecsByQuats(quats, vecs)
matRes = pm.einsumMatVecMul(mats, vecs)
print("Max diff for quat thing:", np.abs(matRes - quatRes).max())

#%% Testing circle code:
circ_angles = np.linspace(0, 2*np.pi, 100)
circ_radius = np.random.uniform(0.35, 3.35, 1)[0]
circ_translations = circ_radius * np.stack([
    np.cos(circ_angles), np.sin(circ_angles), np.zeros(len(circ_angles))
], axis = -1)
circ_diffs = np.diff(circ_translations, 1, axis=0)


cma = pex.CircularMotionAnalysis(circ_translations, circ_diffs)

circ_2d_pred_dot_self = pm.einsumDot(
    cma.vel_deg1_preds_2D, cma.vel_deg1_preds_2D
)

circ_centre_diff_dot = pm.einsumDot(
    cma.diffs_from_centres[1], cma.diffs_from_centres[1]
)

if np.abs(cma._sq_radii - circ_2d_pred_dot_self).max() > 0.0001:
    raise Exception("Prediction radii are mismatched!")
if np.abs(cma._sq_radii - circ_centre_diff_dot).max() > 0.0001:
    raise Exception("Circle centres yield mismatched radii!")

circ_pred_errs_d1 = circ_translations[3:] - cma.vel_deg1_preds_3D
circ_pred_err_dists_d1 = np.linalg.norm(circ_pred_errs_d1, axis=-1)
print("Max circ vel pred err:", np.max(circ_pred_err_dists_d1))

circ_noise_shape = (circ_translations[3:-1].shape[0], 1)
circ_noise = np.random.uniform(-0.1, 0.1, circ_noise_shape)
circ_noise_mags = np.linalg.norm(circ_noise, axis=-1, keepdims=True)

circ_noise_added = circ_translations[3:-1].copy()
circ_noise_added[:, 2:] += circ_noise
circ_noise_mag_mean = circ_noise_mags.mean()
circ_noise_thresh = circ_noise_mag_mean / circ_radius
circ_dist_res = cma.isMotionStillCircular(circ_noise_added, circ_noise_thresh)
circ_noise_over = circ_noise_mags.flatten() > circ_noise_mag_mean
c_gatekeep_agree = (circ_noise_over == circ_dist_res.non_circ_bool_inds)
print("circ gatekeeping works:", np.all(c_gatekeep_agree))
dists_close = np.allclose(circ_dist_res.dists, circ_noise_mags.flatten())
print("circ dists match:", dists_close)
print("circ ratios match:", np.allclose(
    circ_dist_res.dist_radius_ratios, circ_noise_mags.flatten() / circ_radius
))

#%%



def randomUnitAxis():
    unscaled = np.random.uniform(-1.0, 1.0, 3)
    return unscaled / np.linalg.norm(unscaled)

def randomSmallQuaternion(maxAngle):
    half_angle = 0.5 * maxAngle * np.random.sample(1)[0]
    axis = np.sin(half_angle) * randomUnitAxis()
    return np.array([np.cos(half_angle), axis[0], axis[1], axis[2]])

small_angle_max = (np.pi / 2.0) - 0.001
rand_cam_rot = randomSmallQuaternion(small_angle_max) # Actually the transpose.
rand_obj_rot = randomSmallQuaternion(small_angle_max)
rand_initial_camobj = randomSmallQuaternion(small_angle_max)
numCamObjTests = 100
camObjframes = np.empty((numCamObjTests, 4))
camObjframes[0] = rand_initial_camobj

camTotal = np.array([1.0, 0.0, 0.0, 0.0])
objTotal = rand_initial_camobj.copy()
for i in range(1, numCamObjTests):
    camTotal = pm.multiplyLoneQuats(rand_cam_rot, camTotal)
    objTotal = pm.multiplyLoneQuats(rand_obj_rot, objTotal)
    camObjframes[i] = pm.multiplyLoneQuats(camTotal, objTotal)
#%%
cs, ms, camObjPreds = pex.camObjConstAngularVelPreds(camObjframes[:-1])


print("camobjmaxerr:", np.abs(camObjPreds[1:] - camObjframes[4:]).max())
print("Ct diffs:", np.abs(cs[1:] - rand_cam_rot).max())

#%%
def randomQuats(num):
    return pm.normalizeAll(np.random.uniform(-1, 1, (num, 4)))

qs_to_reconstruct = randomQuats(100)
q_to_ax, q_to_ang = pm.axisAnglesFromQuats(qs_to_reconstruct)
aas_to_qs_test = pm.quatsFromAxisAngles(q_to_ax, q_to_ang)
aa_quat_diffs = np.abs(np.stack([
    aas_to_qs_test - qs_to_reconstruct, -aas_to_qs_test - qs_to_reconstruct
], axis = 0))
quat_diffs_per_q = np.sum(aa_quat_diffs, axis = -1)
if np.max(np.min(quat_diffs_per_q), axis = 0) > 0.0001:
    raise Exception("Quaternion reconstruction failed!")

#%% Testing the Runge-Kutta quaternion integration.

num_const_a_vals = 360
start_ang_vel = np.random.uniform(-0.1, 0.1, 3)
start_ang_vel_angle = np.linalg.norm(start_ang_vel)
const_a_ax = start_ang_vel / start_ang_vel_angle
const_a = np.random.sample(1)[0] * 0.0015 * const_a_ax
const_a_angle = np.linalg.norm(const_a)
const_a_times = np.arange(num_const_a_vals).reshape(-1, 1)[::3]
const_a_ang_vel_deltas = const_a * const_a_times
const_a_ang_vels = start_ang_vel + const_a_ang_vel_deltas

const_a_v_deltas = start_ang_vel_angle * const_a_times
const_a_a_deltas = 0.5 * const_a_angle * const_a_times**2
const_a_disp_angles = start_ang_vel_angle + const_a_v_deltas + const_a_a_deltas 

half_const_a_angs = const_a_disp_angles.reshape(-1, 1) / 2
const_a_delta_qs = np.hstack((
    np.cos(half_const_a_angs), const_a_ax * np.sin(half_const_a_angs)
))

start_quat = randomQuats(1)[0]
gt_const_a_qs = pm.multiplyQuatLists(const_a_delta_qs, start_quat)

rk_detail = 1001
interp_rk_vels = np.linspace(const_a_ang_vels[:-1], const_a_ang_vels[1:], rk_detail, axis=1)

const_a_angles = pm.anglesBetweenQuats(gt_const_a_qs[1:], gt_const_a_qs[:-1]).flatten()

const_a_q_diffs = pm.multiplyQuatLists(
    gt_const_a_qs[1:], pm.conjugateQuats( gt_const_a_qs[:-1])
)
fixed_axes = const_a_q_diffs[:, 1:] / np.linalg.norm(const_a_q_diffs[:, 1:], axis=-1, keepdims=True)# np.sin(angles/2)[..., np.newaxis]
ang_vel_vecs = pm.scalarsVecsMul(const_a_angles, fixed_axes)
ang_acc_vecs = np.diff(ang_vel_vecs, 1, axis=0)
ang_vel_vecs[1:] += 0.5 * ang_acc_vecs
ang_vel_vecs[0] = ang_vel_vecs[1] - ang_acc_vecs[0]
extrap_ang_vel_vecs = ang_vel_vecs[1:] + ang_acc_vecs
interp_ang_vels = np.linspace(ang_vel_vecs[1:], extrap_ang_vel_vecs, rk_detail, axis=1)
   

# pred_const_a_qs = pm.integrateAngularVelocityRK(interp_rk_vels, gt_const_a_qs[:-1], 4)[2:]
pred_const_a_qs = pm.integrateAngularVelocityRK(interp_ang_vels[:-1], gt_const_a_qs[2:-1], 1)

const_a_pred_errs = np.abs(pred_const_a_qs - gt_const_a_qs[3:])
print("Max RK diff:", const_a_pred_errs.max())
const_a_pred_angerrs = pm.anglesBetweenQuats(pred_const_a_qs, gt_const_a_qs[3:])
print("Avg RK angle diff:", const_a_pred_angerrs.mean())
