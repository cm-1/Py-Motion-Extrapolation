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
circ_translations = np.stack([
    np.cos(circ_angles), np.sin(circ_angles), np.zeros(len(circ_angles))
], axis = -1)
circ_diffs = np.diff(circ_translations, 1, axis=0)
circle_plane_info = pm.getPlaneInfo(
    circ_diffs[1:-1], -circ_diffs[:-2], circ_translations[:-3]
)
circle_axes = circle_plane_info.plane_axes
circle_pts_2D = []
for j in range(3):
    circle_pts_2D.append(pm.vecsTo2D(
        circ_translations[j:(-3 + j)], *circle_axes
    ))
circle_centres = pm.circleCentres2D(*circle_pts_2D)

diffs_from_centres = []
for j in range(3):
    diffs_from_centres.append(circle_pts_2D[j] - circle_centres)

sq_radii = pm.einsumDot(diffs_from_centres[0], diffs_from_centres[0])
c_cosines_0 = pm.einsumDot(diffs_from_centres[1], diffs_from_centres[0])/sq_radii
c_cosines_1 = pm.einsumDot(diffs_from_centres[2], diffs_from_centres[1])/sq_radii

c_angles_1 = np.arccos(c_cosines_1)
c_sines_1 = np.sin(c_angles_1)

c_trans_preds_2D = pm.rotateBySinCos2D(
    diffs_from_centres[2], c_cosines_1, c_sines_1
)

c_trans_preds = pm.vecsTo3DUsingPlaneInfo(
    circle_centres + c_trans_preds_2D, circle_plane_info
)

if np.abs(sq_radii - pm.einsumDot(c_trans_preds_2D, c_trans_preds_2D)).max() > 0.0001:
    raise Exception("Prediction radii are mismatched!")
if np.abs(sq_radii - pm.einsumDot(diffs_from_centres[1], diffs_from_centres[1])).max() > 0.0001:
    raise Exception("Circle centres yield mismatched radii!")


c_cosines_2 = pm.einsumDot(c_trans_preds_2D, diffs_from_centres[2])

#%%

def randomRotationMat():
    scaled_axis = np.random.uniform(-np.pi, np.pi, 3)
    return pm.matFromAxisAngle(scaled_axis)

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
