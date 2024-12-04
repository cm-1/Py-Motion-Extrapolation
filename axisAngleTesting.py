import numpy as np
from gtCommon import matFromAxisAngle, axisAngleFromMatArray
import gtCommon as gtc


#%%
numAAs = 4096
axisAngles = np.ones((numAAs, 3))
axisAngles[0] = np.ones(3)
distorts = np.random.sample((numAAs - 1, 3)) - 0.5
axisAngles[1:] = np.cumsum(distorts, axis=0)

#%%
mats = np.ones((numAAs, 3,3))

for i in range(numAAs):
    mats[i] = matFromAxisAngle(axisAngles[i])
    
newAAs = axisAngleFromMatArray(mats)

newMats = np.ones((numAAs, 3,3))

for i in range(numAAs):
    newMats[i] = matFromAxisAngle(newAAs[i])


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

quats2 = gtc.quatsFromAxisAngles(axisAngles2)

angles2 = gtc.anglesBetweenQuats(quats2[:-1], quats2[1:])

reproducedNorms = 4 - 4*np.cos(angles2)

# Sum of the dot products of the axes of two frames should be 2*cos(angle) + 1.
dotSums = np.zeros(angles2.shape)
for i in range(3):
    dotSums += gtc.einsumDot(mats2[:-1, :, i], mats2[1:, :, i])

diffFromExpected = dotSums - (2*np.cos(angles2) + 1)

#%%
fixed_axes = np.random.uniform(-1, 1, (numAAs, 3))
fixed_axes /= np.linalg.norm(fixed_axes, axis=-1, keepdims=True)

numNormsToTest = 100
possibleNorms = np.linspace(0, 2*np.pi, numNormsToTest)
investigation_data = np.empty((numAAs - 2, 3))
for i in range(numAAs - 2):
    rot_diff = mats[i+1] @ mats[i].transpose()
    # vel_axis = gtc.axisAngleFromMatArray(rot_diff.reshape((-1,3,3)))
    # vel_axis /= np.linalg.norm(vel_axis)
    # vel_axis = vel_axis.flatten()
    prev = mats[i+1]
    target = mats[i+2]
    near = gtc.closestFrameAboutAxis(prev, target, fixed_axes[i])#vel_axis)
    between = near @ (target).transpose()
    vel_pred = rot_diff @ prev
    angle = np.arccos((np.trace(between) - 1)/2)
    orig_angle = np.arccos((np.trace(prev @ target.transpose()) - 1)/2)
    other_limit = np.arccos(np.dot(fixed_axes[i], mats2[i, :, 2]))
    norm_results = np.empty(numNormsToTest)
    for j, n in enumerate(possibleNorms):

        rot_attempt = gtc.matFromAxisAngle(n * fixed_axes[i])
        mat_from_attempt = rot_attempt @ prev
        norm_results[j] = np.arccos((np.trace(mat_from_attempt @ target.transpose()) - 1)/2)


    
    investigation_data[i] = [angle, orig_angle, norm_results.min()] #vel_limit]

print("Any unexpecteds:", np.any(investigation_data[:, 0] > investigation_data[:, 1]))
print("Any unexpecteds(2):", np.any(investigation_data[:, 0] > investigation_data[:, 2]))
#%%

#%%

vecs = np.random.sample((numAAs, 3))
quats = gtc.quatsFromAxisAngles(axisAngles)
quatRes = gtc.rotateVecsByQuats(quats, vecs)
matRes = gtc.einsumMatVecMul(mats, vecs)
print("Max diff for quat thing:", np.abs(matRes - quatRes).max())
