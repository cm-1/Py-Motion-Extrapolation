import numpy as np
from gtCommon import matFromAxisAngle, axisAngleFromMatArray



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
