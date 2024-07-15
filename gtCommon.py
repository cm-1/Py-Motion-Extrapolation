import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import math
import pathlib
import re

import cinpact

# Makes rotation matrix from an axis (with angle being encoded in axis length).
# Uses common formula that you can google if need-be.
def matFromAxisAngle(scaledAxis):
    angle = np.linalg.norm(scaledAxis)
    unitAxis = scaledAxis / angle
    x, y, z = unitAxis.flatten()
    skewed = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])
    return np.identity(3) + np.sin(angle) * skewed + (1.0 - np.cos(angle)) * (skewed @ skewed)

# 1st return value: rotation angle
# 2nd return value: axis pre-scaled by rotation angle (so NOT unit axis!)
def axisAngleFromMat(matrix, lastAngle, lastAxis):
    # Vec3 representing the direction of our rotation axis. Not yet unit length.
    nonUnitAxisDir = np.array([
        matrix[2,1] - matrix[1,2],
        matrix[0,2] - matrix[2,0],
        matrix[1,0] - matrix[0,1]
    ])
    
    # The angle of rotation about the above axis direction.
    # Outputs of acos are constrained to [0, pi], which will impact later code. 
    angle = math.acos((matrix.trace() - 1)/2.0)

    # TL;DR: Detect axis flips and increment/decrement angle to prevent jumps.
    # Can skip longer explanation below if you want or need to.
    # ------------------------------------------------------------------------
    # Technically, we can now just return `angle/(2*sin(angle)) * axis``, and
    # then we have an **accurate** angle-axis result stored as a vec3.
    # **However**, because acos only returns values in [0, pi], rotations above
    # pi about an axis will be represented by rotations under pi about the
    # flipped version of the axis. That means that, on frame "n", you have a
    # rotation of (pi - epsilon) about axis "v", and then on frame "n+1",
    # instead of a rotation of (pi + epsilon) about axis "v", you get back a
    # rotation of (pi - epsilon) about axis "-v".
    # Technically correct, but very jumpy....
    # So, we perform the following steps:
    #  1. Detect if axis flipped, via dot product.
    #  2. If axis was flipped, negate angle to maintain accuracy.
    #     We also need to flip the axis, but we'll do it later, at the same time
    #     that we scale it, to slightly save on operations.
    #  3. Add necessary multiples of 2pi to angle to prevent large angle jump.
    #     (May be necessary even if axis did not flip on this frame!)
    #     (E.g., corrections to previous frames could lead to lastAngle > 2pi)
    # ------------------------------------------------------------------------
    # TODO: Calculating (via floor/ceil/mult/div) number of 2pi incs needed
    # may be faster than while loop? A worst vs. avg case trade-off, I guess.
    # Is there a way where, e.g., an object on a carousel won't just have the
    # angle accumulate to be higher and higher? I guess you could keep the angle
    # uncorrected/unnegated and within [0,pi], just flip the axis if need-be,
    # and store flip sign in some 4th component scaled by sin(angle) so as to be
    # continuous... which may be just a worse version of quaternions.
    if np.dot(lastAxis, nonUnitAxisDir) < 0.0:
        angle = -angle
    while lastAngle - angle > math.pi:
        angle += 2.0 * math.pi
    while angle - lastAngle > math.pi:
        angle -= 2.0 * math.pi
        
    # (!!!) PLEASE READ THIS COMMENT BEFORE MOVING/EDITING THIS LINE!
    # Our final vec3 output will be `angle * unitAxis`,
    # where `unitAxis = nonUnitAxis/2sin(angle)`.
    # Without flip correction, angle is in [0, pi] and so sin(angle) >= 0; thus,
    # if we say:
    # `originalAxis = nonUnitAxis/2sin(originalAngle)`
    # and if we do a flip correction, our new axis will have to be:
    # `newAxis = -originalAxis`
    #        ` = nonUnitAxis/2sin(-originalAngle) = nonUnitAxis/2sin(newAngle)`
    # What this means is that, if we calculate the sin() scaling after modifying
    # the angle AND we use this scaling to flip the axis dir if need-be, then
    # we save on a few extra calculations.
    nonUnitAxisLen = 2.0 * math.sin(angle)
    
    return angle, (angle / nonUnitAxisLen) * nonUnitAxisDir



def axisAngleListFromMats(matList):
    # differences = []
    lastAngle = 0.0
    lastDir = np.array([0.0, 0.0, 0.0])
    retList = []
    for m in matList:
        lastAngle, val = axisAngleFromMat(m, lastAngle, lastDir)
        lastDir = val
        retList.append(val)
        # recovered = matFromAxisAngle(val)
        # differences.append(np.linalg.norm(recovered - m))
    # print("Max difference:", np.max(np.array(differences)))
    
    return np.array(retList)
