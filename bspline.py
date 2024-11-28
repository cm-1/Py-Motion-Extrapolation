
import numpy as np
# Class for returning curve points and related info (like radii)
class CurvePoints:
    def __init__(self, params, points, radii):
        self.params = params
        self.points = points
        self.radii = radii
    
    def reverseSelf(self):
        self.params = self.params[::-1]
        self.points = self.points[::-1]
        self.radii = self.radii[::-1]

class MotionBSpline:
    # controlPoints is a list of MotionPoint objects.
    def __init__(self, controlPoints, order: int, standardKnots: bool):
        self.order_u = order
        self.points = controlPoints
        self.use_endpoint_u = standardKnots
        
class MotionPoint:
    def __init__(self, coord):
        self.co = coord
        self.radius = 1
    
def bSplineSubdivCtrlPts(originalCtrlPts, order, isClosed, numSubdivs):
    newPts = np.array(originalCtrlPts)
    for _ in range(numSubdivs):
        newPts = _bSplineSubdivCtrlPtsOnce(newPts, order, isClosed)
    return newPts



# Here, we assume mask is flat/1D, but "pts" could have any shape so long as
# len(pts) (i.e., pts.shape[0]) equals len(mask).
# Going by the documentation for np.dot(a, b), if b is 1D (like our mask), then
# np.dot() will do a weighted sum along the last axis of a. So, to have that
# sum be along the FIRST axis instead, we transpose a before dotting, and then
# to get the desired output shape, we transpose again at the end.
def applyMaskToPts(pts, mask):
    return np.dot(pts.transpose(), mask).transpose()

def _bSplineSubdivCtrlPtsOnce(originalCtrlPts, order, isClosed):
    # Filters for orders 2-5
    regMasksDict = {
        2: ([1], [1/2, 1/2]),
        3: ([3/4, 1/4], [1/4, 3/4]),
        4: ([1/2, 1/2], [1/8, 3/4, 1/8]),
        5: ([5/8, 5/16, 1/16], [1/16, 5/16, 5/8])
    }

    startMasksDict = {
        2: (),
        3: regMasksDict[2],
        4: tuple(list(regMasksDict[2]) + [[0, 3/4, 1/4], [0, 3/16, 11/16, 1/8]]),
    }
    startMasksDict[5] = tuple(list(startMasksDict[4]) + [[0, 0, 5/12, 25/48, 1/16], [0, 0, 1/12, 29/48, 5/16]])

    startMasks = startMasksDict.get(order)

    if startMasks is None:
        raise Exception("Order of B-Spline for subdivision must be 2-5, not {}.".format(order))
    

    regMask0 = np.array(regMasksDict[order][0])
    regMask1 = np.array(regMasksDict[order][1])
    regMask0Len = len(regMask0)
    regMask1Len = len(regMask1)

    startPts = []
    endPts = []
    startInputIndForRegCalcs = 0
    lastInputIndForRegCalcs = 0
    # numOutPts = 0
    # numRegPts = 0

    maxMaskLen = (order >> 1) + 1 # I.e., floor(k/2) + 1
    if isClosed:
        startPts = []
        # numOutPts = 2 * len(originalCtrlPts)
        startInputIndForRegCalcs = 0
        lastInputIndForRegCalcs = len(originalCtrlPts) - maxMaskLen
        # numRegPts = 2 * (lastInputIndForRegCalcs + 1)

        # And then for the points where the indexing "wraps around" and you use
        # modulo operations, we'll store those in "endPts".
        #
        # endPts = ...
        raise NotImplementedError("Did not finish implementing B-Spline Subdivision for closed curves!")
    else:
        # Can derive this by looking at subdivision matrices for k=2, k=3, k=4,
        # etc. (here, I use "k" to represent "order") and reasoning out how many
        # columns of "regular"/"inner" filters there will be, reasoning how many
        # rows there will be for a single inner col, noting each additional
        # inner col adds 2 rows, and then simplifying.
        # Result: 2 * len(input) - (k-1)
        # CURRENTLY, IT'S UNUSED... but I'm keeping it here in case in the future
        # it can be used, e.g., to make calculations more efficient or something.
        #
        # numOutPts = 2 * len(originalCtrlPts) - (order - 1)
        
        startPts = [
            applyMaskToPts(originalCtrlPts[:len(m)], np.array(m)) for m in startMasks
        ]
        endPts = [
            applyMaskToPts(originalCtrlPts[-len(m):], np.array(m)[::-1]) for m in startMasks[::-1]
        ]
        # Similarly, we can observe the pattern for the first input point on which
        # we can start using the "regular" masks on.
        startInputIndForRegCalcs = order - 2
        # We use symmetry to find the stopping index. This will be "inclusive" here.
        lastInputIndForRegCalcs = len(originalCtrlPts) - (order - 2) - maxMaskLen

        # If the order is even, then we need to apply one (but not both) of the
        # regular masks a final time.
        if not (order & 1):
            lastRegMaskStartInd = lastInputIndForRegCalcs + 1
            lastRegMaskEndInd = lastRegMaskStartInd + len(regMask0)
            lastRegInputs = originalCtrlPts[lastRegMaskStartInd:lastRegMaskEndInd]
            endPts = [applyMaskToPts(lastRegInputs, regMask0)] + endPts
    
    # Now that we've handled "endpoints", we can handle the interior indices
    regPts = []

    # Because we made the upper index inclusive, we need to "+1" for it.
    for i in range(startInputIndForRegCalcs, lastInputIndForRegCalcs + 1):
        regPts.append(applyMaskToPts(originalCtrlPts[i:i+regMask0Len], regMask0))
        regPts.append(applyMaskToPts(originalCtrlPts[i:i+regMask1Len], regMask1))
        
    return np.array(startPts + regPts + endPts)



# u is a float indicating how far along the curve we are
# k is an int and the order of the curve
# delta is an int and ...
# eList ("elementList") is the elements (any type) we are using as control points
# uList is a list of floats making up the knot vector
def bSplineInner(u, k, delta, eList, uList):
    numPts = len(uList) - k
    c = [0 for _ in range(k)]
    for index in range(k):
        if (delta - index < numPts):
            # Choose the eList element if index is in range.
            c[index] = eList[delta - index]
        else:
            zeroElement = 0 # Otherwise, set to 0
            if isinstance(eList[0], np.ndarray):
                zeroElement = np.zeros(eList[0].shape)
            c[index] = zeroElement
    for r in range(k, 1, -1):
        i = delta
        for s in range(r - 1):
            w = 0.0
            # If we'd divide by 0, or exceed range of uList, keep w as 0 instead.
            if (i + r - 1 < len(uList) and uList[i + r - 1] - uList[i] != 0.0):
                w = (u - uList[i]) / (uList[i + r - 1] - uList[i])
            c[s] = w * c[s] + (1 - w)*c[s + 1]
            i -= 1
    return c[0]

# Get the weights for each control point for the last u value in a curve.
# Assumes each knot has multiplicity 1 for now.
def lastSplineWeightsFilter(k, uList):
    if k == 1:
        return np.array([1.0]) # Special case.
    
    # Could just use identity matrix columns here, but I wanted something a bit
    # more explicit for now.
    c = [np.zeros(k) for _ in range(k)]
    for i in range(k):
        c[i][i] = 1
    # k-1 control points will have nonzero weights at the end.

    m = len(uList) - (k + 1) # m = last ctrl-pt index when starting from zero.
    u = uList[m + 1] # Last u value for which basis functions sum to 1.
    for r in range(k, 1, -1):
        i = m + 1 # Usually is i = delta, but for last pt, delta = m + 1.
        for s in range(r - 1):
            w = 0.0
            # If we'd divide by 0, or exceed range of uList, keep w as 0 instead.
            if (i + r - 1 < len(uList) and uList[i + r - 1] - uList[i] != 0.0):
                w = (u - uList[i]) / (uList[i + r - 1] - uList[i])
            c[s] = w * c[s] + (1 - w)*c[s + 1]
            i -= 1
    return c[0][1:] # First value of c[0] should always be redundant, zero!



# If we want the POINTS to be evenly spaced, we'll have to first roughly
# calculate the arclen of the curve at each of our current sample points and
# then use these arclens and some linear approximation to pick parameter
# values that should generate points separated by roughly equal arclen
# See:
#  * https://www.geometrictools.com/Documentation/MovingAlongCurveSpecifiedSpeed.pdf
#  * https://gamedev.stackexchange.com/questions/5373/moving-ships-between-two-planets-along-a-bezier-missing-some-equations-for-acce/5427#5427
#  * https://www.planetclegg.com/projects/WarpingTextToSplines.html
# At least one of the sources mentions a binary search, which is NOT NECESSARY; since
# we're moving along the curve from start to finish, we can just increment
# the indices in the "source" array we look at instead.
# This could certainly be improved for greater accuracy, but visually, this
# looks fine, and that's all that matters for the current use case.
def approximateEquidistantParamValues(paramVals, initialPoints):
    numSegments = len(paramVals) - 1
    cumulativeArcLens = [0]
    for i in range(1, len(initialPoints)):
        pt0 = initialPoints[i - 1]
        pt1 = initialPoints[i]
        cumulativeArcLens.append(cumulativeArcLens[-1] + np.linalg.norm(pt1 - pt0))
    totalLen = cumulativeArcLens[-1]
    arcLenStep = totalLen/numSegments
    arcLenParamVals = [paramVals[0]]
    lenIndex = 0
    targetArcLen = arcLenStep
    for i in range(1, numSegments):
        while lenIndex < len(cumulativeArcLens) - 1 and cumulativeArcLens[lenIndex + 1] < targetArcLen:
            lenIndex += 1
        
        len0 = cumulativeArcLens[lenIndex]
        len1 = cumulativeArcLens[lenIndex + 1]
        fracBetween = (targetArcLen - len0)/(len1 - len0)
        arcLenParamVals.append(paramVals[lenIndex] + fracBetween*(paramVals[lenIndex + 1] - paramVals[lenIndex]))
        
        targetArcLen += arcLenStep
    arcLenParamVals.append(paramVals[-1])
    return np.array(arcLenParamVals)

def ptsFromNURBS(spline, numSegments, genEquidistantPoints):

    # Weights are the fourth/"W" coordinate following XYZ
    ctrlPtsWithWeights = np.array([np.array(pt.co) for pt in spline.points])
    ctrlPtCoords = ctrlPtsWithWeights[:, :-1]
    weights = ctrlPtsWithWeights[:, -1]

    ctrlPtRadii = np.array([pt.radius for pt in spline.points])
    

    m = len(ctrlPtsWithWeights) - 1
    k = spline.order_u
    numKnots = m + k + 1
    # spaceBetween assumes even spacing of knots.
    spaceBetween = 1.0 / (m - k + 2.0)

    # First, generate uList (list of knots) and knot multiplicities list
    muList = []
    uList = []
    currentU = 0.0
    if spline.use_endpoint_u:
        # Knot sequence where curve "touches" end ctrl points
        for i in range(numKnots):
            if (i < k or i >= m + 1):
                muList.append(k)
                if i == m + 1:
                    currentU += spaceBetween
            else:
                currentU += spaceBetween
                muList.append(1)
            uList.append(currentU)
    else:
        spaceBetween = 1.0
        # Basic/default/simple uniform knot sequence
        for i in range(numKnots):
            muList.append(1)
            uList.append(currentU)
            currentU += spaceBetween

    paramVals = np.zeros(numSegments + 1)
    # Check that there's enough ctrl points to make a curve.
    if (m + 1 > k - 1):
        # If so, create the curve. Make a line segment for each "step" of u, as dictated by curveUStep
        u = uList[k - 1]
        uSpan = uList[m + 1] - u
        curveUStep = uSpan/numSegments
        # Generate parameter values that are evenly-spaced.
        # Does not mean the generated POINTS will be evenly spaced.
        for paramIndex in range(numSegments + 1):
            # The below is useful if we set a step size like 0.01 without a
            # target number of points. But in our case, we choose a step size
            # based on the number of points we want. Therefore, we don't need
            # this, but in case the design changes later, I'm still keeping
            # it here as a comment.
            # if (u > uList[m + 1]):
            #     u = uList[m + 1]
            paramVals[paramIndex] = u
            u += curveUStep
    if genEquidistantPoints:
        initialPoints = specifiedPtsFromNURBS(ctrlPtCoords, weights, m, k, uList, muList, paramVals).points
        paramVals = approximateEquidistantParamValues(paramVals, initialPoints)
        

    return specifiedPtsFromNURBS(ctrlPtCoords, weights, m, k, uList, muList, paramVals, ctrlPtRadii)
    


def specifiedPtsFromNURBS(ctrlPtCoords, weights, m, k, uList, muList, paramVals, ctrlPtRadii = []):
    pointsToReturn = []
    radiiToReturn = []

    delta = k - 1
    # Check that there's enough ctrl points to make a curve.
    if (m + 1 > k - 1):
        weightedPoints = ctrlPtCoords
        weightedRadii = ctrlPtRadii
        for i in range(len(weightedPoints)):
            weightedPoints[i] = ctrlPtCoords[i] * weights[i]
            if len(ctrlPtRadii) > 0:
                weightedRadii[i] = ctrlPtRadii[i] * weights[i]

        for u in paramVals:
            # Update delta value.
            # We stop when delta >= m since, standard or regular knot sequence,
            # that's as far as is reasonable to do.
            #
            # TODO: This would probably break if we increased the multiplicity
            # at the very end of the curve. Will have to make larger fix if we
            # want to accomodate that!!! 
            while (delta < m and u >= uList[delta + 1]):
                delta = delta + muList[delta + 1]

            lineVertex = bSplineInner(u, k, delta, weightedPoints, uList)
            currentWeightScale = (1.0/(bSplineInner(u, k, delta, weights, uList)))
            lineVertex = currentWeightScale * lineVertex

            pointsToReturn.append(lineVertex)

            if len(ctrlPtRadii) > 0:
                currentRadius = bSplineInner(u, k, delta, weightedRadii, uList)
                radiiToReturn.append(currentWeightScale * currentRadius)

    return CurvePoints(paramVals, np.array(pointsToReturn), np.array(radiiToReturn))

    # spline = bpy.context.object.data.splines[0]
    # pts = spline.points
    # bpy.context.scene.objects.active


# Tangents are calculated as the derivatives of the interpolating polynomial
# for five consecutive points containing a given point.
def tangentsFromPoints(pVals):
    numPts = pVals.shape[0]

    tangentVals = []

    for i in range(numPts):
        estT = None
        if i == 0:
            estT = (-25)*pVals[0] + 48*pVals[1] - 36*pVals[2] + 16*pVals[3] - 3*pVals[4]
        elif i == 1:
            estT = (-3)*pVals[0] - 10*pVals[1] + 18*pVals[2] - 6*pVals[3] + pVals[4]
        elif i == (numPts - 1) - 1:
            estT = 3*pVals[numPts-1] + 10*pVals[numPts-2] - 18*pVals[numPts-3] + 6*pVals[numPts-4] - pVals[numPts-5]
        elif i == (numPts - 1):
            estT = 25*pVals[numPts-1] - 48*pVals[numPts-2] + 36*pVals[numPts-3] - 16*pVals[numPts-4] + 3*pVals[numPts-5]
        else:
            estT = pVals[i-2] - 8*pVals[i-1] + 8*pVals[i+1] - pVals[i+2]

        estT = estT/np.linalg.norm(estT)
        tangentVals.append(estT)
    
    tangentValsNP = np.array(tangentVals)
    if pVals.shape != tangentValsNP.shape:
        raise Exception("I messed up the tangents creation.")

    return tangentValsNP

# Only calculates the first normal vector, with the expectation that it would
# be propogated.
def firstNormalFromUnitTangents(tangentVals):
    # We estimate the first normal using the same technique to estimate tangents.
    estN = (-25)*tangentVals[0] + 48*tangentVals[1] - 36*tangentVals[2] + 16*tangentVals[3] - 3*tangentVals[4]

    # The estimated normal may not actually be orthogonal to the first tangent.
    # So we need to use cross products to get the normal.
    Tk = tangentVals[0].flatten()
    Sk = np.cross(Tk, estN.flatten())
    unnormedRk = np.cross(Sk, Tk).reshape(Tk.shape)

    return unnormedRk/np.linalg.norm(unnormedRk)

def singleFrenetNorm(unitTangent0, unitTangent1):
    b = np.cross(unitTangent0.flatten(), unitTangent1.flatten())
    n = np.cross(b, unitTangent1)
    n = n/np.linalg.norm(n)
    return n

def discreteFrenetNormalsApprox(unitTangentVals, closed=False):
    retList = []

    if closed:
        # If the curve is closed, the "previous tangent" at index i=0 is the last one.
        retList.append(singleFrenetNorm(unitTangentVals[-1], unitTangentVals[0]))
    else:
        # Otherwise, we'll reuse the normal for the next pair.
        retList.append(singleFrenetNorm(unitTangentVals[0], unitTangentVals[1]))

    for i in range(1, len(unitTangentVals)):
        retList.append(singleFrenetNorm(unitTangentVals[i], unitTangentVals[i-1]))
    
    return retList