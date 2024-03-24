import math
import numpy as np

def getSplinePts(ctrlPts, supportRad: float, k: float, numSubdivPts: int):
    numCtrlPts = len(ctrlPts)
    maxRad = float((numCtrlPts - 1) >> 1)

    step = (numCtrlPts - 1.0)/(numSubdivPts - 1.0)

    # We'll find and keep track of which Ctrl Pts have nonzero weights.
    ctrlPtRadius = int(math.ceil(supportRad))
    startCtrlPt = 0
    midCtrlPt = 0
    endCtrlPt = ctrlPtRadius + 1 # non-inclusive
    if endCtrlPt > numCtrlPts:
        endCtrlPt = numCtrlPts



    u = 0.0
    retPts = []

    for p in range(numSubdivPts):
        ptSum = np.array([0.0, 0.0, 0.0])
        weightSum = 0.0
        for i in range(startCtrlPt, endCtrlPt):
            w = weightFunc(u, i, supportRad, k)
            weightSum += w
            ptSum += w * ctrlPts[(i + numCtrlPts) % numCtrlPts]

        retPts.append((1.0/weightSum) * ptSum)


        u += step
        if (u >= midCtrlPt + 1):
            midCtrlPt += 1
            startCtrlPt = midCtrlPt - ctrlPtRadius
            endCtrlPt = midCtrlPt + ctrlPtRadius + 1
    
    return np.array(retPts)

def normSinc(u: float, i: int):
    x = math.pi * (u - float(i))
    if (x == 0.0):
        return 1.0
    return math.sin(x)/x

def temperingFunc(u: float, i: int, supportRad: float, k: float):
    umi = u - float(i)
    if (abs(umi) >= supportRad):
        return 0.0
    expon = -k*umi*umi/(supportRad*supportRad - umi*umi)
    return math.exp(expon)

def weightFunc(u: float, i: int, supportRad: float, k: float):
    return normSinc(u, i) * temperingFunc(u, i, supportRad, k)
