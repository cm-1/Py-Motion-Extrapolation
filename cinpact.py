import math
import sys
import numpy as np

class CinpactCurve:
    def __init__(self, ctrlPts, isOpen: bool, supportRad: float, k: float, numSubdivPts: int):
        numCtrlPts = len(ctrlPts)

        # If ctrlPts is empty, just leave the curve empty.
        if numCtrlPts == 0:
            self.curvePoints = np.zeros((0,3), dtype=np.float64)
            self.paramVals = np.array([], dtype=np.float64)
            return
        
        maxRad = float((numCtrlPts - 1) >> 1)
        if (not isOpen) and (supportRad > maxRad):
            warn_str = "Truncating support radius for closed curve; have not \
                decided how to handle when it wraps around more than once yet!"
            print(warn_str, file=sys.stderr)

        lastToFirst = 0.0 if isOpen else 1.0

        step = (numCtrlPts - 1.0 + lastToFirst)/(numSubdivPts - 1.0)

        # We'll find and keep track of which Ctrl Pts have nonzero weights.
        ctrlPtRadius = int(math.ceil(supportRad))
        startCtrlPt = 0 if isOpen else -ctrlPtRadius
        midCtrlPt = 0
        endCtrlPt = ctrlPtRadius + 1 # non-inclusive
        if endCtrlPt > numCtrlPts:
            endCtrlPt = numCtrlPts



        u = 0.0
        ctrlPtDim = len(ctrlPts[0])
        self.curvePoints = np.empty((numSubdivPts, ctrlPtDim))

        self.paramVals = np.empty(numSubdivPts)

        for p in range(numSubdivPts):
            ptSum = np.zeros(ctrlPtDim)
            weightSum = 0.0
            for i in range(startCtrlPt, endCtrlPt):
                w = CinpactCurve.weightFunc(u, i, supportRad, k)
                weightSum += w
                ptSum += w * ctrlPts[(i + numCtrlPts) % numCtrlPts]

            self.curvePoints[p] = (1.0/weightSum) * ptSum

            self.paramVals[p] = u
            u += step
            if (u >= midCtrlPt + 1):
                midCtrlPt += 1
                startCtrlPt = midCtrlPt - ctrlPtRadius
                endCtrlPt = midCtrlPt + ctrlPtRadius + 1
                if isOpen:
                    startCtrlPt = max(startCtrlPt, 0)
                    endCtrlPt = min(endCtrlPt, numCtrlPts)
        
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
        return CinpactCurve.normSinc(u, i) * CinpactCurve.temperingFunc(u, i, supportRad, k)
