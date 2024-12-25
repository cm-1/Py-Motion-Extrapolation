import sys
import numpy as np

class CinpactLogic:
    def __init__(self, numCtrlPts: int, isOpen: bool, supportRad: float, k: float):
        self.numCtrlPts = numCtrlPts
        self.isOpen = isOpen
        self.supportRad = supportRad
        self.k = k
        self.ctrlPtRadius = int(np.ceil(supportRad))

    def ctrlPtBoundsInclusive(self, u: float):
        midCtrlPt = int(np.floor(u))
        startCtrlPt = midCtrlPt - self.ctrlPtRadius
        endCtrlPt = midCtrlPt + self.ctrlPtRadius
        if self.isOpen:
            startCtrlPt = max(startCtrlPt, 0)
            endCtrlPt = min(endCtrlPt, self.numCtrlPts - 1)
        return (startCtrlPt, endCtrlPt)


class CinpactCurve(CinpactLogic):
    def __init__(self, ctrlPts, isOpen: bool, supportRad: float, k: float, numSubdivPts: int):
        super().__init__(len(ctrlPts), isOpen, supportRad, k)
        numCtrlPts = len(ctrlPts)
        self.ctrlPts = ctrlPts
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

        last_u = numCtrlPts - 1.0 + lastToFirst
        self.paramVals = np.linspace(0.0, last_u, numSubdivPts)


        ctrlPtRadius = int(np.ceil(self.supportRad))
        startCtrlPt = 0 if self.isOpen else -ctrlPtRadius
        midCtrlPt = 0
        endCtrlPt = ctrlPtRadius + 1 # non-inclusive
        if endCtrlPt > self.numCtrlPts:
            endCtrlPt = self.numCtrlPts
        self.ctrlPtDim = len(ctrlPts[0])
        self.curvePoints = np.empty((numSubdivPts, self.ctrlPtDim))
        for p in range(len(self.paramVals)):
            u = self.paramVals[p]
            ptSum = np.zeros(self.ctrlPtDim)
            weightSum = 0.0
            for i in range(startCtrlPt, endCtrlPt):
                w = CinpactCurve.weightFunc(u, i, supportRad, k)
                weightSum += w
                ctrlInd = (i + self.numCtrlPts) % self.numCtrlPts
                ptSum += w * self.ctrlPts[ctrlInd]

            self.curvePoints[p] = (1.0/weightSum) * ptSum

            if (u >= midCtrlPt + 1):
                midCtrlPt += 1
                startCtrlPt = midCtrlPt - ctrlPtRadius
                endCtrlPt = midCtrlPt + ctrlPtRadius + 1
                if self.isOpen:
                    startCtrlPt = max(startCtrlPt, 0)
                    endCtrlPt = min(endCtrlPt, self.numCtrlPts)
    
    # sin(pi(u-i))/pi(u-i)
    def normSinc(u: float, i: int):
        x = np.asarray(np.pi * (u - i))
        zero_inds = (x == 0.0)
        x[zero_inds] = 1.0 # Prevent division by zero warnings
        retVal = np.asarray(np.sin(x)/x)
        retVal[zero_inds] = 1.0
        if np.isscalar(retVal):
            retVal = retVal.item()
        return retVal

    # exp[-(k(u-i)^2)/(c^2 - (u-i)^2)]
    def temperingFunc(u: float, i: int, supportRad: float, k: float):
        umi_all = np.asarray(u - i)
        retVal = np.empty_like(u)
        inside_inds = np.abs(umi_all) < supportRad
        umi = umi_all[inside_inds]
        expon = -k*umi*umi/(supportRad*supportRad - umi*umi)
        retVal[inside_inds] = np.exp(expon)
        retVal[np.invert(inside_inds)] = 0.0
        if np.isscalar(retVal):
            retVal = retVal.item()
        return retVal
        

    def weightFunc(u: float, i: int, supportRad: float, k: float):
        nsinc = CinpactCurve.normSinc(u, i)
        tempering = CinpactCurve.temperingFunc(u, i, supportRad, k)
        return nsinc * tempering

    def weightAndDerivative(u: float, i: int, supportRad: float, k: float):
        # We need the derivative of:
        # e^(...)sin(...)/(...)
        # Let's start with the derivative of e^(...)
        # At the front's the derivative of (-k(u-i)^2)(c^2 - (u-i)^2)^-1
        # Which is (-k(u-i)^2)2(u-i)/(c^2 - (u-i)^2)^-2 + (-2k(u-i))(...)^-1
        # Then the derivative of sin(...)(...)^-1 is:
        # pi*cos(pi(u-i))/pi(u-i) - pi*sin(pi(u-i))/(pi(u-i))^2
        
        u_np = np.asarray(u)


        umi_all = u_np - i

        inside_inds = np.abs(umi_all) < supportRad    

        umi = umi_all[inside_inds]
        umi_sq = umi**2

        e_pt = CinpactCurve.temperingFunc(u_np[inside_inds], i, supportRad, k)
        sinc_pt = CinpactCurve.normSinc(u_np[inside_inds], i)

        e_denom = supportRad*supportRad - umi_sq
        frac_pt = umi_sq/(e_denom**2) + 1.0/e_denom
        e_d_pt = -2*k*umi*(frac_pt)
        e_d_pt *= e_pt*sinc_pt

        zero_inds = umi == 0.0
        umi[zero_inds] = 1.0 # Prevent divide-by-zero warnings.
        piumi = np.pi * umi

        sinc_d_pt = np.zeros_like(umi)
        sinc_d_pt = np.cos(piumi)/umi - np.sin(piumi)/(umi*piumi)
        sinc_d_pt[zero_inds] = 0.0
        sinc_d_pt *= e_pt

        weights = np.zeros_like(u_np)
        derivs = np.zeros_like(u_np)
        weights[inside_inds] = e_pt * sinc_pt
        derivs[inside_inds] = e_d_pt + sinc_d_pt
        if np.isscalar(u_np):
            weights = weights.item()
            derivs = derivs.item()
        return (weights, derivs)


    def weightDerivativeFilter(self, u:float):
        indRangeInclusive = self.ctrlPtBoundsInclusive(u)
        derivs = np.empty(indRangeInclusive[1] + 1 - indRangeInclusive[0])
        weights = np.empty(derivs.shape)
        for i in range(indRangeInclusive[0], indRangeInclusive[1] + 1):
            si = i - indRangeInclusive[0] # Storage index
            weights[si], derivs[si] = CinpactCurve.weightAndDerivative(
                u, i, self.supportRad, self.k
            )
        w_sum = weights.sum()
        d_sum = derivs.sum()
        return -weights*d_sum/(w_sum**2) + derivs/w_sum
    
