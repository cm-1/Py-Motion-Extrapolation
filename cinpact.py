import sys
import numpy as np
from curvetools import applyMaskToPts, convolveFilter

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

    def _quotientDeriv(f, g, fd, gd, g2=None):
        if g2 is None:
            g2 = g**2
        return fd/g - f*gd/g2

    def _quotientDeriv2(f, g, fd, gd, fd2, gd2, g2 = None):
        # For quotient rule, derivative of f/g is f'/g - fg'/g^2. Which means 
        # the 2nd derivative is:
        #   f''/g - f'g'/g^2 - [(f'g' + fg'')/g^2 - 2fg(g')^2/g^4]
        #   = f''/g + (-2f'g' - fg'' + 2f(g')^2/g)/g^2
        if g2 is None:
            g2 = g**2
        two_gd = 2*gd
        return fd2/g + (-fd*two_gd - f*gd2 + f*gd*two_gd/g)/g2
        
    def weightAndDerivative(u: float, i: int, supportRad: float, k: float, include_2nd_deriv: bool):
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

        e_numer = -k*umi_sq
        e_denom = supportRad*supportRad - umi_sq
        exp_frac = e_numer/e_denom
        e_pt = np.exp(exp_frac)
        e_denom2 = e_denom**2
        e_denom_d = -2 * umi
        e_numer_d = k*e_denom_d
        e_exp_d = CinpactCurve._quotientDeriv(
            e_numer, e_denom, e_numer_d, e_denom_d, e_denom2
        )
        e_exp_d2 = None
        if include_2nd_deriv:
            e_numer_d2 = -2*k
            e_denom_d2 = -2
            e_exp_d2 = CinpactCurve._quotientDeriv2(
                e_numer, e_denom, e_numer_d, e_denom_d, e_numer_d2, e_denom_d2,
                e_denom2
            )

        zero_inds = umi == 0.0
        umi[zero_inds] = umi_sq[zero_inds] = 1 # Stops divide-by-zero warnings.
        piumi = np.pi * umi

        sinc_num = np.sin(piumi)
        sinc_pt = sinc_num/piumi
        sinc_pt[zero_inds] = 1.0
        sinc_num_d_nopi = np.cos(piumi)
        sinc_d = np.empty_like(umi)
        sinc_d_pt1 = sinc_num_d_nopi/umi
        piumiumi = (umi*piumi)
        sinc_d_pt2 = sinc_num/piumiumi
        sinc_d = sinc_d_pt1 - sinc_d_pt2
        sinc_d[zero_inds] = 0.0
        sinc_d2 = None
        if include_2nd_deriv:
            sinc_d2_pt11 = -np.pi * sinc_num / umi
            sinc_d2_mid = sinc_num_d_nopi / umi_sq
            # d/du sin(pi*x)/(pi*x^2) = ... - 2pi*x*sin(...)/(pi*x^2)^2
            sinc_d2_pt22 = 2 * piumi * sinc_num / (piumiumi**2)
            sinc_d2 = sinc_d2_pt11 - 2*sinc_d2_mid + sinc_d2_pt22
            sinc_d2[zero_inds] = -(np.pi**2)/3 # Limit as u-i -> 0.
        e_d_pt = e_exp_d*e_pt*sinc_pt
        sinc_d_pt = sinc_d * e_pt

        weights = np.zeros_like(u_np)
        derivs = np.zeros_like(u_np)
        weights[inside_inds] = e_pt * sinc_pt
        derivs[inside_inds] = e_d_pt + sinc_d_pt

        derivs2 = None
        if include_2nd_deriv:
            derivs2 = np.zeros_like(derivs)
            # Derivative was d(frac)*e*s + e*d(s)
            # New part is thus:
            # d2(frac)*e*s + d(frac)^2*e*s + 2*d(frac)*e*d(s) * e*d2(s)
            derivs2[inside_inds] = e_exp_d2*e_pt*sinc_pt + e_exp_d*e_d_pt
            derivs2[inside_inds] += 2*e_exp_d*sinc_d_pt + e_pt*sinc_d2 

        if np.isscalar(u_np):
            weights = weights.item()
            derivs = derivs.item()
            if include_2nd_deriv:
                derivs2 = derivs2.item()
        if include_2nd_deriv:
            return (weights, derivs, derivs2)
        
        return (weights, derivs)


    def weightDerivativeFilter(self, u: float, include_2nd_deriv: bool):
        indRangeInclusive = self.ctrlPtBoundsInclusive(u)
        derivs = np.empty(indRangeInclusive[1] + 1 - indRangeInclusive[0])
        weights = np.empty(derivs.shape)
        derivs2 = None
        if include_2nd_deriv:
            derivs2 = np.empty_like(derivs)
        for i in range(indRangeInclusive[0], indRangeInclusive[1] + 1):
            si = i - indRangeInclusive[0] # Storage index
            deriv_info = CinpactCurve.weightAndDerivative(
                u, i, self.supportRad, self.k, include_2nd_deriv
            )
            weights[si], derivs[si] = deriv_info[:2]
            if include_2nd_deriv:
                derivs2[si] = deriv_info[2]
        w_sum = weights.sum()
        d_sum = derivs.sum()
        w_sum_sq = w_sum**2
        deriv_full = -weights*d_sum/w_sum_sq + derivs/w_sum
        deriv2_full = None
        if include_2nd_deriv:
            deriv2_full = CinpactCurve._quotientDeriv2(
                weights, w_sum, derivs, d_sum, derivs2, derivs2.sum(), w_sum_sq 
            )
            return (deriv_full, deriv2_full)
        return(deriv_full,)
