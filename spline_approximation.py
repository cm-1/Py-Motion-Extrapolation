import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
import typing
from enum import Enum

import numpy as np

import bspline
from cinpact import CinpactLogic, CinpactCurve
import curvetools

ndarray_couple_type = typing.Tuple[np.ndarray, np.ndarray]


def bSplineFittingMat(numCtrlPts, order, numInputPts, uVals, knotVals):
    mat = np.empty((numInputPts, numCtrlPts))

    iden = np.identity(numCtrlPts)

    delta = order - 1
    for r in range(numInputPts):
        u = uVals[r]
        while (delta < numCtrlPts - 1 and u >= knotVals[delta + 1]):
            delta = delta + 1 # muList[delta + 1]
        for c in range(numCtrlPts):
            mat[r][c] = bspline.bSplineInner(u, order, delta, iden[c], knotVals)

    return mat

def bSincpactFittingMat(numInputPts: int, numCtrlPts: int, uVals: np.ndarray,
                       half_order: float, support_rad_sq: float,
                       support_rad: float, k: float):
    # raise NotImplementedError()
    umi_inputs = uVals[:numInputPts, np.newaxis] - np.arange(numCtrlPts)
    unnormed = CinpactCurve.getBCinpactWeights(
        umi_inputs, half_order, support_rad_sq, support_rad, k
    )
    return unnormed / np.sum(unnormed, axis=1, keepdims=True)


def fitBSpline(numCtrlPts, order, ptsToFit, uVals, knotVals, numUnknownPts = 0):
    if (len(ptsToFit) + numUnknownPts) != len(uVals):
        raise Exception(
            "Number of u values does not match number of known + unknown points!"
        )

    mat = bSplineFittingMat(numCtrlPts, order, len(ptsToFit), uVals, knotVals)

    fitted_ctrl_pts = np.linalg.lstsq(mat, ptsToFit, rcond = None)[0]
    return fitted_ctrl_pts

@dataclass 
class PseudoInvAndFilterInfo:
    data_u_vals: np.ndarray
    pseudo_inv: np.ndarray
    filter_on_input: np.ndarray = None # Might need to be filled in later

class SplinePredictionMode(Enum):
    EXTRAPOLATE = 1
    SMOOTH = 2
    CONST_ACCEL = 3
    SMOOTH_AND_ACCEL = 4

@dataclass
class SplineFiltersStruct:
    ctrlPtsToLastFilter: np.ndarray = None
    ctrlPtsToDerivFilter: np.ndarray = None
    ctrlPtsTo2ndDerivFilter: np.ndarray = None

class ABCSplineFittingCase(ABC):
    def __init__(self, degree, num_inputs, num_ctrl_pts,
                 precalced_filters: SplineFiltersStruct,
                 modes: typing.List[SplinePredictionMode]):

        need_extrap = False
        smooth_ctrl_keys = []
        calc_last_smooth_pt = False
        save_last_smooth_pt = False
        calc_non_smoothed_accel = False
        accel_keys = []
        for mode in modes:
            if mode == SplinePredictionMode.EXTRAPOLATE:
                need_extrap = True
            else:
                smooth_ctrl_keys.append(mode)
                if mode == SplinePredictionMode.SMOOTH:
                    calc_last_smooth_pt = True
                    save_last_smooth_pt = True
                elif mode == SplinePredictionMode.CONST_ACCEL:
                    accel_keys.append(mode)
                    calc_non_smoothed_accel = True
                elif mode == SplinePredictionMode.SMOOTH_AND_ACCEL:
                    calc_last_smooth_pt = True
                    accel_keys.append(mode)
                else:
                    raise ValueError(f"Unknown SplinePredictionMode: {mode}")

        self.filters_on_ctrl_pts = precalced_filters

        self.num_ctrl_pts = num_ctrl_pts

        self.uInterval = (degree, self.num_ctrl_pts) 

        self.degree = degree
        self.num_inputs = num_inputs

        self.by_mode_dict: \
            typing.Dict[SplinePredictionMode, PseudoInvAndFilterInfo] = dict()
        
        if need_extrap:
            ipii = self._getInitialPseudoInvInfo(num_inputs + 1)
            self.by_mode_dict[SplinePredictionMode.EXTRAPOLATE] = ipii
        if len(smooth_ctrl_keys) > 0:
            ipii = self._getInitialPseudoInvInfo(num_inputs)
            for i, k in enumerate(smooth_ctrl_keys):
                if i > 0:
                    ipii = copy.copy(ipii)
                self.by_mode_dict[k] = ipii

        lastPtFilterMat = None
        if need_extrap or calc_last_smooth_pt:    
            lastPtFilterMat = precalced_filters.ctrlPtsToLastFilter.reshape(
                1, -1
            )
        if need_extrap:
            info = self.by_mode_dict[SplinePredictionMode.EXTRAPOLATE]
            pseudoInvLastRows = info.pseudo_inv[-lastPtFilterMat.shape[1]:]
            inputsToLastPtFilterRowVec = lastPtFilterMat @ pseudoInvLastRows
            info.filter_on_input = inputsToLastPtFilterRowVec.flatten()
        
        last_smooth_filter = None
        if calc_last_smooth_pt:
            info = self.by_mode_dict[smooth_ctrl_keys[0]]
            pseudoInvLastRows = info.pseudo_inv[-lastPtFilterMat.shape[1]:]
            inputsToLastPtFilterRowVec = lastPtFilterMat @ pseudoInvLastRows
            last_smooth_filter = inputsToLastPtFilterRowVec.flatten()
            if save_last_smooth_pt:
                info = self.by_mode_dict[SplinePredictionMode.SMOOTH]
                info.filter_on_input = last_smooth_filter

        if len(accel_keys) > 0:
            shared_acc_info = self.by_mode_dict[accel_keys[0]]
            ctrlToVel = precalced_filters.ctrlPtsToDerivFilter.reshape(1, -1)
            ctrlToAcc = precalced_filters.ctrlPtsTo2ndDerivFilter.reshape(1, -1)

            assert ctrlToVel.shape[1] == ctrlToAcc.shape[1], \
            "Spline velocity and acceleration filters must have same shape!"

            pseudo_inv = shared_acc_info.pseudo_inv
            velAccPseudoInvRows = pseudo_inv[-ctrlToVel.shape[1]:]
            velFilter = (ctrlToVel @ velAccPseudoInvRows).flatten()
            accFilter = (ctrlToAcc @ velAccPseudoInvRows).flatten()

            
            du = shared_acc_info.data_u_vals[1] - shared_acc_info.data_u_vals[0]


            # From the perspective of the "Spline", each time step is not 1 
            # unit, but instead the distance between u values we chose for our
            # points when doing the least-squares fitting. Because it is simpler
            # to, e.g., have consecutive integer knots than consecutive integer 
            # sample locations, the result is that the space between these 
            # sampled u values, and thus the "time", will not be 1. So, the time
            # step we multiply the velocity by to obtain displacement will be
            # this u sample step.
            velFilter *= du # delta_time*velocity
            
            # Since our time step is not 1, we must explicity square it.
            accFilter *= du**2

            accDeltaFilter = velFilter + 0.5 * accFilter

            for mode in accel_keys:
                if mode == SplinePredictionMode.SMOOTH_AND_ACCEL:
                    # A filter that represents smoothing previous points, then 
                    # extrapolating using constant acceleration.
                    smooth_filter_len = len(last_smooth_filter)
                    acc_filter_len = len(accFilter)
                    filter_len = max(acc_filter_len, smooth_filter_len)
                    smooth_accel_filter = np.zeros(filter_len)
                    acc_filter_len_diff = filter_len - acc_filter_len
                    smooth_accel_filter[acc_filter_len_diff:] += accDeltaFilter
                    smooth_filter_len_diff = filter_len - smooth_filter_len
                    smooth_accel_filter[smooth_filter_len_diff:] += last_smooth_filter
                    self.by_mode_dict[mode].filter_on_input = smooth_accel_filter
                elif mode == SplinePredictionMode.CONST_ACCEL:
                    self.by_mode_dict[mode].filter_on_input = accDeltaFilter
            # We do this at the end to avoid a copy of the "delta" filter that
            # might also be used in the smooth+accel case. Here, adding 1.0
            # represents adding the result of the "delta" filter to the last pt.
            if calc_non_smoothed_accel:
                item = self.by_mode_dict[SplinePredictionMode.CONST_ACCEL]
                item.filter_on_input[-1] += 1.0

    def _getInitialPseudoInvInfo(self, data_u_val_count):
        data_u_vals = np.linspace(*self.uInterval, data_u_val_count)
            
        matA = self._getFittingMat(data_u_vals)

        matAtA = matA.transpose() @ matA
        pseudo_inv = np.linalg.inv(matAtA) @ matA.transpose()
        
        return PseudoInvAndFilterInfo(data_u_vals, pseudo_inv)
    
    @abstractmethod
    def _getFittingMat(self, data_u_vals) -> np.ndarray:
        ...
    
class BSplineFittingCase(ABCSplineFittingCase):
    def _getFittingMat(self, data_u_vals):
        # Just the number of pts in any B-Spline; nothing fitting-specific.
        knot_list = np.arange(self.num_ctrl_pts + self.degree + 1)
        matA = bSplineFittingMat(
            self.num_ctrl_pts, self.degree + 1, self.num_inputs, data_u_vals,
            knot_list
        )
        return matA

class BSincpactFittingCase(ABCSplineFittingCase):
    def __init__(self, degree, num_inputs, num_ctrl_pts,
                 precalced_filters: SplineFiltersStruct,
                 modes: typing.List[SplinePredictionMode],
                 cinpact_logic: CinpactLogic, support_rad_sq: float,
                 half_order: float):
        # CINPACT logic will be required to create the point-fitting matrix,
        # which in turn means it must be available before the super constructor!
        self.cinpact_logic = cinpact_logic
        self.support_rad_sq = support_rad_sq
        self.half_order = half_order
        super().__init__(
            degree, num_inputs, num_ctrl_pts, precalced_filters, modes
        )

    def _getFittingMat(self, data_u_vals):
        return bSincpactFittingMat(
            self.num_inputs, self.num_ctrl_pts, data_u_vals, self.half_order, 
            self.support_rad_sq, self.cinpact_logic.supportRad,
            self.cinpact_logic.k        
        )


class ABCSplineFitCalculator(ABC):

    # This might not need to be its own method. I was thinking of overriding it
    # in subclasses, but then decided against it.
    @staticmethod
    def get_min_inputs_req(degree: float):
        return int(np.ceil(degree)) + 1

    def __init__(self, spline_degree: typing.Union[float, int],
                 max_num_ctrl_pts: int, max_num_input_data: int,
                 modes: typing.List[SplinePredictionMode]):
        if (max_num_input_data < spline_degree + 1):
            raise Exception("Need at least order=k=(degree + 1) input pts for calc!")
        if max_num_ctrl_pts > max_num_input_data:
            raise Exception("Need at least as many input points as control points!")
        if max_num_ctrl_pts < spline_degree + 1:
            raise Exception("Need at least order=k=(degree + 1) control points!")


        self.cases: typing.List[ABCSplineFittingCase] = []
        
        self.spline_degree: float = spline_degree
        self.max_num_ctrl_pts: int = max_num_ctrl_pts
        self.max_num_input_data: int = max_num_input_data
        self.modes = modes

        self.reqs_ctrl_to_last_pt = False
        self.reqs_ctrl_to_derivs = False
        for mode in modes:
            if mode == SplinePredictionMode.EXTRAPOLATE:
                self.reqs_ctrl_to_last_pt = True
            elif mode == SplinePredictionMode.SMOOTH:
                self.reqs_ctrl_to_last_pt = True
            elif mode == SplinePredictionMode.SMOOTH_AND_ACCEL:
                self.reqs_ctrl_to_last_pt = True
                self.reqs_ctrl_to_derivs = True
            elif mode == SplinePredictionMode.CONST_ACCEL:
                self.reqs_ctrl_to_derivs = True
            else:
                raise ValueError("Unexpected mode!")
        
        self.min_inputs_req = self.get_min_inputs_req(spline_degree)
        self._createAllCases()

    @abstractmethod
    def createCase(self, num_inputs: int, num_ctrl_pts: int, 
                   modes: typing.List[SplinePredictionMode]) -> ABCSplineFittingCase:
        ...

    def _createAllCases(self):
        num_input_bound = self.max_num_input_data + 1
        for num_inputs in range(self.min_inputs_req, num_input_bound):
            num_ctrl_pts = min(num_inputs, self.max_num_ctrl_pts)
            self.cases.append(self.createCase(
                num_inputs, num_ctrl_pts, self.modes
            ))

    def fitAllData(self, all_input_pts: np.ndarray):
        if all_input_pts.ndim > 2:
            str_dim = str(all_input_pts.ndim)
            raise Exception("Array dimension " + str_dim + " not supported for \
                            Spline fitting!")
        
        mode = SplinePredictionMode.EXTRAPOLATE
        num_spline_preds = len(all_input_pts) - self.min_inputs_req 
                                    
        # The below is just to accomodate either an array of floats or of vecs.
        output_shape = (num_spline_preds,) + all_input_pts.shape[1:]
        spline_preds = np.empty(output_shape)

        # How many predictions we've calculated so far.
        prediction_num = 0
            
        min_input_ind = self.min_inputs_req - 1
        for last_input_ind in range(min_input_ind, self.max_num_input_data - 1):
            case_filter = self.cases[prediction_num].by_mode_dict[mode].filter_on_input
            filter_len = len(case_filter)
            input_start = (last_input_ind + 1) - filter_len

            ptsToFit = all_input_pts[input_start:(last_input_ind + 1)]

        
            spline_preds[prediction_num] = curvetools.applyMaskToPts(
                ptsToFit, case_filter
            )

            prediction_num += 1

        main_filter = self.cases[-1].by_mode_dict[mode].filter_on_input
        main_filter_len = len(main_filter)
        len_diff = self.max_num_input_data - main_filter_len

        spline_preds[prediction_num:] = curvetools.convolveFilter(
            main_filter, all_input_pts[len_diff:-1] 
        )

        return spline_preds
    
    def smoothAllData(self, all_input_pts: np.ndarray, all_preds: np.ndarray):
        if all_input_pts.ndim > 2:
            str_dim = str(all_input_pts.ndim)
            raise Exception("Array dimension " + str_dim + " not supported for Spline smoothing!")

        mode = SplinePredictionMode.SMOOTH
        # We'll output as many predictions as were inputted.
        # We may not be able to smooth all of them; if some cannot be smoothed,
        # they will just be returned as they are.
        spline_smooths = np.empty(all_preds.shape)

        inputs_for_1st_pred = len(all_input_pts) - len(all_preds)

        # If there are not enough available points for spline fitting, then just
        # copy the respective non-smoothed prediction as-is.
        copy_pred_amt = self.min_inputs_req - 1
        for pred_ind in range(len(all_preds)):
            available_inputs = inputs_for_1st_pred + pred_ind + 1
            if available_inputs <= copy_pred_amt:
                spline_smooths[pred_ind] = all_preds[pred_ind]
                continue

            startInd = max(0, available_inputs - self.max_num_input_data)
            
            ptsToFit_measured = all_input_pts[startInd:(available_inputs - 1)]
            ptToFit_pred = all_preds[pred_ind:(pred_ind + 1)]

            max_case_ind = len(self.cases) - 1
            inputs_beyond_min_req = available_inputs - (self.spline_degree + 1)
            case = self.cases[min(max_case_ind, inputs_beyond_min_req)]
            case_pseudo_inv = case.by_mode_dict[mode].pseudo_inv

            pseudoInv_measured = case_pseudo_inv[:, :-1]
            ctrlPts = pseudoInv_measured @ ptsToFit_measured
            pseudoInv_pred = case_pseudo_inv[:, -1:]
            ctrlPts += pseudoInv_pred @ ptToFit_pred

            # The documentation for np.dot(a, b) states that if b is 1D like our
            # filter, then np.dot() will do a weighted sum along a's last axis.
            smooth_filter = case.filters_on_ctrl_pts.ctrlPtsToLastFilter
            spline_smooths[pred_ind] = np.dot(
                ctrlPts[-len(smooth_filter):].transpose(), smooth_filter
            ).flatten()

        return spline_smooths

    def constantAccelPreds(self, all_input_pts: np.ndarray, smooth_before_accel_add: bool):
        if all_input_pts.ndim > 2:
            str_dim = str(all_input_pts.ndim)
            raise Exception("Array dimension " + str_dim + " not supported for Spline smoothing!")
        mode = None
        if smooth_before_accel_add:
            mode = SplinePredictionMode.SMOOTH_AND_ACCEL
        else:
            mode = SplinePredictionMode.CONST_ACCEL
        # We will have a prediction for all points except for the first.
        output_shape = (len(all_input_pts) - 1, ) + all_input_pts.shape[1:]
        spline_preds = np.empty(output_shape)

        # First prediction assumes no motion.
        spline_preds[0] = all_input_pts[0]
        # Second prediction assumes constant velocity.
        spline_preds[1] = all_input_pts[1] + all_input_pts[1] - all_input_pts[0]

        # Remaining predictions until enough inputs for B-SPline fit (deg + 1)
        # exist will be const-accel using quadratic polynomial fitting. 
        
        # Because we have no prediction for the first input, the index==degree
        # prediction array val will have the degree+1 input pts required for 
        # spline fitting. So we'll fill in the predictions up until this.
        spline_ind_min = self.min_inputs_req - 1
        # spline_ind_min - 1 is guaranteed to be at least 0, but - 2 might not.
        spline_ind_min_sub_2 = max(spline_ind_min - 2, 0)
        # Quadratic polynomial fit result can be shown to be 3x_2 - 3x_1 + x_0.
        spline_preds[2:spline_ind_min] = 3 * all_input_pts[2:spline_ind_min] - 3 * all_input_pts[1:spline_ind_min - 1] + all_input_pts[:spline_ind_min_sub_2]

        # For the rest, calculate velocity and acceleration using derivatives of
        # the fitted B-Spline curve.
        for pred_ind in range(spline_ind_min, self.max_num_input_data - 1):
            # Predictions[0] corresponds to position[1]. Hence: 
            available_inputs = pred_ind + 1

            # We have different B-Spline fitting cases depending on how many
            # inputs are available. Case[0] corresponds to degree+1 inputs.
            inputs_beyond_min_req = available_inputs - self.min_inputs_req
            acc_filter = self.cases[inputs_beyond_min_req].by_mode_dict[mode].filter_on_input
            filter_len = len(acc_filter)

            startInd = available_inputs - filter_len
            
            ptsToFit = all_input_pts[startInd:available_inputs]

            spline_preds[pred_ind] = curvetools.applyMaskToPts(
                ptsToFit, acc_filter
            )

        main_filter = self.cases[-1].by_mode_dict[mode].filter_on_input

        main_filter_len = len(main_filter)
        len_diff = self.max_num_input_data - main_filter_len

        main_pred_start = self.max_num_input_data - 1
        spline_preds[main_pred_start:] = curvetools.convolveFilter(
            main_filter, all_input_pts[len_diff:-1] 
        )

        return spline_preds


class BSplineFitCalculator(ABCSplineFitCalculator):
    deg_to_pt_filter: typing.Dict[int, np.ndarray] = dict()

    def get_deg_ctrl_pt_filter(deg: int):
        if deg in BSplineFitCalculator.deg_to_pt_filter.keys():
            return BSplineFitCalculator.deg_to_pt_filter[deg]
        elif deg < 0:
            return np.empty(0) # i.e. np.array([])
        
        # A B-Spline curve has m + k + 1 knots. Min value for m is k - 1.
        # In that case, would have 2k = 2*(deg+1) = (2*deg + 2) knots.
        sample_knots = np.arange(2*deg + 2) # Uniform knot seq.
        f = bspline.lastSplineWeightsFilter(deg + 1, sample_knots)        
        BSplineFitCalculator.deg_to_pt_filter[deg] = f
        return f


    def createCase(self, num_inputs, num_ctrl_pts, modes):
        return BSplineFittingCase(
            self.spline_degree, num_inputs, num_ctrl_pts, 
            self.filters_on_ctrl_pts, modes
        )

    def _createAllCases(self):
        last_pt_filter = None
        deriv_filter = None
        deriv2_filter = None
        if self.reqs_ctrl_to_last_pt:
            last_pt_filter = BSplineFitCalculator.get_deg_ctrl_pt_filter(
                self.spline_degree
            )
        if self.reqs_ctrl_to_derivs:
            deriv_filter, deriv2_filter = self._get1st2ndDerivativeFilters()                
        
        self.filters_on_ctrl_pts = SplineFiltersStruct(
            last_pt_filter, deriv_filter, deriv2_filter
        )

        return super()._createAllCases()        

    def _diffFilter(orig_filter):
        # [a b c d e] [-1 1 / 0 -1 1 / 0 0 -1 1 ...]
        # [-a a-b c-b d-c e-d e] 
        ret_filter = np.empty(len(orig_filter) + 1)        
        ret_filter[:-1] = -orig_filter
        ret_filter[-1] = 0.0
        ret_filter[1:] += orig_filter
        return ret_filter
    
    def _get1st2ndDerivativeFilters(self) -> ndarray_couple_type:
        
        # Formula for derivatives of B-Spline curves comes from:
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html
        # Because we are assuming uniform integer knots, the
        # p/(u_{i+p+1} - u_{i+1}) part simplifies to 1.    
        f1 = BSplineFitCalculator.get_deg_ctrl_pt_filter(self.spline_degree - 1)
        f2 = BSplineFitCalculator.get_deg_ctrl_pt_filter(self.spline_degree - 2)
        
        fd1 = BSplineFitCalculator._diffFilter(f1)
        fd2 = BSplineFitCalculator._diffFilter(
            BSplineFitCalculator._diffFilter(f2
        ))

        # For degree 2, both f1 and f2 will be [1.0], meaning fd1 and fd2 will
        # have different lengths. This should be corrected. 
        if self.spline_degree == 2:
            fd1 = np.insert(fd1, 0, 0.0) # Prepend with a 0.0

        return (fd1, fd2)

    def getCtrlPts(self, in_data):
        case = self.cases[-1]

        if SplinePredictionMode.EXTRAPOLATE not in case.by_mode_dict.keys():
            raise NotImplementedError("Only EXTRAPOLATE mode supported so far!")

        info = case.by_mode_dict[SplinePredictionMode.EXTRAPOLATE]

        if len(in_data) != len(info.data_u_vals) - 1:
            raise NotImplementedError("Only handles max input count for now!")

        return (info.pseudo_inv @ in_data)
    
class BSincpactFitCalculator(ABCSplineFitCalculator):
    def __init__(self, spline_degree: typing.Union[float, int],
                 max_num_ctrl_pts: int, max_num_input_data: int, 
                 support_rad: float, k: float,
                 modes: typing.List[SplinePredictionMode]):
        # The support radius and k will be used by other methods that the super
        # constructor calls, so they must be set first!
        self.k = k
        self.support_rad = support_rad
        self.support_rad_sq = support_rad * support_rad
        self.half_order = (spline_degree + 1.0)/2.0
        super().__init__(
            spline_degree, max_num_ctrl_pts, max_num_input_data, modes
        )
        
    def createCase(self, num_inputs, num_ctrl_pts, modes):
        start_ind = max(0, self.unnormed_filter_len - num_ctrl_pts)
        
        individ_weights = self.unnormed_filter_info[0][start_ind:]
        w_sum = individ_weights.sum()
        weights = individ_weights / w_sum
        
        filters_on_ctrl_pts = None
        if self.reqs_ctrl_to_derivs:
            derivs, derivs2 = (
                ufi[start_ind:] for ufi in self.unnormed_filter_info[1:]
            )
            d_sum = derivs.sum()
            w_sum_sq = w_sum**2
            deriv_full = -weights*d_sum/w_sum_sq + derivs/w_sum
            deriv2_full = CinpactCurve._quotientDeriv2(
                weights, w_sum, derivs, d_sum, derivs2, derivs2.sum(), w_sum_sq 
            )
            filters_on_ctrl_pts = SplineFiltersStruct(
                weights, deriv_full, deriv2_full
            )

        else:
            filters_on_ctrl_pts = SplineFiltersStruct(weights, None, None)
        
        return BSincpactFittingCase(
            self.spline_degree, num_inputs, num_ctrl_pts,
            filters_on_ctrl_pts, modes, self.last_cinpact_logic,
            self.support_rad_sq, self.half_order
        )

    def _createAllCases(self):
        # The B-Cinpact basis function will "start" at coordinate i.
        # It will actually be nonzero for some points before this, but its
        # "main" and B-Spline-emulating portion will start at i. Then, its
        # centre will be at i + half_order. From the centre, it'll be nonzero
        # for half-order * support-radius further units. So, the nonzero basis 
        # functions at the last point, which is u=m+1, will be the ones s.t.
        # i + half_order (1 + support_rad) > m + 1 
        # => i > m + 1 - half_order (1 + support_rad)
        first_i_float = self.max_num_ctrl_pts - self.half_order * (1 + self.support_rad)
        first_i = max(0, np.ceil(first_i_float))
        i_vals = np.arange(first_i, self.max_num_ctrl_pts)

        self.unnormed_filter_info = None
        if self.reqs_ctrl_to_derivs:
            self.unnormed_filter_info = CinpactCurve.weightAndDerivative(
                self.max_num_ctrl_pts, i_vals, self.support_rad, self.k, True,
                True, self.half_order
            )
        else:
            umi = self.max_num_ctrl_pts - i_vals
            unnormed_weights = CinpactCurve.getBCinpactWeights(
                umi, self.half_order, self.support_rad_sq, self.support_rad,
                self.k
            )
            self.unnormed_filter_info = (unnormed_weights, )
        self.unnormed_filter_len = len(self.unnormed_filter_info[0])


        num_input_bound = self.max_num_input_data + 1
        last_num_ctrl_pts = -1
        self.last_cinpact_logic = None
        for num_inputs in range(self.min_inputs_req, num_input_bound):
            num_ctrl_pts = min(num_inputs, self.max_num_ctrl_pts)
            if num_ctrl_pts != last_num_ctrl_pts:
                last_num_ctrl_pts = num_ctrl_pts
                self.last_cinpact_logic = CinpactLogic(
                    num_ctrl_pts, True, self.support_rad, self.k
                )
            self.cases.append(self.createCase(
                num_inputs, num_ctrl_pts, self.modes
            ))



    
