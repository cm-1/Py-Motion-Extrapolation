from abc import ABC, abstractmethod
# from dataclasses import dataclass
import typing
# from enum import Enum

import numpy as np

import bspline
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

def fitBSpline(numCtrlPts, order, ptsToFit, uVals, knotVals, numUnknownPts = 0):
    if (len(ptsToFit) + numUnknownPts) != len(uVals):
        raise Exception(
            "Number of u values does not match number of known + unknown points!"
        )

    mat = bSplineFittingMat(numCtrlPts, order, len(ptsToFit), uVals, knotVals)

    fitted_ctrl_pts = np.linalg.lstsq(mat, ptsToFit, rcond = None)[0]
    return fitted_ctrl_pts

# class SplinePredictionMode(Enum):
#     EXTRAPOLATE = 1
#     SMOOTH = 2
#     CONST_ACCEL = 3

class SplineFiltersStruct:
    def __init__(self, ctrlPtsToLastFilter: np.ndarray,
                 ctrlPtsToDerivFilter: np.ndarray,
                 ctrlPtsTo2ndDerivFilter: np.ndarray):
        self.ctrlPtsToLastFilter = ctrlPtsToLastFilter.reshape(1, -1)
        self.ctrlPtsToDerivFilter = ctrlPtsToDerivFilter.reshape(1, -1)
        self.ctrlPtsTo2ndDerivFilter = ctrlPtsTo2ndDerivFilter.reshape(1, -1)

class ABCSplineFittingCase(ABC):
    def __init__(self, degree, num_inputs, num_outputs, max_num_ctrl_pts,
                 precalced_filters: SplineFiltersStruct):

        num_u_vals = num_inputs + num_outputs

        self.num_ctrl_pts = min(num_inputs, max_num_ctrl_pts)

        uInterval = (degree, self.num_ctrl_pts) 

        self.data_u_vals = np.linspace(uInterval[0], uInterval[1], num_u_vals)
        self.data_u_step = self.data_u_vals[1] - self.data_u_vals[0]

        self.degree = degree
        self.num_inputs = num_inputs
        matA = self._getFittingMat()

        matAtA = matA.transpose() @ matA
        self.pseudoInv = np.linalg.inv(matAtA) @ matA.transpose()

        lastPtFilterMat = precalced_filters.ctrlPtsToLastFilter
        pseudoInvLastRows = self.pseudoInv[-lastPtFilterMat.shape[1]:]
        inputsToLastPtFilterRowVec = lastPtFilterMat @ pseudoInvLastRows
        self.inputsToLastPtFilter = inputsToLastPtFilterRowVec.flatten()

        ctrlToVelFilter = precalced_filters.ctrlPtsToDerivFilter
        ctrlToAccFilter = precalced_filters.ctrlPtsTo2ndDerivFilter

        assert ctrlToVelFilter.shape[1] == ctrlToAccFilter.shape[1], \
        "Spline velocity and acceleration filters must have same shape!"

        velAccPseudoInvRows = self.pseudoInv[-ctrlToVelFilter.shape[1]:]
        velFilter = (ctrlToVelFilter @ velAccPseudoInvRows).flatten()
        accFilter = (ctrlToAccFilter @ velAccPseudoInvRows).flatten()

        # From the perspective of the "Spline", each time step is not 1 unit,
        # but instead the distance between u values we chose for our points when
        # doing the least-squares fitting. Because it is simpler to, e.g., have
        # consecutive integer knots than consecutive integer sample locations,
        # the result is that the space between these sampled u values, and thus 
        # the "time", will not be 1. So, the time step we multiply the velocity
        # by to obtain displacement will be this u sample step.
        velFilter *= self.data_u_step # delta_time*velocity
        
        # Since our time step is not 1, we must explicity square it.
        accFilter *= self.data_u_step**2

        self.inputsToConstAccelFilter = velFilter + 0.5 * accFilter

    @abstractmethod
    def _getFittingMat(self) -> np.ndarray:
        ...
    
class BSplineFittingCase(ABCSplineFittingCase):
    def _getFittingMat(self):
        # Just the number of pts in any B-Spline; nothing fitting-specific.
        knot_list = np.arange(self.num_ctrl_pts + self.degree + 1)
        matA = bSplineFittingMat(
            self.num_ctrl_pts, self.degree + 1, self.num_inputs, self.data_u_vals,
            knot_list
        )
        return matA


class ABCSplineFitCalculator(ABC):

    # This might not need to be its own method. I was thinking of overriding it
    # in subclasses, but then decided against it.
    @staticmethod
    def get_min_inputs_req(degree: float):
        return int(np.ceil(degree)) + 1

    def __init__(self, spline_degree, max_num_ctrl_pts, max_num_input_data):
        if (max_num_input_data < spline_degree + 1):
            raise Exception("Need at least order=k=(degree + 1) input pts for calc!")
        if max_num_ctrl_pts > max_num_input_data:
            raise Exception("Need at least as many input points as control points!")
        if max_num_ctrl_pts < spline_degree + 1:
            raise Exception("Need at least order=k=(degree + 1) control points!")


        self.fit_cases: typing.List[ABCSplineFittingCase] = []
        self.smooth_cases: typing.List[ABCSplineFittingCase] = []
        
        self.spline_degree: float = spline_degree
        self.max_num_ctrl_pts: int = max_num_ctrl_pts
        self.max_num_input_data: int = max_num_input_data

        self.ctrlToLastPtFilter = self._getLastPtFilter()
        derivFilters = self._get1st2ndDerivativeFilters()
        self.ctrlToDerivFilter = derivFilters[0]
        self.ctrlTo2ndDerivFilter = derivFilters[1]
        self.filter_row_mats = SplineFiltersStruct(
            self.ctrlToLastPtFilter, derivFilters[0], derivFilters[1]
        )
        self.min_inputs_req = self.get_min_inputs_req(spline_degree)
        for num_inputs in range(self.min_inputs_req, max_num_input_data + 1):
            self.fit_cases.append(self.createCase(num_inputs, 1))
            self.smooth_cases.append(self.createCase(num_inputs, 0))

    @abstractmethod
    def createCase(self, num_inputs: int, num_outputs: int) -> ABCSplineFittingCase:
        ...

    @abstractmethod
    def _getLastPtFilter(self) -> np.ndarray:
        ...

    @abstractmethod
    def _get1st2ndDerivativeFilters(self) -> ndarray_couple_type:
        ...

    def fitAllData(self, all_input_pts: np.ndarray):
        if all_input_pts.ndim > 2:
            str_dim = str(all_input_pts.ndim)
            raise Exception("Array dimension " + str_dim + " not supported for \
                            Spline fitting!")
        
        num_spline_preds = len(all_input_pts) - self.min_inputs_req 
                                    
        # The below is just to accomodate either an array of floats or of vecs.
        output_shape = (num_spline_preds,) + all_input_pts.shape[1:]
        spline_preds = np.empty(output_shape)

        # How many predictions we've calculated so far.
        prediction_num = 0
            
        min_input_ind = self.min_inputs_req - 1
        for last_input_ind in range(min_input_ind, self.max_num_input_data - 1):
            case = self.fit_cases[prediction_num]
            filter_len = len(case.inputsToLastPtFilter)
            input_start = (last_input_ind + 1) - filter_len

            ptsToFit = all_input_pts[input_start:(last_input_ind + 1)]

        
            spline_preds[prediction_num] = curvetools.applyMaskToPts(
                ptsToFit, case.inputsToLastPtFilter
            )

            prediction_num += 1

        main_filter = self.fit_cases[-1].inputsToLastPtFilter
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

            max_case_ind = len(self.smooth_cases) - 1
            inputs_beyond_min_req = available_inputs - (self.spline_degree + 1)
            case = self.smooth_cases[min(max_case_ind, inputs_beyond_min_req)]

            pseudoInv_measured = case.pseudoInv[:, :-1]
            ctrlPts = pseudoInv_measured @ ptsToFit_measured
            pseudoInv_pred = case.pseudoInv[:, -1:]
            ctrlPts += pseudoInv_pred @ ptToFit_pred

            # The documentation for np.dot(a, b) states that if b is 1D like our
            # filter, then np.dot() will do a weighted sum along a's last axis.
            spline_smooths[pred_ind] = np.dot(
                ctrlPts[-len(self.ctrlToLastPtFilter):].transpose(),
                self.ctrlToLastPtFilter
            ).flatten()

        return spline_smooths

    def constantAccelPreds(self, all_input_pts: np.ndarray):
        if all_input_pts.ndim > 2:
            str_dim = str(all_input_pts.ndim)
            raise Exception("Array dimension " + str_dim + " not supported for Spline smoothing!")

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
            case = self.smooth_cases[inputs_beyond_min_req]
            filter_len = len(case.inputsToLastPtFilter)

            startInd = available_inputs - filter_len
            
            ptsToFit = all_input_pts[startInd:available_inputs]

            delta = curvetools.applyMaskToPts(
                ptsToFit, case.inputsToConstAccelFilter
            )

            spline_preds[pred_ind] = all_input_pts[pred_ind] + delta

        main_filter = self.smooth_cases[-1].inputsToConstAccelFilter
        main_filter_len = len(main_filter)
        len_diff = self.max_num_input_data - main_filter_len

        convolution_delta = curvetools.convolveFilter(
            main_filter, all_input_pts[len_diff:-1] 
        )

        main_pred_start = self.max_num_input_data - 1
        main_start_pts = all_input_pts[main_pred_start:-1]
        spline_preds[main_pred_start:] = main_start_pts + convolution_delta
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


    def createCase(self, num_inputs, num_outputs):
        return BSplineFittingCase(
            self.spline_degree, num_inputs, num_outputs, self.max_num_ctrl_pts,
            self.filter_row_mats
        )

    def _getLastPtFilter(self) -> np.ndarray:
        return BSplineFitCalculator.get_deg_ctrl_pt_filter(self.spline_degree)

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
        case = self.fit_cases[-1]

        if len(in_data) != len(case.data_u_vals) - 1:
            raise NotImplementedError("Only handles max input count for now!")

        return (case.pseudoInv @ in_data)
    