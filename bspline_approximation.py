import numpy as np
import bspline
import typing

def bSplineFittingMat(numCtrlPts, order, numInputPts, uVals, knotVals):
    mat = np.zeros((numInputPts, numCtrlPts))

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

class BSplineFitCalculator:
    class BSplineFittingCase:
        def __init__(self, degree, num_input_pts, max_num_ctrl_pts):
    
            num_u_vals = num_input_pts + 1 # One output point.

            self.num_ctrl_pts = min(num_input_pts, max_num_ctrl_pts)
    
            uInterval = (degree, self.num_ctrl_pts) 

            # Just the number of pts in any B-Spline; nothing fitting-specific.
            self.knot_list = np.arange(self.num_ctrl_pts + degree + 1)

            self.data_u_vals = np.linspace(uInterval[0], uInterval[1], num_u_vals)

            matA = bSplineFittingMat(
                self.num_ctrl_pts, degree + 1, num_input_pts, self.data_u_vals,
                self.knot_list
            )

            matAtA = matA.transpose() @ matA
            self.pseudoInv = np.linalg.inv(matAtA) @ matA.transpose()

    def __init__(self, spline_degree, max_num_ctrl_pts, max_num_input_data):
        if (max_num_input_data < spline_degree + 1):
            raise Exception("Need at least order=k=(degree + 1) input pts for calc!")
        if max_num_ctrl_pts > max_num_input_data:
            raise Exception("Need at least as many input points as control points!")
        if max_num_ctrl_pts < spline_degree + 1:
            raise Exception("Need at least order=k=(degree + 1) control points!")


        self.cases: typing.List[BSplineFitCalculator.BSplineFittingCase] = []
        self.spline_degree = spline_degree
        self.max_num_ctrl_pts = max_num_ctrl_pts
        self.max_num_input_data = max_num_input_data

        for num_inputs in range(spline_degree + 1, max_num_input_data + 1):
            self.cases.append(BSplineFitCalculator.BSplineFittingCase(
                spline_degree, num_inputs, max_num_ctrl_pts
            ))
    

    def fitAllData(self, all_input_pts: np.ndarray):
        if all_input_pts.ndim > 2:
            str_dim = str(all_input_pts.ndim)
            raise Exception("Array dimension " + str_dim + " not supported for B-Spline fitting!")
        
        num_spline_preds = (len(all_input_pts) - 1) - (self.spline_degree)
        
        # The below is just to accomodate either an array of floats or of vecs.
        output_shape = (num_spline_preds,) + all_input_pts.shape[1:]
        spline_preds = np.empty(output_shape)

        for last_input_ind in range(self.spline_degree, len(all_input_pts) - 1):
            # How many predictions we've calculated so far.
            prediction_num = last_input_ind - (self.spline_degree)
            
            startInd = max(0, last_input_ind - self.max_num_input_data + 1)
            
            ptsToFit = all_input_pts[startInd:(last_input_ind + 1)]

            case = self.cases[min(prediction_num, len(self.cases) - 1)]

            # Note: Using pseudoinverse is way faster than calling linalg.lstsqr
            # each iteration and gives results that, so far, seem identical.
            ctrlPts = (case.pseudoInv @ ptsToFit)

            next_spline_pt = bspline.bSplineInner(
                case.data_u_vals[-1], self.spline_degree + 1,
                case.num_ctrl_pts - 1, ctrlPts, case.knot_list
            )
            spline_preds[prediction_num] = next_spline_pt
        
        return spline_preds
            
    def getCtrlPts(self, in_data):
        case = self.cases[-1]

        if len(in_data) != len(case.data_u_vals) - 1:
            raise NotImplementedError("Only handles max input count for now!")

        return (case.pseudoInv @ in_data)