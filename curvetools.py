import numpy as np

def applyMaskToPts(pts, mask):
    return np.einsum('i,ij->j', mask, pts)

def convolveFilter(filter, data: np.ndarray):
    """
    Apply a 1D convolution filter to a set of multi-dimensional points.
    The filter is applied to a sliding window of data rows. 

    Parameters:
        filter (array-like): 1D array acting as convolution filter.
        data (np.ndarray): 2D array of points, where each row is a point.

    Returns:
        np.ndarray: The convolution result.
    """
    # Note: Function was originally generated with ChatGPT but was verified
    # (and heavily modified/documented) manually.
    # TODO: Uncomment this once slight efficiency difference is known to no
    # longer be worth it!
    # assert len(data) >= len(filter) and len(filter) > 0, \
    #     "Must have len(data) >= len(filter) > 0!"

    # Calculate the number of sliding windows for a step size of 1.
    num_windows = len(data) + 1 - len(filter)
    # The shape of the strided array view we will create, where each element
    # along axis 0 is one instance of the sliding window. 
    stride_shape = (num_windows, len(filter), data.shape[1])
    # The strides, in bytes, for each axis of our strided view of the data.
    # In case of non-contiguous data (e.g., a view like x[::2] for some x),
    # the numpy documentation STRONGLY recommends using the original
    # data.strides values to construct this set of strides.
    # My testing confirms that this works on a variety of non-contiguous inputs.
    strides = (data.strides[0], data.strides[0], data.strides[1])
    # Create a view that offers multiple sliding window instances into the data.
    # More information about as_strided can be found in its documentation and at
    # this helpful post: https://jessicastringham.net/2017/12/31/stride-tricks/
    strided = np.lib.stride_tricks.as_strided(
        data, stride_shape, strides, writeable=False
    )
    # Apply the filter. Here, b is the window index, i the data row index, and j
    # the vector column index.
    return np.einsum('i,bij->bj', filter, strided)

# Tangents are calculated as the derivatives of the interpolating polynomial
# for five consecutive points containing a given point.
def tangentsFromPoints(pVals, shouldNormalize: bool = True):
    numPts = pVals.shape[0]

    tangentVals = np.empty(pVals.shape)

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

        if shouldNormalize:
            estT = estT/np.linalg.norm(estT)
        else:
            estT /= 12.0 # Omitted denominator from above calculations.
        tangentVals[i] = estT
        
    return tangentVals