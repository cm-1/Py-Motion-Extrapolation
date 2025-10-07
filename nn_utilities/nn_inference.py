import typing

import numpy as np
from numpy.typing import NDArray

from motiontools.posefeatures import (
    HypotheticalInputsForNN, getWorldFrameDisplacements
)

def getOutputsNN(model, scaler, hyp_calcer: HypotheticalInputsForNN,
                 last_pts: NDArray, prev_pts: typing.Optional[NDArray] = None,
                 rot_mats: typing.Optional[NDArray] = None,
                 return_inputs: str = 'none'):
    '''
        For a given set of prior transformations, gets the world frame vec3
        positions predicted by a model that outputs vec12 JAV multipliers.

        Parameters:
            model: The model (e.g., a neural net) that outputs the multipliers.
                Any class that has a predict() method that takes in a 2D array
                of prediction data matching the scaler's outputs and produces
                an output array of shape (n, 12) can be used here.
            scaler: A scaler that transforms the "raw" data columns generated.
            hyp_calcer (HypotheticalInputsForNN): Used to calculate the raw
                columns, pre-scaling, for input into the model.
            last_pts (NDArray): Positions at the final timestep pre-prediction.
                If None, it is assumed that hyp_calcer is set up with this
                information already. Defaults to None.
            prev_pts (NDArray): Positions leading up to last_pts.
            rot_mats (NDArray): Rotation matrices up to and including final
                timestep pre-prediction. If None, it is assumed that hyp_calcer
                is set up with this information already. Defaults to None.
            return_inputs (str): Specifies whether to just return predicted
                vec3s ("none"), or to also return scaled ("scaled") or unscaled
                ("unscaled") columns generated as the model's input. Defaults
                to "none".

        Returns:
            NDArray or tuple: Depending on the value of return_inputs, either
                an NDArray of shape (n, 3) of world frame positions predicted
                by the model or else a 2-tuple which contains:    
                    - The above-mentioned vec3s (ndarray).
                    - The data columns generated as model input (ndarray).
        '''

    last_pts = np.atleast_2d(last_pts)
    orig_write_status = last_pts.flags.writeable

    last_pts.setflags(write=False)

    unscaled_cols: NDArray
    jav_mags: NDArray
    w2ls: NDArray
    if prev_pts is None or rot_mats is None:
        if not (prev_pts is None and rot_mats is None):
            raise ValueError("Cannot have just one of pts and mats be None!")
        unscaled_cols, jav_mags, w2ls = hyp_calcer.getHypotheticalCalcs(
            last_pts
        )
    else:
        all_pts = np.concatenate([prev_pts, [last_pts]], axis=0)
        unscaled_cols, jav_mags, w2ls = hyp_calcer.getSeparateCalcs(
            all_pts, rot_mats
        )

    scaled_inputs = scaler.transform(unscaled_cols)
    out_JAV = model.predict(scaled_inputs, batch_size=1024, verbose=0)
    displacements = getWorldFrameDisplacements(jav_mags, out_JAV, w2ls)

    final_positions = last_pts + displacements

    last_pts.setflags(write=orig_write_status)
    if return_inputs == "unscaled":
        return final_positions, unscaled_cols
    elif return_inputs == "scaled":
        return final_positions, scaled_inputs
    elif return_inputs != "none":
        raise ValueError("Invalid return_inputs value of: " + return_inputs)
    return final_positions

