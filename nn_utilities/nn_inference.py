import typing

import numpy as np
from numpy.typing import NDArray

from motiontools.posefeatures import (
    HypotheticalInputsForNN, getWorldFrameDisplacements
)

def getOutputsNN(model, scaler, hyp_calcer: HypotheticalInputsForNN,
                 last_pts: NDArray, prev_pts: typing.Optional[NDArray] = None,
                 rot_mats: typing.Optional[NDArray] = None):

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
    return final_positions

