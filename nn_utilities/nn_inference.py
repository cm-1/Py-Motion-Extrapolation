import numpy as np
from numpy.typing import NDArray

from motiontools.posefeatures import (
    HypotheticalInputsForNN, getWorldFrameDisplacements
)

def getHypotheticalOutputsNN(model, scaler, hyp_calcer: HypotheticalInputsForNN, hypothetical_pts: NDArray):

    hypothetical_pts = np.atleast_2d(hypothetical_pts)
    orig_write_status = hypothetical_pts.flags.writeable

    hypothetical_pts.setflags(write=False)

    unscaled_cols, jav_mags, w2ls = hyp_calcer.calculateInputsForNN(
        hypothetical_pts
    )
    scaled_inputs = scaler.transform(unscaled_cols)
    out_JAV = model.predict(scaled_inputs, batch_size=1024, verbose=0)
    displacements = getWorldFrameDisplacements(jav_mags, out_JAV, w2ls)

    final_positions = hypothetical_pts + displacements

    hypothetical_pts.setflags(write=orig_write_status)
    return final_positions