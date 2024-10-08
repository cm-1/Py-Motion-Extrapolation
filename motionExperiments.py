import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import cinpact

from gtCommon import BCOT_Data_Calculator

bodIndex = 1
seqIndex = 11
skipAmount = 2
calculator = BCOT_Data_Calculator(bodIndex, seqIndex, skipAmount)

translations = calculator.getTranslationsGTNP(True)
rotations = calculator.getRotationsGTNP(True)

numTimestamps = len(translations)
timestamps = np.arange(numTimestamps)

translations_vel = translations[1:-1] - translations[:-2]
rotations_vel = rotations[1:-1] - rotations[:-2]

t_vel_pred = translations[1:-1] + translations_vel
r_vel_pred = rotations[1:-1] + rotations_vel

tx_coords = np.stack((timestamps, translations[:, 0]), axis = -1)
ptx_coords = np.stack((timestamps[2:], t_vel_pred[:, 0]), axis = -1)

v_line_coords = np.hstack((tx_coords[:-2], ptx_coords)).reshape((len(ptx_coords), 2, 2))

# For predictions that require multiple prior points, they may lack a prediction
# for the 2nd point, 3rd point, etc. In this case, the CV algorithm would just
# assume no motion for those frames, so we'll reflect that here as well.
def prependMissingPredictions(values, predictions):
    numMissing = (len(values) - 1) - len(predictions)
    return np.vstack((values[0:numMissing], predictions))

vlc = LineCollection(v_line_coords)


fig, ax = plt.subplots()
ax.add_collection(vlc)
ax.autoscale()
ax.margins(0.1)
plt.show()

