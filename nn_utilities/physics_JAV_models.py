import numpy as np
from abc import ABC

class ConstFirstMultipliersModel(ABC):
    def __init__(self, num_of_1_multipliers: int):
        self._num_of_1_multipliers = num_of_1_multipliers

    def predict(self, _, **kwargs):
        ret_val = np.zeros((1, 12))
        ret_val[:, :self._num_of_1_multipliers] = 1.0
        return ret_val

class ZeroMultipliersModel(ConstFirstMultipliersModel):
    def __init__(self):
        super(ZeroMultipliersModel, self).__init__(0)

class ConstVelMultipliersModel(ConstFirstMultipliersModel):
    def __init__(self):
        super(ConstVelMultipliersModel, self).__init__(1)

class ConstAccMultipliersModel(ConstFirstMultipliersModel):
    def __init__(self):
        super(ConstAccMultipliersModel, self).__init__(3)

class ConstJerkMultipliersModel(ConstFirstMultipliersModel):
    def __init__(self):
        super(ConstJerkMultipliersModel, self).__init__(6)
