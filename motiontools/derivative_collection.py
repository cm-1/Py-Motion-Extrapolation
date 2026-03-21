import typing

import numpy as np
from numpy.typing import NDArray, ArrayLike

class DerivativeCollection:
    def __init__(self, displacements: NDArray, max_derivative_order: int,
                 time_info: typing.Optional[ArrayLike] = 1):
        
        assert max_derivative_order >= 1, "max_derivative_order >= 1 required!"
        
        self.velocities = displacements.copy()

        if time_info is None:
            time_info = 1
        self.dt_not_const: bool = not isinstance(time_info, (int, float))
        self.div_by_time: bool = self.dt_not_const or (time_info != 1)
        
        self.unflat_time_info = time_info
        if self.dt_not_const:
            # Reshape for broadcasting.
            coeff_shape = self.velocities.shape[:-1] + (1, )
            self.unflat_time_info = time_info.reshape(coeff_shape)

        if self.div_by_time:
            time_deltas = time_info
            if self.dt_not_const:
                time_deltas = np.diff(self.unflat_time_info, 1, axis=0)
            self.velocities = displacements / time_deltas

        if max_derivative_order >= 2:
            self.accelerations = self._recursiveDeriv(self.velocities, 2)
        if max_derivative_order >= 3:
            self.jerks = self._recursiveDeriv(self.accelerations, 3)
        if max_derivative_order >= 4:
            self.snaps = self._recursiveDeriv(self.jerks, 4)
        if max_derivative_order >= 5:
            self.crackles = self._recursiveDeriv(self.snaps, 5)
        if max_derivative_order >= 6:
            raise NotImplementedError("Derivative orders >= 6 not supported!")

    def _recursiveDeriv(self, prev_vals: NDArray, deriv_power: int):
        """Compute higher-order derivatives recursively.
        E.g., prev_vals are the last accelerations, deriv_power is 3 
        (for jerk)."""
        ret_val = np.diff(prev_vals, 1, axis=0)
        if self.div_by_time:
            if self.dt_not_const:
                time_div = self.unflat_time_info[deriv_power:] - self.unflat_time_info[:-deriv_power]
                # Do scalar math first for efficiency, as otherwise you perform
                # a division on vec3s and then a mul on vec3s, instead of doing
                # the division on "vec1s". Could use parentheses, but this is
                # more explicit.
                scalars = deriv_power / time_div
                ret_val = scalars * ret_val
            else:
                ret_val = ret_val / self.unflat_time_info
        return ret_val
