"""A uniformly random kernel.

The UniformlyRandom kernel is equivalent to a Metropolis kernel with infinite temperature.
"""

__author__ = "Luis Barroso-Luque"


from math import log

import numpy as np

from smol.moca.kernel.base import ALL_BIAS, ALL_MCUSHERS, MCKernel


class UniformlyRandom(MCKernel):
    """A Kernel that accepts all proposed steps.

    This kernel samples the random limit distribution where all states have the
    same probability (corresponding to an infinite temperature). If a
    bias is added then the corresponding distribution will be biased
    accordingly.
    """

    valid_mcushers = ALL_MCUSHERS
    valid_bias = ALL_BIAS

    def _accept_step(self, occupancy, step):
        exponent = self.mcusher.compute_log_priori_factor(occupancy, step)
        if self._bias is not None:
            exponent += self.trace.delta_trace.bias

        self.trace.accepted = np.array(
            True if exponent >= 0 else exponent > log(self._rng.random())
        )
        return self.trace.accepted
