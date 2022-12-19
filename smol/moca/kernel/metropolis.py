"""Implementation of Metropolis and UniformlyRandom kernels.

A Metropolis-Hastings kernel and a UniformlyRandom kernel (which can be thought of
as an infinite temperature Metropols kernel).
"""

__author__ = "Luis Barroso-Luque"

from math import log

import numpy as np

from smol.moca.kernel._base import ALL_BIAS, ALL_MCUSHERS, MCKernel, ThermalKernel


class MetropolisCriterionMixin:
    """Metropolis Criterion Mixin class.

    Simple mixin class impolementing the Metropolis-Hastings criterion for reause in
    MCKernels.

    Must be used in conjunction with inheritance from an MCKernel or derived class.
    """

    def _accept_step(self, occupancy, step):
        """Accept/reject a given step and populate trace and aux states accordingly.

        Args:
            occupancy (ndarray):
                Current occupancy
            step (Sequence of tuple):
                Sequence of tuples of ints representing an MC step
        """
        log_factor = self.mcusher.compute_log_priori_factor(occupancy, step)
        exponent = -self.beta * self.trace.delta_trace.enthalpy + log_factor

        if self._bias is not None:
            exponent += self.trace.delta_trace.bias

        self.trace.accepted = np.array(
            True if exponent >= 0 else exponent > log(self._rng.random())
        )
        return self.trace.accepted


class Metropolis(MetropolisCriterionMixin, ThermalKernel):
    """A Metropolis-Hastings kernel.

    The classic and nothing but the classic.
    """

    valid_mcushers = ALL_MCUSHERS
    valid_bias = ALL_BIAS


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
