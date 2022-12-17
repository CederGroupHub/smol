"""Implementation of Metropolis and UniformlyRandom kernels.

A Metropolis-Hastings kernel and a UniformlyRandom kernel (which can be thought of
as an infinite temperature Metropols kernel).
"""

__author__ = "Luis Barroso-Luque"

from math import log

import numpy as np

from smol.moca.kernel._base import ALL_BIAS, ALL_MCUSHERS, MCKernel, ThermalKernel


class Metropolis(ThermalKernel):
    """A Metropolis-Hastings kernel.

    The classic and nothing but the classic.
    """

    valid_mcushers = ALL_MCUSHERS
    valid_bias = ALL_BIAS

    def single_step(self, occupancy):
        """Attempt an MC step.

        Returns the next occupancies in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            StepTrace
        """
        step = self.mcusher.propose_step(occupancy)
        self._compute_step_trace(occupancy, step)

        log_factor = self.mcusher.compute_log_priori_factor(occupancy, step)
        exponent = -self.beta * self.trace.delta_trace.enthalpy + log_factor
        if self._bias is not None:
            exponent += self.trace.delta_trace.bias

        self.trace.accepted = np.array(
            True if exponent >= 0 else exponent > log(self._rng.random())
        )

        if self.trace.accepted:
            for tup in step:
                occupancy[tup[0]] = tup[1]
            self.mcusher.update_aux_state(step)
        self.trace.occupancy = occupancy

        return self.trace


class UniformlyRandom(MCKernel):
    """A Kernel that accepts all proposed steps.

    This kernel samples the random limit distribution where all states have the
    same probability (corresponding to an infinite temperature). If a
    bias is added then the corresponding distribution will be biased
    accordingly.
    """

    valid_mcushers = ALL_MCUSHERS
    valid_bias = ALL_BIAS

    def single_step(self, occupancy):
        """Attempt an MCMC step.

        Returns the next occupancies in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            StepTrace
        """
        step = self.mcusher.propose_step(occupancy)
        self._compute_step_trace(occupancy, step)

        exponent = self.mcusher.compute_log_priori_factor(occupancy, step)
        if self._bias is not None:
            exponent += self.trace.delta_trace.bias

        self.trace.accepted = np.array(
            True if exponent >= 0 else exponent > log(self._rng.random())
        )

        if self.trace.accepted:
            for tup in step:
                occupancy[tup[0]] = tup[1]
            self.mcusher.update_aux_state(step)

        self.trace.occupancy = occupancy
        return self.trace
