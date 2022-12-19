"""Implementation of Metropolis and UniformlyRandom kernels.

A Metropolis-Hastings kernel and a UniformlyRandom kernel (which can be thought of
as an infinite temperature Metropols kernel).
"""

__author__ = "Luis Barroso-Luque"

from math import log

import numpy as np

from smol.moca.kernel._base import ALL_BIAS, ALL_MCUSHERS, MCKernel, ThermalKernelMixin


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

        if self.bias is not None:
            exponent += self.trace.delta_trace.bias

        self.trace.accepted = np.array(
            True if exponent >= 0 else exponent > log(self._rng.random())
        )
        return self.trace.accepted


class Metropolis(MetropolisCriterionMixin, ThermalKernelMixin, MCKernel):
    """A Metropolis-Hastings kernel.

    The classic and nothing but the classic.
    """

    valid_mcushers = ALL_MCUSHERS
    valid_bias = ALL_BIAS

    def __init__(
        self,
        ensemble,
        step_type,
        temperature,
        *args,
        seed=None,
        bias_type=None,
        bias_kwargs=None,
        **kwargs
    ):
        """Initialize Metropolis Kernel.

        Args:
            ensemble (Ensemble):
                an Ensemble instance to obtain the features and parameters
                used in computing log probabilities.
            step_type (str):
                string specifying the MCMC step type.
            temperature (float):
                temperature at which the MCMC sampling will be carried out.
            args:
                positional arguments to instantiate the MCUsher for the
                corresponding step size.
            kwargs:
                keyword arguments to instantiate the MCUsher for the
                corresponding step size.
        """
        # order of arguments here matters based on the MRO, temperature must be first
        super().__init__(
            temperature,
            ensemble,
            step_type,
            *args,
            seed=seed,
            bias_type=bias_type,
            bias_kwargs=bias_kwargs,
            **kwargs
        )


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
