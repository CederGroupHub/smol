"""Implementation of Metropolis and UniformlyRandom kernels.

A simple Metropolis-Hastings kernel and a Multicell Metropolis kernel that can hop
among different supercell shapes.
"""

__author__ = "Luis Barroso-Luque"

from math import log

import numpy as np

from smol.moca.kernel.base import (
    ALL_BIAS,
    ALL_MCUSHERS,
    MCKernel,
    MulticellKernel,
    ThermalKernelMixin,
)


class MetropolisAcceptMixin:
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


class Metropolis(MetropolisAcceptMixin, ThermalKernelMixin, MCKernel):
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
        **kwargs,
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
            **kwargs,
        )


class MulticellMetropolis(MetropolisAcceptMixin, ThermalKernelMixin, MulticellKernel):
    """A Multicell Metropolis Hastings kernel.

    This is useful for example when one wants to sample over different supercell shapes,
    i.e. for generating special quasi-random structures.

    The ensembles in all kernels must have supercells of the same size, only the shape
    can change. All ensembles must be created from the same Hamiltonian. And all kernels
    must be of the same kind.

    There is no requirement for the individual kernels to have the same step type, bias,
    or any other setup parameter, which allows very flexible sampling strategies.
    However, users are responsible for ensuring that the overall sampling strategy is
    correct! (i.e. satisfies balance or detailed balance).
    """

    valid_mcushers = None
    valid_bias = None

    def __init__(
        self,
        mckernels,
        temperature,
        kernel_probabilities=None,
        kernel_hop_periods=5,
        kernel_hop_probabilities=None,
        seed=None,
        **kwargs,
    ):
        """Initialize MCKernel.

        Args:
            mckernels (list of MCKernels):
                a list of MCKernels instances to obtain the features and parameters
                used in computing log probabilities.
            temperature (float):
                Temperature that kernel hops are attempted at.
            seed (int): optional
                non-negative integer to seed the PRNG
            kernel_probabilities (Sequence of floats): optional
                Probability for choosing each of the supplied kernels
            kernel_hop_periods (int or Sequence of ints): optional
                number of steps between kernel hop attempts.
            kernel_hop_probabilities (Sequence of floats): optional
                probabilities for choosing each of the specified hop periods.
        """
        if not all(isinstance(kernel, Metropolis) for kernel in mckernels):
            raise ValueError("All kernels must be of type Metropolis")

        # order of arguments here matters based on the MRO, temperature must be first
        super().__init__(
            temperature,
            mckernels,
            kernel_probabilities=kernel_probabilities,
            kernel_hop_periods=kernel_hop_periods,
            kernel_hop_probabilities=kernel_hop_probabilities,
            seed=seed,
            **kwargs,
        )

    @ThermalKernelMixin.temperature.setter
    def temperature(self, temperature):
        """Set temperature and update all kernels."""
        self.trace.temperature = np.array(temperature, dtype=np.float64)
        self.beta = 1.0 / (self.kB * temperature)
        for kernel in self.mckernels:
            kernel.temperature = temperature
