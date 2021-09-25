"""Implementation of MCMC transition kernel classes.

A kernel essentially is an implementation of the MCMC algorithm that is used
to generate states for sampling an MCMC chain.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod
from math import log
from random import random
import numpy as np
import time

from smol.constants import kB
from smol.utils import derived_class_factory
from smol.moca.sampler.mcusher import mcusher_factory
from smol.moca.sampler.bias import mcbias_factory


class MCMCKernel(ABC):
    """Abtract base class for transition kernels.

    A kernel is used to implement a specific MCMC algorithm used to sampler
    the ensemble classes. For an illustrtive example of how to derive from this
    and write a specific sampler see the MetropolisSampler.
    """

    valid_mcushers = None  # set this in derived kernels
    valid_bias = None  # set this in derived kernels

    def __init__(self, ensemble, temperature, step_type, bias_type,
                 *args, **kwargs):
        """Initialize MCMCKernel.

        Args:
            ensemble (Ensemble):
                An Ensemble instance to obtain the feautures and parameters
                used in computing log probabilities.
            temperature (float):
                Temperature at which the MCMC sampling will be carried out.
            step_type (str):
                String specifying the MCMC step type.
            bias_type (str|List[str]):
                String specifying the bias term type. If null, no bias sampling
                will be used.
            args:
                positional arguments to instantiate the mcusher for the
                corresponding step size.
            kwargs:
                Keyword arguments to instantiate the mcusher for the
                corresponding step size, or to instantiate the bias term.
        Note that passing values in args will not be used toinstantiate
        bias terms, so when you intend to pass parameters tobias terms,
        be sure to specify argument name so it goes into kwargs.
        """
        self.natural_params = ensemble.natural_parameters
        self.feature_fun = ensemble.compute_feature_vector
        self._feature_change = ensemble.compute_feature_vector_change
        self.temperature = temperature

        try:
            if step_type in ['table-flip', 'subchain-walk']:
                sublattices = ensemble.all_sublattices
            else:
                sublattices = ensemble.sublattices

            self._usher = mcusher_factory(self.valid_mcushers[step_type],
                                          sublattices,
                                          *args, **kwargs)
        except KeyError:
            raise ValueError(f"Step type {step_type} is not valid for a "
                             f"{type(self)} MCMCKernel.")

        try:
            self._bias = mcbias_factory(self.valid_bias[bias_type],
                                        ensemble.all_sublattices,
                                        **kwargs)
        except KeyError:
            raise ValueError(f"Step type {bias_type} is not valid for a "
                             f"{type(self)} MCMCKernel.")

    @property
    def temperature(self):
        """Get the temperature of kernel."""
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        """Set the temperature and beta accordingly."""
        self._temperature = temperature
        self.beta = 1.0 / (kB * temperature)

    def set_aux_state(self, state, *args, **kwargs):
        """Set the auxiliary state from initial or checkpoint values."""
        self._usher.set_aux_state(state, *args, **kwargs)

    @abstractmethod
    def single_step(self, occupancy):
        """Attempt an MCMC step.

        Returns the next state in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            tuple: (acceptance, occupancy, enthalpy change, features change)
        """
        return tuple()


class Metropolis(MCMCKernel):
    """A Metropolis-Hastings kernel.

    The classic and nothing but the classic.
    """

    valid_mcushers = {'flip': 'Flipper', 'swap': 'Swapper',
                      'table-flip': 'Tableflipper',
                      'subchain-walk': 'Subchainwalker'}
    valid_bias = {'null': 'Nullbias',
                  'square-charge': 'Squarechargebias',
                  'square-comp-constraint': 'Squarecompconstraintbias'}

    def single_step(self, occupancy):
        """Attempt an MC step.

        Returns the next state in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            tuple: (acceptance, occupancy, bias, dt, features change,
                    enthalpy change)
        """
        cur_time = time.time()
        step = self._usher.propose_step(occupancy)
        factor = self._usher.compute_a_priori_factor(occupancy, step)
        delta_features = self._feature_change(occupancy, step)
        delta_enthalpy = np.dot(self.natural_params, delta_features)
        delta_bias = self._bias.compute_bias_change(occupancy, step)
        bias = self._bias.compute_bias(occupancy)
        accept = ((-self.beta * delta_enthalpy - delta_bias) +
                  log(factor)
                  > log(random()))
        if accept:
            for f in step:
                occupancy[f[0]] = f[1]
            self._usher.update_aux_state(step)
            bias += delta_bias

        dt = time.time()-cur_time
        # Record the current bias term.
        return (accept, occupancy, bias, dt, delta_enthalpy,
                delta_features)


def mcmckernel_factory(kernel_type, ensemble, temperature, step_type,
                       bias_type, *args, **kwargs):
    """Get a MCMC Kernel from string name.

    Args:
        kernel_type (str):
            String specifying step to instantiate.
        ensemble (Ensemble)
            An Ensemble object to create the MCMC kernel from.
        temperature (float)
            Temperature at which the MCMC sampling will be carried out.
        step_type (str):
            String specifying the step type (ie key to mcusher type).
        bias_type (str):
            String specifying MCMCBias type.
        *args:
            Positional arguments passed to class constructor
        **kwargs:
            Keyword arguments passed to class constructor

    Returns:
        MCMCKernel: instance of derived class.
    """
    return derived_class_factory(kernel_type.capitalize(), MCMCKernel,
                                 ensemble, temperature, step_type, bias_type,
                                 *args, **kwargs)
