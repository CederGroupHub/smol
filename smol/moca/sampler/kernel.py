"""Implementation of MCMC transition kernel classes.

A kernel essentially is an implementation of the MCMC algorithm that is used
to generate states for sampling an MCMC chain.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod
from math import log
from random import random
import numpy as np

from smol.constants import kB
from smol.utils import derived_class_factory
from smol.moca.sampler.mcusher import mcusher_factory


class MCMCKernel(ABC):
    """Abtract base class for transition kernels.

    A kernel is used to implement a specific MCMC algorithm used to sampler
    the ensemble classes. For an illustrtive example of how to derive from this
    and write a specific sampler see the MetropolisSampler.
    """

    valid_mcushers = None  # set this in derived kernels

    def __init__(self, ensemble, temperature, step_type, *args, **kwargs):
        """Initialize MCMCKernel.

        Args:
            ensemble (Ensemble):
                An Ensemble instance to obtain the feautures and parameters
                used in computing log probabilities.
            temperature (float):
                Temperature at which the MCMC sampling will be carried out.
            step_type (str): optional
                String specifying the MCMC step type, e.g. 'flip', 'swap'
            args:
                positional arguments to instantiate the mcusher for the
                corresponding step size.
            kwargs:
                Keyword arguments to instantiate the mcusher for the
                corresponding step size.
        """
        self.natural_params = ensemble.natural_parameters
        self.feature_fun = ensemble.compute_feature_vector
        self._feature_change = ensemble.compute_feature_vector_change
        self.temperature = temperature
        try:
            self._usher = mcusher_factory(self.valid_mcushers[step_type],
                                          ensemble.sublattices,
                                          *args, **kwargs)
        except KeyError:
            raise ValueError(f"Step type {step_type} is not valid for a "
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

    def set_aux_state(self, state, **kwargs):
        """Set the auxiliary state from a checkpoint values."""
        pass

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
                      'table_swap': 'SublatticeSwapper'}

    def single_step(self, occupancy):
        """Attempt an MC step.

        Returns the next state in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            tuple: (acceptance, occupancy, features change, enthalpy change)
        """
        step = self._usher.propose_step(occupancy)
        delta_features = self._feature_change(occupancy, step)
        delta_enthalpy = np.dot(self.natural_params, delta_features)
        accept = (True if delta_enthalpy <= 0
                  else -self.beta * delta_enthalpy > log(random()))
        if accept:
            for f in step:
                occupancy[f[0]] = f[1]
            self._usher.update_aux_state(step)

        return accept, occupancy, delta_enthalpy, delta_features

def mcmckernel_factory(kernel_type, ensemble, temperature, step_type,
                       *args, **kwargs):
    """Get a MCMC Kernel from string name.

    Args:
        kernel_type (str):
            String specifying step to instantiate.
        ensemble (Ensemble)
            An Ensemble object to create the MCMC kernel from.
        temperature (float)
            Temperature at which the MCMC sampling will be carried out.
        step_type (str):
            String specifying the step type (ie key to mcusher type)
        *args:
            Positional arguments passed to class constructor
        **kwargs:
            Keyword arguments passed to class constructor

    Returns:
        MCMCKernel: instance of derived class.
    """
    return derived_class_factory(kernel_type.capitalize(), MCMCKernel,
                                 ensemble, temperature, step_type,
                                 *args, **kwargs)
