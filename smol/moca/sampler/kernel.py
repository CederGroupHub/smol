"""Implementation of MCMC transition kernel classes.

A kernel essentially is an implementation of the MCMC algorithm that is used
to generate states for sampling an MCMC chain.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod
from math import log, sqrt
from random import random
from collections import defaultdict
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

    def __init__(self, ensemble, temperature, step_type, nwalkers, *args,
                 **kwargs):
        """Initialize MCMCKernel.

        Args:
            ensemble (Ensemble):
                An Ensemble instance to obtain the feautures and parameters
                used in computing log probabilities.
            temperature (float):
                Temperature at which the MCMC sampling will be carried out.
            step_type (str): optional
                String specifying the MCMC step type.
            nwalkers (int): optional
                Number of walkers/chains to sampler.
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
        self._nwalkers = nwalkers
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

    def set_aux_state(self, occupancies, *args, **kwargs):
        """Set the auxiliary occupancies from initial or checkpoint values."""
        self._usher.set_aux_state(occupancies, *args, **kwargs)

    @abstractmethod
    def single_step(self, occupancy):
        """Attempt an MCMC step.

        Returns the next occupancies in the chain and if the attempted step was
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

    valid_mcushers = {'flip': 'Flipper', 'swap': 'Swapper'}

    def single_step(self, occupancy):
        """Attempt an MC step.

        Returns the next occupancies in the chain and if the attempted step was
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


class WangLandau(MCMCKernel):
    """
    A Kernel for Wang Landau Sampling.

    Inheritance naming is probably a misnomer, since WL is non-Markovian. But
    alas we can be code descriptivists.
    """

    valid_mcushers = {'flip': 'Flipper', 'swap': 'Swapper'}

    def __init__(self, ensemble, step_type, nwalkers, bin_size, min_energy,
                 max_energy, flatness=0.8, mod_factor=np.e, check_period=1000,
                 fixed_window=False, mod_update=None):
        """Initialize a WangLandau Kernel

        Args:
            ensemble (Ensemble):
                The ensemble object to use to generate samples
            step_type (str):
                An MC step type corresponding to an MCUsher. See valid_mcushers
            nwalkers (int): optional
                Number of walkers/chains to sampler. Default is 1.
            bin_size (float):
                The energy bin size to determine different states.
            min_energy (float):
                The minimum energy to sample. Energy value should be given per
                supercell (i.e. same order as what will be sampled).
            max_energy (float):
                The maximum energy to sample.
            flatness (float): optional
                The flatness factor used when checking histogram flatness.
                Must be between 0 and 1.
            mod_factor (float):
                The modification factor used to update the DOS/entropy.
                Default is e^1.
            check_period (int): optional
                The period in number of steps for the histogram flatness to be
                checked.
            fixed_window (bool): optional
                Whether to update the max of min energy if an energy outside of
                range is sampled. Default False.
            mod_update (Callable): optional
                A function used to update the fill factor when the histogram
                satisfies the flatness criteria. The function must
                monotonically decrease to 1.
        """
        if min_energy > max_energy:
            raise ValueError("min_energy can not be larger than max_energy.")
        elif mod_factor <= 1:
            raise ValueError("mod_factor must be greater than 1.")
        self._mfactor = mod_factor
        self.flatness = flatness
        self.check_period = check_period
        self.range = (min_energy, max_energy)
        self.fixed_window = fixed_window
        self._bin_size = bin_size
        self._update = mod_update if mod_update is not None \
            else lambda f: sqrt(f)

        # default dict of arrays with [energy, DOS, histogram]
        # keys are generated by _get_key method
        self._aux_states = defaultdict(
            lambda: np.array(nwalkers * [[np.inf, 1, 0],]))
        self._prev_states = np.array(nwalkers * [[np.inf, 1, 0],])
        super().__init__(ensemble=ensemble, step_type=step_type, temperature=1,
                         nwalkers=nwalkers)

    def _get_key(self, energy):
        """Get key for _aux_states dict from given energy."""
        return int((energy - self.range[0])//self._bin_size)

    def single_step(self, occupancy):
        """Attempt an MC step.

        Returns the next occupancies in the chain and if the attempted step was
        successful based on the WL algorithm.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            tuple: (acceptance, occupancy, features change, enthalpy change)
        """
        step = self._usher.propose_step(occupancy)
        delta_features = self._feature_change(occupancy, step)
        delta_energy = np.dot(self.natural_params, delta_features)
        # TODO need to make this work for aux states with many walkers...

    def set_aux_state(self, occupancies, *args, **kwargs):
        """Set the auxiliary occupancies based on an occupancy"""
        energies = np.dot(self.feature_fun(occupancies), self.natural_params)
        for i, energy in enumerate(energies):
            key = self._get_key(energy)
            self._aux_states[key][:, 0] = energy
            # update only in corresponding chain
            self._aux_states[key][i, 1] *= self._mfactor
            self._aux_states[key][i, 2] += 1
            self._prev_states[i] = self._aux_states[key][i]

        self._usher.set_aux_state(occupancies, *args, **kwargs)


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
