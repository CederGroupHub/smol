"""Implementation of MCMC transition kernel classes.

A kernel essentially is an implementation of the MCMC algorithm that is used
to generate states for sampling an MCMC chain.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod
from math import log, sqrt
from random import random
from collections import defaultdict
from operator import itemgetter
from itertools import repeat
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

    def __init__(self, ensemble, step_type, temperature, nwalkers=1, *args,
                 **kwargs):
        """Initialize MCMCKernel.

        Args:
            ensemble (Ensemble):
                An Ensemble instance to obtain the feautures and parameters
                used in computing log probabilities.
            step_type (str): optional
                String specifying the MCMC step type.
            temperature (float):
                Temperature at which the MCMC sampling will be carried out.
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

    def iter_steps(self, occupancies):
        """Iterate steps for each walker over an array of occupancies."""
        for occupancy in occupancies:
            yield self.single_step(occupancy)

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

        accept = True if delta_enthalpy <= 0 \
            else -self.beta * delta_enthalpy > log(random())

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

    def __init__(self, ensemble, step_type, bin_size, min_energy,
                 max_energy, flatness=0.8, mod_factor=1.0, check_period=1000,
                 fixed_window=False, mod_update=None, nwalkers=1):
        """Initialize a WangLandau Kernel

        Args:
            ensemble (Ensemble):
                The ensemble object to use to generate samples
            step_type (str):
                An MC step type corresponding to an MCUsher. See valid_mcushers
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
                Whether to accept energies below the minimum and above the
                maximum energy provided. Default is False
            mod_update (Callable): optional
                A function used to update the fill factor when the histogram
                satisfies the flatness criteria. The function is used to update
                the entropy value for an energy level and so must monotonically
                decrease to 0.
            nwalkers (int): optional
                Number of walkers/chains to sampler. Default is 1.
        """
        if min_energy > max_energy:
            raise ValueError("min_energy can not be larger than max_energy.")
        elif mod_factor <= 0:
            raise ValueError("mod_factor must be greater than 0.")
        elif fixed_window is True:
            raise NotImplementedError(
                "fixed_window=True is not implemented."
                " If you need this, consider doing a PR.")
        
        self.flatness = flatness
        self.check_period = check_period
        self.fixed_window = fixed_window
        self._mfactors = np.array(nwalkers * [mod_factor, ])
        self._window = (min_energy, max_energy)
        self._bin_size = bin_size
        self._update_fun = mod_update if mod_update is not None \
            else lambda f: f / 2.0

        # default dict of arrays with [entropy, histogram]
        # keys are generated by _get_key method
        nbins = int((max_energy - min_energy) // bin_size)
        self._aux_states = defaultdict(
            lambda: np.array(nwalkers * [[0.0, 0.0], ]),
            {
                bin_num: np.array(nwalkers * [[0.0, 0.0], ])
                for bin_num in range(nbins)
            }
        )
        self._current_energy = np.zeros(nwalkers)
        self._counter = 0  # count steps to check flatness at given intervals

        # a little annoying to keep the temperature there...
        super().__init__(ensemble=ensemble, step_type=step_type, temperature=1,
                         nwalkers=nwalkers)

    @property
    def energy_levels(self):
        """Get energies that have been visited or are inside initial window."""
        energies = [self._get_bin_energy(b) for b in
                    sorted(self._aux_states.keys())]
        return np.array(energies)

    @property
    def entropy(self):
        """Return entropy values for each walker."""
        return np.array(
            [state[:, 0] for _, state in
             sorted(self._aux_states.items(), key=itemgetter(0))]).T

    @property
    def dos(self):
        """Get current DOS for each walker"""
        # TODO look into handling this to avoid overflow issues...
        return np.exp(self.entropy)

    @property
    def histograms(self):
        """Get current histograms for each walker."""
        return np.array(
            [state[:, 1] for _, state in
             sorted(self._aux_states.items(), key=itemgetter(0))]).T

    @property
    def mod_factors(self):
        """Get the entropy modification factors"""
        return self._mfactors

    def _get_bin(self, energy):
        """Get the histogram bin for _aux_states dict from given energy."""
        return int((energy - self._window[0]) // self._bin_size)

    def _get_bin_energy(self, bin_number):
        """Get the bin energy for _aux_states dict from given energy."""
        return self._window[0] + bin_number * self._bin_size

    def iter_steps(self, occupancies):
        """Iterate steps for each walker over an array of occupancies."""
        for walker, occupancy in enumerate(occupancies):
            yield self.single_step(occupancy, walker)

        self._counter += 1
        # check if histograms are flat and reset accordingly
        if self._counter % self.check_period == 0:
            histograms = self.histograms  # cache it
            for i, flat in enumerate(
                    (histograms > self.flatness * histograms.mean(axis=1)).all(axis=1)):  # noqa
                if flat:
                    self._mfactors[i] = self._update_fun(self._mfactors[i])
                    for level in self._aux_states.values():
                        level[i, 1] = 0  # reset histogram

    def single_step(self, occupancy, walker=0):
        """Attempt an MC step.

        Returns the next occupancies in the chain and if the attempted step was
        successful based on the WL algorithm.

        Args:
            occupancy (ndarray):
                encoded occupancy.
            walker (int):
                index of walker to take step.

        Returns:
            tuple: (acceptance, occupancy, features change, enthalpy change)
        """
        energy = self._current_energy[walker]
        bin_num = self._get_bin(energy)
        state = self._aux_states[bin_num][walker]

        step = self._usher.propose_step(occupancy)
        delta_features = self._feature_change(occupancy, step)
        delta_energy = np.dot(self.natural_params, delta_features)
        new_energy = energy + delta_energy
        new_bin_num = self._get_bin(new_energy)
        new_state = self._aux_states[new_bin_num][walker]

        accept = state[0] - new_state[0] >= log(random())
        if accept:
            for f in step:
                occupancy[f[0]] = f[1]
            bin_num = new_bin_num
            self._current_energy[walker] = new_energy
            self._usher.update_aux_state(step)

        mfactor = self._mfactors[walker]
        # update DOS and histogram
        self._aux_states[bin_num][walker][0] += mfactor
        self._aux_states[bin_num][walker][1] += 1

        return accept, occupancy, delta_energy, delta_features

    def set_aux_state(self, occupancies):
        """Set the auxiliary occupancies based on an occupancy"""
        energies = np.dot(
            list(map(self.feature_fun, occupancies)), self.natural_params)
        self._current_energy[:] = energies
        self._usher.set_aux_state(occupancies)


def mcmckernel_factory(kernel_type, ensemble, step_type, *args, **kwargs):
    """Get a MCMC Kernel from string name.

    Args:
        kernel_type (str):
            String specifying step to instantiate.
        ensemble (Ensemble)
            An Ensemble object to create the MCMC kernel from.
        step_type (str):
            String specifying the step type (ie key to mcusher type)
        *args:
            Positional arguments passed to class constructor
        **kwargs:
            Keyword arguments passed to class constructor

    Returns:
        MCMCKernel: instance of derived class.
    """

    kernel_name = ''.join([s.capitalize() for s in kernel_type.split('-')])
    return derived_class_factory(kernel_name, MCMCKernel,
                                 ensemble, step_type, *args, **kwargs)
