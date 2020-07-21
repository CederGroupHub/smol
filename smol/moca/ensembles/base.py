"""Abstract Base class for Monte Carlo Ensembles."""

__author__ = "Luis Barroso-Luque"

import random
from copy import deepcopy
import os
import json
from abc import ABC, ABCMeta, abstractmethod
from math import exp
import numpy as np

from smol.constants import kB

# TODO it would be great to use the design paradigm of observers to extract
#  a variety of information during a montecarlo run


class BaseEnsemble(ABC):
    """Abstract Base Class for Monte Carlo Ensembles."""

    def __init__(self, processor, sample_interval, initial_occupancy,
                 sublattices=None, seed=None):
        """Initialize class instance.

        Args:
            processor (Processor):
                A processor that can compute the change in a property given
                a set of flips.
            sample_interval (int):
                interval of steps to save the current occupancy and property
            initial_occupancy (ndarray or list):
                Initial occupancy vector. The occupancy can be encoded
                according to the processor or the species names directly.
            sublattices (dict): optional
                dictionary with keys identifying the active sublattices
                (i.e. "anion" or the allowed species in that sublattice
                "Li+/Vacancy". The values should be a dictionary
                with two items {'sites': array with the site indices for all
                sites corresponding to that sublattice in the occupancy vector,
                'site_space': OrderedDict of allowed species in sublattice}
                All sites in a sublattice need to have the same set of allowed
                species.
            seed (int): optional
                seed for random number generator
        """
        if len(initial_occupancy) != len(processor.structure):
            raise ValueError('The given initial occupancy does not match '
                             'the underlying processor size')

        if isinstance(initial_occupancy[0], str):
            initial_occupancy = processor.encode_occupancy(initial_occupancy)

        if sublattices is None:
            sublattices = {'/'.join(site_space.keys()):
                           {'sites': np.array([i for i, sp in
                                               enumerate(processor.allowed_species)  # noqa
                                               if sp == list(site_space.keys())]),  # noqa
                            'site_space': site_space}
                           for site_space in processor.unique_site_spaces}

        self.processor = processor
        self.sample_interval = sample_interval
        self.num_atoms = len(initial_occupancy)

        self._sublattices = sublattices
        self._active_sublatts = deepcopy(sublattices)
        self._init_occupancy = initial_occupancy
        self._occupancy = self._init_occupancy.copy()
        self._property = processor.compute_property(self._occupancy)
        self._prod_start = 0
        self._step = 0
        self._ssteps = 0
        self._data = []
        self.restricted_sites = []

        # Set and save the seed for random. This allows reproducible results.
        if seed is None:
            seed = random.randint(1, 1E24)

        self._seed = seed
        random.seed(seed)

    @property
    def sublattices(self):
        """Get names of sublattices.

        Useful if allowing flips only from certain sublattices is needed.
        """
        return list(self._sublattices.keys())

    @property
    def current_occupancy(self):
        """Get the occupancy string for current interation."""
        return self.processor.decode_occupancy(self._occupancy)

    @property
    def current_property(self):
        """Get the property from current iteration."""
        return deepcopy(self._property)

    @property
    def current_structure(self):
        """Get the structure from current iteration."""
        return self.processor.structure_from_occupancy(self._occupancy)

    @property
    def current_step(self):
        """Get the iteration number of current step."""
        return self._step

    @property
    def initial_occupancy(self):
        """Get encoded occupancy string for the initial structure."""
        return self.processor.decode_occupancy(self._init_occupancy)

    @property
    def initial_structure(self):
        """Get the initial structure."""
        return self.processor.structure_from_occupancy(self._init_occupancy)

    @property
    def accepted_steps(self):
        """Get the number of accepted/successful MC steps."""
        return self._ssteps

    @property
    def acceptance_ratio(self):
        """Get the ratio of accepted/successful MC steps."""
        return self.accepted_steps/self.current_step

    @property
    def production_start(self):
        """Get the iteration number for production samples and values."""
        return self._prod_start*self.sample_interval

    @production_start.setter
    def production_start(self, val):
        """Set the iteration number for production samples and values."""
        self._prod_start = val//self.sample_interval

    @property
    def data(self):
        """List of sampled data."""
        return self._data

    @property
    def seed(self):
        """Seed for the random number generator."""
        return self._seed

    def restrict_sites(self, sites):
        """Restricts (freezes) the given sites.

        This will exclude those sites from being flipped during a Monte Carlo
        run. If some of the given indices refer to inactive sites, there will
        be no effect.

        Args:
            sites (Sequence):
                indices of sites in the occupancy string to restrict.
        """
        for sublatt in self._active_sublatts.values():
            sublatt['sites'] = np.array([i for i in sublatt['sites']
                                         if i not in sites])
        self.restricted_sites += [i for i in sites
                                  if i not in self.restricted_sites]

    def reset_restricted_sites(self):
        """Unfreeze all previously restricted sites."""
        self._active_sublatts = deepcopy(self._sublattices)
        self.restricted_sites = []

    def run(self, iterations, sublattices=None):
        """Run the ensembles for the given number of iterations.

        Samples are taken at the set intervals specified in constructur.

        Args:
            iterations (int):
                Total number of monte carlo steps to attempt
            sublattices (list of str):
                List of sublattice names to consider in site flips.
        """
        write_loops = iterations//self.sample_interval
        if iterations % self.sample_interval > 0:
            write_loops += 1

        start_step = self.current_step

        for _ in range(write_loops):
            remaining = iterations - self.current_step + start_step
            no_interrupt = min(remaining, self.sample_interval)

            for _ in range(no_interrupt):
                success = self._attempt_step(sublattices)
                self._ssteps += success

            self._step += no_interrupt
            self._save_data()

    def reset(self):
        """Reset the ensemble by returning it to its initial state.

        This will also clear the sample data.
        """
        self._occupancy = self._init_occupancy.copy()
        self._property = self.processor.compute_property(self._occupancy)
        self._step, self._ssteps = 0, 0
        self._data = []

    def dump(self, filename):
        """Write data into a text file in json format, and clear data."""
        with open(filename, 'a') as fp:
            for d in self.data:
                json.dump(d, fp)
                fp.write(os.linesep)
        self._data = []

    @abstractmethod
    def _attempt_step(self, sublattices=None):
        """Attempt a MC step and return if the step was accepted or not."""
        return

    def _get_current_data(self):
        """Extract the ensembles data from the current state.

        Returns: ensembles data
            dict
        """
        return {'occupancy': self.current_occupancy}

    def _save_data(self):
        """
        Save the current sample and properties.

        Args:
            step (int):
                Current montecarlo step
        """
        data = self._get_current_data()
        data['step'] = self.current_step
        self._data.append(data)


class ThermoEnsemble(BaseEnsemble, metaclass=ABCMeta):
    """
    A base class for thermodynamic ensembles classes to run Thermodynamic
    Monte Carlo Simulations.

    Derived classes must implement the _attempt_step method, where the specific
    flip type and acceptance criteria can be implemented.

    This allows easy implementation of any fancy ensemble that your heart
    desires: Metropolis, waiting time, cluster algos...the sky is the limit

    Attributes:
        temperature (float): temperature in Kelvin
    """

    def __init__(self, processor, temperature, sample_interval,
                 initial_occupancy, sublattices=None, seed=None):
        """Initialize CanonicalEnemble.

        Args:
            processor (Processor):
                A processor that can compute the change in a property given
                a set of flips.
            temperature (float):
                Temperature of ensemble
            sample_interval (int):
                Interval of steps to save the current occupancy and energy
            initial_occupancy (ndarray or list):
                Initial occupancy vector. The occupancy can be encoded
                according to the processor or the species names directly.
            sublattices (dict): optional
                dictionary with keys identifying the active sublattices
                (i.e. "anion" or the allowed species in that sublattice
                "Li+/Vacancy". The values should be a dictionary
                with two items {'sites': array with the site indices for all
                sites corresponding to that sublattice in the occupancy vector,
                'site_space': OrderedDict of allowed species in sublattice}
                All sites in a sublattice need to have the same set of allowed
                species.
            seed (int): optional
                Seed for random number generator.
        """
        super().__init__(processor, initial_occupancy=initial_occupancy,
                         sample_interval=sample_interval, seed=seed,
                         sublattices=sublattices)
        self.temperature = temperature
        self._min_energy = self._property
        self._min_occupancy = self._init_occupancy

    @property
    def temperature(self):
        """Get the temperature of ensemble."""
        return self._temperature

    @temperature.setter
    def temperature(self, T):
        """Set the temperature and beta accordingly."""
        self._temperature = T
        self._beta = 1.0 / (kB * T)

    @property
    def beta(self):
        """Get 1/kBT."""
        return self._beta

    @property
    def current_energy(self):
        """Get the energy of structure in the last iteration."""
        return self.current_property

    @property
    def average_energy(self):
        """Get the average of sampled energies."""
        return self.energy_samples.mean()

    @property
    def energy_variance(self):
        """Get the variance of samples energies."""
        return self.energy_samples.var()

    @property
    def energy_samples(self):
        """Get an array with the sampled energies."""
        return np.array([d['energy'] for d
                         in self.data[self._prod_start:]])

    @property
    def minimum_energy(self):
        """Get the minimum energy in samples."""
        return self._min_energy

    @property
    def minimum_energy_occupancy(self):
        """Get the occupancy for of the structure with minimum energy."""
        return self.processor.decode_occupancy(self._min_occupancy)

    @property
    def minimum_energy_structure(self):
        """Get the structure with minimum energy in samples."""
        return self.processor.structure_from_occupancy(self._min_occupancy)

    def anneal(self, start_temperature, steps, mc_iterations,
               cool_function=None, set_min_occu=True):
        """Carry out a simulated annealing procedure.

        Uses the total number of temperatures given by "steps" interpolating
        between the start and end temperature according to a cooling function.
        The start temperature is the temperature set for the ensemble.

        Args:
           start_temperature (float):
               Starting temperature. Must be higher than the current ensemble
               temperature.
           steps (int):
               Number of temperatures to run MC simulations between start and
               end temperatures.
           mc_iterations (int):
               number of Monte Carlo iterations to run at each temperature.
           cool_function (str):
               A (monotonically decreasing) function to interpolate
               temperatures.
               If none is given, linear interpolation is used.
           set_min_occu (bool):
               When True, sets the current occupancy and energy of the
               ensemble to the minimum found during annealing.
               Otherwise, do a full reset to initial occupancy and energy.

        Returns:
           tuple: (minimum energy, occupancy, annealing data)
        """
        if start_temperature < self.temperature:
            raise ValueError('End temperature is greater than start '
                             f'temperature {self.temperature} > '
                             f'{start_temperature}.')
        if cool_function is None:
            temperatures = np.linspace(start_temperature, self.temperature,
                                       steps)
        else:
            raise NotImplementedError('No other cooling functions implemented '
                                      'yet.')

        anneal_data = {}
        for T in temperatures:
            self.temperature = T
            self.run(mc_iterations)
            anneal_data[T] = self.data
            self._data = []

        min_energy = self._min_energy
        min_occupancy = self._min_occupancy

        self.reset()
        if set_min_occu:
            self._occupancy = min_occupancy.copy()
            self._min_occupancy = min_occupancy
            self._property = min_energy
            self._min_energy = min_energy

        min_occupancy = self.processor.decode_occupancy(min_occupancy)
        return min_energy, min_occupancy, anneal_data

    def reset(self):
        """Reset the ensemble by returning it to its initial state.

        This will also clear the all the sample data.
        """
        super().reset()
        self._min_occupancy = self._init_occupancy
        self._min_energy = self.processor.compute_property(self._min_occupancy)

    @abstractmethod
    def _attempt_step(self, sublattices=None):
        """Attempt a MC step and return if the step was accepted or not."""
        return

    @staticmethod
    def _accept(delta_e, beta=1.0):
        """Evaluate the metropolis acceptance criterion.

        Args:
            delta_e (float):
                potential change
            beta (float):
                1/kBT

        Returns:
            bool
        """
        return True if delta_e < 0 else exp(-beta*delta_e) >= random.random()

    def _get_current_data(self):
        """Get ensemble specific data for current MC step."""
        data = super()._get_current_data()
        data['energy'] = self.current_energy
        return data

    def as_dict(self) -> dict:
        """Json-serialization dict representation.

        Returns:
            serializable dict
        """
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             'processor': self.processor.as_dict(),
             'temperature': self.temperature,
             'sample_interval': self.sample_interval,
             'initial_occupancy': self.current_occupancy,
             'seed': self.seed,
             '_min_energy': self.minimum_energy,
             '_min_occupancy': self._min_occupancy.tolist(),
             '_sublattices': self._sublattices,
             '_active_sublatts': self._active_sublatts,
             'restricted_sites': self.restricted_sites,
             'data': self.data,
             '_step': self.current_step,
             '_ssteps': self.accepted_steps,
             '_energy': self.current_energy,
             '_occupancy': self._occupancy.tolist()}
        return d
