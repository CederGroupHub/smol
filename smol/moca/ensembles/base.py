"""Abstract Base class for Monte Carlo Ensembles."""

__author__ = "Luis Barroso-Luque"

import random
from copy import deepcopy
import os
import json
import numpy as np
from math import exp
from abc import ABC, abstractmethod


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
                'site_domain': OrderedDict of allowed species in sublattice}
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
            sublattices = {'/'.join(domain.keys()):
                           {'sites': np.array([i for i, sp in
                                        enumerate(processor.allowed_species)  # noqa
                                               if sp == list(domain.keys())]),
                            'domain': domain}
                           for domain in processor.unique_site_domains}

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
