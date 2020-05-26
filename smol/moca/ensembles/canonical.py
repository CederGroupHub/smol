"""Implementation of a Canonical Ensemble Class for running Monte Carlo
simulations for fixed number of sites and fixed concentration of species.
"""

__author__ = "Luis Barroso-Luque"

import random
import numpy as np
from monty.json import MSONable
from smol.moca.ensembles.base import BaseEnsemble
from smol.moca.processor import CEProcessor
from smol.globals import kB


class CanonicalEnsemble(BaseEnsemble, MSONable):
    """
    A Canonical Ensemble class to run Monte Carlo Simulations
    """

    def __init__(self, processor, temperature, sample_interval,
                 initial_occupancy=None, seed=None):
        """
        Args:
            processor (Processor):
                A processor that can compute the change in a property given
                a set of flips.
            temperature (float):
                Temperature of ensemble
            sample_interval (int):
                Interval of steps to save the current occupancy and property
            inital_occupancy (ndarray):
                Initial occupancy vector. If none is given then a random one
                will be used.
            seed (int):
                Seed for random number generator.
        """

        super().__init__(processor, initial_occupancy=initial_occupancy,
                         sample_interval=sample_interval, seed=seed)
        self.temperature = temperature
        self._min_energy = self._property
        self._min_occupancy = self._init_occupancy

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, T):
        self._temperature = T
        self._beta = 1.0/(kB*T)

    @property
    def beta(self):
        return self._beta

    @property
    def current_energy(self):
        return self.current_property

    @property
    def average_energy(self):
        return self.energy_samples.mean()

    @property
    def energy_variance(self):
        return self.energy_samples.var()

    @property
    def energy_samples(self):
        return np.array([d['energy'] for d
                         in self.data[self.production_start:]])

    @property
    def minimum_energy(self):
        return self._min_energy

    @property
    def minimum_energy_occupancy(self):
        return self.processor.decode_occupancy(self._min_occupancy)

    @property
    def minimum_energy_structure(self):
        return self.processor.structure_from_occupancy(self._min_occupancy)

    def anneal(self, start_temperature, steps, mc_iterations,
               cool_function=None):
        """
        Carries out a simulated annealing procedure for a total number of
        temperatures given by "steps" interpolating between the start and end
        temperature according to a cooling function. The start temperature is
        the temperature set for the ensemble.

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

        Returns: (minimum energy, occupation)
           tuple
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

        min_energy = self.minimum_energy
        min_occupancy = self.minimum_energy_occupancy
        anneal_data = {}

        for T in temperatures:
            self.temperature = T
            self.run(mc_iterations)
            anneal_data[T] = self.data
            self._data = []

        min_energy = self._min_energy
        min_occupancy = self.processor.decode_occupancy(self._min_occupancy)
        self.reset()  # should we do full reset or keep min energy?
        # TODO Save annealing data?
        return min_energy, min_occupancy, anneal_data

    def reset(self):
        """
        Resets the ensemble by returning it to its initial state. This will
        also clear the data.
        """
        super().reset()
        self._min_energy = self.processor.compute_property(self._occupancy)
        self._min_occupancy = self._occupancy

    def _attempt_step(self, sublattice_name=None):
        """
        Attempts flips corresponding to an elementary canonical swap.
        Will pick a sublattice at random and then a canonical swap at random
        from that sublattice.

        Args:
            sublattice_name (str): optional
                If only considering one sublattice.

        Returns: Flip acceptance
            bool
        """
        flips = self._get_flips(sublattice_name)
        delta_e = self.processor.compute_property_change(self._occupancy,
                                                         flips)
        accept = self._accept(delta_e, self.beta)

        if accept:
            self._property += delta_e
            for f in flips:
                self._occupancy[f[0]] = f[1]
            if self._property < self._min_energy:
                self._min_energy = self._property
                self._min_occupancy = self._occupancy.copy()

        return accept

    def _get_flips(self, sublattice_name=None):
        """
        Gets a possible canonical flip. A swap between two sites.

        Args:
            sublattice_name (str): optional
                If only considering one sublattice.
        Returns:
            tuple
        """
        if sublattice_name is None:
            sublattice_name = random.choice(list(self._sublattices.keys()))

        sites = self._sublattices[sublattice_name]['sites']
        site1 = random.choice(sites)
        swap_options = [i for i in sites
                        if self._occupancy[i] != self._occupancy[site1]]
        if swap_options:
            site2 = random.choice(swap_options)
            return ((site1, self._occupancy[site2]),
                    (site2, self._occupancy[site1]))
        else:
            # inefficient, maybe re-call method? infinite recursion problem
            return tuple()

    def _get_current_data(self):
        """
        Get ensemble specific data for current MC step
        """
        return {'energy': self.current_energy,
                'occupancy': self.current_occupancy}

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation.

        Returns:
            MSONable dict
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
             '_data': self.data,
             '_step': self.current_step,
             '_ssteps': self.accepted_steps,
             '_energy': self.current_energy,
             '_occupancy': self._occupancy.tolist()}
        return d

    @classmethod
    def from_dict(cls, d):
        """
        Creates a CanonicalEnsemble from MSONable dict representation.
        """
        eb = cls(CEProcessor.from_dict(d['processor']),
                 temperature=d['temperature'],
                 sample_interval=d['sample_interval'],
                 initial_occupancy=d['initial_occupancy'],
                 seed=d['seed'])
        eb._min_energy = d['_min_energy']
        eb._min_occupancy = np.array(d['_min_occupancy'])
        eb._sublattices = d['_sublattices']
        eb._data = d['_data']
        eb._step = d['_step']
        eb._ssteps = d['_ssteps']
        eb._property = d['_energy']
        eb._occupancy = np.array(d['_occupancy'])
        return eb
