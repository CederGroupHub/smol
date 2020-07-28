"""Implementation of sampler classes.

A samples essentially is an implementation of the MCMC algorithm that is used
by the corresponding ensemble to generate Monte Carlo samples.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod
import random
import numpy as np


class Sampler(ABC):
    """Abtract base class for sampler.

    A sampler is used to implement a specific MCMC algorithm used to sample
    the ensemble classes.
    """

    def __init__(self, seed=None):
        """Initialize BaseSampler.

        Args:
            seed (int): optional
                seed for random number generator.
        """
        # Set and save the seed for random. This allows reproducible results.
        if seed is None:
            seed = random.randint(1, 1E24)

        self.__seed = seed
        random.seed(seed)

    @property
    def seed(self):
        """Seed for the random number generator."""
        return self.__seed
    
    @seed.setter
    def seed(self, seed):
        """Set the seed for the PRNG."""
        random.seed(seed)
        self.__seed = seed



    def anneal(self, start_temperature, steps, mc_iterations,
               cool_function=None):
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

        min_occupancy = self.processor.decode_occupancy(min_occupancy)
        return min_energy, min_occupancy, anneal_data


        # TODO check for site space overlap and warn
        if move_type is None:
            move_type = 'swap'
        elif move_type not in self.valid_move_types:
            raise ValueError(f'Provided move type {move_type} is not a valid '
                             'option for a Canonical Ensemble. Valid options '
                             f'are {self.valid_move_types}.')