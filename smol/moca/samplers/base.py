"""Implementation of sampler classes.

A samples essentially is an implementation of the MCMC algorithm that is used
by the corresponding ensemble to generate Monte Carlo samples.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod
import random


class BaseSampler(ABC):
    """Abtract base class for samplers.

    A sampler is used to implement a specific MCMC algorithm for use in the
    ensemble classes.
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

        self._prod_start = 0
        self._step = 0
        self._ssteps = 0
        self._seed = seed
        random.seed(seed)

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

    def reset(self):
        """Reset the ensemble by returning it to its initial state.

        This will also clear the sample data.
        """
        self._occupancy = self._init_occupancy.copy()
        self._property = self.processor.compute_property(self._occupancy)
        self._step, self._ssteps = 0, 0
        self._data = []

