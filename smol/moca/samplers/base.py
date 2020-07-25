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
        self._seed = seed
        random.seed(seed)

