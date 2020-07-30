"""Implementation of a Metropolis-Hastings sampler."""

__author__ = "Luis Barroso-Luque"

from math import log
from random import random
import numpy as np

from .base import Sampler
from .usher import mcmc_usher_factory


class MetropolisSampler(Sampler):
    """A Metropolis-Hastings sampler.

    The classic and nothing but the classic.
    """
    valid_mcmc_ushers = {'flip': 'Flipper', 'swap': 'Swapper'}

    def __init__(self, ensemble, step_type=None, sublattices=None,
                 sublattice_probabilities=None, nwalkers=1, container=None,
                 seed=None):
        """Initialize a Metropolis-Hastings sampler.

        Args:
            ensemble (Ensemble):
                An Ensemble instance to sample from.
            step_type (str): optional
                String specifying the MCMC step type.
            sublattices (list of Sublattice): optional
                List of Sublattices to propose steps for. This must be a subset
                of the ensemble.sublattices. If not given all sublattices in
                the ensemble will be used. Use this if you want to restrict
                or freeze some sublattices.
            sublattice_probabilities (list of float): optional
                List of probability to pick a site from a specific sublattice.
                The order must correspond to the same order of the sublattices.
            nwalkers (int): optional
                Number of walkers used to generate chain. Default is 1
            container (SampleContainer): optional
                A containter to store samples. If given num_walkers is taken
                from the container.
            seed (int): optional
                Seed for random number generator.
        """
        if sublattices is None:
            sublattices = ensemble.sublattices
        if step_type is None:
            usher = mcmc_usher_factory(ensemble.valid_mcmc_ushers[0],
                                       sublattices, sublattice_probabilities)
        else:
            if step_type not in self.valid_mcmc_ushers.keys():
                raise ValueError(f'{step_type} step type is not a valid option'
                                 '. Options are '
                                 f'{self.valid_mcmc_ushers.keys()}')
            usher = mcmc_usher_factory(self.valid_mcmc_ushers[step_type],
                                       sublattices, sublattice_probabilities)
        super().__init__(ensemble, usher, nwalkers, container, seed)

    def _attempt_step(self, occupancy):
        """Attempts a MC step.

        Returns the next state in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            tuple: (acceptance, occupancy, features change, enthalpy change)
        """
        step = self._usher.propose_step(occupancy)
        delta_features = self.ensemble.compute_feature_vector_change(occupancy,
                                                                     step)
        delta_enthalpy = np.dot(self.ensemble.natural_parameters,
                                delta_features)
        accept = (True if delta_enthalpy <= 0
                  else -self.ensemble.beta * delta_enthalpy > log(random()))
        if accept:
            for f in step:
                occupancy[f[0]] = f[1]

        return accept, occupancy, delta_features, delta_enthalpy
