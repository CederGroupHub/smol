"""Implementation of base sampler classes.

A sampler essentially is an implementation of the MCMC algorithm that is used
by the corresponding ensemble to generate Monte Carlo samples.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod
import random
from warnings import warn
import numpy as np

from smol.moca.sampler.container import SampleContainer


class Sampler(ABC):
    """Abtract base class for sampler.

    A sampler is used to implement a specific MCMC algorithm used to sample
    the ensemble classes. For an illustrtive example of how to derive from this
    and write a specific sampler see the MetropolisSampler.
    """

    def __init__(self, ensemble, usher, nwalkers=1, samples=None,
                 seed=None):
        """Initialize BaseSampler.

        Args:
            ensemble (Ensemble):
                an Ensemble instance to sample from.
            usher (MCUsher)
                MC Usher to suggest MCMC steps.
            nwalkers (int):
                Number of walkers used to generate chain. Default is 1
            samples (SampleContainer)
                A sample containter to store samples. If given num_walkers is
                taken from the container.
            seed (int): optional
                seed for random number generator.
        """
        # Set and save the seed for random. This allows reproducible results.
        self.__ensemble = ensemble
        self._usher = usher
        if samples is None:
            ensemble_metadata = {'name': type(ensemble).__name__}
            ensemble_metadata.update(ensemble.thermo_boundaries)
            samples = SampleContainer(ensemble.temperature,
                                      ensemble.num_sites,
                                      ensemble.sublattices,
                                      ensemble.natural_parameters,
                                      ensemble.num_energy_coefs,
                                      ensemble_metadata, nwalkers)
        self.__container = samples
        self.__nwalkers = nwalkers
        if seed is None:
            seed = random.randint(1, np.iinfo(np.uint64).max)
        #  Save the seed for reproducibility
        self.__container.metadata['seed'] = seed
        self.__seed = seed
        random.seed(seed)

    @property
    def ensemble(self):
        """Get the underlying ensemble."""
        return self.__ensemble

    @property
    def seed(self):
        """Seed for the random number generator."""
        return self.__seed

    @seed.setter
    def seed(self, seed):
        """Set the seed for the PRNG."""
        random.seed(seed)
        self.__seed = seed

    @property
    def samples(self):
        return self.__container

    @property
    def efficiency(self):
        """Return the sampling efficiency for each walker."""
        return self.samples.sampling_efficiency

    @abstractmethod
    def _attempt_step(self, occupancy):
        """Attempts a MC step.

        Returns the next state in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            tuple: (acceptance, occupancy, enthalpy change, features change)
        """
        return tuple()

    def sample(self, nsteps, initial_occupancies, thin_by=1):
        """Generate samples

        Yield a sample state every thin_by iterations. A state is give by
        a tuple of (occupancies, feature_blob, enthalpy)

        Args:
            nsteps (int):

                Number of iterations to run.
            initial_occupancies (ndarray):
                array of occupancies
            thin_by (int): optional
                Number to thin iterations by and provide samples.

        Yields:
            tuple: accepted, occupancies, features change, enthalpies change
        """
        occupancies = initial_occupancies.copy()
        if occupancies.shape != self.samples.shape:
            occupancies = self.__reshape_occu(occupancies)
        if nsteps % thin_by != 0:
            warn(f'The number of steps {nsteps} is not a multiple of thin_by '
                 f' {thin_by}. The last {nsteps % thin_by} will be ignored.',
                 category=RuntimeWarning)
        # TODO check that initial states are independent if num_walkers > 1

        self.samples.total_mc_steps += nsteps  # update iterations in samples
        # allocate arrays for states
        occupancies = np.ascontiguousarray(occupancies, dtype=int)
        accepted = np.zeros(occupancies.shape[0], dtype=int)
        feature_blob = list(map(self.ensemble.compute_feature_vector,
                                occupancies))
        feature_blob = np.ascontiguousarray(feature_blob)
        enthalpy = np.dot(self.ensemble.natural_parameters, feature_blob.T)

        for _ in range(nsteps // thin_by):
            for _ in range(thin_by):
                for i, (accept, occupancy, delta_enthalpy, delta_features) in \
                  enumerate(map(self._attempt_step, occupancies)):
                    accepted[i] += accept
                    occupancies[i] = occupancy
                    if accept:
                        enthalpy[i] = enthalpy[i] + delta_enthalpy
                        feature_blob[i] = feature_blob[i] + delta_features
            # yield copies
            yield (accepted.copy(), occupancies.copy(), enthalpy.copy(),
                   feature_blob.copy())

    # TODO add progress option and streaming
    def run(self, nsteps, initial_occupancies=None, thin_by=1):
        """Run an MCMC sampling simulations.

        This will run and save the samples every thin_by into a
        SampleContainer.

        Args:
            nsteps (int):
                number of total MC steps.
            initial_occupancies (ndarray):
                array of occupancies. If None, the last sample will be taken.
                You should only provide this the first time you call run. If
                you want to reset then you should call reset before to start
                a fresh run.
            thin_by (int): optional
                the amount to thin by for saving samples.
        """
        if initial_occupancies is None:
            try:
                initial_occupancies = self.samples.get_occupancies()[-1]
            except IndexError:
                raise RuntimeError('There are no saved samples to obtain the '
                                   'initial occupancies. These must be '
                                   'provided.')
        elif self.samples.num_samples > 0:
            warn('Initial occupancies where provided with a pre-existing '
                 'set of samples. This basically breaks the existing chain. '
                 'Make real sure that is what you want. If not, reset the '
                 'sampler', RuntimeWarning)
        else:
            if initial_occupancies.shape != self.samples.shape:
                initial_occupancies = self.__reshape_occu(initial_occupancies)

        self.samples.allocate(nsteps)
        for state in self.sample(nsteps, initial_occupancies, thin_by=thin_by):
            self.samples.save_sample(*state)

    def __reshape_occu(self, occupancies):
        """Reshape occupancies for the single walker case."""
        # check if this is only a single walker.
        if len(occupancies.shape) == 1 and self.__nwalkers == 1:
            occupancies = np.reshape(occupancies, (1, len(occupancies)))
        else:
            raise AttributeError('The given initial occcupancies have '
                                 'incompompatible dimensions. Shape should'
                                 f' be {self.samples.shape}')
        return occupancies


# TODO write an embrassingly parallel sampler base class with a _run_parallel
#  or something and flag in run to call that instead.
