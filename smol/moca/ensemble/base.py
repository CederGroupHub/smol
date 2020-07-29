"""Abstract Base class for Monte Carlo Ensembles."""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod
import numpy as np

from smol.constants import kB
from .sublattice import Sublattice


class Ensemble(ABC):
    """Abstract Base Class for Monte Carlo Ensembles."""

    def __init__(self, processor, temperature, sublattices=None):
        """Initialize class instance.

        Args:
            processor (Processor):
                A processor that can compute the change in a property given
                a set of flips.
            sublattices (list of Sublattice): optional
                list of Lattice objects representing sites in the processor
                supercell with same site spaces.
        """
        if sublattices is None:
            sublattices = [Sublattice(site_space,
                                      np.array([i for i, sp in
                                                enumerate(processor.allowed_species)  # noqa
                                                if sp == list(site_space.keys())]))  # noqa
                           for site_space in processor.unique_site_spaces]

        self.temperature = temperature
        self._processor = processor
        self._sublattices = sublattices
        self.restricted_sites = []
        self.thermo_boundaries = {}  # not pretty way to save general info

    @property
    def temperature(self):
        """Get the temperature of ensemble."""
        return self.__temperature

    @temperature.setter
    def temperature(self, temperature):
        """Set the temperature and beta accordingly."""
        self.__temperature = temperature
        self.__beta = 1.0 / (kB * temperature)

    @property
    def beta(self):
        """Get 1/kBT."""
        return self.__beta

    @property
    def num_sites(self):
        """Get the total number of atoms in supercell."""
        return self.processor.num_sites

    @property
    def system_size(self):
        """Get size of supercell in number of prims."""
        return self.processor.size

    @property
    def processor(self):
        """Get the system processor."""
        return self._processor

    # TODO make a setter for this that checks sublattices are correct and
    #  all sites are included.
    @property
    def sublattices(self):
        """Get names of sublattices.

        Useful if allowing flips only from certain sublattices is needed.
        """
        return self._sublattices

    @property
    @abstractmethod
    def natural_parameters(self):
        """Get the vector of natural parameters.

        The natural parameters correspond to the fit coeficients of the
        underlying processor plus any additional terms involved in the Legendre
        transformation corresponding to the ensemble.
        """
        return

    @abstractmethod
    def compute_feature_vector(self, occupancy):
        """Compute the feature vector for a give occupancy

        The feature vector is the necessary features required to compute
        the exponent determining in the relative probability for the given
        occupancy (i.e. a generalized enthalpy). The feature vector for
        ensembles represents the sufficient statistics.

        Args:
            occupancy (ndarray):
                encoded occupancy string

        Returns:
            ndarray: feature vector
        """
        return

    @abstractmethod
    def compute_feature_vector_change(self, occupancy, step):
        """Return the change in the feature vector from a step.

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            step (list of tuple):
                A sequence of flips given my the MCMCUsher.propose_step

        Returns:
            ndarray: difference in feature vector
        """
        return

    def restrict_sites(self, sites):
        """Restricts (freezes) the given sites.

        This will exclude those sites from being flipped during a Monte Carlo
        run. If some of the given indices refer to inactive sites, there will
        be no effect.

        Args:
            sites (Sequence):
                indices of sites in the occupancy string to restrict.
        """
        for sublattice in self.sublattices:
            sublattice.restrict_sites(sites)

    def reset_restricted_sites(self):
        """Unfreeze all previously restricted sites."""
        for sublattice in self.sublattices:
            sublattice.reset_restricted_sites()
