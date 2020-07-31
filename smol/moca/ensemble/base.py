"""Abstract Base class for Monte Carlo Ensembles."""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod
import numpy as np

from smol.constants import kB
from smol.moca import CompositeProcessor, CEProcessor, EwaldProcessor
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
                              np.array([i for i, sp in  # noqa
                                        enumerate(processor.allowed_species)  # noqa
                                        if sp == list(site_space.keys())]))  # noqa
                           for site_space in processor.unique_site_spaces]

        self.temperature = temperature
        self.__processor = processor
        self.__sublattices = sublattices
        self.num_energy_coefs = len(processor.coefs)
        self.restricted_sites = []
        self.thermo_boundaries = {}  # not pretty way to save general info

    @classmethod
    def from_cluster_expansion(cls, cluster_expansion, supercell_matrix,
                               temperature, optimize_indicator=False,
                               **kwargs):
        """Initialize an ensemble from a cluster expansion.

        Convenience constructor to instantiate an ensemble. This will take
        care of initializing the correct processor based on the
        ClusterExpansion.

        Args:
            cluster_expansion (ClusterExpansion):
                A cluster expansion object.
            supercell_matrix (ndarray):
                Supercell matrix defining the system size.
            temperature (float):
                Ensemble temperature.
            optimize_indicator (bool): optional
                Wether to optimize calculations for indicator basis.
            **kwargs:
                Keyword arguments to pass to ensemble constructor. Such as
                sublattices, sublattice_probabilities, chemical_potentials,
                fugacity_fractions.

        Returns:
            Ensemble
        """
        if len(cluster_expansion.cluster_subspace.external_terms) > 0:
            processor = CompositeProcessor(cluster_expansion.cluster_subspace,
                                           supercell_matrix)
            processor.add_processor(CEProcessor, cluster_expansion.coefs[:-1],
                                    optimize_indicator=optimize_indicator)
            # at some point determine term and spinup processor maybe with a
            # factory, if we ever implement more external terms.
            ewald_term = cluster_expansion.cluster_subspace.external_terms[0]
            processor.add_processor(EwaldProcessor, ewald_term=ewald_term,
                                    coefficient=cluster_expansion.coefs[-1])
        else:
            processor = CEProcessor(cluster_expansion.cluster_subspace,
                                    supercell_matrix, cluster_expansion.coefs,
                                    optimize_indicator=optimize_indicator)
        return cls(processor, temperature, **kwargs)

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
        return self.__processor

    # TODO make a setter for this that checks sublattices are correct and
    #  all sites are included.
    @property
    def sublattices(self):
        """Get names of sublattices.

        Useful if allowing flips only from certain sublattices is needed.
        """
        return self.__sublattices

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
        """Compute the change in the feature vector from a step.

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
