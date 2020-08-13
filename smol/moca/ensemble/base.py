"""Abstract Base class for Monte Carlo Ensembles."""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod

from smol.constants import kB
from smol.moca import CompositeProcessor, CEProcessor, EwaldProcessor
from .sublattice import get_sublattices


class Ensemble(ABC):
    """Abstract Base Class for Monte Carlo Ensembles.

    Attributes:
        num_energy_coefs (int):
            Number of coefficients in the natural parameters array that
            correspond to energy only.
        thermo_boundaries (dict):
            dictionary with corresponding thermodynamic boundaries, i.e.
            chemical potentials or fugacity fractions. This is kept only for
            descriptive purposes.
    """

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
            sublattices = get_sublattices(processor)
        self.temperature = temperature
        self.num_energy_coefs = len(processor.coefs)
        self.thermo_boundaries = {}  # not pretty way to save general info
        self._processor = processor
        self._sublattices = sublattices

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
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        """Set the temperature and beta accordingly."""
        self._temperature = temperature
        self._beta = 1.0 / (kB * temperature)

    @property
    def beta(self):
        """Get 1/kBT."""
        return self._beta

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
    def restricted_sites(self):
        """Get indices of all restricted sites."""
        sites = []
        for sublattice in self.sublattices:
            sites += sublattice.restricted_sites
        return sites

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

    def as_dict(self):
        """Get dictionary representation."""
        d = {'temperature': self.temperature,
             'thermo_boundaries': self.thermo_boundaries,
             'processor': self._processor.as_dict(),
             'sublattices': [s.as_dict() for s in self._sublattices]}
        return d
