"""Abstract base class for Monte Carlo Ensembles."""

__author__ = "Luis Barroso-Luque"

import warnings
from abc import ABC, abstractmethod

from smol.moca.processor import (
    ClusterExpansionProcessor,
    CompositeProcessor,
    EwaldProcessor,
)


class BaseEnsemble(ABC):
    """Abstract base class for Monte Carlo Ensembles.

    Attributes:
        num_energy_coefs (int):
            Number of coefficients in the natural parameters array that
            correspond to energy only.
        thermo_boundaries (dict):
            Dictionary with corresponding thermodynamic boundaries, i.e.
            chemical potentials or fugacity fractions. This is kept only for
            descriptive purposes.
    """

    def __init__(self, processor, sublattices=None):
        """Initialize class instance.

        Args:
            processor (Processor):
                a processor that can compute the change in property given
                a set of flips.
            sublattices (list of Sublattice): optional
                list of Sublattice objects representing sites in the processor
                supercell with same site spaces.
        """
        # deprecation warning
        warnings.warn(
            f"{type(self).__name__} is deprecated; use Ensemble in smol.moca instead.",
            category=FutureWarning,
            stacklevel=2,
        )

        if sublattices is None:
            sublattices = processor.get_sublattices()
        self.num_energy_coefs = len(processor.coefs)
        self.thermo_boundaries = {}  # not pretty way to save general info
        self._processor = processor
        self._sublattices = sublattices

    @classmethod
    def from_cluster_expansion(cls, cluster_expansion, supercell_matrix, **kwargs):
        """Initialize an ensemble from a cluster expansion.

        Convenience constructor to instantiate an ensemble. This will take
        care of initializing the correct processor based on the
        ClusterExpansion.

        Args:
            cluster_expansion (ClusterExpansion):
                A cluster expansion object.
            supercell_matrix (ndarray):
                Supercell matrix defining the system size.
            **kwargs:
                Keyword arguments to pass to ensemble constructor. Such as
                sublattices, sublattice_probabilities, chemical_potentials,
                fugacity_fractions.

        Returns:
            Ensemble
        """
        if len(cluster_expansion.cluster_subspace.external_terms) > 0:
            processor = CompositeProcessor(
                cluster_expansion.cluster_subspace, supercell_matrix
            )
            ceprocessor = ClusterExpansionProcessor(
                cluster_expansion.cluster_subspace,
                supercell_matrix,
                cluster_expansion.coefs[:-1],
            )
            processor.add_processor(ceprocessor)
            # at some point determine term and spinup processor maybe with a
            # factory, if we ever implement more external terms.
            ewald_term = cluster_expansion.cluster_subspace.external_terms[0]
            ewprocessor = EwaldProcessor(
                cluster_expansion.cluster_subspace,
                supercell_matrix,
                ewald_term=ewald_term,
                coefficient=cluster_expansion.coefs[-1],
            )
            processor.add_processor(ewprocessor)
        else:
            processor = ClusterExpansionProcessor(
                cluster_expansion.cluster_subspace,
                supercell_matrix,
                cluster_expansion.coefs,
            )
        return cls(processor, **kwargs)

    @property
    def num_sites(self):
        """Get the total number of sites in the supercell."""
        return self.processor.num_sites

    @property
    def system_size(self):
        """Get size of supercell in number of prims."""
        return self.processor.size

    @property
    def processor(self):
        """Get the ensemble processor."""
        return self._processor

    # TODO make a setter for these that checks sublattices are correct and
    #  all sites are included.
    @property
    def sublattices(self):
        """Get list of Sublattices included in ensemble."""
        return self._sublattices

    @property
    def active_sublattices(self):
        """Get list of active sub-lattices."""
        return [s for s in self.sublattices if s.is_active]

    @property
    def restricted_sites(self):
        """Get indices of all restricted sites."""
        sites = []
        for sublattice in self.sublattices:
            sites += sublattice.restricted_sites
        return sites

    @property
    def species(self):
        """Species on active sublattices.

        These are minimal species required in setting chemical potentials.
        """
        return list(
            {sp for sublatt in self.active_sublattices for sp in sublatt.site_space}
        )

    def split_sublattice_by_species(self, sublattice_id, occu, species_in_partitions):
        """Split a sub-lattice in system by its occupied species.

        An example use case might be simulating topotactic Li extraction
        and insertion, where we want to consider Li/Vac, TM and O as
        different sub-lattices that can not be mixed by swapping.

        Args:
            sublattice_id (int):
                The index of sub-lattice to split in self.sublattices.
            occu (np.ndarray[int]):
                An occupancy array to reference with.
            species_in_partitions (List[List[int|Species|Vacancy|Element|str]]):
                Each sub-list contains a few species or encodings of species in
                the site space to be grouped as a new sub-lattice, namely,
                sites with occu[sites] == specie in the sub-list, will be
                used to initialize a new sub-lattice.
                Sub-lists will be pre-sorted to ascending order.
        """
        splits = self.sublattices[sublattice_id].split_by_species(
            occu, species_in_partitions
        )
        self._sublattices = (
            self._sublattices[:sublattice_id]
            + splits
            + self._sublattices[sublattice_id + 1 :]
        )

    @property
    @abstractmethod
    def natural_parameters(self):
        """Get the vector of natural parameters.

        The natural parameters correspond to the fit coefficients of the
        underlying processor plus any additional terms involved in the Legendre
        transformation corresponding to the ensemble.
        """
        return

    @abstractmethod
    def compute_feature_vector(self, occupancy):
        """Compute the feature vector for a given occupancy.

        The feature vector includes the necessary features required to compute
        the exponent determining the relative probability for the given
        occupancy (i.e. a generalized enthalpy). The feature vector for
        ensembles represents the sufficient statistics.

        For a cluster expansion, the feature vector is the
        correlation vector x system size

        Args:
            occupancy (ndarray):
                encoded occupancy string

        Returns:
            ndarray: feature vector
        """
        return

    @abstractmethod
    def compute_feature_vector_change(self, occupancy, step):
        """Compute the change in the feature vector from a given step.

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            step (list of tuple):
                a sequence of flips given by MCUsher.propose_step

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
        """Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        ensemble_d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "thermo_boundaries": self.thermo_boundaries,
            "processor": self._processor.as_dict(),
            "sublattices": [s.as_dict() for s in self._sublattices],
        }
        return ensemble_d
