"""Implementation of Thermodynamic Ensemble for MC sampling."""

__author__ = "Luis Barroso-Luque"

from collections import Counter

import numpy as np
from monty.json import MSONable, jsanitize
from pymatgen.core.composition import ChemicalPotential

from smol.cofe.space.domain import get_species
from smol.moca.processor import (
    ClusterDecompositionProcessor,
    ClusterExpansionProcessor,
    CompositeProcessor,
    EwaldProcessor,
)
from smol.moca.processor.base import Processor
from smol.moca.sublattice import Sublattice


class ChemicalPotentialManager:
    """Chemical potential descriptor for use Ensemble class."""

    natural_parameter: float = -1.0

    def __set_name__(self, owner, name):
        """Set the private variable names."""
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        """Return the chemical potentials if set None otherwise."""
        value = getattr(obj, self.private_name, None)
        return value if value is None else value["value"]

    def __set__(self, obj, value):
        """Set the table given the owner and value."""
        if value is None:  # call delete if set to None
            self.__delete__(obj)
            return

        for spec, count in Counter(map(get_species, value.keys())).items():
            if count > 1:
                raise ValueError(
                    f"{count} values of the chemical potential for the same "
                    f"species {spec} were provided.\n Make sure the dictionary "
                    "you are using has only string keys or only Species "
                    "objects as keys."
                )
        value = {
            get_species(k): v for k, v in value.items() if get_species(k) in obj.species
        }
        if set(value.keys()) != set(obj.species):
            raise ValueError(
                "Chemical potentials given are missing species. "
                "Values must be given for each of the following:"
                f" {obj.species}"
            )

        # if first instantiation concatenate the natural parameter
        if not hasattr(obj, self.private_name):
            obj.natural_parameters = np.append(
                obj.natural_parameters, self.natural_parameter
            )
        setattr(
            obj,
            self.private_name,
            {"value": ChemicalPotential(value), "table": self._build_table(obj, value)},
        )
        # update the ensemble dictionary and _boundaries list
        if hasattr(obj, "thermo_boundaries"):
            obj.thermo_boundaries.update({self.public_name: value})
        else:
            setattr(obj, "thermo_boundaries", {self.public_name: value})

    def __delete__(self, obj):
        """Delete the boundary condition."""
        if hasattr(obj, self.private_name):
            del obj.__dict__[self.private_name]
        if (
            hasattr(obj, "thermo_boundaries")
            and self.public_name in obj.thermo_boundaries
        ):
            del obj.thermo_boundaries[self.public_name]
        if obj.num_energy_coefs < len(obj.natural_parameters):
            obj.natural_parameters = obj.natural_parameters[:-1]  # remove last entry

    @staticmethod
    def _build_table(obj, value):
        """Set the chemical potentials and update table."""
        num_cols = max(max(sl.encoding) for sl in obj.sublattices) + 1

        # Sublattice can only be initialized as default, or split from default.
        table = np.zeros((obj.num_sites, num_cols))
        for sublatt in obj.active_sublattices:
            ordered_pots = [value[sp] for sp in sublatt.site_space]
            table[sublatt.sites[:, None], sublatt.encoding] = ordered_pots
        return table


class Ensemble(MSONable):
    """Thermodynamic ensemble class.

    Attributes:
        thermo_boundaries (dict):
            Dictionary with corresponding thermodynamic boundaries, i.e.
            chemical potentials. This is kept only for descriptive purposes.
    """

    chemical_potentials = ChemicalPotentialManager()

    def __init__(self, processor, sublattices=None, chemical_potentials=None):
        """Initialize class instance.

        Args:
            processor (Processor):
                a processor that can compute the change in property given
                a set of flips.
            sublattices (list of Sublattice): optional
                list of Sublattice objects representing sites in the processor
                supercell with same site spaces.
        """
        if sublattices is None:
            sublattices = processor.get_sublattices()
        self.thermo_boundaries = {}  # not pretty way to save general info
        self._params = processor.coefs  # natural parameters
        self._processor = processor
        self._sublattices = sublattices
        self.chemical_potentials = chemical_potentials

    @classmethod
    def from_cluster_expansion(
        cls,
        cluster_expansion,
        supercell_matrix,
        processor_type="decomposition",
        **kwargs,
    ):
        """Initialize an ensemble from a cluster expansion.

        Convenience constructor to instantiate an ensemble. This will take
        care of initializing the correct processor based on the
        ClusterExpansion.

        Args:
            cluster_expansion (ClusterExpansion):
                A cluster expansion object.
            supercell_matrix (ndarray):
                Supercell matrix defining the system size.
            processor_type(str): optional
                Type of processor to be used besides external term.
                Can use "decomposition" (default) or "expansion".
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
            if processor_type == "decomposition":
                ceprocessor = ClusterDecompositionProcessor(
                    cluster_expansion.cluster_subspace,
                    supercell_matrix,
                    cluster_expansion.cluster_interaction_tensors,
                )
            elif processor_type == "expansion":
                ceprocessor = ClusterExpansionProcessor(
                    cluster_expansion.cluster_subspace,
                    supercell_matrix,
                    cluster_expansion.coefs[:-1],
                )
            else:
                raise ValueError(f"Processor type {processor_type}" f" not supported!")
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
            if processor_type == "decomposition":
                processor = ClusterDecompositionProcessor(
                    cluster_expansion.cluster_subspace,
                    supercell_matrix,
                    cluster_expansion.cluster_interaction_tensors,
                )
            elif processor_type == "expansion":
                processor = ClusterExpansionProcessor(
                    cluster_expansion.cluster_subspace,
                    supercell_matrix,
                    cluster_expansion.coefs,
                )
            else:
                raise ValueError(f"Processor type {processor_type}" f" not supported!")
        return cls(processor, **kwargs)

    @property
    def num_sites(self):
        """Get the total number of sites in the supercell."""
        return self.processor.num_sites

    @property
    def num_energy_coefs(self):
        """Return the number of coefficients used in the energy expansion."""
        return len(self._processor.coefs)

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
        return np.concatenate(
            [sublattice.restricted_sites for sublattice in self.sublattices]
        )

    @property
    def species(self):
        """Species on active sublattices.

        These are minimal species required in setting chemical potentials.
        """
        return list(
            {sp for sublatt in self.active_sublattices for sp in sublatt.site_space}
        )

    @property
    def natural_parameters(self):
        """Get the vector of natural parameters.

        The natural parameters correspond to the fit coefficients of the
        underlying processor plus any additional terms involved in the Legendre
        transformation corresponding to the ensemble.
        """
        return self._params

    @natural_parameters.setter
    def natural_parameters(self, value):
        """Set the value of the natural parameters.

        Should only allow appending to the original energy coefficients
        """
        if not np.array_equal(self.processor.coefs, value[: self.num_energy_coefs]):
            raise ValueError("The original expansion coefficients can not be changed!")
        self._params = value

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
        if self.chemical_potentials is not None:
            # Species in active sub-lattices may change after split.
            # Need to reset and rebuild chemical potentials.
            chemical_potentials = {
                spec: self.chemical_potentials[spec] for spec in self.species
            }
            self.chemical_potentials = chemical_potentials

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
        features = self.processor.compute_feature_vector(occupancy)

        if self.chemical_potentials is not None:
            chemical_work = sum(
                self._chemical_potentials["table"][site][species]
                for site, species in enumerate(occupancy)
            )
            features = np.append(features, chemical_work)

        return features

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
        delta_features = self.processor.compute_feature_vector_change(occupancy, step)

        if self.chemical_potentials is not None:
            delta_work = sum(
                self._chemical_potentials["table"][f[0]][f[1]]
                - self._chemical_potentials["table"][f[0]][occupancy[f[0]]]
                for f in step
            )
            delta_features = np.append(delta_features, delta_work)

        return delta_features

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
            "thermo_boundaries": jsanitize(self.thermo_boundaries),
            "processor": self._processor.as_dict(),
            "sublattices": [s.as_dict() for s in self._sublattices],
        }
        return ensemble_d

    @classmethod
    def from_dict(cls, ensemble_d):
        """Instantiate a CanonicalEnsemble from dict representation.

        Args:
            ensemble_d (dict):
                dictionary representation.
        Returns:
            CanonicalEnsemble
        """
        ensemble = cls(
            Processor.from_dict(ensemble_d["processor"]),
            [Sublattice.from_dict(s) for s in ensemble_d["sublattices"]],
        )
        chemical_potentials = ensemble_d["thermo_boundaries"].get("chemical_potentials")
        if chemical_potentials is not None:
            ensemble.chemical_potentials = chemical_potentials

        return ensemble
