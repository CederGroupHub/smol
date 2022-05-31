"""Implementation of Semi-Grand Canonical Ensemble classes.

These are used to run Monte Carlo sampling for systems with
a fixed number of sites but variable concentration of species.
"""

__author__ = "Luis Barroso-Luque"


from collections import Counter

import numpy as np
from monty.json import MSONable
from pymatgen.core import DummySpecies, Element, Species

from smol.cofe.space.domain import Vacancy, get_species
from smol.moca.processor.base import Processor
from smol.moca.sublattice import Sublattice

from .base import BaseEnsemble


class SemiGrandEnsemble(BaseEnsemble, MSONable):
    """Relative chemical potential-based SemiGrand Ensemble.

    A Semi-Grand Canonical Ensemble for Monte Carlo simulations where species'
    relative chemical potentials are predefined. Note that in the SGC Ensemble
    implemented here, only the differences in chemical potentials with
    respect to a reference species on each sublattice are fixed, and not the
    absolute values. To obtain the absolute values you must calculate the
    reference chemical potential and then subtract it from the given values.

    Attributes:
        thermo_boundaries (dict):
            dict of chemical potentials.
    """

    def __init__(self, processor, chemical_potentials, sublattices=None):
        """Initialize MuSemiGrandEnsemble.

        Args:
            processor (Processor):
                a processor that can compute the change in property given
                a set of flips.
            chemical_potentials (dict):
                Dictionary with species and chemical potentials.
            sublattices (list of Sublattice): optional
                list of Sublattice objects representing sites in the processor
                supercell with same site spaces.
        """
        super().__init__(processor, sublattices=sublattices)
        self._params = np.append(self.processor.coefs, -1.0)
        # check that species are valid
        chemical_potentials = {
            get_species(k): v for k, v in chemical_potentials.items()
        }
        # Excessive species not appeared on active sub-lattices
        # will be dropped.
        for spec in self.species:
            if spec not in chemical_potentials.keys():
                raise ValueError(
                    f"Species {spec} was not assigned a chemical "
                    " potential, a value must be provided."
                )

        # preallocate this for slight speed improvements
        self._dfeatures = np.empty(len(processor.coefs) + 1)
        self._features = np.empty(len(processor.coefs) + 1)

        self._mus = {k: v for k, v in chemical_potentials.items() if k in self.species}
        self._mu_table = self._build_mu_table(self._mus)
        self.thermo_boundaries = {
            "chemical-potentials": {str(k): v for k, v in self._mus.items()}
        }

    @property
    def natural_parameters(self):
        """Get the vector of natural parameters.

        For SGC an extra -1 is added for the chemical part of the Legendre
        transform.
        """
        return self._params

    @property
    def chemical_potentials(self):
        """Get the chemical potentials for species in the system."""
        return self._mus

    @chemical_potentials.setter
    def chemical_potentials(self, value):
        """Set the chemical potentials and update table."""
        for spec, count in Counter(map(get_species, value.keys())).items():
            if count > 1:
                raise ValueError(
                    f"{count} values of the chemical potential for the same "
                    f"species {spec} were provided.\n Make sure the dictionary "
                    "you are using has only string keys or only Species "
                    "objects as keys."
                )
        value = {get_species(k): v for k, v in value.items() if k in self.species}
        if set(value.keys()) != set(self.species):
            raise ValueError(
                "Chemical potentials given are missing species. "
                "Values must be given for each of the following:"
                f" {self.species}"
            )
        self._mus = value
        self._mu_table = self._build_mu_table(value)
        self.thermo_boundaries = {
            "chemical-potentials": {str(k): v for k, v in self._mus.items()}
        }

    def compute_feature_vector(self, occupancy):
        """Compute the relevant feature vector for a given occupancy.

        In the semigrand case it is the feature vector and the chemical work
        term.

        Args:
            occupancy (ndarray):
                encoded occupancy string

        Returns:
            ndarray: feature vector
        """
        self._features[:-1] = self.processor.compute_feature_vector(occupancy)
        self._features[-1] = self.compute_chemical_work(occupancy)
        return self._features

    def compute_feature_vector_change(self, occupancy, step):
        """Return the change in the feature vector from a given step.

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            step (list of tuple):
                a sequence of flips given by MCUsher.propose_step

        Returns:
            ndarray: difference in feature vector
        """
        self._dfeatures[:-1] = self.processor.compute_feature_vector_change(
            occupancy, step
        )
        self._dfeatures[-1] = sum(
            self._mu_table[f[0]][f[1]] - self._mu_table[f[0]][occupancy[f[0]]]
            for f in step
        )  # Can be wrong if step has two same site indices.
        return self._dfeatures

    def compute_chemical_work(self, occupancy):
        """Compute sum of mu * N for given occupancy."""
        return sum(
            self._mu_table[site][species] for site, species in enumerate(occupancy)
        )

    def split_sublattice_by_species(self, sublattice_id, occu, species_in_partitions):
        """Split a sub-lattice in system by its occupied species.

        An example use case might be simulating topotactic Li extraction
        and insertion, where we want to consider Li/Vac, TM and O as
        different sub-lattices that can not be mixed by swapping.

        In the grand canonical ensemble, the mu table will also be updated
        after split.

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
        super().split_sublattice_by_species(sublattice_id, occu, species_in_partitions)
        # Species in active sub-lattices may change after split.
        # Need to reset and rebuild mu table.
        new_chemical_potentials = {spec: self._mus[spec] for spec in self.species}
        self.chemical_potentials = new_chemical_potentials

    def _build_mu_table(self, chemical_potentials):
        """Build an array for chemical potentials for all sites in system.

        Rows represent sites and columns species. This allows quick evaluation
        of chemical potential changes from flips. Not that the total number
        of columns will be the number of species in the largest site space. For
        smaller site spaces the values at those rows are meaningless and will
        be given values of 0. Also rows representing sites with not partial
        occupancy will have all 0 values and should never be used.
        """
        # Mu table should be built with ensemble, rather than processor data.
        # Otherwise you may get wrong species encoding if the sub-lattices are
        # split.
        num_cols = max(max(sl.encoding) for sl in self.sublattices) + 1
        # Sublattice can only be initialized as default, or splitted from default.
        table = np.zeros((self.num_sites, num_cols))
        for sublatt in self.active_sublattices:
            ordered_pots = [chemical_potentials[sp] for sp in sublatt.site_space]
            table[sublatt.sites[:, None], sublatt.encoding] = ordered_pots
        return table

    def as_dict(self):
        """Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        sgce_d = super().as_dict()
        sgce_d["chemical_potentials"] = tuple(
            (s.as_dict(), c) for s, c in self.chemical_potentials.items()
        )
        return sgce_d

    @classmethod
    def from_dict(cls, d):
        """Instantiate a SemiGrandEnsemble from dict representation.

        Returns:
            CanonicalEnsemble
        """
        chemical_potentials = {}
        for spec, chem_pot in d["chemical_potentials"]:
            if "oxidation_state" in spec and Element.is_valid_symbol(spec["element"]):
                spec = Species.from_dict(spec)
            elif "oxidation_state" in spec:
                if spec["@class"] == "Vacancy":
                    spec = Vacancy.from_dict(spec)
                else:
                    spec = DummySpecies.from_dict(spec)
            else:
                spec = Element(spec["element"])
            chemical_potentials[spec] = chem_pot

        sublatts = d.get("sublattices")  # keep backwards compatibility
        if sublatts is not None:
            sublatts = [Sublattice.from_dict(sl_d) for sl_d in sublatts]

        return cls(
            Processor.from_dict(d["processor"]),
            chemical_potentials=chemical_potentials,
            sublattices=sublatts,
        )
