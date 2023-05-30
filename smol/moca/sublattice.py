"""Implementation of Sublattice class.

A sublattice represents a set of sites in a supercell that all have
the same domain/site space. More rigourously it represents a substructure
of the random structure supercell being sampled in a Monte Carlo
simulation.
"""

__author__ = "Luis Barroso-Luque, Fengyu Xie"

import itertools
import warnings
from dataclasses import dataclass, field

import numpy as np
from monty.json import MSONable
from pymatgen.core import Composition

from smol.cofe.space.domain import SiteSpace, Vacancy


@dataclass
class Sublattice(MSONable):
    """Sublattice class.

    A Sublattice is used to represent a subset of supercell sites that have
    the same site space. Rigorously it represents a set of sites in a
    "substructure" of the total structure.

    As implemented in the Sublattice class, you are allowed to have multiple
    species on an inactive sub-lattice, but it is not the suggested approach.
    For simplicity, especially when you generate a CompositionSpace, you had better
    always split inactive sub-lattices until each inactive sub-lattice have
    only 1 species in it.

    Attributes:
     site_space (SiteSpace):
        SiteSpace with the allowed species and their random
        state composition.
     sites (ndarray):
        array of site indices for all sites in sublattice
     active_sites (ndarray):
        array of site indices for all unrestricted sites in the sublattice.
     encoding (ndarray):
        array of species encoding in integer indices. By default,
        will be initialized as range(len(site_space)). Might be different
        if a sub-lattice was created from the split of another sub-lattice.
    """

    site_space: SiteSpace
    sites: np.array
    active_sites: np.array = field(init=False)
    encoding: np.array = field(init=False)

    def __post_init__(self):
        """Copy sites into active_sites, and initial setup."""
        self.sites = np.unique(self.sites)
        self.active_sites = self.sites.copy()
        if len(self.site_space) <= 1:
            # A single-species sub-lattice should not be active at all.
            self.restrict_sites(self.sites)

        self.encoding = np.arange(len(self.site_space), dtype=int)

    @property
    def is_active(self):
        """Whether sub-lattice has active sites."""
        if len(self.active_sites) == 0 and len(self.species) > 1:
            warnings.warn(
                "Sub-lattice is inactive, but have multiple "
                "allowed species. You'd better split it."
            )
        return len(self.active_sites) > 0

    @property
    def species(self):
        """Get allowed species for sites in sublattice."""
        return tuple(self.site_space.keys())

    @property
    def composition(self):
        """Get composition of sublattice."""
        return self.site_space.composition

    @property
    def restricted_sites(self):
        """Get restricted sites for species."""
        return np.setdiff1d(self.sites, self.active_sites)

    def restrict_sites(self, sites):
        """Restricts (freezes) the given sites.

        Once a site is restricted, no Metropolis step can be proposed
        with it, including flipping, swapping, etc.

        Args:
            sites (Sequence):
                indices of sites in the occupancy string to restrict.
        """
        self.active_sites = np.array([i for i in self.active_sites if i not in sites])

    def reset_restricted_sites(self):
        """Reset all restricted sites to active."""
        # Single species sub-lattice can never be active.
        if len(self.site_space) > 1:
            self.active_sites = self.sites.copy()

    def split_by_species(self, occu, species_in_partitions):
        """Split a sub-lattice into multiple by specie.

        An example use case might be simulating topotactic Li extraction
        and insertion, where we want to consider Li/Vac, TM and O as
        different sub-lattices that can not be mixed by swapping.
        Args:
            occu (np.ndarray[int]):
                An occupancy array to reference with.
            species_in_partitions (List[List[int|Species|Vacancy|Element|str]]):
                Each sub-list contains a few species or encodings of species in
                the site space to be grouped as a new sub-lattice, namely,
                sites with occu[sites] == specie in the sub-list, will be
                used to initialize a new sub-lattice.
                Sub-lists will be pre-sorted to ascending order.
        Returns:
            List of split sub-lattices:
                List[Sublattice]
        """
        part_sublattices = []
        # Codes given
        if all(
            isinstance(sp, (int, np.int32, np.int64))
            for sp in itertools.chain(*species_in_partitions)
        ):
            codes_in_partitions = species_in_partitions
        # Species or species strings given.
        else:
            # Treat vacancies more carefully. One sub-lattice can only
            # have one vacancy species.
            def get_index(sp, species):
                if isinstance(sp, Vacancy):
                    for i, sp2 in enumerate(species):
                        if isinstance(sp2, Vacancy):
                            return i
                return species.index(sp)

            codes_in_partitions = [
                [self.encoding[get_index(sp, self.species)] for sp in partition]
                for partition in species_in_partitions
            ]

        for species_codes in codes_in_partitions:
            part_comp = {}
            part_sites = []
            part_actives = []
            # Because site space species were sorted.
            part_codes = sorted(species_codes)
            for code in part_codes:
                sp_id = np.where(self.encoding == code)[0][0]
                sp = self.species[sp_id]
                part_comp[sp] = self.site_space[sp]
                part_sites.extend(self.sites[occu[self.sites] == code].tolist())
                part_actives.extend(
                    self.active_sites[occu[self.active_sites] == code].tolist()
                )
            # Re-weighting partitioned site-space
            part_n = sum(list(part_comp.values()))
            part_comp = {
                sp: part_comp[sp] / part_n
                for sp in part_comp
                if not isinstance(sp, Vacancy)
            }
            part_comp = Composition(part_comp)
            part_space = SiteSpace(part_comp)
            part_sites = np.array(part_sites, dtype=int)
            part_actives = np.array(part_actives, dtype=int)
            part_codes = np.array(part_codes, dtype=int)
            part_sublatt = Sublattice(part_space, part_sites)
            part_sublatt.active_sites = part_actives
            part_sublatt.encoding = part_codes
            if len(part_codes) == 1:
                part_sublatt.restrict_sites(part_sublatt.sites)
            part_sublattices.append(part_sublatt)
        return part_sublattices

    def __eq__(self, other):
        """Check equality of sublattices.

        Does not check equality of restricted sites!
        """
        if (
            other.site_space == self.site_space
            and np.array_equal(other.encoding, self.encoding)
            and np.array_equal(other.sites, self.sites)
        ):
            return True
        return False

    def as_dict(self):
        """Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        sl_d = {
            "site_space": self.site_space.as_dict(),
            "sites": self.sites.tolist(),
            "encoding": self.encoding.tolist(),
            "active_sites": self.active_sites.tolist(),
        }
        return sl_d

    @classmethod
    def from_dict(cls, d):
        """Instantiate a sublattice from dict representation.

        Returns:
            Sublattice
        """
        sublattice = cls(
            SiteSpace.from_dict(d["site_space"]), sites=np.array(d["sites"], dtype=int)
        )
        sublattice.active_sites = np.array(d["active_sites"], dtype=int)
        sublattice.encoding = np.array(d["encoding"], dtype=int)
        return sublattice
