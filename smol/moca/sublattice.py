"""Implementation of Sublattice class.

A sublattice represents a set of sites in a supercell that have all have
the same site space. More rigourously it represents a substructure of the
random structure supercell being sampled in a Monte Carlo simulation.
"""

__author__ = "Luis Barroso-Luque"

from dataclasses import dataclass, field
import numpy as np
from monty.json import MSONable
from smol.cofe.space.domain import SiteSpace, Vacancy

from pymatgen.core import Composition

@dataclass
class Sublattice(MSONable):
    """Sublattice class.

    A Sublattice is used to represent a subset of supercell sites that have
    the same site space. Rigorously it represents a set of sites in a
    "substructure" of the total structure.

    Attributes:
     site_space (SiteSpace):
        SiteSpace with the allowed species and their random
        state composition.
     sites (ndarray):
        array of site indices for all sites in sublattice
     active_sites (ndarray):
        array of site indices for all unrestricted sites in the sublattice.
     is_active (ndarray):
     encoding (ndarray):
        array of species encoding in integer indices. If not
        given, will initialize as range(len(site_space)).
    """

    site_space: SiteSpace
    sites: np.array
    active_sites: np.array = field(init=False)
    is_active: bool = field(init=False)
    encoding: np.array = field(default=None)

    def __post_init__(self):
        """Copy sites into active_sites."""
        self.active_sites = self.sites.copy()
        if len(self.site_space) <= 1:
            self.is_active = False
        else:
            self.is_active = True

        if self.encoding is None:
            self.encoding = np.arange(len(self.site_space), dtype=int)
        elif len(self.encoding) != len(self.site_space):
            raise ValueError(f"Encoding size: {len(self.encoding)}"
                             " is not equal "
                             f"to number of species: {len(self.site_space)}!")

    @property
    def species(self):
        """Get allowed species for sites in sublattice."""
        return tuple(self.site_space.keys())

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
        self.active_sites = np.array([i for i in self.active_sites
                                      if i not in sites])
        if len(self.active_sites) == 0:
            self.is_active = False

    def reset_restricted_sites(self):
        """Reset all restricted sites to active."""
        self.active_sites = self.sites.copy()
        self.is_active = True

    def mute(self):
        """Restrict all sites."""
        self.restrict_sites(self.sites)

    def split_by_species_at_occupancy(self, occu, species_in_partitions):
        """Split a sublattice into multiple by specie.

        An example use case might be simulating topotactic Li extraction
        and insertion, where we want to consider Li/Vac, TM and O as
        different sub-lattices that can not be mixed by swapping.

        Args:
            occu (np.ndarray):
                An occupancy array to reference with.
            species_in_partitions (List[List[int]]):
                Each sub-list contains integer encodings of species in
                the site space. For each sub-list, create a new sub-lattice
                with site-space including species in the sub-list, and
                with sites including occu[sites] == specie in the sub-list.
        Returns:
            List of new sub-lattices.
                List[Sublattice]
        """
        part_sublattices = []
        for species_codes in species_in_partitions:
            part_comp = {}
            part_sites = []
            # Because site space species were sorted.
            part_codes = sorted(species_codes)
            for code in part_codes:
                sp = self.species[code]
                part_comp[sp] = self.site_space[sp]
                part_sites.extend(self.sites(occu[self.sites] == code)
                                  .tolist())
            part_n = sum(list(part_comp.values()))
            part_comp = {sp: part_comp[sp] / part_n for sp in part_comp
                         if not isinstance(sp, Vacancy)}
            part_comp = Composition(part_comp)
            part_space = SiteSpace(part_comp)
            part_sites = np.array(part_sites, dtype=int)
            part_codes = np.array(part_codes, dtype=int)
            part_sublattices.append(Sublattice(part_space,
                                               part_sites, part_codes))
        return part_sublattices

    def as_dict(self):
        """Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {'site_space': self.site_space.as_dict(),
             'sites': self.sites.tolist(),
             'encoding': self.encoding.tolist(),
             'active_sites': self.active_sites.tolist()}
        return d

    @classmethod
    def from_dict(cls, d):
        """Instantiate a sublattice from dict representation.

        Returns:
            Sublattice
        """
        sublattice = cls(SiteSpace.from_dict(d['site_space']),
                         sites=np.array(d['sites']),
                         encoding=np.array(d.get('encoding'), dtype=int))
        sublattice.active_sites = np.array(d['active_sites'])
        return sublattice

# The InactiveSublattice class is not necessary, thus removed.
