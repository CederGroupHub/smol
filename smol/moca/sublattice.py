"""Implementation of Sublattice class.

A sublattice represents a set of sites in a supercell that have all have
the same site space. More rigourously it represents a substructure of the
random structure supercell being sampled in a Monte Carlo simulation.
"""

__author__ = "Luis Barroso-Luque"

from dataclasses import dataclass, field
import numpy as np
from monty.json import MSONable
from smol.cofe.space.domain import SiteSpace


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
    """

    site_space: SiteSpace
    sites: np.ndarray
    active_sites: np.ndarray = field(init=False)

    def __post_init__(self):
        """Copy sites into active_sites."""
        self.active_sites = self.sites.copy()

    @property
    def species(self):
        """Get allowed species for sites in sublattice."""
        return tuple(self.site_space.keys())

    @property
    def encoding(self):
        """Get the encoding for the allowed species."""
        return list(range(len(self.site_space)))

    @property
    def restricted_sites(self):
        """Get restricted sites for species."""
        return np.setdiff1d(self.sites, self.active_sites)

    def restrict_sites(self, sites):
        """Restricts (freezes) the given sites.

        Args:
            sites (Sequence):
                indices of sites in the occupancy string to restrict.
        """
        self.active_sites = np.array([i for i in self.active_sites
                                      if i not in sites])

    def reset_restricted_sites(self):
        """Reset all restricted sites to active."""
        self.active_sites = self.sites.copy()

    def as_dict(self):
        """Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {'site_space': self.site_space.as_dict(),
             'sites': self.sites.tolist(),
             'active_sites': self.active_sites.tolist()}
        return d

    @classmethod
    def from_dict(cls, d):
        """Instantiate a sublattice from dict representation.

        Returns:
            Sublattice
        """
        sublattice = cls(SiteSpace.from_dict(d['site_space']),
                         sites=np.array(d['sites']))
        sublattice.active_sites = np.array(d['active_sites'])
        return sublattice


@dataclass
class InactiveSublattice(MSONable):
    """Same as above but for sublattices with no configuration DOFs.

    Attributes:
     site_space (SiteSpace):
        SiteSpace with the allowed species and their random
        state composition.
     sites (ndarray):
        array of site indices for all sites in sublattice
    """

    site_space: SiteSpace
    sites: np.ndarray

    def as_dict(self):
        """Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {'site_space': self.site_space.as_dict(),
             'sites': self.sites.tolist()}
        return d

    @classmethod
    def from_dict(cls, d):
        """Instantiate a sublattice from dict representation.

        Returns:
            Sublattice
        """
        return cls(SiteSpace.from_dict(d['site_space']), np.array(d['sites']))
