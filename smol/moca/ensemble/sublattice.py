"""Implementation of Sublattice class.

A sublattice represents a set of sites in a supercell that have all have
the same site space. It more rigourously represents a substructure of the
random structure supercell being sampled in a Monte Carlo simulation.
"""

__author__ = "Luis Barroso-Luque"

from collections import OrderedDict
import numpy as np
from monty.json import MSONable


def get_sublattices(processor):
    """Get a list of sublattices from a processor

    Args:
        processor (Processor):
            A processor object to extract sublattices from
    Returns:
        list of Sublattice
    """
    return [Sublattice(site_space,
                       np.array([i for i, sp in
                                 enumerate(processor.allowed_species)
                                if sp == list(site_space.keys())]))
            for site_space in processor.unique_site_spaces]


# TODO consider adding the inactive sublattices?
class Sublattice(MSONable):
    """Sublattice class.

     A Sublattice is used to represent a subset of supercell sites that have
     the same site space.

     Attributes:
         site_space (OrderedDict):
            Ordered dict with the allowed species and their random
            state composition. See definitions in cofe.cofigspace.basis
         species (tuple)
            Allowed species at each site. A tuple of site_space keys
         sites (ndarray):
            array of site indices for all sites in sublattice
         active_sites (ndarray):
            array of site indices for all unrestricted sites in the sublattice.
         restricted_sites (ndarray):
            list of site indices for all restricted sites in the sublattice.
            restricted sites are excluded from flip proposals.

    """

    def __init__(self, site_space, sites):
        """Initialize Sublattice.

        Args:
            site_space (OrderedDict):
                An ordered dict with the allowed species and their random
                state composition. See definitions in cofe.cofigspace.basis
            sites (ndarray):
                array with the site indices
        """
        self.sites = sites
        self.site_space = site_space
        self.species = tuple(site_space.keys())
        self.active_sites = sites.copy()
        self.restricted_sites = []

    def restrict_sites(self, sites):
        """Restricts (freezes) the given sites.
        Args:
            sites (Sequence):
                indices of sites in the occupancy string to restrict.
        """
        self.active_sites = np.array([i for i in self.active_sites
                                      if i not in sites])
        self.restricted_sites += [i for i in sites
                                  if i not in self.restricted_sites]

    def reset_restricted_sites(self):
        """Resets all restricted sites to active."""
        self.active_sites = self.sites.copy()
        self.restricted_sites = []

    def __str__(self):
        """Pretty print the sublattice species."""
        string = f'Sublattice\n Site space: {dict(self.site_space)}\n'
        string += f' Number of sites: {len(self.sites)}\n'
        return string

    def __repr__(self):
        """Repr for nice viewing."""
        rep = f'Sublattice Summary \n\n   site_space: {self.site_space}\n\n'
        rep += f'   sites: {self.sites}\n\n active_sites: {self.active_sites}'
        return rep

    def as_dict(self):
        """Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {'site_space': self.site_space,
             'sites': self.sites.tolist(),
             'active_sites': self.active_sites.tolist(),
             'restricted_sites': self.restricted_sites}
        return d

    @classmethod
    def from_dict(cls, d):
        """Instantiate a sublattice from dict representation.

        Args:
            d (dict):
                dictionary representation.
        Returns:
            Sublattice
        """
        sublattice = cls(OrderedDict(d['site_space']),  # order conserved?
                         sites=np.array(d['sites']))
        sublattice.active_sites = np.array(d['active_sites'])
        sublattice.restricted_sites = d['restricted_sites']
        return sublattice
