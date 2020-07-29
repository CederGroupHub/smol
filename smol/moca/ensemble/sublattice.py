"""Implementation of Sublattice class.

A sublattice represents a set of sites in a supercell that have all have
the same site space. It more rigourously represents a substructure of the
random structure supercell being sampled in a Monte Carlo simulation.
"""

__author__ = "Luis Barroso-Luque"

import numpy as np


class Sublattice:
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
        return '/'.join(self.species)
