"""Implementation of Sublattice class.

A sublattice represents a set of sites in a supercell that have all have
the same site space. More rigourously it represents a substructure of the
random structure supercell being sampled in a Monte Carlo simulation.
"""

__author__ = "Luis Barroso-Luque"

import numpy as np
from monty.json import MSONable
from smol.cofe.space.domain import SiteSpace, get_site_spaces


def get_sublattices(processor):
    """Get a list of sublattices from a processor.

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


def get_all_sublattices(processor):
    """Get a list of all sublattices from a processor.

    Will include all sublattices, active or not.

    This is only to be used by the charge neutral ensembles.

    Args:
        processor (Processor):
            A processor object to extract sublattices from.
    Returns:
        list of Sublattice, containing all sites, even
        if only occupied by one specie.
    """
    # Must keep the same order as processor.unique_site_spaces.
    unique_site_spaces = tuple(set(get_site_spaces(
                               processor.cluster_subspace.structure)))

    return [Sublattice(site_space,
            np.array([i for i, sp in enumerate(processor.allowed_species)
                     if sp == list(site_space.keys())]))
            for site_space in unique_site_spaces]


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

    def __init__(self, site_space, sites):
        """Initialize Sublattice.

        Args:
            site_space (SiteSpace):
                A site space object representing the sites in the sublattice
            sites (ndarray):
                array with the site indices
        """
        self.sites = sites
        self.site_space = site_space
        self.active_sites = sites.copy()

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
