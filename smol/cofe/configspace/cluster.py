"""
Implementation of the Cluster class, which represents a group of sites of a
given lattice. Aka the building blocks for a cluster basis of functions over
configurational space
"""

from __future__ import division
import numpy as np
from monty.json import MSONable
from pymatgen.util.coord import is_coord_subset
from pymatgen import Lattice

from smol.cofe.configspace.utils import SITE_TOL, _repr


class Cluster(MSONable):
    """
    An undecorated (no occupancies) cluster
    """

    def __init__(self, sites, lattice):
        """
        Args:
            sites (list):
                list of frac coords for the sites
            lattice (pymatgen.Lattice):
                pymatgen Lattice object
        """
        sites = np.array(sites)
        centroid = np.average(sites, axis=0)
        shift = np.floor(centroid)
        self.centroid = centroid - shift
        self.sites = sites - shift
        self.lattice = lattice
        self.id = None

    @staticmethod
    def from_sites(sites):
        return Cluster([s.frac_coords for s in sites], sites[0].lattice)

    @property
    def size(self):
        return len(self.sites)

    @property
    def radius(self):
        coords = self.lattice.get_cartesian_coords(self.sites)
        all_d2 = np.sum((coords[None, :, :] - coords[:, None, :])**2, axis=-1)
        return np.max(all_d2) ** 0.5

    def assign_ids(self, id):
        """
        Method to recursively assign ids to clusters after initialization.
        """
        self.id = id
        return id + 1

    def __eq__(self, other):
        if self.sites.shape != other.sites.shape:
            return False
        othersites = other.sites + np.round(self.centroid - other.centroid)
        return is_coord_subset(self.sites, othersites, atol=SITE_TOL)

    def __neq__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        points = str(np.round(self.sites, 2))
        points = points.replace('\n', ' ').ljust(len(self.sites)*21)
        centroid = str(np.round(self.centroid, 2))
        return (f'[Cluster] id: {self.id:<4} Radius: {self.radius:<5.3} '
                f'Centroid: {centroid:<18} Points: {points}')

    def __repr__(self):
        return _repr(self, c_id=self.id, radius=self.radius,
                     centroid=self.centroid, lattice=self.lattice)

    @classmethod
    def from_dict(cls, d):
        """
        Creates a cluster from serialized dict
        """
        return cls(d['sites'], Lattice.from_dict(d['lattice']))

    def as_dict(self):
        """
        Json-serialization dict representation

        Returns:
            MSONable dict
        """
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "lattice": self.lattice.as_dict(),
             "sites": self.sites.tolist()}
        return d
