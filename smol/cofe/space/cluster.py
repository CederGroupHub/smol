"""Implementation of the Cluster class.

Represents a group of sites of a given lattice. These are the building blocks
for a cluster basis of functions over configurational space.
"""

__author__ = "Luis Barroso-Luque, William Davidson Richard"

import numpy as np
from monty.json import MSONable
from pymatgen.core import Lattice
from pymatgen.util.coord import is_coord_subset

from smol.cofe.space.constants import SITE_TOL
from smol.utils import _repr


class Cluster(MSONable):
    """An undecorated (no occupancies) cluster.

    Represented simply by a list of sites, its centroid, and the underlying
    lattice.

    You probably never need to instantiate this class directly. Look at
    ClusterSubspace to create orbits and clusters necessary for a CE.

    Attributes:
        sites (list): List of fractional coordinates of each site.
        lattice (Lattice): Underlying lattice of cluster.
        centroid (float): Geometric centroid of included sites.
        id (int): ID of cluster.
            Used to identify the Cluster in a given ClusterSubspace.
    """

    def __init__(self, sites, lattice):
        """Initialize Cluster.

        Args:
            sites (list):
                list of frac coords for the sites
            lattice (Lattice):
                pymatgen Lattice object
        """
        sites = np.array(sites)
        centroid = np.average(sites, axis=0)
        shift = np.floor(centroid)
        self.centroid = centroid - shift
        self.sites = sites - shift
        self.lattice = lattice
        self.id = None

    @classmethod
    def from_sites(cls, sites):
        """Create a cluster from a list of pymatgen Sites."""
        return cls([s.frac_coords for s in sites], sites[0].lattice)

    @property  # TODO deprecate this
    def size(self):
        """Get number of sites in the cluster."""
        return len(self.sites)

    @property
    def diameter(self):
        """Get maximum distance between any 2 sites in the cluster."""
        coords = self.lattice.get_cartesian_coords(self.sites)
        all_d2 = np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1)
        return np.max(all_d2) ** 0.5

    @property
    def radius(self):
        """Get half the maximum distance between any 2 sites in the cluster."""
        return self.diameter / 2.0

    def assign_ids(self, cluster_id):
        """Recursively assign IDs to clusters after initialization."""
        self.id = cluster_id
        return cluster_id + 1

    def __len__(self):
        """Get size of a cluster. The number of sites."""
        return len(self.sites)

    def __eq__(self, other):
        """Check equivalency of clusters considering symmetry."""
        if self.sites.shape != other.sites.shape:
            return False
        othersites = other.sites + np.round(self.centroid - other.centroid)
        return is_coord_subset(self.sites, othersites, atol=SITE_TOL)

    def __neq__(self, other):
        """Non equivalency."""
        return not self.__eq__(other)

    def __str__(self):
        """Pretty print a cluster."""
        points = str(np.round(self.sites, 2))
        points = points.replace("\n", " ").ljust(len(self.sites) * 21)
        centroid = str(np.round(self.centroid, 2))
        return (
            f"[Base Cluster] Radius: {self.radius:<5.3} "
            f"Centroid: {centroid:<18} Points: {points}"
        )

    def __repr__(self):
        """Pretty representation."""
        return _repr(
            self,
            c_id=self.id,
            diameter=self.diameter,
            centroid=self.centroid,
            lattice=self.lattice,
        )

    @classmethod
    def from_dict(cls, d):
        """Create a Cluster from serialized dict."""
        return cls(d["sites"], Lattice.from_dict(d["lattice"]))

    def as_dict(self):
        """Get json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "lattice": self.lattice.as_dict(),
            "sites": self.sites.tolist(),
        }
        return d
