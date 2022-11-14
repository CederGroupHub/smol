"""Implementation of the Cluster class.

Represents a group of sites of a given lattice. These are the building blocks
for a cluster basis of functions over configurational space.
"""

__author__ = "Luis Barroso-Luque, William Davidson Richard"

import json
import os
from fnmatch import fnmatch
from functools import cached_property
from io import StringIO

import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Lattice, Site
from pymatgen.core.structure import Composition, SiteCollection
from pymatgen.util.coord import is_coord_subset
from ruamel.yaml import YAML

from smol.cofe.space.constants import SITE_TOL
from smol.cofe.space.domain import Vacancy, get_site_spaces


class Cluster(SiteCollection, MSONable):
    """An undecorated (no occupancies) cluster.

    Represented simply by a list of sites, its centroid, and the underlying
    lattice.

    You probably never need to instantiate this class directly. Look at
    ClusterSubspace to create orbits and clusters necessary for a CE.

    Attributes:
        frac_coords (ndarray): fractional coordinates of each site.
        lattice (Lattice): Underlying lattice of cluster.
        centroid (float): Geometric centroid of included sites.
        id (int): ID of cluster.
            Used to identify the Cluster in a given ClusterSubspace.
    """

    def __init__(self, site_spaces, frac_coords, lattice):
        """Initialize Cluster.

        Args:
            site_spaces (list of SiteSpace):
                list of site spaces for the cluster
            frac_coords (Sequence):
                Sequence of frac coords for the site spaces
            lattice (Lattice):
                pymatgen Lattice object
        """
        frac_coords = np.array(frac_coords)
        centroid = np.average(frac_coords, axis=0)
        shift = np.floor(centroid)
        self._centroid = centroid - shift
        self._frac_coords = frac_coords - shift
        self._sites = tuple(
            Site(site_space, coords)
            for site_space, coords in zip(
                site_spaces, lattice.get_cartesian_coords(frac_coords)
            )
        )
        self._lattice = lattice
        self.id = None

    @property
    def centroid(self):
        """Return the centroid of cluster."""
        return self._centroid

    @property
    def frac_coords(self):
        """Return the fractional coordinates of cluster w.r.t the underlying lattice."""
        return self._frac_coords

    @cached_property
    def diameter(self):
        """Get maximum distance between any 2 sites in the cluster."""
        coords = self.lattice.get_cartesian_coords(self.frac_coords)
        all_d2 = np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1)
        return np.max(all_d2) ** 0.5

    @property
    def radius(self):
        """Get half the maximum distance between any 2 sites in the cluster."""
        return self.diameter / 2.0

    @property
    def lattice(self):
        """Return the underlying lattice."""
        return self._lattice

    @property
    def sites(self):
        """Return the list of sites."""
        return self._sites

    def get_distance(self, i: int, j: int) -> float:
        """Return distance between sites at index i and j.

        Args:
            i: Index of first site
            j: Index of second site
        Returns:
            Distance between sites at index i and index j.
        """
        return self[i].distance(self[j])

    def assign_ids(self, cluster_id):
        """Recursively assign IDs to clusters after initialization."""
        self.id = cluster_id
        return cluster_id + 1

    def to(self, fmt: str = None, filename: str = None):
        """
        Output the cluster  to a file or string.

        this is basically a watered down version of pymatgen.Molecule.to

        Args:
            fmt (str): Format to output to. Defaults to JSON unless filename
                is provided. If fmt is specifies, it overrides whatever the
                filename is. Options include "yaml" and "json" only.
                 Non-case sensitive.
            filename (str): If provided, output will be written to a file. If
                fmt is not specified, the format is determined from the
                filename. Defaults is None, i.e. string output.
        Returns:
            (str) if filename is None. None otherwise.
        """
        fmt = "" if fmt is None else fmt.lower()
        fname = os.path.basename(filename or "")
        if fmt == "json" or fnmatch(fname, "*.json*") or fnmatch(fname, "*.mson*"):
            if filename:
                with zopen(filename, "wt", encoding="utf8") as f:
                    return json.dump(self.as_dict(), f)
            else:
                return json.dumps(self.as_dict())

        if fmt == "yaml" or fnmatch(fname, "*.yaml*"):
            yaml = YAML()
            if filename:
                with zopen(fname, "wt", encoding="utf8") as f:
                    return yaml.dump(self.as_dict(), f)
            else:
                sio = StringIO()
                yaml.dump(self.as_dict(), sio)
                return sio.getvalue()

        raise ValueError(f"Invalid format: `{str(fmt)}`")

    @classmethod
    def from_str(cls, input_string: str, fmt):
        """
        Read a cluster from a string.

        Args:
            input_string (str): String to parse.
            fmt (str): Format to output to. Defaults to JSON unless filename
                is provided. If fmt is specifies, it overrides whatever the
                filename is. Options include "yaml", "json". Non-case sensitive.
        Returns:
            Cluster
        """
        if fmt == "json":
            d = json.loads(input_string)
            return cls.from_dict(d)
        if fmt == "yaml":
            yaml = YAML()
            d = yaml.load(input_string)
            return cls.from_dict(d)

        raise ValueError(f"Invalid format: `{str(fmt)}`")

    @classmethod
    def from_file(cls, filename: str):
        """
        Read a cluster from a file. Supported formats are json and yaml only.

        Args:
            filename (str): The filename to read from.
        Returns:
            Cluster
        """
        filename = str(filename)

        with zopen(filename) as f:
            contents = f.read()
        fname = filename.lower()

        if fnmatch(fname, "*.json*") or fnmatch(fname, "*.mson*"):
            return cls.from_str(contents, fmt="json")
        if fnmatch(fname, "*.yaml*"):
            return cls.from_str(contents, fmt="yaml")

        raise ValueError("Cannot determine file type.")

    @classmethod
    def from_sites(cls, sites, lattice):
        """Create a cluster from a list of sites and lattice object."""
        frac_coords = [lattice.get_fractional_coords(site.coords) for site in sites]
        site_spaces = get_site_spaces(sites)
        return cls(site_spaces, frac_coords, lattice)

    def __eq__(self, other):
        """Check equivalency of clusters considering symmetry."""
        if self.frac_coords.shape != other.frac_coords.shape:
            return False
        othersites = other.frac_coords + np.round(self.centroid - other.centroid)
        return is_coord_subset(self.frac_coords, othersites, atol=SITE_TOL)

    def __str__(self):
        """Pretty print a cluster."""
        centroid_str = " ".join(
            [
                f"{j:0.6f}".rjust(12)
                for j in self.lattice.get_cartesian_coords(self.centroid)
            ]
        )
        centroid_str += "  -> " + " ".join(
            [f"{j:0.6f}".rjust(12) for j in self.centroid]
        )
        outs = [
            f"Diameter : {self.diameter:0.4f}",
            f"  Charge : {self.charge}",
            f"Centroid : {centroid_str}",
            f"Sites ({len(self)})",
        ]
        site_outs = []
        for i, site in enumerate(self):
            site_outs.append(
                " ".join(
                    [
                        str(i),
                        site.species_string,
                        " ".join([f"{j:0.6f}".rjust(12) for j in site.coords]),
                        "  ->",
                        " ".join(
                            [
                                f"{j:0.6f}".rjust(12)
                                for j in self.lattice.get_fractional_coords(site.coords)
                            ]
                        ),
                    ]
                )
            )
        width = len(site_outs[0])
        outs.append(width * "-")
        outs += site_outs
        return "\n".join(outs)

    def __repr__(self):
        """Get cluster summary."""
        centroid_str = "({:.4f}, {:.4f}, {:.4f})".format(
            *self.lattice.get_cartesian_coords(self.centroid)
        ) + " -> [{:.4f}, {:.4f}, {:.4f}]".format(*self.centroid)
        outs = [
            "Cluster",
            f"No. sites: {len(self._sites)}   Diameter: {self.diameter:0.4f}",
            f"Centroid: {centroid_str}",
        ]
        for s in self:
            outs.append(
                s.__repr__()
                + " -> [{:.4f}, {:.4f}, {:.4f}]".format(
                    *self.lattice.get_fractional_coords(s.coords)
                )
            )
        return "\n".join(outs)

    @classmethod
    def from_dict(cls, d):
        """Create a Cluster from serialized dict."""
        sites = [Site.from_dict(item) for item in d["sites"]]
        # Force vacancies back to vacancies
        for symbols, site in zip(d["vacancy_symbols"], sites):
            site.species = Composition(
                {
                    spec
                    if spec.symbol not in symbols
                    else Vacancy(
                        spec.symbol, spec.oxidation_state, spec.properties
                    ): val
                    for spec, val in site.species.items()
                    if spec.symbol not in symbols
                }
            )

        return Cluster.from_sites(sites, Lattice.from_dict(d["lattice"]))

    def as_dict(self):
        """Get json-serialization dict representation.

        Returns:
            MSONable dict
        """
        cluster_d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "lattice": self.lattice.as_dict(),
            "sites": [site.as_dict() for site in self.sites],
            "vacancy_symbols": [
                [spec.symbol for spec in site.species if isinstance(spec, Vacancy)]
                for site in self.sites
            ],
        }
        return cluster_d


# functions to render cluster using crystal-toolkit
try:
    from collections import defaultdict

    from crystal_toolkit.core.legend import Legend
    from crystal_toolkit.core.scene import Scene

    def get_cluster_scene(
        self,
        origin=None,
        legend=None,
    ) -> Scene:
        """
        Create CTK objects for the lattice and sties.

        Args:
            self:  Structure object
            origin: coordinate of the origin
            legend: Legend for the sites

        Returns:
            CTK scene object to be rendered
        """
        origin = origin or list(-self.lattice.get_cartesian_coords([0.5, 0.5, 0.5]))
        legend = legend or Legend(self)
        primitives = defaultdict(list)

        for site in self:
            site_scene = site.get_scene(
                origin=origin,
                legend=legend,
            )
            for scene in site_scene.contents:
                primitives[scene.name] += scene.contents

        primitives["unit_cell"].append(self.lattice.get_scene())

        return Scene(
            name="Cluster",
            origin=origin,
            contents=[
                Scene(name=k, contents=v, origin=origin) for k, v in primitives.items()
            ],
        )

    Cluster.get_scene = get_cluster_scene

except ImportError:
    pass
