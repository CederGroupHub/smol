"""Implementation of an Orbit.

A set of symmetrically equivalent (with respect to the given random structure
symmetry) clusters.
"""

import operator
from functools import reduce
from itertools import chain, combinations, product

import numpy as np
from monty.json import MSONable
from pymatgen.core import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.util.coord import coord_list_mapping, is_coord_subset

from smol.exceptions import SYMMETRY_ERROR_MESSAGE, SymmetryError
from smol.utils import _repr

from .basis import DiscreteBasis, basis_factory
from .cluster import Cluster
from .constants import SITE_TOL

__author__ = "Luis Barroso-Luque, William Davidson Richard"


class Orbit(MSONable):
    """Orbit set of symmetrically equivalent clusters.

    An Orbit represents a set of clusters that are symmetrically equivalent
    in the random structure. The class also includes the possible orderings on
    the clusters in the orbit. The different orderings represent the single
    site function indices to generate all possible orbit functions (correlation
    functions) for the given orbit.

    An orbit usually includes translational and structure symmetry of the
    underlying lattice. But this is not a hard requirement any set of symmetry
    operations can be passed to the constructor (regardless an orbit should at
    a minimum have translational symmetry).

    You probably never need to instantiate this class directly. Look at
    ClusterSubspace to create orbits and clusters necessary for a CE.

    Attributes:
        bits (list of list):
            list of lists describing the posible non-constant site function
            indices at each site of a cluster in the orbit.
        site_bases (list of DiscreteBasis):
            list of the SiteBasis for each site.
        structure_symops (list of Symmops):
            list of underlying structure symmetry operations.
        lattice (Lattice):
            underlying structure lattice.
    """

    def __init__(self, frac_coords, lattice, bits, site_bases, structure_symops):
        """Initialize an Orbit.

        Args:
            frac_coords (list or ndarray):
                list of frac coords for the sites
            lattice (pymatgen.Lattice):
                A lattice object for the given sites
            bits (list of list):
                list describing the possible site function indices for
                each site in cluster. Should be the number of possible
                occupancies minus one. i.e. for a 3 site cluster, each of which
                having one of Li, TM, or Vac, bits are [[0, 1], [0, 1], [0, 1]]
                This is ensures the expansion is not "over-complete" by
                implicitly enforcing that all sites have a site basis function
                phi_0 = 1.
            site_bases (list of DiscreteBasis):
                list of SiteBasis objects for each site in the given sites.
            structure_symops (list of SymmOp):
                list of symmetry operations for the base structure
        """
        if len(frac_coords) != len(bits):
            raise AttributeError(
                f"Number of sites {len(frac_coords)} must be equal to number of "
                f"bits {len(bits)}"
            )
        elif len(frac_coords) != len(site_bases):
            raise AttributeError(
                f"Number of sites {len(frac_coords)} must be equal to number of "
                f"site bases {len(site_bases)}"
            )

        self.bits = bits
        self.site_bases = site_bases
        self.structure_symops = structure_symops

        # ids should be assigned using the assign_id method externally
        self.id = None  # id identifying orbit amongst all other orbits only
        self.bit_id = None  # id for first bit combo in this orbit
        # considering all the bit combos in all orbits.

        # lazy generation of properties
        self._equiv = None
        self._symops = None
        self._permutations = None
        self._bit_combos = None
        self._basis_arrs = None
        self._corr_tensors = None
        self._flat_corr_tensors = None

        # Create basecluster
        self.base_cluster = Cluster(
            [site_basis.site_space for site_basis in site_bases], frac_coords, lattice
        )

    @property
    def basis_type(self):
        """Return the name of basis set used."""
        return self.site_bases[0].flavor

    @property
    def basis_orthogonal(self):
        """Test if the Orbit bases are orthogonal."""
        return all(basis.is_orthogonal for basis in self.site_bases)

    @property
    def basis_orthonormal(self):
        """Test if the orbit bases are orthonormal."""
        return all(basis.is_orthonormal for basis in self.site_bases)

    @property
    def multiplicity(self):
        """Get number of clusters in orbit.

        Number of clusters in the given sites of the lattice object that are in
        the orbit.
        """
        return len(self.clusters)

    @property
    def bit_combos(self):
        """Get tuple of site bit orderings.

        tuple of ndarrays, each array is a set of symmetrically equivalent bit
        orderings represented by row. Bit combos represent non-constant site
        function orderings.
        """
        if self._bit_combos is None:
            # get all the bit symmetry operations
            all_combos = []
            for bit_combo in product(*self.bits):
                if not any(np.array_equal(bit_combo, bc) for bc in chain(*all_combos)):
                    bit_combo = np.array(bit_combo, dtype=np.int_)
                    new_bits = np.unique(bit_combo[self.cluster_permutations], axis=0)
                    all_combos.append(new_bits)
            self._bit_combos = tuple(all_combos)
        return self._bit_combos

    @property
    def site_spaces(self):
        """Get the site spaces for the site basis associate with each site."""
        return [site_basis.site_space for site_basis in self.site_bases]

    @property
    def bit_combo_multiplicities(self):
        """Get the multiplicites of the symmetrically distinct bit ordering."""
        return [bcombo.shape[0] for bcombo in self.bit_combos]

    @property
    def clusters(self):
        """Get symmetrically equivalent clusters."""
        if self._equiv:
            return self._equiv
        equiv = [self.base_cluster]
        for symop in self.structure_symops:
            new_coords = symop.operate_multi(self.base_cluster.frac_coords)
            c = Cluster(
                [site_basis.site_space for site_basis in self.site_bases],
                new_coords,
                self.base_cluster.lattice,
            )
            if c not in equiv:
                equiv.append(c)
        self._equiv = equiv
        if len(equiv) * len(self.cluster_symops) != len(self.structure_symops):
            self._equiv = None  # Unset this
            raise SymmetryError(SYMMETRY_ERROR_MESSAGE)
        return equiv

    @property
    def cluster_symops(self):
        """Get symmetry operations that map a cluster to its periodic image.

        Each element is a pymatgen.core.operations.Symop.
        """
        if self._symops is None:
            self._gen_cluster_symops()
        return self._symops

    @property
    def cluster_permutations(self):
        """Get the symmetrical site permutations that map a cluster to itself.

        A permutation is a mapping such that for a give symop in cluster_symops
        Symop.operate(sites) = sites[mapping] (after translation back to unit
        cell)
        """
        if self._permutations is None:
            self._gen_cluster_symops()
        return self._permutations

    @property
    def basis_arrays(self):
        """Get a tuple of all site function arrays for each site in orbit."""
        if self._basis_arrs is None:
            self._basis_arrs = tuple(sb.function_array for sb in self.site_bases)
        return self._basis_arrs

    @property
    def correlation_tensors(self):
        """Get the array of correlation functions for all possible configs.

        Array of stacked correlation arrays for each symetrically distinct
        set of bit combos.

        The correlations array is a multidimensional array with each dimension
        corresponding to each site space.

        First dimension is for bit combos, the remainding dimensions correspond
        to site spaces.

        i.e. correlation_tensors[0, 1, 0, 2] gives the value of the
        correlation function for bit_combo 0 evaluated for a cluster with
        occupancy [1, 0, 2]
        """
        if self._corr_tensors is None:
            corr_tensors = np.zeros(
                (len(self.bit_combos), *(basis.shape[1] for basis in self.basis_arrays))
            )

            for i, combos in enumerate(self.bit_combos):
                for bits in combos:
                    corr_tensors[i] += reduce(
                        lambda a, b: np.tensordot(a, b, axes=0),
                        (self.basis_arrays[i][b] for i, b in enumerate(bits)),
                    )
                corr_tensors[i] /= len(combos)
            self._flat_corr_tensors = None  # reset
            self._corr_tensors = corr_tensors
        return self._corr_tensors

    @property
    def flat_correlation_tensors(self):
        """Get correlation_tensors flattened to 2D for fast cython."""
        if self._flat_corr_tensors is None:
            self._flat_corr_tensors = np.ascontiguousarray(
                np.reshape(
                    self.correlation_tensors,
                    (
                        self.correlation_tensors.shape[0],
                        np.prod(self.correlation_tensors.shape[1:]),
                    ),
                    order="C",
                )
            )
        return self._flat_corr_tensors

    @property
    def flat_tensor_indices(self):
        """Index multipliers to read data easier from flat corr tensors."""
        indices = np.cumprod(np.append(self.correlation_tensors.shape[2:], 1)[::-1])[
            ::-1
        ]
        return np.ascontiguousarray(indices, dtype=int)

    @property
    def rotation_array(self):
        """Get the rotation array.

        The rotation array is of size len(bit combos) x len(bit combos)
        """
        R = np.empty(2 * (len(self._bit_combos),))
        for (i, j), (bcombos_i, bcombos_j) in zip(
            product(range(len(self._bit_combos)), repeat=2),
            product(self._bit_combos, repeat=2),
        ):
            R[i, j] = sum(
                reduce(
                    operator.mul,
                    (
                        np.dot(
                            self.site_bases[k].rotation_array.T
                            @ self.basis_arrays[k][bj],  # noqa
                            self.site_bases[k].measure_vector
                            * self.basis_arrays[k][bi],  # noqa
                        )
                        for k, (bi, bj) in enumerate(zip(bcombo_i, bcombo_j))
                    ),
                )
                for bcombo_i, bcombo_j in product(bcombos_i, bcombos_j)
            ) / len(bcombos_i)
            # \ (len(bcombos_i) * len(bcombos_j))**0.5 is unitary
        return R

    def remove_bit_combo(self, bits):  # seems like this is no longer used?
        """Remove bit_combos from orbit.

        Only a single set bits in the bit combo (symmetrically equivalent bit
        orderings) needs to be passed.
        """
        bit_combos = []

        for bit_combo in self.bit_combos:
            if not any(np.array_equal(bits, b) for b in bit_combo):
                bit_combos.append(bit_combo)

        if not bit_combos:
            raise RuntimeError(
                "All bit_combos have been removed from orbit with id " f"{self.id}"
            )

        self._bit_combos = tuple(bit_combos)
        self.reset_bases()

    def remove_bit_combos_by_inds(self, inds):
        """Remove bit combos by their indices in the bit_combo list."""
        if max(inds) > len(self.bit_combos) - 1:
            raise ValueError(
                f"Some indices {inds} out of range for total "
                f"{len(self._bit_combos)} bit combos"
            )

        self._bit_combos = tuple(
            b_c for i, b_c in enumerate(self._bit_combos) if i not in inds
        )

        if not self.bit_combos:
            raise RuntimeError(
                "All bit_combos have been removed from orbit with id " f"{self.id}"
            )
        self.reset_bases()

    def transform_site_bases(self, basis_name, orthonormal=False):
        """Transform the Orbits site bases to new basis set.

        Args:
            basis_name (str):
                name of new basis for all site bases
            orthonormal (bool):
                option to orthonormalize all new site bases
        """
        new_bases = []
        for basis in self.site_bases:
            new_basis = basis_factory(basis_name, basis.site_space)
            if orthonormal:
                new_basis.orthonormalize()
            new_bases.append(new_basis)

        self.site_bases = tuple(new_bases)
        self.reset_bases()

    def is_sub_orbit(self, orbit):
        """Check if given orbits clusters are subclusters.

        Note this does not consider bit_combos
        Args:
            orbit (Orbit):
                Orbit object to check if
        Returns:
            bool: True if the clusters of given orbit are subclusters.
        """
        if len(self.base_cluster) <= len(orbit.base_cluster):
            return False
        elif not np.all(sp in self.site_spaces for sp in orbit.site_spaces):
            return False

        match = any(
            Cluster(
                [site_basis.site_space for site_basis in self.site_bases],
                self.base_cluster.frac_coords[inds, :],
                self.base_cluster.lattice,
            )
            in orbit.clusters
            for inds in combinations(
                range(len(self.base_cluster)), len(orbit.base_cluster)
            )
        )

        return match

    def sub_orbit_mappings(self, orbit):
        """Return a mapping of the sites in the orbit to a sub orbit.

        If the given orbit is not a sub-orbit will return an empty list.
        Note this works for mapping between sites, sites spaces, and basis
        functions associated with each site.

        Args:
            orbit (Orbit):
                A sub orbit to return mapping of sites
        Returns:
            list: of indices sucht that
                self.base_cluster.sites[indices] = orbit.base_cluster.sites
        """
        indsets = np.array(
            list(
                combinations(
                    (
                        i
                        for i, space in enumerate(self.site_spaces)
                        if space in orbit.site_spaces
                    ),
                    len(orbit.site_spaces),
                )
            ),
            dtype=int,
        )

        mappings = []
        for cluster in self.clusters:
            for inds in indsets:
                # take the centroid of subset of sites, not all cluster sites
                centroid = np.average(cluster.frac_coords[inds], axis=0)
                recenter = np.round(centroid - orbit.base_cluster.centroid)
                c_sites = orbit.base_cluster.frac_coords + recenter
                if is_coord_subset(c_sites, cluster.frac_coords):
                    mappings.append(
                        coord_list_mapping(c_sites, cluster.frac_coords, atol=SITE_TOL)
                    )

        if len(mappings) == 0 and self.is_sub_orbit(orbit):
            raise RuntimeError(
                "The given orbit is a suborbit, but no site mappings were "
                "found!\n Something is very wrong here!"
            )
        return np.unique(mappings, axis=0)

    def _gen_cluster_symops(self):
        """Generate the cluster SymOps and decoration permutations."""
        symops = []
        permutations = []
        for symop in self.structure_symops:
            new_sites = symop.operate_multi(self.base_cluster.frac_coords)
            cluster = Cluster(
                [site_basis.site_space for site_basis in self.site_bases],
                new_sites,
                self.base_cluster.lattice,
            )
            if cluster == self.base_cluster:
                recenter = np.round(self.base_cluster.centroid - cluster.centroid)
                c_sites = cluster.frac_coords + recenter
                mapping = coord_list_mapping(
                    self.base_cluster.frac_coords, c_sites, atol=SITE_TOL
                )
                symops.append(symop)
                permutations.append(mapping)

        self._permutations = np.unique(permutations, axis=0)
        self._symops = tuple(symops)

        if len(self._symops) * self.multiplicity != len(self.structure_symops):
            raise SymmetryError(SYMMETRY_ERROR_MESSAGE)

    def reset_bases(self):
        """Reset cached basis function array and correlation tensors."""
        self._basis_arrs = None
        self._corr_tensors = None
        self._flat_corr_tensors = None

    def assign_ids(self, orbit_id, orbit_bit_id, start_cluster_id):
        """Assign unique orbit and cluster ids.

        This should be called iteratively for a list of orbits to get a proper
        set of unique id's for the orbits.

        Args:
            orbit_id (int): orbit id
            orbit_bit_id (int): start bit ordering id
            start_cluster_id (int): start cluster id

        Returns:
            (int, int, int):
            next orbit id, next bit ordering id, next cluster id
        """
        self.id = orbit_id
        self.bit_id = orbit_bit_id
        c_id = start_cluster_id
        for c in self.clusters:
            c_id = c.assign_ids(c_id)
        return orbit_id + 1, orbit_bit_id + len(self.bit_combos), c_id

    def __len__(self):
        """Get total number of orbit basis functions.

        The number of symmetrically distinct bit orderings.
        """
        return len(self.bit_combos)

    def __eq__(self, other):
        """Check equality of orbits."""
        # when performing orbit in list, this ordering stops the
        # equivalent structures from generating
        # NOTE: does not compare bit_combos!
        return self.base_cluster in other.clusters

    def __neq__(self, other):
        """Check negation of orbit equality."""
        return not self.__eq__(other)

    def __str__(self):
        """Pretty strings for pretty things."""
        return (
            f"[Orbit] id: {self.id:<3}"
            f"orderings: {len(self):<4}"
            f"multiplicity: {self.multiplicity:<4}"
            f" no. symops: {len(self.cluster_symops):<4}\n"
            f"        {self.site_spaces}\n"
            f"        {str(self.base_cluster)}"
        )

    def __repr__(self):
        """Get Orbit representation."""
        return _repr(
            self,
            orb_id=self.id,
            orb_b_id=self.bit_id,
            radius=self.base_cluster.radius,
            lattice=self.base_cluster.lattice,
            basecluster=self.base_cluster,
        )

    @classmethod
    def from_dict(cls, d):
        """Create Orbit from serialized MSONable dict."""
        structure_symops = [SymmOp.from_dict(so_d) for so_d in d["structure_symops"]]
        site_bases = [DiscreteBasis.from_dict(sd) for sd in d["site_bases"]]
        o = cls(
            d["sites"],
            Lattice.from_dict(d["lattice"]),
            d["bits"],
            site_bases,
            structure_symops,
        )

        o._bit_combos = (
            tuple(np.array(c, dtype=int) for c in d["_bit_combos"])
            if "_bit_combos" in d
            else None
        )
        # This is to ensure that, after removing some bit_combos, an orbit
        # can still be correctly recorded and reloaded.
        return o

    def as_dict(self):
        """Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "sites": self.base_cluster.frac_coords.tolist(),
            "lattice": self.base_cluster.lattice.as_dict(),
            "bits": self.bits,
            "site_bases": [sb.as_dict() for sb in self.site_bases],
            "structure_symops": [so.as_dict() for so in self.structure_symops],
            "_bit_combos": tuple(c.tolist() for c in self.bit_combos),
        }
        return d
