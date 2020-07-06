"""Implementation of an Orbit.

A set of symmetrically equivalent (with respect to the given undecorated
lattice symmetry) clusters.
"""

__author__ = "Luis Barroso-Luque, William Davidson Richard"

import itertools
import numpy as np
from monty.json import MSONable
from pymatgen import Lattice, SymmOp
from pymatgen.util.coord import coord_list_mapping
from smol.cofe.configspace.utils import SITE_TOL, _repr
from smol.exceptions import SymmetryError, SYMMETRY_ERROR_MESSAGE
from .cluster import Cluster
from .basis import basis_factory


class Orbit(MSONable):
    """Orbit set of symmetrically equivalent clusters.

    An Orbit represents a set of clusters that are symmetrically equivalent
    (when undecorated). The class also includes the possible orderings on the
    clusters in the orbit.

    An orbit usually includes translational and structure symmetry of the
    underlying lattice. But this is not a hard requirement any set of symmetry
    operations can be passed to the constructor. (regardless an orbit should at
    a minimum have translational symmetry).

    You probably never need to instantiate this class directly. Look at
    ClusterSubspace to create orbits and clusters necessary for a CE.

    Attributes:
        bits (list): list describing occupancy in each cluster.
        site_bases (list of SiteBasis): list of the SiteBasis for each site.
        structure_symops (list of Symmops):
            list of underlying structure symmetry operations.
        radius (float): max distance between two sites in a cluster.
        size (int): number of sites in cluster.
        lattice (Lattice): underlying structure lattice.
    """

    def __init__(self, sites, lattice, bits, site_bases, structure_symops):
        """Initialize an Orbit.

        Args:
            sites (list or ndarray):
                list of frac coords for the sites
            lattice (pymatgen.Lattice):
                A lattice object for the given sites
            bits (list):
                list describing the possible site function orderings for
                each site in cluster. Should be the number of possible
                occupancies minus one. i.e. for a 3 site cluster, each of which
                having one of Li, TM, or Vac, bits are [[0, 1], [0, 1], [0, 1]]
                This is ensures the expansion is not "over-complete" by
                implicitly enforcing that all sites have a site basis function
                phi_0 = 1.
            site_bases (list of SiteBasis):
                list of SiteBasis objects for each site in the given sites.
            structure_symops (list of SymmOp):
                list of symmetry operations for the base structure
        """
        if len(sites) != len(bits):
            raise AttributeError(f'Number of sites {len(sites)} must be equal '
                                 f'to number of bits {len(bits)}')
        elif len(sites) != len(site_bases):
            raise AttributeError(f'Number of sites {len(sites)} must be equal '
                                 f'to number of site bases {len(site_bases)}')
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
        self._bit_combos = None
        self._basis_arrs = None
        self._bases_arr = None

        # Create basecluster
        self.base_cluster = Cluster(sites, lattice)
        self.radius = self.base_cluster.radius
        self.size = self.base_cluster.size
        self.lattice = lattice

    @property
    def n_bit_orderings(self):
        """Get number of symmetrically distinct bit orderings."""
        return len(self.bit_combos)

    @property
    def multiplicity(self):
        """Get number of clusters in orbit.

        Number of clusters in the given sites of the lattice object that are in
        the orbit.
        """
        return len(self.clusters)

    @property
    def bit_combos(self):
        """Get list of site bit orderings.

        List of lists, each inner list is of symmetrically equivalent bit
        orderings.
        """
        if self._bit_combos is not None:
            return self._bit_combos
        # get all the bit symmetry operations
        bit_ops = []
        for _, bitop in self.cluster_symops:
            if bitop not in bit_ops:
                bit_ops.append(bitop)
        all_combos = []
        for bit_combo in itertools.product(*self.bits):
            if bit_combo not in itertools.chain(*all_combos):
                bit_combo = np.array(bit_combo)
                new_bits = []
                for b_o in bit_ops:
                    new_bit = tuple(bit_combo[np.array(b_o)])
                    if new_bit not in new_bits:
                        new_bits.append(new_bit)
                all_combos.append(new_bits)
        self._bit_combos = tuple(np.array(c, dtype=np.int) for c in all_combos)
        return self._bit_combos

    @property
    def clusters(self):
        """Get symmetrically equivalent clusters."""
        if self._equiv:
            return self._equiv
        equiv = [self.base_cluster]
        for symop in self.structure_symops:
            new_sites = symop.operate_multi(self.base_cluster.sites)
            c = Cluster(new_sites, self.lattice)
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

        Each element is a tuple of (pymatgen.core.operations.Symop, mapping)
        where mapping is a tuple such that
        Symop.operate(sites) = sites[mapping]
        (after translation back to unit cell)
        """
        if self._symops:
            return self._symops
        self._symops = []
        for symop in self.structure_symops:
            new_sites = symop.operate_multi(self.base_cluster.sites)
            c = Cluster(new_sites, self.base_cluster.lattice)
            if c == self.base_cluster:
                recenter = np.round(self.base_cluster.centroid - c.centroid)
                c_sites = c.sites + recenter
                mapping = tuple(coord_list_mapping(self.base_cluster.sites,
                                                   c_sites, atol=SITE_TOL))
                self._symops.append((symop, mapping))
        if len(self._symops) * self.multiplicity != len(self.structure_symops):
            raise SymmetryError(SYMMETRY_ERROR_MESSAGE)
        return self._symops

    @property
    def basis_arrays(self):
        """Get a tuple of all site function arrays for each site in orbit."""
        if self._basis_arrs is None:
            self._basis_arrs = tuple(sb.function_array
                                     for sb in self.site_bases)
        return self._basis_arrs

    @property
    def bases_array(self):
        """Get bases array.

        3D array with all basis arrays. Since each basis array can be of
        different dimension the 3D array is the size of the largest array.
        Smaller arrays are padded with ones. Doing this allows using numpy
        fancy indexing which can be faster than for loops?
        """
        if self._bases_arr is None or self._basis_arrs is None:
            max_dim = max(len(fa) for fa in self.basis_arrays)
            self._bases_arr = np.ones((len(self.basis_arrays),
                                       max_dim, max_dim + 1))
            for i, fa in enumerate(self.basis_arrays):
                j, k = fa.shape
                self._bases_arr[i, :j, :k] = fa
        return self._bases_arr

    @property
    def basis_orthogonal(self):
        """Test if the Orbit bases are orthogonal."""
        return all(basis.is_orthogonal for basis in self.site_bases)

    @property
    def basis_orthonormal(self):
        """Test if the orbit bases are orthonormal."""
        return all(basis.is_orthonormal for basis in self.site_bases)

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
            raise RuntimeError('All bit_combos have been removed from orbit '
                               f'with id {self.id}')

        self._bit_combos = tuple(bit_combos)

    def remove_bit_combos_by_inds(self, inds):
        """Remove bit combos by their indices in the bit_combo list."""
        if max(inds) > len(self.bit_combos) - 1:
            raise RuntimeError(f'Some indices {inds} out of ranges for total '
                               f'{len(self._bit_combos)} bit combos')

        self._bit_combos = tuple(b_c for i, b_c in enumerate(self._bit_combos)
                                 if i not in inds)

        if not self.bit_combos:
            raise RuntimeError('All bit_combos have been removed from orbit '
                               f'with id {self.id}')

    def eval(self, bits, species_encoding):  # is this used anymore?
        """Evaluate a cluster function defined for this orbit.

        Args:
            bits (list):
                list of the cluster bits specifying which site basis function
                to evaluate for the corresponding site.
            species_encoding (list):
                list of lists of species encoding for each site. (index of
                species in species bits)

        Returns: orbit function evaluated for the corresponding structure
            float
        """
        p = 1
        for i, (b, sp) in enumerate(zip(bits, species_encoding)):
            p *= self.basis_arrays[i][b, sp]

        return p

    def transform_site_bases(self, basis_name, orthonormal=False):
        """Transform the Orbits site bases to new basis set.

        Args:
            basis_name (str):
                name of new basis for all site bases
            orthormal (bool):
                option to orthonormalize all new site bases
        """
        new_bases = []
        for basis in self.site_bases:
            new_basis = basis_factory(basis_name, basis.site_space)
            if orthonormal:
                new_basis.orthonormalize()
            new_bases.append(new_basis)

        self.site_bases = tuple(new_bases)
        self._basis_arrs, self._bases_arr = None, None

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
        """Get size of a the base cluster. The number of sites."""
        return self.size

    def __eq__(self, other):
        """Check equality of orbits."""
        # when performing orbit in list, this ordering stops the
        # equivalent structures from generating
        return self.base_cluster in other.clusters

    def __neq__(self, other):
        """Check negation of orbit equality."""
        return not self.__eq__(other)

    def __str__(self):
        """Pretty strings for pretty things."""
        return f'[Orbit] id: {self.id:<3}' \
               f'orderings: {self.n_bit_orderings:<4}' \
               f'multiplicity: {self.multiplicity:<4}' \
               f' no. symops: {len(self.cluster_symops):<4}\n' \
               f'              {str(self.base_cluster)}'

    def __repr__(self):
        """Get Orbit representation."""
        return _repr(self, orb_id=self.id,
                     orb_b_id=self.bit_id,
                     radius=self.radius,
                     lattice=self.lattice,
                     basecluster=self.base_cluster)

    @classmethod
    def from_dict(cls, d):
        """Create Orbit from serialized MSONable dict."""
        structure_symops = [SymmOp.from_dict(so_d)
                            for so_d in d['structure_symops']]
        site_bases = [basis_factory(*sb_d) for sb_d in d['site_bases']]
        return cls(d['sites'], Lattice.from_dict(d['lattice']),
                   d['bits'], site_bases, structure_symops)

    def as_dict(self):
        """Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "sites": self.base_cluster.sites.tolist(),
             "lattice": self.lattice.as_dict(),
             "bits": self.bits,
             "site_bases": [(sb.__class__.__name__[:-5].lower(),
                             sb.species) for sb in self.site_bases],
             "structure_symops": [so.as_dict() for so in self.structure_symops]
             }
        return d
