from __future__ import division
import itertools
import numpy as np
from monty.json import MSONable
from pymatgen import Lattice, SymmOp
from pymatgen.util.coord import coord_list_mapping

from .cluster import Cluster
from ..utils import SymmetryError, SYMMETRY_ERROR_MESSAGE, SITE_TOL, _repr
from .basis import basis_factory


class Orbit(MSONable):
    """
    An Orbit represents a set of clusters that are symmetrically equivalent (when undecorated).
    This usually includes translational and structure symmetry of the underlying lattice. But this is not
    as the symmetry operations are required by the constructor. (regardless and orbit should at a minimum have
    translational symmetry).
    Also includes the possible ordering on the clusters
    """

    def __init__(self, sites, lattice, bits, site_bases, structure_symops):
        """
        Args:
            sites (list(pymatgen.Sites)):
                list of sites used in defining the orbit.
            lattice (pymatgen.Lattice):
                A lattice object for the given sites
            bits (list):
                list describing the occupancy of each site in cluster. For each site, should
                be the number of possible occupancies minus one. i.e. for a 3 site cluster,
                each of which having one of Li, TM, or Vac, bits should be
                [[0, 1], [0, 1], [0, 1]]. This is ensures the expansion is not "over-complete"
                by implicitly enforcing that all sites have a site basis function phi_0 = 1.
            site_bases (list(SiteBasis)):
                list of SiteBasis objects for each site in the given sites.
            structure_symops (list(pymatgen.SymmOps)):
                list of symmetry operations for the base structure
        """

        self.bits = bits
        self.site_bases = site_bases
        self.structure_symops = structure_symops
        self.orb_id = None
        self.orb_b_id = None

        #lazy generation of properties
        self._equiv = None
        self._symops = None
        self._bit_combos = None

        # Create basecluster
        self.basecluster = Cluster(sites, lattice)
        self.radius = self.basecluster.radius
        self.lattice = lattice

    @property
    def cluster_symops(self):
        """
        Symmetry operations that map a cluster to its periodic image.
        each element is a tuple of (pymatgen.core.operations.Symop, mapping)
        where mapping is a tuple such that
        Symop.operate(sites) = sites[mapping] (after translation back to unit cell)
        """
        if self._symops:
            return self._symops
        self._symops = []
        for symop in self.structure_symops:
            new_sites = symop.operate_multi(self.basecluster.sites)
            c = Cluster(new_sites, self.basecluster.lattice)
            if self.clusters[0] == c:
                c_sites = c.sites + np.round(self.basecluster.centroid - c.centroid)
                self._symops.append((symop, tuple(coord_list_mapping(self.basecluster.sites, c_sites, atol=SITE_TOL))))
        if len(self._symops) * self.multiplicity != len(self.structure_symops):
            raise SymmetryError(SYMMETRY_ERROR_MESSAGE)
        return self._symops

    @property
    def bit_combos(self):
        """
        List of arrays, each array is of symmetrically equivalent bit orderings
        """
        if self._bit_combos is not None:
            return self._bit_combos
        #get all the bit symmetry operations
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
                all_combos += [new_bits]
        self._bit_combos = all_combos
        return self._bit_combos

    @property
    def clusters(self):
        """
        Returns symmetrically equivalent clusters
        """
        if self._equiv:
            return self._equiv
        equiv = [self.basecluster]
        for symop in self.structure_symops:
            new_sites = symop.operate_multi(self.basecluster.sites)
            c = Cluster(new_sites, self.lattice)
            if c not in equiv:
                equiv.append(c)
        self._equiv = equiv
        if len(equiv) * len(self.cluster_symops) != len(self.structure_symops):
            self._equiv = None # Unset this
            raise SymmetryError(SYMMETRY_ERROR_MESSAGE)

        return equiv

    @property
    def multiplicity(self):
        """
        Number of clusters in the given sites of the lattice object that are in the orbit
        Returns:
            int
        """
        return len(self.clusters)

    def eval(self, bits, species):
        """
        Evaluates a cluster function defined for this orbit

        Args:
            bits (list):
                list of the cluster bits specifying which site basis function to evaluate for the
                corresponding site
            species (list):
                list of lists of species names for each site

        Returns: orbit function evaluated for the corresponding structure
            float
        """
        p = 1
        for i, (b, sp) in enumerate(zip(bits, species)):
            p *= self.site_bases[i].eval(b, sp)
        return p

    def assign_ids(self, o_id, o_b_id, start_c_id):
        """
        Used to assign unique orbit and cluster id's when creating a cluster subspace.

        Args:
            o_id: symmetrized cluster id
            o_b_id: start bit ordering id
            start_c_id: start cluster id

        Returns:
            next orbit id, next bit ordering id, next cluster id
        """
        self.orb_id = o_id
        self.orb_b_id = o_b_id
        c_id = start_c_id
        for c in self.clusters:
            c_id = c.assign_ids(c_id)
        return o_id + 1, o_b_id + len(self.bit_combos), c_id

    def __eq__(self, other):
        try:
        # when performing Orbit in list, this ordering stops the equivalent structures from generating
            return any(self.basecluster == cluster for cluster in other.clusters)
        except AttributeError as e:
            print(e.message)
            raise NotImplementedError

    def __neq__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f'[Orbit] id: {self.orb_id:<4} bit_id: {self.orb_b_id:<4}'\
               f'multiplicity: {self.multiplicity:<4}'\
               f' no. symops: {len(self.cluster_symops):<4} {str(self.basecluster)}'

    def __repr__(self):
        return _repr(self, orb_id=self.orb_id,
                     orb_b_id=self.orb_b_id,
                     radius=self.radius,
                     lattice=self.lattice,
                     basecluster=self.basecluster)

    @classmethod
    def from_dict(cls, d):
        """
        Creates Orbit from serialized MSONable dict
        """

        structure_symops = [SymmOp.from_dict(so_d) for so_d in d['structure_symops']]
        site_bases = [basis_factory(*sb_d) for sb_d in d['site_bases']]
        return cls(d['sites'], Lattice.from_dict(d['lattice']), d['bits'],
                   site_bases, structure_symops)

    def as_dict(self):
        """
        Json-serialization dict representation

        Returns:
            MSONable dict
        """
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "sites": self.basecluster.sites.tolist() ,
             "lattice": self.lattice.as_dict(),
             "bits": self.bits,
             "site_bases": [(sb.__class__.__name__[:-5].lower(), sb.species) for sb in self.site_bases],
             "structure_symops": [so.as_dict for so in self.structure_symops]}
        return d
