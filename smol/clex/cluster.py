from __future__ import division
import itertools
import numpy as np
from pymatgen.util.coord import is_coord_subset, coord_list_mapping
from monty.json import MSONable

from .utils import SITE_TOL


class Cluster(MSONable):
    """
    An undecorated (no occupancies) cluster with translational symmetry
    """

    def __init__(self, sites, lattice):
        """
        Args:
            sites: list of frac coords for the sites
            symops: list of symops from pymatgen.symmetry
            lattice: pymatgen Lattice object
        """
        sites = np.array(sites)
        centroid = np.average(sites, axis=0)
        shift = np.floor(centroid)
        self.centroid = centroid - shift
        self.sites = sites - shift
        self.lattice = lattice
        self.c_id = None

    def assign_ids(self, c_id):
        """
        Method to recursively assign ids to clusters after initialization.
        """
        self.c_id = c_id
        return c_id + 1

    @property
    def size(self):
        return len(self.sites)

    @property
    def max_radius(self):
        coords = self.lattice.get_cartesian_coords(self.sites)
        all_d2 = np.sum((coords[None, :, :] - coords[:, None, :]) ** 2, axis=-1)
        return np.max(all_d2) ** 0.5

    def __eq__(self, other):
        if self.sites.shape != other.sites.shape:
            return False
        other_sites = other.sites + np.round(self.centroid - other.centroid)
        return is_coord_subset(self.sites, other_sites, atol=SITE_TOL)

    def __str__(self):
        points = str(np.round(self.sites,2)).replace("\n", " ").ljust(len(self.sites) * 21)
        return "Cluster: id: {:<3} Radius: {:<4.3} Points: {} Centroid: {}".format(self.c_id,
                                                                                   self.max_radius,
                                                                                   points,
                                                                                   np.round(self.centroid,2))

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_sites(sites):
        return Cluster([s.frac_coords for s in sites], sites[0].lattice)


class SymmetrizedCluster(MSONable):
    """
    Cluster with translational and structure symmetry. Also includes the possible orderings
    on the cluster
    """
    def __init__(self, base_cluster, bits, structure_symops):
        """
        Args:
            base_cluster: a Cluster object.
            bits: list describing the occupancy of each site in cluster. For each site, should
                    be the number of possible occupancies minus one. i.e. for a 3 site cluster,
                    each of which having one of Li, TM, or Vac, bits should be
                    [[0, 1], [0, 1], [0, 1]]. This is because the bit combinations that the
                    methodology *seems* to be missing are in fact linear combinations of other smaller
                    clusters. With least squares fitting, it can be verified that reintroducing these
                    bit combos doesn't improve the quality of the fit (though Bregman can do weird things
                    because of the L1 norm).
                    In any case, we know that pairwise ECIs aren't sparse in an ionic system, so
                    not sure how big of an issue this is.
            structure_symops: list of symmetry operations for the base structure
        """
        self.base_cluster = base_cluster
        self.bits = bits
        self.structure_symops = structure_symops
        self.sc_id = None
        self.sc_b_id = None
        #lazy generation of properties
        self._equiv = None
        self._symops = None
        self._bit_combos = None

    @property
    def equivalent_clusters(self):
        """
        Returns symmetrically equivalent clusters
        """
        if self._equiv:
            return self._equiv
        equiv = [self.base_cluster]
        for symop in self.structure_symops:
            new_sites = symop.operate_multi(self.base_cluster.sites)
            c = Cluster(new_sites, self.base_cluster.lattice)
            if c not in equiv:
                equiv.append(c)
        self._equiv = equiv
        if len(equiv) * len(self.cluster_symops) != len(self.structure_symops):
            raise SYMMETRY_ERROR
        return equiv

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
                all_combos.append(new_bits)
        self._bits = [np.array(x, dtype=np.int) for x in all_combos] # this shouldn't be an array
        return self._bits

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
            new_sites = symop.operate_multi(self.base_cluster.sites)
            c = Cluster(new_sites, self.base_cluster.lattice)
            if self.base_cluster == c:
                c_sites = c.sites + np.round(self.base_cluster.centroid - c.centroid)
                self._symops.append((symop, tuple(coord_list_mapping(self.base_cluster.sites, c_sites, atol=SITE_TOL))))
        if len(self._symops) * self.multiplicity != len(self.structure_symops):
            raise SYMMETRY_ERROR
        return self._symops

    @property
    def max_radius(self):
        return self.base_cluster.max_radius

    @property
    def sites(self):
        return self.base_cluster.sites

    @property
    def multiplicity(self):
        return len(self.equivalent_clusters)

    def assign_ids(self, sc_id, sc_b_id, start_c_id):
        """
        Args:
            sc_id: symmetrized cluster id
            sc_b_id: start bit ordering id
            start_c_id: start cluster id

        Returns:
            next symmetrized cluster id, next bit ordering id, next cluster id
        """
        self.sc_id = sc_id
        self.sc_b_id = sc_b_id
        c_id = start_c_id
        for c in self.equivalent_clusters:
            c_id = c.assign_ids(c_id)
        return sc_id+1, sc_b_id + len(self.bit_combos), c_id

    def __eq__(self, other):
        #when performing SymmetrizedCluster in list, this ordering stops the equivalent structures from generating
        return self.base_cluster in other.equivalent_clusters

    def __str__(self):
        return "SymmetrizedCluster: id: {:<4} bit_id: {:<4} multiplicity: {:<4} symops: {:<4}" \
            " {}".format(str(self.sc_id), str(self.sc_b_id), str(self.multiplicity), str(len(self.cluster_symops)), str(self.base_cluster))

    def __repr__(self):
        return self.__str__()


SYMMETRY_ERROR = ValueError("Error in calculating symmetry operations. Try using a "
                            "more symmetrically refined input structure. "
                            "SpacegroupAnalyzer(s).get_refined_structure().get_primitive_structure() "
                            "usually results in a safe choice")