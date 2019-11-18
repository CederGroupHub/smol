from __future__ import division
import itertools
import numpy as np
from pymatgen.util.coord import coord_list_mapping

from smol.clex.cluster import Cluster
from smol.clex.utils import SymmetryError, SYMMETRY_ERROR_MESSAGE, SITE_TOL


class Orbit(Cluster):
    """
    An Orbit represents a set of clusters that are symmetrically equivalent (when undecorated).
    This usually includes translational and structure symmetry of the underlying lattice. But this is not
    as the symmetry operations are required by the constructor. (regardless and orbit should at a minimum have
    translational symmetry).
    Also includes the possible ordering on the clusters
    """
    def __init__(self, sites, lattice, bits, structure_symops):
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

        self.bits = bits
        self.structure_symops = structure_symops
        self.o_id = None
        self.o_b_id = None

        #lazy generation of properties
        self._equiv = None
        self._symops = None
        self._bit_combos = None

        # Initialize Cluster base class
        super().__init__(sites, lattice)

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
            new_sites = symop.operate_multi(self.sites)
            c = Cluster(new_sites, self.lattice)
            if self.clusters[0] == c:
                c_sites = c.sites + np.round(self.centroid - c.centroid)
                self._symops.append((symop, tuple(coord_list_mapping(self.sites, c_sites, atol=SITE_TOL))))
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
                all_combos.append(new_bits)
        self._bits = [np.array(x, dtype=np.int) for x in all_combos] # this shouldn't be an array
        return self._bits

    @property
    def clusters(self):
        """
        Returns symmetrically equivalent clusters
        """
        if self._equiv:
            return self._equiv
        equiv = [Cluster(self.sites, self.lattice)]
        for symop in self.structure_symops:
            new_sites = symop.operate_multi(self.sites)
            c = Cluster(new_sites, self.lattice)
            if c not in equiv:
                equiv.append(c)
        self._equiv = equiv
        if len(equiv) * len(self.cluster_symops) != len(self.structure_symops):
            raise SYMMETRY_ERROR_MESSAGE
        return equiv

    @property
    def multiplicity(self):
        return len(self.clusters)

    def assign_ids(self, o_id, o_b_id, start_c_id):
        """
        Args:
            o_id: symmetrized cluster id
            o_b_id: start bit ordering id
            start_c_id: start cluster id

        Returns:
            next symmetrized cluster id, next bit ordering id, next cluster id
        """
        self.o_id = o_id
        self.o_b_id = o_b_id
        c_id = start_c_id
        for c in self.clusters:
            c_id = c.assign_ids(c_id)
        return o_id + 1, o_b_id + len(self.bit_combos), c_id

    def __eq__(self, other):
        #try:
        #when performing SymmetrizedCluster in list, this ordering stops the equivalent structures from generating
        return any(super(Orbit, self).__eq__(cluster) for cluster in other.clusters)
        #except:
        #    return NotImplemented

    def __neq__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return "Orbit: id: {:<4}, bit_id: {:<4}, multiplicity: {:<4}, symops: {:<4}" \
            " {}".format(str(self.o_id), str(self.o_b_id), str(self.multiplicity), str(len(self.cluster_symops)),
                         super().__str__())

    #def __repr__(self):
     #   return self.__str__()str(