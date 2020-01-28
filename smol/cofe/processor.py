"""
Implementation of a processor for a fixed size Super Cell optimized to compute
correlation vectors and local changes in correlation vectors. This class allows
the use a cluster expansion hamiltonian to run Monte Carlo based simulations.
The name comes from its use to create the "Markov process" aka Markov Processor
"""

from collections import defaultdict
import numpy as np
from monty.json import MSONable
from pymatgen import Structure, PeriodicSite
from src.ce_utils import delta_corr_single_flip

# TODO consider optimizing by ignoring zeroed eci, by removing those orbits
#  from cluster subspace
# TODO this needs to have a way to determine valid flips, swaps etc.


class ClusterExpansionProcessor(MSONable):
    """
    A processor used to generate Markov processes for sampling thermodynamic
    properties from a cluster expansion hamiltonian. Think of this as fixed
    size supercell optimized to calculate correlation vectors and local changes
    to correlation vectors from site flips.
    """

    def __init__(self, cluster_expansion, supercell_matrix):

        self.cluster_subspace = self.cluster_subspace.wrangler.subspace
        self.structure = cluster_expansion.wrangler.prim_structure.copy()
        self.structure.make_supercell(supercell_matrix)
        self.supercell_matrix = supercell_matrix
        self.size = int(round(np.abs(np.linalg.det(supercell_matrix))))

        # Create a dictionary of orbits by site index and information
        # neccesary to compute local changes in correlation vectors from flips
        self.orbits_by_sites = defaultdict(list)
        # now we store the orbits grouped by site index in the structure,
        # to be used by delta_corr. We also store a reduced index array,
        # where only the rows with the site index are stored. The ratio is
        # needed because the correlations are averages over the full inds
        # array.
        sc = self.cluster_subspace.supercell_from_matrix(supercell_matrix)
        self.orbit_indices = sc.orbit_indices
        for orbit, inds in self.orbit_indices:
            for site_index in np.unique(inds):
                in_inds = np.any(inds == site_index, axis=-1)
                ratio = len(inds) / np.sum(in_inds)
                self.orbits_by_sites[site_index].append((orbit.bit_combos,
                                                         orbit.orb_b_id,
                                                         inds[in_inds], ratio))

    def corr_from_occu(self, occu):
        """
        Each entry in the correlation vector corresponds to a particular
        symmetrically distinct bit ordering
        """
        corr = np.zeros(self.n_bit_orderings)
        corr[0] = 1  # zero point cluster
        occu = np.array(occu)
        for orb, inds in self.orbit_indices:
            c_occu = occu[inds]
            for i, bit_list in enumerate(orb.bit_combos):
                p = [np.fromiter(map(lambda occu: orb.eval(bits, occu),
                                     c_occu[:]), dtype=np.float)
                     for bits in bit_list]
                corr[orb.orb_b_id + i] = np.concatenate(p).mean()

        return corr

    def structure_from_occu(self, occu):
        """Get pymatgen.Structure from an occupancy vector"""
        sites = []
        for sp, s in zip(occu, self.structure):
            if sp != 'Vacancy':
                site = PeriodicSite(sp, s.frac_coords, self.structure.lattice)
                sites.append(site)
        return Structure.from_sites(sites)

    def encode_occu(self, occu):
        """
        Encode occupancy vector of species str to ints.
        This is mainly used to compute delta_corr
        """
        ec_occu = np.array([bit.index(sp) for bit, sp in zip(self.bits, occu)])
        return ec_occu

    def decode_occu(self, enc_occu):
        """Decode encoded occupancy vector of int to species str"""
        occu = [bit[i] for i, bit in zip(enc_occu, self.bits)]
        return occu

    # TODO get rid of this?
    def occu_energy(self, occu, ecis):
        return np.dot(self.corr_from_occupancy(occu), ecis) * self.size

    def delta_corr(self, flips, occu,
                   all_ewalds=np.zeros((0, 0, 0), dtype=np.float),
                   ewald_inds=np.zeros((0, 0), dtype=np.int), debug=False):
        """
        Returns the *change* in the correlation vector from applying a list of
        flips. Flips is a list of (site, new_bit) tuples.
        """

        new_occu = self.encode_occu(occu)
        len_eci = self.n_bit_orderings + len(all_ewalds)
        delta_corr = np.zeros(len_eci)
        # TODO code/decode this so that occu is returned as str not ints?
        #  May slow down though
        new_occu = new_occu

        # TODO need to figure out how to implement delta_corr for different
        #  bases!!!
        for f in flips:
            new_occu_f = new_occu.copy()
            new_occu_f[f[0]] = f[1]
            delta_corr += delta_corr_single_flip(new_occu_f, new_occu,
                                                 self.n_bit_orderings,
                                                 self.clusters_by_sites[f[0]],
                                                 f[0], f[1], all_ewalds,
                                                 ewald_inds, self.size)
            new_occu = new_occu_f

        if debug:
            e = self.corr_from_occupancy(self.decode_occu(new_occu))
            de = e - self.corr_from_occupancy(occu)
            assert np.allclose(delta_corr, de)

        return delta_corr, new_occu

    def from_dict(cls, d):
        pass

    def as_dict(self) -> dict:
        pass
