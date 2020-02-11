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
from smol.cofe.configspace.utils import get_bits
from src.ce_utils import delta_corr_single_flip


class ClusterExpansionProcessor(MSONable):
    """
    A processor used to generate Markov processes for sampling thermodynamic
    properties from a cluster expansion hamiltonian. Think of this as fixed
    size supercell optimized to calculate correlation vectors and local changes
    to correlation vectors from site flips.
    """

    def __init__(self, cluster_expansion, supercell_matrix):

        self.ecis = cluster_expansion.ecis.copy()
        self.subspace = cluster_expansion.subspace
        self.structure = cluster_expansion.prim_structure.copy()
        self.supercell_matrix = supercell_matrix
        self.structure.make_supercell(supercell_matrix)
        self.bits = get_bits(self.structure)
        self.size = self.subspace.num_prims_from_matrix(supercell_matrix)
        self.n_orbit_functions = self.subspace.n_bit_orderings

        self.orbit_inds = self.subspace.supercell_orbit_mappings(supercell_matrix)  # noqa

        # Create a dictionary of orbits by site index and information
        # necessary to compute local changes in correlation vectors from flips
        self.orbits_by_sites = defaultdict(list)
        # Store the orbits grouped by site index in the structure,
        # to be used by delta_corr. We also store a reduced index array,
        # where only the rows with the site index are stored. The ratio is
        # needed because the correlations are averages over the full inds
        # array.
        for orbit, inds in self.orbit_inds:
            for site_ind in np.unique(inds):
                in_inds = np.any(inds == site_ind, axis=-1)
                ratio = len(inds) / np.sum(in_inds)
                self.orbits_by_sites[site_ind].append((orbit.bit_combos,
                                                       orbit.orb_b_id,
                                                       inds[in_inds], ratio))

    def compute_corr(self, occu):
        """
        Each entry in the correlation vector corresponds to a particular
        symmetrically distinct bit ordering
        """
        # This can be optimized by writing a separate corr_from_occu function
        # that used only the encoded occu vector and has an optimized orbit
        # eval loop (ie maybe some JIT or something like that...)
        occu = self.decode_occupancy(occu)
        return self.subspace.corr_from_occupancy(occu, self.orbit_inds)

    def compute_property(self, occu):
        """
        Computes the total value of the property represented by the given ECIs
        for the given occupancy vector
        """
        return np.dot(self.compute_corr(occu), self.ecis) * self.size

    def structure_from_occupancy(self, occu):
        """Get pymatgen.Structure from an occupancy vector"""
        sites = []
        for sp, s in zip(occu, self.structure):
            if sp != 'Vacancy':
                site = PeriodicSite(sp, s.frac_coords, self.structure.lattice)
                sites.append(site)
        return Structure.from_sites(sites)

    def encode_occupancy(self, occu):
        """
        Encode occupancy vector of species str to ints.
        This is mainly used to compute delta_corr
        """
        ec_occu = np.array([bit.index(sp) for bit, sp in zip(self.bits, occu)])
        return ec_occu

    def decode_occupancy(self, enc_occu):
        """Decode encoded occupancy vector of int to species str"""
        occu = [bit[i] for i, bit in zip(enc_occu, self.bits)]
        return occu

    def delta_corr(self, flips, occu,
                   all_ewalds=np.zeros((0, 0, 0), dtype=np.float),
                   ewald_inds=np.zeros((0, 0), dtype=np.int), debug=False):
        """
        Returns the *change* in the correlation vector from applying a list of
        flips. Flips is a list of (site, new_bit) tuples.
        """

        new_occu = occu.copy()
        delta_corr = np.zeros(self.n_orbit_functions + len(all_ewalds))

        # TODO need to figure out how to implement delta_corr for different
        #  bases!!!
        # this loop is candidate for optimization as well
        for f in flips:
            new_occu_f = new_occu.copy()
            new_occu_f[f[0]] = f[1]
            delta_corr += delta_corr_single_flip(new_occu_f, new_occu,
                                                 self.n_orbit_functions,
                                                 self.orbits_by_sites[f[0]],
                                                 f[0], f[1])
            new_occu = new_occu_f

        if debug:
            e = self.compute_corr(new_occu)
            de = e - self.compute_corr(occu)
            assert np.allclose(delta_corr, de)

        return delta_corr

    @classmethod
    def from_dict(cls, d):
        """
        Creates Structure Wrangler from serialized MSONable dict
        """
        pass

    def as_dict(self) -> dict:
        """
        Json-serialization dict representation

        Returns:
            MSONable dict
        """
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             'ecis': self.ecis,
             '_subspace': self.subspace,
             'supercell_matrix': self.supercell_matrix}
        return d
