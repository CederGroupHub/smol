"""
Module implementing ClusterSupercell, which is used to evaluate cluster
correlations on cells beyond the primitive cell used to define the
ClusterSubspace.

This class is used within the ClusterSubspace class and should rarely be needed
directly by a user
"""

from __future__ import division
from collections import defaultdict
import numpy as np
from pymatgen import Structure, PeriodicSite
from pymatgen.analysis.structure_matcher import StructureMatcher,\
    OrderDisorderElementComparator
from pymatgen.util.coord import lattice_points_in_supercell,\
    coord_list_mapping_pbc
from src.ce_utils import delta_corr_single_flip
from smol.cofe.utils import StructureMatchError, SITE_TOL


class ClusterSupercell():
    """
    Used to calculates correlation vectors on a specific supercell lattice.
    """
    # TODO also remove keeping the csubspace as an attribute, just use whats needed, aka the orbits
    def __init__(self, clustersubspace, supercell, supercell_matrix, bits):
        """
        Args:
            clustersubspace (ClusterSubspace):
                A ClusterSubspace object used to compute corresponding
                correlation vectors
            supercell (pymatgen.Structure):
                Structure representing the super cell
            supercell matrix (np.array):
                Matrix representing transformation between prim and supercell
            bits (np.array):
                array describing the occupation of supercell,
                e.g. [[1,0,0],[0,1,0],[0,0,1]]
        """

        self.supercell = supercell
        self.supercell_matrix = supercell_matrix
        self.prim_to_supercell = np.linalg.inv(self.supercell_matrix)
        self.csubspace = clustersubspace
        self.size = int(round(np.abs(np.linalg.det(self.supercell_matrix))))

        self.bits = bits
        self.nbits = np.array([len(b) - 1 for b in self.bits])
        self.fcoords = np.array(self.supercell.frac_coords)

        self.cluster_indices, self.clusters_by_sites = self._generate_mappings()  # noqa

        comparator = OrderDisorderElementComparator()
        self._sm = StructureMatcher(primitive_cell=False,
                                    attempt_supercell=False,
                                    allow_subset=True,
                                    comparator=comparator,
                                    supercell_size=self.csubspace.supercell_size,  # noqa
                                    scale=True,
                                    ltol=self.csubspace.ltol,
                                    stol=self.csubspace.stol,
                                    angle_tol=self.csubspace.angle_tol)

    def _generate_mappings(self):
        """
        Find all the supercell indices associated with each cluster
        """

        ts = lattice_points_in_supercell(self.supercell_matrix)
        cluster_indices = []
        clusters_by_sites = defaultdict(list)
        for orbit in self.csubspace.iterorbits():
            prim_fcoords = np.array([c.sites for c in orbit.clusters])
            fcoords = np.dot(prim_fcoords, self.prim_to_supercell)
            # tcoords contains all the coordinates of the symmetrically
            # equivalent clusters the indices are: [equivalent cluster
            # (primitive cell), translational image, index of site in cluster,
            # coordinate index]
            tcoords = fcoords[:, None, :, :] + ts[None, :, None, :]
            tcs = tcoords.shape
            inds = coord_list_mapping_pbc(tcoords.reshape((-1, 3)),
                                          self.fcoords, atol=SITE_TOL).reshape((tcs[0] * tcs[1], tcs[2]))  # noqa
            # TODO cluster_indices will only be used in cluster_supercell
            #  not in the calculator
            cluster_indices.append((orbit, inds))
            # orbit, 2d array of index groups that correspond to the cluster
            # the 2d array may have some duplicates. This is due to
            # symetrically equivalent groups being matched to the same sites
            # (eg in simply cubic all 6 nn interactions will all be [0, 0]
            # indices. This multiplicity disappears as supercell size
            # increases, so I haven't implemented a more efficient method

            # now we store the orbits grouped by site index in the supercell,
            # to be used by delta_corr. We also store a reduced index array,
            # where only the rows with the site index are stored. The ratio is
            # needed because the correlations are averages over the full inds
            # array.
            # TODO break this apart and put this in the CalculatorClass
            #  cluster_by_sites is only used in delta_corr
            for site_index in np.unique(inds):
                in_inds = np.any(inds == site_index, axis=-1)
                ratio = len(inds) / np.sum(in_inds)
                clusters_by_sites[site_index].append((orbit.bit_combos,
                                                      orbit.orb_b_id,
                                                      inds[in_inds], ratio))

        return cluster_indices, clusters_by_sites

    # TODO this should be a method in the calculator
    def structure_from_occu(self, occu):
        """Get pymatgen.Structure from an occupancy vector"""

        sites = []
        for sp, s in zip(occu, self.supercell):
            if sp != 'Vacancy':
                site = PeriodicSite(sp, s.frac_coords, self.supercell.lattice)
                sites.append(site)
        return Structure.from_sites(sites)

    def corr_from_occupancy(self, occu):
        """
        Each entry in the correlation vector corresponds to a particular
        symmetrically distinct bit ordering
        """
        corr = np.zeros(self.csubspace.n_bit_orderings)
        corr[0] = 1  # zero point cluster
        occu = np.array(occu)
        for orb, inds in self.cluster_indices:
            c_occu = occu[inds]
            for i, bit_list in enumerate(orb.bit_combos):
                p = [np.fromiter(map(lambda occu: orb.eval(bits, occu),
                                     c_occu[:]), dtype=np.float)
                     for bits in bit_list]
                corr[orb.orb_b_id + i] = np.concatenate(p).mean()

        return corr

    def mapping_from_structure(self, structure):
        """
        Obtain the mapping of sites from a given structure to the supercell
        structure
        """
        mapping = self._sm.get_mapping(self.supercell, structure)
        if mapping is None:
            raise StructureMatchError('Mapping could not be found from '
                                      'structure')
        return mapping.tolist()

    def occu_from_structure(self, structure):
        """
        Returns list of occupancies of each site in the structure
        """
        mapping = self.mapping_from_structure(structure)
        occu = []  # np.zeros(len(self.supercell), dtype=np.int)

        for i, bit in enumerate(self.bits):
            # rather than starting with all vacancies and looping
            # only over mapping, explicitly loop over everything to
            # catch vacancies on improper sites
            if i in mapping:
                sp = str(structure[mapping.index(i)].specie)
            else:
                sp = 'Vacancy'
            occu.append(sp)

        return occu

    # TODO write test for this
    # TODO this should be part of the calculator too!
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
        len_eci = self.csubspace.n_bit_orderings + len(all_ewalds)
        delta_corr = np.zeros(len_eci)
        new_occu = new_occu # TODO code/decode this so that occu is returned as str not ints? May slow down though

        # TODO need to figure out how to implement delta_corr for different bases!!!
        for f in flips:
            new_occu_f = new_occu.copy()
            new_occu_f[f[0]] = f[1]
            delta_corr += delta_corr_single_flip(new_occu_f, new_occu,
                                                 self.csubspace.n_bit_orderings,  # noqa
                                                 self.clusters_by_sites[f[0]],
                                                 f[0], f[1], all_ewalds,
                                                 ewald_inds, self.size)
            new_occu = new_occu_f

        if debug:
            e = self.corr_from_occupancy(self.decode_occu(new_occu))
            de = e - self.corr_from_occupancy(occu)
            assert np.allclose(delta_corr, de)

        return delta_corr, new_occu
