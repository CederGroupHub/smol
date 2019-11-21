from __future__ import division
import itertools
import numpy as np
from collections import defaultdict
from pymatgen import Structure, PeriodicSite
from pymatgen.analysis.structure_matcher import StructureMatcher, OrderDisorderElementComparator
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.util.coord import lattice_points_in_supercell, coord_list_mapping_pbc

#TODO fix this import
from ..utils import SITE_TOL
from src.ce_utils import delta_corr_single_flip

#TODO can we simple obtain the cluster vectors based on the clustersubspace (ie get rid of this supercell thing)?
def get_bits(structure):
    """
    Helper method to compute list of species on each site.
    Includes vacancies
    """
    all_bits = []
    for site in structure:
        bits = []
        for sp in sorted(site.species.keys()):
            bits.append(str(sp))
        if site.species.num_atoms < 0.99:
            bits.append("Vacancy")
        all_bits.append(bits)
    return all_bits


class ClusterSupercell(object):
    """
    Calculates correlation vectors on a specific supercell lattice.
    """

    def __init__(self, supercell_matrix, clustersubspace):
        """
        Args:
            supercell matrix: array describing the supercell, e.g. [[1,0,0],[0,1,0],[0,0,1]]
            clustersubspace: ClusterExpansion object
        """
        self.supercell_matrix = np.array(supercell_matrix)
        self.prim_to_supercell = np.linalg.inv(self.supercell_matrix)
        self.clustersubspace = clustersubspace

        self.supercell = clustersubspace.structure.copy()
        self.supercell.make_supercell(self.supercell_matrix)
        self.size = int(round(np.abs(np.linalg.det(self.supercell_matrix))))

        self.bits = get_bits(self.supercell)
        self.nbits = np.array([len(b) - 1 for b in self.bits])
        self.fcoords = np.array(self.supercell.frac_coords)

        self._generate_mappings()

        # JY definition
        self.mapping = None

        if self.clustersubspace.use_ewald:
            # lazily generate the difficult ewald parts
            self.ewald_inds = []
            ewald_sites = []
            for bits, s in zip(self.bits, self.supercell):
                inds = np.zeros(max(self.nbits) + 1) - 1
                for i, b in enumerate(bits):
                    if b == 'Vacancy':
                        # inds.append(-1)
                        continue
                    inds[i] = len(ewald_sites)
                    ewald_sites.append(PeriodicSite(b, s.frac_coords, s.lattice))
                self.ewald_inds.append(inds)
            self.ewald_inds = np.array(self.ewald_inds, dtype=np.int)
            self._ewald_structure = Structure.from_sites(ewald_sites)
            self._ewald_matrix = None
            self._partial_ems = None
            self._all_ewalds = None
            self._range = np.arange(len(self.nbits))
        else:
            self._all_ewalds = np.zeros((0, 0, 0), dtype=np.float)
            self.ewald_inds = np.zeros((0, 0), dtype=np.int)

    @property
    def all_ewalds(self):
        if self._all_ewalds is None:
            if self.clustersubspace.use_ewald:
                ms = [self.ewald_matrix]
            else:
                ms = []
            if self.clustersubspace.use_inv_r:
                ms += self.partial_ems
            self._all_ewalds = np.array(ms)
        return self._all_ewalds

    @property
    def ewald_matrix(self):
        if self._ewald_matrix is None:
            self._ewald = EwaldSummation(self._ewald_structure,
                                         eta=self.clustersubspace.eta)
            self._ewald_matrix = self._ewald.total_energy_matrix
        return self._ewald_matrix

    @property
    def partial_ems(self):
        if self._partial_ems is None:
            # There seems to be an issue with SpacegroupAnalyzer such that making a supercell
            # can actually reduce the symmetry operations, so we're going to group the ewald
            # matrix by the equivalency in self.cluster_indices
            equiv_orb_inds = []
            ei = self.ewald_inds
            n_inds = len(self.ewald_matrix)
            for orb, inds in self.cluster_indices:
                # only want the point terms, which should be first
                if len(orb.bits) > 1:
                    break
                equiv = ei[inds[:, 0]]  # inds is normally 2d, but these are point terms
                for inds in equiv.T:
                    if inds[0] > -1:
                        b = np.zeros(n_inds, dtype=np.int)
                        b[inds] = 1
                        equiv_orb_inds.append(b)

            self._partial_ems = []
            for x in equiv_orb_inds:
                mask = x[None, :] * x[:, None]
                self._partial_ems.append(self.ewald_matrix * mask)
            for x, y in itertools.combinations(equiv_orb_inds, r=2):
                mask = x[None, :] * y[:, None]
                mask = mask.T + mask  # for the love of god don't use a += here, or you will forever regret it
                self._partial_ems.append(self.ewald_matrix * mask)
        return self._partial_ems

    def _get_ewald_occu(self, occu):
        i_inds = self.ewald_inds[self._range, occu]

        # instead of this line:
        #   i_inds = i_inds[i_inds != -1]
        # just make b_inds one longer than it needs to be and don't return the last value
        b_inds = np.zeros(len(self._ewald_structure) + 1, dtype=np.bool)
        b_inds[i_inds] = True
        return b_inds[:-1]

    def _get_ewald_eci(self, occu):
        inds = self._get_ewald_occu(occu)
        ecis = [np.sum(self.ewald_matrix[inds, :][:, inds]) / self.size]

        if self.clustersubspace.use_inv_r:
            for m in self.partial_ems:
                ecis.append(np.sum(m[inds, :][:, inds]) / self.size)

        return np.array(ecis)

    def _get_ewald_diffs(self, new_occu, occu):
        inds = self._get_ewald_occu(occu)
        new_inds = self._get_ewald_occu(new_occu)
        diff = inds != new_inds
        both = inds & new_inds
        add = new_inds & diff
        sub = inds & diff

        ms = [self.ewald_matrix]
        if self.clustersubspace.use_inv_r:
            ms += self.partial_ems

        diffs = []
        for m in ms:
            ma = m[add]
            ms = m[sub]
            v = np.sum(ma[:, add]) - np.sum(ms[:, sub]) + \
                (np.sum(ma[:, both]) - np.sum(ms[:, both])) * 2

            diffs.append(v / self.size)

        return diffs

    def _generate_mappings(self):
        """
        Find all the supercell indices associated with each cluster
        """
        ts = lattice_points_in_supercell(self.supercell_matrix)
        self.cluster_indices = []
        self.clusters_by_sites = defaultdict(list)
        for orbit in self.clustersubspace.orbits:
            prim_fcoords = np.array([c.sites for c in orbit.clusters])
            fcoords = np.dot(prim_fcoords, self.prim_to_supercell)
            # tcoords contains all the coordinates of the symmetrically equivalent clusters
            # the indices are: [equivalent cluster (primitive cell), translational image, index of site in cluster, coordinate index]
            tcoords = fcoords[:, None, :, :] + ts[None, :, None, :]
            tcs = tcoords.shape
            inds = coord_list_mapping_pbc(tcoords.reshape((-1, 3)),
                                          self.fcoords, atol=SITE_TOL).reshape((tcs[0] * tcs[1], tcs[2]))
            self.cluster_indices.append(
                (orbit, inds))  # orbit, 2d array of index groups that correspond to the cluster
            # the 2d array may have some duplicates. This is due to symetrically equivalent
            # groups being matched to the same sites (eg in simply cubic all 6 nn interactions
            # will all be [0, 0] indices. This multiplicity disappears as supercell size
            # increases, so I haven't implemented a more efficient method

            # now we store the orbits grouped by site index in the supercell,
            # to be used by delta_corr. We also store a reduced index array, where only the
            # rows with the site index are stored. The ratio is needed because the correlations
            # are averages over the full inds array.
            for site_index in np.unique(inds):
                in_inds = np.any(inds == site_index, axis=-1)
                ratio = len(inds) / np.sum(in_inds)
                self.clusters_by_sites[site_index].append((orbit.bit_combos, orbit.o_b_id, inds[in_inds], ratio))

    def structure_from_occu(self, occu):
        sites = []
        for b, o, s in zip(self.bits, occu, self.supercell):
            if b[o] != 'Vacancy':
                sites.append(PeriodicSite(b[o], s.frac_coords, self.supercell.lattice))
        return Structure.from_sites(sites)

    def corr_from_occupancy(self, occu):
        """
        Each entry in the correlation vector corresponds to a particular symmetrically distinct bit ordering
        """
        corr = np.zeros(self.clustersubspace.n_bit_orderings)
        corr[0] = 1  # zero point cluster
        occu = np.array(occu)
        for orb, inds in self.cluster_indices:
            c_occu = occu[inds]
            for i, bits in enumerate(orb.bit_combos):
                p = np.all(c_occu[None, :, :] == bits[:, None, :], axis=-1)
                corr[orb.o_b_id + i] = np.average(p)
        if self.clustersubspace.use_ewald:
            corr = np.concatenate([corr, self._get_ewald_eci(occu)])
        return corr

    def occu_from_structure(self, structure, return_mapping=False):
        """
        Calculates the correlation vector. Structure must be on this supercell
        """
        # calculate mapping to supercell
        sm_no_orb = StructureMatcher(primitive_cell=False,
                                    attempt_supercell=False,
                                    allow_subset=True,
                                    comparator=OrderDisorderElementComparator(),
                                    supercell_size=self.clustersubspace.supercell_size,
                                    scale=True,
                                    ltol=self.clustersubspace.ltol,
                                    stol=self.clustersubspace.stol,
                                    angle_tol=self.clustersubspace.angle_tol)
        if self.mapping == None:
            mapping = sm_no_orb.get_mapping(self.supercell, structure).tolist()
        else:
            mapping = self.mapping
        if mapping is None:
            raise ValueError('Structure cannot be mapped to this supercell')

        # cs.supercell[mapping] = structure
        occu = np.zeros(len(self.supercell), dtype=np.int)
        for i, bit in enumerate(self.bits):
            # rather than starting with all vacancies and looping
            # only over mapping, explicitly loop over everything to
            # catch vacancies on improper sites
            if i in mapping:
                sp = str(structure[mapping.index(i)].specie)
            else:
                sp = 'Vacancy'
            occu[i] = bit.index(sp)
        if not return_mapping:
            return occu
        else:
            return occu, mapping

    def corr_from_structure(self, structure):
        occu = self.occu_from_structure(structure)
        return self.corr_from_occupancy(occu)

    def occu_energy(self, occu, ecis):
        return np.dot(self.corr_from_occupancy(occu), ecis) * self.size

    def delta_corr(self, flips, occu, debug=False):
        """
        Returns the *change* in the correlation vector from applying a list of flips.
        Flips is a list of (site, new_bit) tuples.
        """
        new_occu = occu.copy()

        delta_corr = np.zeros(self.clustersubspace.n_bit_orderings + len(self.all_ewalds))
        for f in flips:
            new_occu_f = new_occu.copy()
            new_occu_f[f[0]] = f[1]
            delta_corr += delta_corr_single_flip(new_occu_f, new_occu,
                                                 self.clustersubspace.n_bit_orderings,
                                                 self.clusters_by_sites[f[0]], f[0], f[1], self.all_ewalds,
                                                 self.ewald_inds, self.size)
            new_occu = new_occu_f

        if debug:
            e = self.corr_from_occupancy(new_occu) - self.corr_from_occupancy(occu)
            assert np.allclose(delta_corr, e)
        return delta_corr, new_occu
