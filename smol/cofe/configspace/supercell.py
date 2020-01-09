from __future__ import division
import numpy as np
from collections import defaultdict
from pymatgen import Structure, PeriodicSite
from pymatgen.analysis.structure_matcher import StructureMatcher, OrderDisorderElementComparator, FrameworkComparator
from pymatgen.util.coord import lattice_points_in_supercell, coord_list_mapping_pbc

from ..utils import StructureMatchError, SITE_TOL
from src.ce_utils import delta_corr_single_flip

#TODO can we simple obtain the cluster vectors based on the clustersubspace
# (ie get rid or simplify this supercell thing)?
#TODO the supercell and supercell_matrix should probably be obtained with an undercorated structure/lattice


class ClusterSupercell(object):
    """
    Used to calculates correlation vectors on a specific supercell lattice.
    """

    def __init__(self, clustersubspace, supercell, supercell_matrix, bits):
        """
        Args:
            clustersubspace (ClusterSubspace):
                A ClusterSubspace object used to compute corresponding correlation vectors
            supercell (pymatgen.Structure):
                Structure representing the super cell
            supercell matrix (np.array):
                Matrix representing transformation between prim and supercell
            bits (np.array):
                array describing the occupation of supercell, e.g. [[1,0,0],[0,1,0],[0,0,1]]
        """

        self.supercell = supercell
        self.supercell_matrix = supercell_matrix
        self.prim_to_supercell = np.linalg.inv(self.supercell_matrix)
        self.clustersubspace = clustersubspace
        self.size = int(round(np.abs(np.linalg.det(self.supercell_matrix))))

        self.bits = bits
        self.nbits = np.array([len(b) - 1 for b in self.bits])
        self.fcoords = np.array(self.supercell.frac_coords)

        self.cluster_indices, self.clusters_by_sites = self._generate_mappings()

        # JY definition
        self.mapping = None

    def _generate_mappings(self):
        """
        Find all the supercell indices associated with each cluster
        """

        ts = lattice_points_in_supercell(self.supercell_matrix)
        cluster_indices = []
        clusters_by_sites = defaultdict(list)
        for orbit in self.clustersubspace.iterorbits():
            prim_fcoords = np.array([c.sites for c in orbit.clusters])
            fcoords = np.dot(prim_fcoords, self.prim_to_supercell)
            # tcoords contains all the coordinates of the symmetrically equivalent clusters
            # the indices are: [equivalent cluster (primitive cell), translational image, index of site in cluster, coordinate index]
            tcoords = fcoords[:, None, :, :] + ts[None, :, None, :]
            tcs = tcoords.shape
            inds = coord_list_mapping_pbc(tcoords.reshape((-1, 3)),
                                          self.fcoords, atol=SITE_TOL).reshape((tcs[0] * tcs[1], tcs[2]))
            cluster_indices.append((orbit, inds))
            # orbit, 2d array of index groups that correspond to the cluster
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
                clusters_by_sites[site_index].append((orbit.bit_combos, orbit.orb_b_id, inds[in_inds], ratio))

        return cluster_indices, clusters_by_sites

    def structure_from_occu(self, occu):
        sites = []
        for sp, s in zip(occu, self.supercell):
            if sp != 'Vacancy':
                sites.append(PeriodicSite(sp, s.frac_coords, self.supercell.lattice))
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
                #each bit in bits represents a site that has its own site basis in orb.sbases
                p = np.fromiter(map(lambda occu: orb.eval(bits, occu), c_occu[:]), dtype=np.float)
                corr[orb.orb_b_id + i] = p.sum()
        return corr

    def occu_from_structure(self, structure, return_mapping=False):
        """
        Returns list of occupancies of each site in the structure
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

        #TODO the mapping depends on the given structure. Is being able to short-circuit this by setting an
        # attribute a good idea?
        if self.mapping is None:
            mapping = sm_no_orb.get_mapping(self.supercell, structure)
            if mapping is None:
                raise StructureMatchError('Mapping could not be found from structure')
            mapping = mapping.tolist()
        else:
            mapping = self.mapping

        occu = [] #np.zeros(len(self.supercell), dtype=np.int)
        for i, bit in enumerate(self.bits):
            # rather than starting with all vacancies and looping
            # only over mapping, explicitly loop over everything to
            # catch vacancies on improper sites
            if i in mapping:
                sp = str(structure[mapping.index(i)].specie)
            else:
                sp = 'Vacancy'
            occu.append(sp)
        if not return_mapping:
            return occu
        else:
            return occu, mapping

    #TODO get rid of this?
    def occu_energy(self, occu, ecis):
        return np.dot(self.corr_from_occupancy(occu), ecis) * self.size

    def delta_corr(self, flips, occu, all_ewalds=np.zeros((0, 0, 0), dtype=np.float),
                   ewald_inds=np.zeros((0, 0), dtype=np.int), debug=False):
        """
        Returns the *change* in the correlation vector from applying a list of flips.
        Flips is a list of (site, new_bit) tuples.
        """

        new_occu = occu.copy()

        delta_corr = np.zeros(self.clustersubspace.n_bit_orderings + len(all_ewalds))
        for f in flips:
            new_occu_f = new_occu.copy()
            new_occu_f[f[0]] = f[1]
            delta_corr += delta_corr_single_flip(new_occu_f, new_occu,
                                                 self.clustersubspace.n_bit_orderings,
                                                 self.clusters_by_sites[f[0]], f[0], f[1],
                                                 all_ewalds,
                                                 ewald_inds, self.size)
            new_occu = new_occu_f

        if debug:
            e = self.corr_from_occupancy(new_occu) - self.corr_from_occupancy(occu)
            assert np.allclose(delta_corr, e)
        return delta_corr, new_occu
