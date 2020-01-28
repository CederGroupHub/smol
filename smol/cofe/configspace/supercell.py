"""
Module implementing ClusterSupercell, which is used to evaluate cluster
correlations on cells beyond the primitive cell used to define the
ClusterSubspace.

This class is used within the ClusterSubspace class and should rarely be needed
directly by a user
"""

from __future__ import division
import numpy as np
from pymatgen import Structure, PeriodicSite
from pymatgen.analysis.structure_matcher import StructureMatcher,\
    OrderDisorderElementComparator
from pymatgen.util.coord import lattice_points_in_supercell,\
    coord_list_mapping_pbc
from smol.cofe.configspace.utils import get_bits, StructureMatchError, SITE_TOL


class ClusterSupercell():
    """
    Used to calculates correlation vectors on a specific supercell structure
    lattice.
    """

    def __init__(self, supercell_structure, supercell_matrix, bits,
                 n_bit_orderings, orbits, **matcher_kwargs):
        """
        Args:
            clustersubspace (ClusterSubspace):
                A ClusterSubspace object used to compute corresponding
                correlation vectors
            supercell_structure (pymatgen.Structure):
                Structure representing the super cell
            supercell_struct matrix (np.array):
                Matrix representing transformation between prim and
                supercell_struct
            bits (np.array):
                array describing the occupation of supercell_struct,
                e.g. [[1,0,0],[0,1,0],[0,0,1]]
            n_bit_orderings (int):
                total number of possible orderings of bits for all prim sites.
                This corresponds to the total number of cluster functions in
                the expansion.
            orbits (list(Orbit)):
                list of cluster orbits ordered by increasing size
            matcher_kwargs:
                keyword arguments to be passed to StructureMatcher: ltol, stol,
                atol, supercell_size
        """

        self.supercell_struct = supercell_structure
        self.supercell_matrix = supercell_matrix
        self.prim_to_supercell = np.linalg.inv(self.supercell_matrix)
        self.size = int(round(np.abs(np.linalg.det(supercell_matrix))))

        self.bits = bits
        self.nbits = np.array([len(b)-1 for b in self.bits])
        self.n_bit_orderings = n_bit_orderings

        self.fcoords = np.array(self.supercell_struct.frac_coords)
        self.orbit_indices = self._generate_mappings(orbits)

        comparator = OrderDisorderElementComparator()
        self._sm = StructureMatcher(primitive_cell=False,
                                    attempt_supercell=False,
                                    allow_subset=True,
                                    comparator=comparator,
                                    scale=True,
                                    **matcher_kwargs)

    def _generate_mappings(self, orbits):
        """
        Find all the supercell_structure indices associated with each cluster
        """

        ts = lattice_points_in_supercell(self.supercell_matrix)
        orbit_indices = []
        for orbit in orbits:
            prim_fcoords = np.array([c.sites for c in orbit.clusters])
            fcoords = np.dot(prim_fcoords, self.prim_to_supercell)
            # tcoords contains all the coordinates of the symmetrically
            # equivalent clusters the indices are: [equivalent cluster
            # (primitive cell), translational image, index of site in cluster,
            # coordinate index]
            tcoords = fcoords[:, None, :, :] + ts[None, :, None, :]
            tcs = tcoords.shape
            inds = coord_list_mapping_pbc(tcoords.reshape((-1, 3)),
                                          self.fcoords,
                                          atol=SITE_TOL).reshape((tcs[0] * tcs[1], tcs[2]))  # noqa
            # orbit_indices holds orbit, and 2d array of index groups that
            # correspond to the orbit
            # the 2d array may have some duplicates. This is due to
            # symetrically equivalent groups being matched to the same sites
            # (eg in simply cubic all 6 nn interactions will all be [0, 0]
            # indices. This multiplicity disappears as supercell_structure size
            # increases, so I haven't implemented a more efficient method
            orbit_indices.append((orbit, inds))

        return orbit_indices

    def corr_from_occupancy(self, occu):
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

    def mapping_from_structure(self, structure):
        """
        Obtain the mapping of sites from a given structure to the
        supercell_structure
        """
        mapping = self._sm.get_mapping(self.supercell_struct, structure)
        if mapping is None:
            raise StructureMatchError('Mapping could not be found from '
                                      'structure')
        return mapping.tolist()

    def occu_from_structure(self, structure):
        """
        Returns list of occupancies of each site in the structure
        """
        mapping = self.mapping_from_structure(structure)
        occu = []  # np.zeros(len(self.supercell_structure), dtype=np.int)

        for i, bit in enumerate(self.bits):
            # rather than starting with all vacancies and looping
            # only over mapping, explicitly loop over everything to
            # catch vacancies on improper sites
            if i in mapping:
                sp = str(structure[mapping.index(i)].specie)
            else:
                sp = 'Vacancy'
            if sp not in bit:
                raise StructureMatchError(f'A site in given structure has a'
                                          f' unrecognized specie {sp}. ')
            occu.append(sp)

        return occu

    def structure_from_occu(self, occu):
        """Get pymatgen.Structure from an occupancy vector"""
        sites = []
        for sp, s in zip(occu, self.supercell_struct):
            if sp != 'Vacancy':
                site = PeriodicSite(sp, s.frac_coords,
                                    self.supercell_struct.lattice)
                sites.append(site)
        return Structure.from_sites(sites)
