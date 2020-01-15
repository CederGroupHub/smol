import unittest
import numpy as np
from itertools import combinations
from pymatgen import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from smol.cofe.configspace import ClusterSupercell
from smol.cofe.configspace.clusterspace import get_bits
from smol.cofe import ClusterSubspace


#TODO need to implement tests for missing functions
class TestSuperCell(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = Lattice([[3, 3, 0], [0, 3, 3], [3, 0, 3]])
        species = [{'Li': 0.1, 'Ca': 0.1}] * 3 + ['Br']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                  (0.5, 0.5, 0.5), (0, 0, 0))
        self.structure = Structure(self.lattice, species, coords)

    def test_occu_from_structure(self):
        pass

    def test_generate_mappings(self):
        pass

    def test_vs_CASM_pairs(self):
        species = [{'Li':0.1}] * 3 + ['Br']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                  (0.5, 0.5, 0.5),  (0, 0, 0))
        structure = Structure(self.lattice, species, coords)
        cs = ClusterSubspace.from_radii(structure, {2: 6})

        supercell = cs.structure.copy()

        sc = ClusterSupercell(cs, supercell, [[1,0,0],[0,1,0],[0,0,1]], get_bits(supercell))
        #last two clusters are switched from CASM output (and using occupancy basis)
        #all_li (ignore casm point term)
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Li','Li','Li']), np.array([1]*12)))
        #all_vacancy
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Vacancy','Vacancy','Vacancy']), np.array([1]+[0]*11)))
        #octahedral
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Vacancy','Vacancy','Li']),
                                    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]))
        #tetrahedral
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Li','Li','Vacancy']),
                                   [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0]))
        #mixed
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Li','Vacancy','Li']),
                                   [1, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0, 0.5, 1]))
        #single_tet
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Li','Vacancy','Vacancy']),
                                   [1, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0]))

    def test_vs_CASM_triplets(self):
        """
        test vs casm generated correlation with occupancy basis
        """
        species = [{'Li': 0.1}] * 3 + ['Br']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                  (0.5, 0.5, 0.5), (0, 0, 0))
        structure = Structure(self.lattice, species, coords)
        cs = ClusterSubspace.from_radii(structure, {2: 6, 3: 4.5})
        supercell = cs.structure.copy()
        sc = ClusterSupercell(cs, supercell, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], get_bits(supercell))

        # last two pair terms are switched from CASM output (and using occupancy basis)
        # all_li (ignore casm point term)
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Vacancy', 'Vacancy', 'Vacancy']),
                                    np.array([1] + [0] * 18)))
        # all_vacancy
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Li', 'Li', 'Li']), np.array([1] * 19)))
        # octahedral
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Vacancy', 'Vacancy', 'Li']),
                                    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]))

        # tetrahedral
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Li', 'Li', 'Vacancy']),
                                    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0]))
        # mixed
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Li', 'Vacancy', 'Li']),
                                    [1, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0, 0.5, 1, 0, 0, 0.5, 0.5, 0.5, 0.5, 1]))
        # single_tet
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Li', 'Vacancy', 'Vacancy']),
                                    [1, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.5, 0.5, 0]))

    def test_vs_CASM_multicomp(self):
        cs = ClusterSubspace.from_radii(self.structure, {2: 5})
        supercell = cs.structure.copy()
        sc = ClusterSupercell(cs, supercell, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], get_bits(supercell))

        # mixed
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Vacancy', 'Li', 'Li']),
                                    [1, 0.5, 0, 1, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 1, 0, 0, 0.5, 0, 0, 0]))
        # Li_tet_ca_oct
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Vacancy', 'Li', 'Ca']),
                                    [1, 0.5, 0, 0, 1, 0, 0.5, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1, 0, 0.5, 0, 0]))

    def test_delta_corr(self):
        pass