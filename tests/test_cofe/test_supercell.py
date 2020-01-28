import unittest
import numpy as np
from itertools import combinations
from pymatgen import Lattice, Structure
from pymatgen.util.coord import is_coord_subset_pbc
from smol.cofe.configspace import ClusterSupercell
from smol.cofe.configspace.clusterspace import get_bits
from smol.cofe.utils import SITE_TOL
from smol.cofe import ClusterSubspace


class TestSuperCell(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = Lattice([[3, 3, 0], [0, 3, 3], [3, 0, 3]])
        species = [{'Li': 0.1, 'Ca': 0.1}] * 3 + ['Br']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                  (0.5, 0.5, 0.5), (0, 0, 0))
        self.structure = Structure(self.lattice, species, coords)

    # TODO Implement this?
    def test_occu_from_structure(self):
        pass

    def test_generate_mappings(self):
        # check that all supercell index groups map to the correct primitive
        # cell sites, and check that max distance under supercell pbc is
        # less than the max distance without pbc

        cs = ClusterSubspace.from_radii(self.structure, {2: 6, 3: 5})
        m = np.array([[2, 0, 0], [0, 2, 0], [0, 1, 1]])
        supercell = cs.structure.copy()
        supercell.make_supercell(m)

        sc = ClusterSupercell(supercell, supercell_matrix=m,
                              bits=get_bits(supercell),
                              n_bit_orderings=cs.n_bit_orderings,
                              orbits=cs.orbits)
        for orb, inds in sc.cluster_indices:
            for x in inds:
                pbc_radius = np.max(sc.supercell.lattice.get_all_distances(
                    sc.fcoords[x], sc.fcoords[x]))
                # primitive cell fractional coordinates
                new_fc = np.dot(sc.fcoords[x], sc.supercell_matrix)
                self.assertGreater(orb.radius + 1e-7, pbc_radius)
                found = False
                for equiv in orb.clusters:
                    if is_coord_subset_pbc(equiv.sites, new_fc, atol=SITE_TOL):
                        found = True
                        break
                self.assertTrue(found)

    def test_periodicity(self):
        # Check to see if a supercell of a smaller supercell gives the same corr
        m = np.array([[2, 0, 0], [0, 2, 0], [0, 1, 1]])
        supercell = self.structure.copy()
        supercell.make_supercell(m)
        s = Structure(supercell.lattice, ['Ca', 'Li', 'Li', 'Br', 'Br', 'Br', 'Br'],
                      [[0.125, 1, 0.25], [0.125, 0.5, 0.25], [0.375, 0.5, 0.75], [0, 0, 0], [0, 0.5, 1],
                       [0.5, 1, 0], [0.5, 0.5, 0]])
        cs = ClusterSubspace.from_radii(self.structure, {2: 6, 3: 5})
        a = cs.corr_from_structure(s)
        s.make_supercell([2, 1, 1])
        b = cs.corr_from_structure(s)
        self.assertTrue(np.allclose(a, b))

    def test_vs_CASM_pairs(self):
        species = [{'Li':0.1}] * 3 + ['Br']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                  (0.5, 0.5, 0.5),  (0, 0, 0))
        structure = Structure(self.lattice, species, coords)
        cs = ClusterSubspace.from_radii(structure, {2: 6})

        supercell = cs.structure.copy()

        sc = ClusterSupercell(supercell,
                              supercell_matrix=[[1,0,0],[0,1,0],[0,0,1]],
                              bits=get_bits(supercell),
                              n_bit_orderings=cs.n_bit_orderings,
                              orbits=cs.orbits)
        # last two clusters are switched from CASM output (and using occupancy basis)
        # all_li (ignore casm point term)
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Li', 'Li', 'Li']),
                                    np.array([1]*12)))
        # all_vacancy
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Vacancy', 'Vacancy', 'Vacancy']),
                                    np.array([1]+[0]*11)))
        # octahedral
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Vacancy', 'Vacancy', 'Li']),
                                    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]))
        # tetrahedral
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Li', 'Li', 'Vacancy']),
                                    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0]))
        # mixed
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Li', 'Vacancy', 'Li']),
                                    [1, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0, 0.5, 1]))
        # single_tet
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Li', 'Vacancy', 'Vacancy']),
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
        sc = ClusterSupercell(supercell,
                              supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              bits=get_bits(supercell),
                              n_bit_orderings=cs.n_bit_orderings,
                              orbits=cs.orbits)

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
        sc = ClusterSupercell(supercell,
                              supercell_matrix= [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              bits=get_bits(supercell),
                              n_bit_orderings=cs.n_bit_orderings,
                              orbits=cs.orbits)

        # mixed
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Vacancy', 'Li', 'Li']),
                                    [1, 0.5, 0, 1, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 1, 0, 0, 0.5, 0, 0, 0]))
        # Li_tet_ca_oct
        self.assertTrue(np.allclose(sc.corr_from_occupancy(['Vacancy', 'Li', 'Ca']),
                                    [1, 0.5, 0, 0, 1, 0, 0.5, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1, 0, 0.5, 0, 0]))

    # TODO write this too!
    def test_delta_corr(self):
        pass