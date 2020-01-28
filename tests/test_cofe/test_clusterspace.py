import unittest
import random
import numpy as np
from itertools import combinations
from pymatgen import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from smol.cofe import ClusterSubspace


class TestClusterSubSpace(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = Lattice([[3, 3, 0], [0, 3, 3], [3, 0, 3]])
        self.species = [{'Li': 0.1, 'Ca': 0.1}] * 3 + ['Br']
        self.coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                       (0.5, 0.5, 0.5), (0, 0, 0))
        self.structure = Structure(self.lattice, self.species, self.coords)
        sf = SpacegroupAnalyzer(self.structure)
        self.symops = sf.get_symmetry_operations()
        self.cs = ClusterSubspace.from_radii(self.structure, {2: 6, 3: 5})

    def test_numbers(self):
        # Test the total generated orbits, orderings and clusters are as expected.
        self.assertEqual(self.cs.n_orbits, 27)
        self.assertEqual(self.cs.n_bit_orderings, 124)
        self.assertEqual(self.cs.n_clusters, 377)

    def test_orbits(self):
        self.assertEqual(len(self.cs.orbits) + 1, self.cs.n_orbits)  # +1 for empty cluster
        for o1, o2 in combinations(self.cs.orbits, 2):
            self.assertNotEqual(o1, o2)

    def test_iterorbits(self):
        orbits = [o for o in self.cs.iterorbits()]
        for o1, o2 in zip(orbits, self.cs.orbits):
            self.assertEqual(o1, o2)

    #  These can probably be improved to check odd and specific cases we want to watch out for
    def test_supercell_matrix_from_structure(self):
        # Simple scaling
        supercell = self.structure.copy()
        supercell.make_supercell(2)
        sc_matrix = self.cs.supercell_matrix_from_structure(supercell)
        self.assertAlmostEqual(np.linalg.det(sc_matrix), 8)

        # A more complex supercell_structure
        m = np.array([[ 0,  5,  3],
                      [-2,  0,  2],
                      [-2,  4,  3]])
        supercell = self.structure.copy()
        supercell.make_supercell(m)
        sc_matrix = self.cs.supercell_matrix_from_structure(supercell)
        self.assertAlmostEqual(np.linalg.det(sc_matrix), abs(np.linalg.det(m)))

        # Test a slightly distorted structure
        lattice = Lattice([[2.95, 3, 0], [0, 3, 2.9], [3, 0, 3]])
        structure = Structure(lattice, self.species, self.coords)
        supercell = structure.copy()
        supercell.make_supercell(2)
        sc_matrix = self.cs.supercell_matrix_from_structure(supercell)
        self.assertAlmostEqual(np.linalg.det(sc_matrix), 8)

        m = np.array([[0, 5, 3],
                      [-2, 0, 2],
                      [-2, 4, 3]])
        supercell = structure.copy()
        supercell.make_supercell(m)
        sc_matrix = self.cs.supercell_matrix_from_structure(supercell)
        self.assertAlmostEqual(np.linalg.det(sc_matrix), abs(np.linalg.det(m)))

    def test_supercell_from_matrix(self):
        m = np.array([[0, 5, 3],
                      [-2, 0, 2],
                      [-2, 4, 3]])
        sc = self.cs.supercell_from_matrix(m)
        supercell = self.structure.copy()
        supercell.make_supercell(m)
        self.assertEqual(sc.supercell_struct, supercell)

    def test_refine_structure(self):
        lattice = Lattice([[2.95, 3, 0], [0, 3, 2.9], [3, 0, 3]])
        structure = Structure(lattice, ['Li', ] * 2 + ['Ca'] + ['Br'], self.coords)
        structure.make_supercell(2)
        structure = self.cs.refine_structure(structure).get_primitive_structure()

        # This is failing in pymatgen 2020.1.10, since lattice matrices are not the same,
        # but still equivalent
        # self.assertEqual(self.lattice, structure.lattice)
        self.assertTrue(np.allclose(self.lattice.parameters, structure.lattice.parameters))

    def test_corr_from_structure(self):
        structure = Structure(self.lattice, ['Li',] * 2 + ['Ca'] + ['Br'], self.coords)
        corr = self.cs.corr_from_structure(structure)
        self.assertEqual(len(corr), self.cs.n_bit_orderings + len(self.cs.external_terms))
        self.assertEqual(corr[0], 1)

        cs = ClusterSubspace.from_radii(self.structure, {2: 5})

        # make an ordered supercell_structure
        s = self.structure.copy()
        s.make_supercell([2, 1, 1])
        species = ('Li', 'Ca', 'Li', 'Ca', 'Br', 'Br')
        coords = ((0.125, 0.25, 0.25),
                  (0.625, 0.25, 0.25),
                  (0.375, 0.75, 0.75),
                  (0.25, 0.5, 0.5),
                  (0, 0, 0),
                  (0.5, 0, 0))
        s = Structure(s.lattice, species, coords)
        self.assertEqual(len(cs.corr_from_structure(s)), 22)
        self.assertTrue(np.allclose(cs.corr_from_structure(s),
                                    [1, 0.5, 0.25, 0, 0.5, 0, 0.375, 0, 0.0625, 0.25, 0.125,
                                     0, 0.25, 0.125, 0.125, 0, 0, 0.25, 0, 0.125, 0, 0.1875]))

        for _ in range(10):
            random.shuffle(s)
            self.assertTrue(np.allclose(cs.corr_from_structure(s),
                                        [1, 0.5, 0.25, 0, 0.5, 0, 0.375, 0, 0.0625, 0.25, 0.125,
                                         0, 0.25, 0.125, 0.125, 0, 0, 0.25, 0, 0.125, 0, 0.1875]))

    def test_repr(self):
        repr(self.cs)

    def test_str(self):
        str(self.cs)

    def test_msonable(self):
        d = self.cs.as_dict()
        cs = ClusterSubspace.from_dict(d)
        self.assertEqual(cs.n_orbits, 27)
        self.assertEqual(cs.n_bit_orderings, 124)
        self.assertEqual(cs.n_clusters, 377)
        self.assertEqual(str(cs), str(self.cs))
