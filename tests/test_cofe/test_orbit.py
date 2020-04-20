import unittest
from itertools import combinations_with_replacement
import numpy as np
from pymatgen import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from smol.cofe.configspace import Orbit, Cluster
from smol.cofe.configspace.basis import basis_factory


class TestOrbit(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = Lattice([[3, 3, 0], [0, 3, 3], [3, 0, 3]])
        species = [{'Li': 0.1, 'Ca': 0.1}] * 3 + ['Br']
        self.coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                  (0.5, 0.5, 0.5), (0, 0, 0))
        structure = Structure(self.lattice, species, self.coords)
        sf = SpacegroupAnalyzer(structure)
        self.symops = sf.get_symmetry_operations()
        self.bits = [['Li', 'Ca', 'Vacancy'],
                    ['Li', 'Ca', 'Vacancy']]
        self.bases = [basis_factory('indicator', bit) for bit in self.bits]
        self.basecluster = Cluster(self.coords[:2], self.lattice)
        self.orbit = Orbit(self.coords[:2], self.lattice, [[0, 1], [0, 1]],
                           self.bases, self.symops)
        self.orbit.assign_ids(1, 1, 1)

    def test_constructor(self):
        self.assertRaises(AttributeError, Orbit, self.coords[:3], self.lattice,
                          [[0, 1], [0, 1]], self.bases, self.symops)
        self.assertRaises(AttributeError, Orbit, self.coords[:3], self.lattice,
                          [[0, 1]], self.bases, self.symops)

    def test_basecluster(self):
        self.assertEqual(self.orbit.base_cluster, self.basecluster)

    def test_clusters(self):
        self.assertEqual(len(self.orbit.clusters), 4)
        self.assertEqual(self.orbit.clusters[0], self.basecluster)
        for cluster in self.orbit.clusters[1:]:
            self.assertNotEqual(self.orbit.base_cluster, cluster)

    def test_multiplicity(self):
        self.assertEqual(self.orbit.multiplicity, 4)

    def test_cluster_symops(self):
        self.assertEqual(len(self.orbit.cluster_symops), 12)

    def test_eq(self):
        orbit1 = Orbit(self.coords[:2], self.lattice, [[0, 1], [0, 1]],
                       self.bases, self.symops)
        orbit2 = Orbit(self.coords[:3], self.lattice, [[0, 1], [0, 1], [0, 1]],
                       self.bases + [None], self.symops)
        self.assertEqual(orbit1, self.orbit)
        self.assertNotEqual(orbit2, self.orbit)

    def test_bit_combos(self):
        bit_combos = self.orbit.bit_combos # orbit with two symmetrically equivalent sites
        self.assertEqual(len(bit_combos), 3)
        orbit = Orbit(self.coords[1:3], self.lattice, [[0, 1], [0, 1]],
                      self.bases, self.symops)
        bit_combos = orbit.bit_combos  # orbit with two symmetrically distinct sites
        self.assertEqual(len(bit_combos), 4)

    def test_is_orthonormal(self):
        self.assertFalse(self.orbit.basis_orthogonal)
        self.assertFalse(self.orbit.basis_orthonormal)
        for b in self.bases:
            b.orthonormalize()
            self.assertTrue(b.is_orthogonal)
        orbit1 = Orbit(self.coords[:2], self.lattice, [[0, 1], [0, 1]],
                       self.bases, self.symops)
        self.assertTrue(orbit1.basis_orthogonal)
        self.assertTrue(orbit1.basis_orthonormal)

    def _test_eval(self, bases):
        occu = list(range(len(self.bits[0])))
        for s1, s2 in combinations_with_replacement(occu, 2):
            for i, j in combinations_with_replacement([0, 1], 2):
                self.assertEqual(bases[0].function_array[i, s1]*bases[1].function_array[j, s2],
                                 self.orbit.eval([i,j], [s1, s2]))

    def test_remove_bit_combo(self):
        bits = [0, 0]
        self.orbit.remove_bit_combo(bits)
        self.assertFalse(any(any(np.array_equal(bits, b) for b in b_c)
                             for b_c in self.orbit.bit_combos))
        bits = [0, 1]
        equiv_bits = [1, 0]
        self.orbit.remove_bit_combo(bits)
        self.assertFalse(any(any(np.array_equal(equiv_bits, b) for b in b_c)
                             for b_c in self.orbit.bit_combos))

        # check that it complains if we remove the last remaining combo
        self.assertRaises(RuntimeError, self.orbit.remove_bit_combo, [1, 1])

    def test_remove_bit_combo_by_inds(self):
        orb1 = Orbit(self.coords[:2], self.lattice, [[0, 1], [0, 1]],
              self.bases, self.symops)
        orb1.assign_ids(1, 1, 1)
        orb2 = Orbit(self.coords[:2], self.lattice, [[0, 1], [0, 1]],
                     self.bases, self.symops)
        orb2.assign_ids(1, 1, 1)

        bit = orb1.bit_combos[1][0]
        orb1.remove_bit_combos_by_inds([1])
        orb2.remove_bit_combo(bit)
        self.assertTrue(all(np.array_equal(bc1, bc2) for bc1, bc2 in
                             zip(orb1.bit_combos, orb2.bit_combos)))
        self.assertRaises(RuntimeError, orb1.remove_bit_combos_by_inds, [0,1])

    def test_eval(self):
        # Test cluster function evaluation with indicator basis
        bases = [basis_factory('indicator', bit) for bit in self.bits]
        self._test_eval(bases)

    def test_transform_basis(self):
        for basis in ('sinusoid', 'chebyshev', 'legendre'):
            self.orbit.transform_site_bases(basis)
            bases = [basis_factory(basis, bit) for bit in self.bits]
            self._test_eval(bases)

    def test_exceptions(self):
        self.assertRaises(AttributeError, Orbit, self.coords[:3],
                          self.lattice, [[0, 1], [0, 1]], self.bases, self.symops)
        self.assertRaises(RuntimeError, self.orbit.remove_bit_combos_by_inds,
                          [4])


    def test_repr(self):
        repr(self.orbit)

    def test_str(self):
        str(self.orbit)

    def test_msonable(self):
        d = self.orbit.as_dict()
        self.assertEqual(self.orbit, Orbit.from_dict(d))
