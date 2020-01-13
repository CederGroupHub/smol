import unittest
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
        bits = [['Li', 'Ca', 'Vacancy'],
                ['Li', 'Ca', 'Vacancy']]
        self.bases = tuple(basis_factory('indicator', bit) for bit in bits)
        self.basecluster = Cluster(self.coords[:2], self.lattice)
        self.orbit = Orbit(self.coords[:2], self.lattice, [[0, 1], [0, 1]],
                           self.bases, self.symops)
        self.orbit.assign_ids(1, 1, 1)

    def test_basecluster(self):
        self.assertEqual(self.orbit.basecluster, self.basecluster)

    def test_clusters(self):
        self.assertEqual(len(self.orbit.clusters), 4)
        self.assertEqual(self.orbit.clusters[0], self.basecluster)
        for cluster in self.orbit.clusters[1:]:
            self.assertNotEqual(self.orbit.basecluster, cluster)

    def test_multiplicity(self):
        self.assertEqual(self.orbit.multiplicity, 4)

    def test_cluster_symops(self):
        self.assertEqual(len(self.orbit.cluster_symops), 12)

    def test_eq(self):
        orbit1 = Orbit(self.coords[:2], self.lattice, [[0, 1], [0, 1]],
                       self.bases, self.symops)
        orbit2 = Orbit(self.coords[:3], self.lattice, [[0, 1], [0, 1], [0, 1]],
                       self.bases, self.symops)
        self.assertEqual(orbit1, self.orbit)
        self.assertNotEqual(orbit2, self.orbit)

    #TODO write these tests
    def test_bit_combos(self):
        pass

    def test_eval(self):
        pass

    def test_repr(self):
        repr(self.orbit)

    def test_str(self):
        str(self.orbit)

    def test_msonable(self):
        d = self.orbit.as_dict()
        self.assertEqual(self.orbit, Orbit.from_dict(d))
