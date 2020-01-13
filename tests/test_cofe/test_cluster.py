import unittest
from itertools import combinations
import numpy as np
from pymatgen import Structure, Lattice
from smol.cofe.configspace import Cluster


class TestCluster(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = Lattice([[3, 3, 0], [0, 3, 3], [3, 0, 3]])
        species = [{'Li': 0.1, 'Ca': 0.1}] * 3 + ['Br']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                  (0.5, 0.5, 0.5), (0, 0, 0))
        self.structure = Structure(self.lattice, species, coords)

        # Create cluster
        c1 = np.array([0.25, 0.25, -1.75])
        c2 = np.array([0.75, 0.75, -1.25])
        c3 = np.array([0, 0, -2])
        c4 = np.array([0.5, 0.5, -1.5])
        self.cluster = Cluster([c1, c2, c3, c4], self.lattice)

    def test_from_sites(self):
        clust1 = Cluster.from_sites(self.structure.sites)
        self.assertEqual(clust1, self.cluster)

    def test_size(self):
        self.assertEqual(self.cluster.size, 4)

    def test_radius(self):
        coords = self.lattice.get_cartesian_coords(self.cluster.sites)
        radius = max([np.sum((i - j)**2) for i, j in combinations(coords, 2)])**0.5
        self.assertEqual(self.cluster.radius, radius)

    def test_periodicity(self):
        """Test periodicity of clusters"""
        c1 = np.array([0.25, 0.25, 0.25])
        c2 = np.array([0.75, 0.75, -1.25])
        c3 = np.array([0.75, 0.75, 0.75])
        c4 = np.array([0.25, 0.25, 2.25])
        clust1 = Cluster([c1, c2], self.lattice)
        clust2 = Cluster([c3, c4], self.lattice)
        self.assertEqual(clust1, clust2)

    def test_edge_case(self):
        """Test edge case handling"""
        c1 = np.array([0.25, 0.25, 0.25])
        c2 = np.array([0, 0, 1])
        c3 = np.array([0.25, 1.25, -0.75])
        c4 = np.array([0, 1, 0])
        clust1 = Cluster([c1, c2], self.lattice)
        clust2 = Cluster([c3, c4], self.lattice)
        self.assertEqual(clust1, clust2)

    def test_repr(self):
        repr(self.cluster)

    def test_str(self):
        str(self.cluster)

    def test_msonable(self):
        d = self.cluster.as_dict()
        self.assertEqual(self.cluster, Cluster.from_dict(d))

