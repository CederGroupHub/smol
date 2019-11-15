from __future__ import division
import unittest

from pymatgen import Lattice, Structure
from pyabinitio.cluster_expansion.ce import Cluster,\
        SymmetrizedCluster, ClusterSupercell,\
        ClusterExpansion, SITE_TOL
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.util.coord_utils import is_coord_subset_pbc

from monty.serialization import MontyDecoder, MontyEncoder
import json
import numpy as np

class CETest(unittest.TestCase):

    def setUp(self):
        self.lattice = Lattice([[3,3,0],[0,3,3],[3,0,3]])
        species = [{'Li':0.1, 'Ca':0.1}] * 3 + ['Br']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                  (0.5, 0.5, 0.5),  (0, 0, 0))
        self.structure = Structure(self.lattice, species, coords)
        self.sf = SpacegroupAnalyzer(self.structure)
        self.symops = self.sf.get_symmetry_operations()

    def test_cluster(self):
        #test periodicity of cluster
        c1 = np.array([0.25, 0.25, 0.25])
        c2 = np.array([0.75, 0.75, -1.25])
        c3 = np.array([0.75, 0.75, 0.75])
        c4 = np.array([0.25, 0.25, 2.25])
        clust1 = Cluster([c1,c2], self.lattice)    
        clust2 = Cluster([c3,c4], self.lattice)
        self.assertEqual(clust1, clust2)

        #test edge case handling
        c1 = np.array([0.25, 0.25, 0.25])
        c2 = np.array([0, 0, 1])
        c3 = np.array([0.25, 1.25, -0.75])
        c4 = np.array([0, 1, 0])
        clust1 = Cluster([c1,c2], self.lattice)
        clust2 = Cluster([c3,c4], self.lattice)
        self.assertEqual(clust1, clust2)

        #test from sites
        c1 = np.array([0.25, 0.25, -1.75])
        c2 = np.array([0.75, 0.75, -1.25])
        c3 = np.array([0, 0, -2])
        c4 = np.array([0.5, 0.5, -1.5])
        clust1 = Cluster.from_sites(self.structure)
        self.assertEqual(clust1, Cluster([c1,c2,c3,c4], self.lattice))

        #test representations
        repr(clust1)
        str(clust1)

    def test_SymmetrizedCluster(self):
        clust = Cluster.from_sites(self.structure[:2])
        sc = SymmetrizedCluster(clust, [1,2], self.symops)
        self.assertEqual(len(sc.clusters), 4)
        for c in sc.clusters[1:]:
            self.assertNotEqual(sc.base_cluster, c)
            self.assertEqual(sc, SymmetrizedCluster(c, [1,2], self.symops))
        self.assertEqual(sc.multiplicity, 4)
        self.assertEqual(len(sc.cluster_symops), 12)

        #test representations
        repr(sc)
        str(sc)

    def test_ClusterExpansion(self):
        cb = ClusterExpansion.from_radii(self.structure, {2:6, 3:5})
        self.assertEqual(cb.n_sclusters, 27)
        self.assertEqual(cb.n_bit_orderings, 124)
        self.assertEqual(cb.n_clusters, 377)

        #test representations
        repr(cb)
        str(cb)

        #test as/from_dict
        d = cb.as_dict()
        self.assertEqual(str(cb), str(ClusterExpansion.from_dict(d)))

        #test monty encoding
        new = json.loads(json.dumps(d, cls=MontyEncoder), cls=MontyDecoder)
        self.assertEqual(str(cb), str(new))
    
    def test_ClusterSupercell(self):
        """
        Since the cluster indices do not contain information on which image
        each cluster site is in, cannot use the Cluster.__eq__ method and
        instead allow periodic boundary conditions
        """
        self.structure.add_oxidation_state_by_element({'Br': -1, 'Ca': 2, 'Li': 1})
        cb = ClusterExpansion.from_radii(self.structure, {2:6, 3:5}, use_ewald=True, use_inv_r=True,
                              eta=0.15)
        matrix = np.array([[2,0,0],[0,2,0],[0,1,1]])
        cs = ClusterSupercell(matrix, cb)
        #check that all supercell index groups map to the correct primitive
        #cell sites, and check that max distance under supercell pbc is
        #less than the max distance without pbc
        for sc, inds in cs.cluster_indices:
            for x in inds:
                pbc_radius = np.max(cs.supercell.lattice.get_all_distances(
                                            cs.fcoords[x], cs.fcoords[x]))
                #primitive cell fractional coordinates
                new_fc = np.dot(cs.fcoords[x], cs.supercell_matrix)
                self.assertGreater(sc.max_radius + 1e-7, pbc_radius)
                found = False
                for equiv in sc.clusters:
                    if is_coord_subset_pbc(equiv.sites, new_fc, atol=SITE_TOL):
                        found = True
                        break
                self.assertTrue(found)

        s = Structure(cs.supercell.lattice, ['Ca2+', 'Li+', 'Li+', 'Br-', 'Br-', 'Br-', 'Br-'],
                      [[0.125, 1, 0.25], [0.125, 0.5, 0.25], [0.375, 0.5, 0.75], [0, 0, 0], [0, 0.5, 1],
                       [0.5, 1, 0], [0.5, 0.5, 0]])

        self.assertAlmostEqual(cs.corr_from_structure(s)[cb.n_bit_orderings] * cs.size,
                               EwaldSummation(s, eta=cs._ewald._eta).total_energy, places=5)
        expected = [-9.31110885, -1.12141621, -2.40223921, 0, 0, 0.58210444, 0, 0, 0, 0, 0]
        self.assertTrue(np.allclose(expected, cs.corr_from_structure(s)[cb.n_bit_orderings:]))
        a = cb.corr_from_structure(s)
        s.make_supercell([2,1,1])
        b = cb.corr_from_structure(s)
        self.assertTrue(np.allclose(a, b))

    def test_delta_corr(self):
        self.structure.add_oxidation_state_by_element({'Br': -1, 'Ca': 2, 'Li': 1})
        cb = ClusterExpansion.from_radii(self.structure, {2: 6, 3: 5}, use_ewald=True, use_inv_r=True)
        matrix = np.array([[2,0,0],[0,2,0],[0,1,1]])
        cs = ClusterSupercell(matrix, cb)

        s = Structure(cs.supercell.lattice, ['Ca2+', 'Li+', 'Li+', 'Br-', 'Br-', 'Br-', 'Br-'],
                      [[0.125, 1, 0.25], [0.125, 0.5, 0.25], [0.375, 0.5, 0.75], [0, 0, 0], [0, 0.5, 1],
                       [0.5, 1, 0], [0.5, 0.5, 0]])

        occu = cs.occu_from_structure(s)
        new_occu = occu.copy()
        flips = [(0, 1), (1, 1), (3, 2)]
        for f in flips:
            new_occu[f[0]] = f[1]
        expected = cs.corr_from_occupancy(new_occu) - cs.corr_from_occupancy(occu)
        d_corr, next_occu = cs.delta_corr(flips, occu, True)
        self.assertTrue(np.allclose(d_corr, expected))
        self.assertTrue(np.all(next_occu == new_occu))

    def test_delta_corr_no_ewald(self):
        self.structure.add_oxidation_state_by_element({'Br': -1, 'Ca': 2, 'Li': 1})
        cb = ClusterExpansion.from_radii(self.structure, {2: 6, 3: 5}, use_ewald=False, use_inv_r=False)
        matrix = np.array([[2,0,0],[0,2,0],[0,1,1]])
        cs = ClusterSupercell(matrix, cb)

        s = Structure(cs.supercell.lattice, ['Ca2+', 'Li+', 'Li+', 'Br-', 'Br-', 'Br-', 'Br-'],
                      [[0.125, 1, 0.25], [0.125, 0.5, 0.25], [0.375, 0.5, 0.75], [0, 0, 0], [0, 0.5, 1],
                       [0.5, 1, 0], [0.5, 0.5, 0]])

        occu = cs.occu_from_structure(s)
        new_occu = occu.copy()
        flips = [(0, 1), (1, 1), (3, 2)]
        for f in flips:
            new_occu[f[0]] = f[1]
        expected = cs.corr_from_occupancy(new_occu) - cs.corr_from_occupancy(occu)
        d_corr, next_occu = cs.delta_corr(flips, occu, True)
        self.assertTrue(np.allclose(d_corr, expected))
        self.assertTrue(np.all(next_occu == new_occu))

    def test_vs_CASM_pairs(self):
        species = [{'Li':0.1}] * 3 + ['Br']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75), 
                  (0.5, 0.5, 0.5),  (0, 0, 0))
        structure = Structure(self.lattice, species, coords)
        cb = ClusterExpansion.from_radii(structure, {2:6})
        cs = ClusterSupercell([[1,0,0],[0,1,0],[0,0,1]], cb)
        #last two clusters are switched from CASM output (and using occupancy basis)
        #all_li (ignore casm point term)
        self.assertTrue(np.allclose(cs.corr_from_occupancy([0,0,0]), np.array([1]*12)))
        #all_vacancy
        self.assertTrue(np.allclose(cs.corr_from_occupancy([1,1,1]), np.array([1]+[0]*11)))
        #octahedral
        self.assertTrue(np.allclose(cs.corr_from_occupancy([1,1,0]),
                                    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]))
        #tetrahedral
        self.assertTrue(np.allclose(cs.corr_from_occupancy([0,0,1]),
                                    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0]))
        #mixed
        self.assertTrue(np.allclose(cs.corr_from_occupancy([1,0,0]),
                                    [1, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0, 0.5, 1]))
        #single_tet
        self.assertTrue(np.allclose(cs.corr_from_occupancy([1,0,1]),
                                    [1, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0]))
        
    def test_vs_CASM_triplets(self):
        """
        test vs casm generated correlation with occupancy basis
        """
        species = [{'Li':0.1}] * 3 + ['Br']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75), 
                  (0.5, 0.5, 0.5),  (0, 0, 0))
        structure = Structure(self.lattice, species, coords)
        cb = ClusterExpansion.from_radii(structure, {2:6, 3:4.5})
        cs = ClusterSupercell([[1,0,0],[0,1,0],[0,0,1]], cb)
        #last two pair terms are switched from CASM output (and using occupancy basis)
        #all_li (ignore casm point term)
        self.assertTrue(np.allclose(cs.corr_from_occupancy([0,0,0]), np.array([1]*19)))
        #all_vacancy
        self.assertTrue(np.allclose(cs.corr_from_occupancy([1,1,1]), np.array([1]+[0]*18)))
        #octahedral
        self.assertTrue(np.allclose(cs.corr_from_occupancy([1,1,0]),
                                    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]))
        
        #tetrahedral
        self.assertTrue(np.allclose(cs.corr_from_occupancy([0,0,1]),
                                    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0]))
        #mixed
        self.assertTrue(np.allclose(cs.corr_from_occupancy([1,0,0]),
                                    [1, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0, 0.5, 1, 0, 0, 0.5, 0.5, 0.5, 0.5, 1]))
        #single_tet
        self.assertTrue(np.allclose(cs.corr_from_occupancy([1,0,1]),
                                    [1, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.5, 0.5, 0]))
        
    def test_vs_CASM_multicomp(self):
        cb = ClusterExpansion.from_radii(self.structure, {2:5})
        cs = ClusterSupercell([[1,0,0],[0,1,0],[0,0,1]], cb)
        
        #mixed
        self.assertTrue(np.allclose(cs.corr_from_occupancy([2,0,0]),
                [1, 0.5, 0, 1, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 1, 0, 0, 0.5, 0, 0, 0]))
        #Li_tet_ca_oct
        self.assertTrue(np.allclose(cs.corr_from_occupancy([2,0,1]),
                [1, 0.5, 0, 0, 1, 0, 0.5, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1, 0, 0.5, 0, 0]))
    
    def test_ClusterExpansion2(self):
        ce = ClusterExpansion.from_radii(self.structure, {2: 5})
        #make an ordered supercell
        s = self.structure.copy()
        s.make_supercell([2,1,1])
        species = ('Li', 'Ca', 'Li', 'Ca', 'Br', 'Br')
        coords = ((0.125, 0.25, 0.25),
                  (0.625, 0.25, 0.25),
                  (0.375, 0.75, 0.75),
                  (0.25, 0.5, 0.5),
                  (0,0,0),
                  (0.5,0,0))
        s = Structure(s.lattice, species, coords)
        self.assertTrue(np.allclose(ce.corr_from_structure(s),
                [1, 0.5, 0.25, 0, 0.5, 0, 0.375, 0, 0.0625, 0.25, 0.125, 
                 0, 0.25, 0.125, 0.125, 0, 0, 0.25, 0, 0.125, 0, 0.1875]))
        
        import random
        for _ in range(10):
            random.shuffle(s)
            self.assertTrue(np.allclose(ce.corr_from_structure(s),
                [1, 0.5, 0.25, 0, 0.5, 0, 0.375, 0, 0.0625, 0.25, 0.125, 
                 0, 0.25, 0.125, 0.125, 0, 0, 0.25, 0, 0.125, 0, 0.1875]))
                                
    def test_symmetry_error(self):
        # this tests a slightly off symmetry structure, and makes sure
        # it raises a useful error message
        l = Lattice.from_parameters(2.882900, 2.882900, 10.492700,
                                    90, 90, 120)

        s = Structure(l,
                      [{'Na': 0.5}] * 4 + ['O'] * 4,
                      [[0, 0, 0.25],
                       [0, 0, 0.75],
                       [0.6667, 0.3334, 0.25],
                       [0.3333, 0.6666, 0.75],
                       [0.3333, 0.6666, 0.4064],
                       [0.6667, 0.3334, 0.5936],
                       [0.6667, 0.3334, 0.9064],
                       [0.3333, 0.6666, 0.0936]])

        self.assertRaisesRegexp(ValueError, 'symmetry operations',
                                ClusterExpansion.from_radii, s, {2: 1})
