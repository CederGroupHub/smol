import unittest
import warnings
import numpy as np
from smol.cofe import StructureWrangler, ClusterSubspace
from smol.cofe.configspace import EwaldTerm
from tests.data import lno_prim, lno_data

class TestStructureWrangler(unittest.TestCase):
    def setUp(self) -> None:
        self.cs = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1},
                                             ltol=0.15, stol=0.2,
                                             angle_tol=5, supercell_size='O2-')
        self.sw = StructureWrangler(self.cs)

    def test_add_data(self):
        for struct, energy in lno_data[:-1]:
            self.sw.add_data(struct, energy, weight=2.0)
        self.assertTrue(all(w == 2.0 for w in self.sw.weights))
        self.sw.add_data(*lno_data[-1], weight=3.0)
        self.assertTrue(len(self.sw.weights) == len(self.sw.items))
        self.assertTrue(self.sw.weights[-1] == 3.0)
        self.sw.remove_all_data()
        self.assertTrue(len(self.sw.items) == 0)
        for struct, energy in lno_data[:5]:
            self.sw.add_data(struct, energy)
        self.assertTrue(self.sw.weights is None)

    def test_update_features(self):
        for struct, energy in lno_data[:5]:
            self.sw.add_data(struct, energy)
        shape = self.sw.feature_matrix.shape
        self.cs.add_external_term(EwaldTerm)
        self.sw.update_features()
        self.assertEqual(shape[1] + 1, self.sw.feature_matrix.shape[1])

    # TODO write a better test. One that actually checks the structures
    #  expected to be removed are removed
    def test_filter_by_ewald(self):
        for struct, energy in lno_data:
            self.sw.add_data(struct, energy)
        len_total = len(self.sw.items)
        self.sw.filter_by_ewald(1)
        len_filtered = len(self.sw.items)
        self.assertNotEqual(len_total, len_filtered)
        self.assertEqual(self.sw.metadata['applied_filters'][0]['Ewald']['nstructs_removed'],
                         len_total - len_filtered)
        self.assertEqual(self.sw.metadata['applied_filters'][0]['Ewald']['nstructs_total'],
                         len_total)

    def test_weights_e_above_comp(self):
        sw = StructureWrangler(self.cs, weights='composition',
                               temperature=1000)
        for struct, energy in lno_data[:-1]:
            sw.add_data(struct, energy)
        expected = np.array([0.85637358, 0.98816678, 1., 0.59209449, 1.,
                    0.92882071, 0.87907454, 0.94729315, 0.40490513, 0.82484222,
                    0.81578984, 1., 0.89615121, 0.92893004, 0.81650693,
                    0.6080223 , 0.94848913, 0.92135297, 0.92326977, 0.83995635,
                    1., 0.94663979, 1., 0.9414506 , 1.])
        self.assertTrue(np.allclose(expected[:-1], sw.weights))
        self.assertWarns(Warning, sw.add_data, *lno_data[-1], 2)
        self.assertTrue(np.allclose(expected, sw.weights))

    def test_weights_e_above_hull(self):
        sw = StructureWrangler(self.cs, weights='hull', temperature=1000)
        for struct, energy in lno_data[:-1]:
            sw.add_data(struct, energy)
        expected = np.array([0.85637358, 0.98816678, 1., 0.56916328, 0.96127103,
           0.89284844, 0.84502889, 0.91060546, 0.40490513, 0.82484222,
           0.81578984, 1., 0.89615121, 0.92893004, 0.81650693,
           0.58819251, 0.91755548, 0.89130433, 0.89315862, 0.81256235,
           0.9673864 , 0.91576647, 1., 0.9414506 , 1])
        self.assertTrue(np.allclose(expected[:-1], sw.weights))
        sw.add_data(*lno_data[-1])
        self.assertTrue(np.allclose(expected, sw.weights))

    def test_weight_exception(self):
        self.assertRaises(ValueError, StructureWrangler, self.cs, 'blab')

    def test_msonable(self):
        d = self.sw.as_dict()
        sw = StructureWrangler.from_dict(d)
        self.assertTrue(np.array_equal(sw.properties, self.sw.properties))
        self.assertTrue(all([s1 == s2 for s1, s2 in zip(sw.structures, self.sw.structures)]))
        self.assertTrue(all([s1 == s2 for s1, s2 in zip(sw.refined_structures, self.sw.refined_structures)]))
        self.assertTrue(np.array_equal(sw.feature_matrix, self.sw.feature_matrix))
