import unittest
import warnings
import numpy as np
from smol.cofe import StructureWrangler, ClusterSubspace
from smol.cofe.configspace import EwaldTerm
from tests.data import lno_prim, lno_data

class TestStructureWrangler(unittest.TestCase):
    def setUp(self) -> None:
        self.cs = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1}, ltol=0.15, stol=0.2,
                                             angle_tol=5, supercell_size='O2-')
        self.sw = StructureWrangler(self.cs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  #  Supress warnings for structures not mapped to prim
            self.sw.add_data(lno_data)

    def test_add_data_w_weights(self):
        sw = StructureWrangler(self.cs)
        sw.add_data(lno_data, weights='hull')
        self.assertTrue(np.all([w is not None for w in sw.weights]))
        sw.remove_all_data()
        sw.add_data(lno_data, weights=['hull', {'temperature':1000}])
        self.assertTrue(np.all([w is not None for w in sw.weights]))
        weights = np.ones_like(sw.weights)
        sw.remove_all_data()
        sw.add_data(lno_data, weights=weights)
        self.assertTrue(np.all(w == 1 for w in sw.weights))
        sw.remove_all_data()
        self.assertRaises(AttributeError, sw.add_data, lno_data, weights='bla')

    def test_update_features(self):
        shape = self.sw.feature_matrix.shape
        self.cs.add_external_term(EwaldTerm)
        self.sw.update_features()
        self.assertEqual(shape[1] + 1, self.sw.feature_matrix.shape[1])

    # TODO write a better test. One that actually checks the structures
    #  expected to be removed are removed
    def test_filter_by_ewald(self):
        len_total = len(self.sw.items)
        self.sw.filter_by_ewald(1)
        len_filtered = len(self.sw.items)
        self.assertNotEqual(len_total, len_filtered)

    def test_weights_e_above_comp(self):
        self.sw._set_weights(self.sw.items, 'composition', temperature=1000)
        expected = np.array([0.85637358, 0.98816678, 1., 0.59209449, 1.,
                    0.92882071, 0.87907454, 0.94729315, 0.40490513, 0.82484222,
                    0.81578984, 1., 0.89615121, 0.92893004, 0.81650693,
                    0.6080223 , 0.94848913, 0.92135297, 0.92326977, 0.83995635,
                    1., 0.94663979, 1., 0.9414506 , 1.])
        self.assertTrue(np.allclose(expected, self.sw.weights))
        sw = StructureWrangler(self.cs, ['composition', {'temperature': 1000}])
        sw.add_data(lno_data)
        self.assertTrue(np.allclose(expected, sw.weights))

    def test_weights_e_above_hull(self):
        self.sw._set_weights(self.sw.items, 'hull', temperature=1000)
        expected = np.array([0.85637358, 0.98816678, 1., 0.56916328, 0.96127103,
           0.89284844, 0.84502889, 0.91060546, 0.40490513, 0.82484222,
           0.81578984, 1., 0.89615121, 0.92893004, 0.81650693,
           0.58819251, 0.91755548, 0.89130433, 0.89315862, 0.81256235,
           0.9673864 , 0.91576647, 1., 0.9414506 , 1])
        self.assertTrue(np.allclose(expected, self.sw.weights))
        sw = StructureWrangler(self.cs, ['hull', {'temperature': 1000}])
        sw.add_data(lno_data)
        self.assertTrue(np.allclose(expected, sw.weights))

    def test_msonable(self):
        d = self.sw.as_dict()
        sw = StructureWrangler.from_dict(d)
        self.assertTrue(np.array_equal(sw.properties, self.sw.properties))
        self.assertTrue(all([s1 == s2 for s1, s2 in zip(sw.structures, self.sw.structures)]))
        self.assertTrue(all([s1 == s2 for s1, s2 in zip(sw.refined_structures, self.sw.refined_structures)]))
        self.assertTrue(np.array_equal(sw.feature_matrix, self.sw.feature_matrix))
