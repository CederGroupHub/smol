import unittest
import warnings
import numpy as np
from ..data import lno_prim, lno_data
from smol.cofe import StructureWrangler, ClusterSubspace

class TestStructureWrangler(unittest.TestCase):
    def setUp(self) -> None:
        self.cs = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1}, ltol=0.15, stol=0.2,
                                             angle_tol=5, supercell_size='O2-')
        self.sw = StructureWrangler(self.cs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  #  Supress warnings for structures not mapped to prim
            self.sw.add_data(lno_data)

    # TODO write a better test. One that actually checks the structures expected to be removed are removed
    def test_filter_by_ewald(self):
        len_total = len(self.sw.items)
        self.sw.filter_by_ewald(1)
        len_filtered = len(self.sw.items)
        self.assertNotEqual(len_total, len_filtered)

    def test_weights_e_above_comp(self):
        self.sw.set_weights('composition', temperature=1000)
        expected = np.array([0.85636844, 0.98816632, 1., 0.59208249, 1.,
                             0.92881806, 0.87907016, 0.94729116, 0.40489097, 0.82483607,
                             0.81578342, 1., 0.89614741, 0.9289274, 0.81650053,
                             0.6080106, 0.94848719, 0.92135005, 0.92326692, 0.83995068,
                             1., 0.94663779, 1., 0.9414484, 1.])
        self.assertTrue(np.allclose(expected, self.sw.weights))

    def test_weights_e_above_hull(self):
        self.sw.set_weights('hull', temperature=1000)
        expected = np.array([0.85636844, 0.98816632, 1., 0.56915087, 0.96126956,
                             0.89284453, 0.84502339, 0.91060216, 0.40489097, 0.82483607,
                             0.81578342, 1., 0.89614741, 0.9289274, 0.81650053,
                             0.58818044, 0.91755243, 0.89130037, 0.89315472, 0.81255583,
                             0.96738516, 0.91576335, 1., 0.9414484, 1.])
        self.assertTrue(np.allclose(expected, self.sw.weights))

    def test_msonable(self):
        d = self.sw.as_dict()
        sw = StructureWrangler.from_dict(d)
        self.assertTrue(all(sw.properties == self.sw.properties))
        self.assertTrue(sw.structures == self.sw.structures)
        self.assertTrue(np.array_equal(sw.feature_matrix, self.sw.feature_matrix))
