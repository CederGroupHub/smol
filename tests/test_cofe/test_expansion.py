import unittest
import numpy as np
from smol.cofe import StructureWrangler, ClusterSubspace, ClusterExpansion
from smol.cofe.configspace import EwaldTerm
from smol.cofe.regression import constrain_dielectric
from smol.cofe.regression.estimator import CVXEstimator, BaseEstimator
from tests.data import lno_prim, lno_data

#TODO Change this to synthetic data fitting to synthetic data. Test on binary, ternary
#Add another test file to test vs pyabinitio binary, ternary structure


class TestClusterExpansion(unittest.TestCase):
    def setUp(self) -> None:
        cs = ClusterSubspace.from_radii(lno_prim, {2: 6, 3: 4.1, 4: 4},
                                        ltol=0.15, stol=0.2,
                                        angle_tol=5, supercell_size='O2-')
        self.sw = StructureWrangler(cs)
        for struct, energy in lno_data:
            self.sw.add_data(struct, {'energy': energy})
        ecis = np.linalg.lstsq(self.sw.feature_matrix,
                               self.sw.get_property_vector('energy', True),
                               rcond=None)[0]
        self.ce = ClusterExpansion(cs, ecis, self.sw.feature_matrix)

    def test_predict(self):
        preds = self.ce.predict(self.sw.structures)
        self.assertTrue(np.allclose(preds,
                                    self.sw.get_property_vector('energy'),
                                    atol=.07))
        preds = self.ce.predict(self.sw.structures, normalize=True)
        self.assertTrue(np.allclose(preds,
                                    self.sw.get_property_vector('energy', True),
                                    atol=.01))

    def test_convert_eci(self):
        cs = self.ce.cluster_subspace.copy()
        cs.change_site_bases('indicator', orthonormal=True)
        feature_matrix = np.array([cs.corr_from_structure(s)
                                   for s in self.sw.refined_structures])
        eci = self.ce.convert_eci('indicator',
                                  self.sw.refined_structures,
                                  self.sw.supercell_matrices,
                                  orthonormal=True)
        self.assertTrue(np.allclose(np.dot(self.sw.feature_matrix, self.ce.ecis),
                                    np.dot(feature_matrix, eci)))

    # TODO Finish writing this test
    def test_prune(self):
        pass

    def test_print(self):
        print(self.ce)

    def test_msonable(self):
        # ce.print_ecis()
        self.ce.metadata['somethingimportant'] = 75
        d = self.ce.as_dict()
        ce1 = ClusterExpansion.from_dict(d)
        self.assertTrue(np.array_equal(self.ce.ecis, ce1.ecis))
        self.assertIsInstance(self.ce.cluster_subspace, ClusterSubspace)
        self.assertEqual(ce1.metadata, self.ce.metadata)
