import unittest
import json
import numpy as np
from smol.cofe import StructureWrangler, ClusterSubspace, ClusterExpansion
from tests.data import synthetic_CE_binary


# TODO add tests with synthetic ternary dataset
# TODO add tests with synthetic dataset that includes ewald-like energy


class TestClusterExpansionBinary(unittest.TestCase):
    def setUp(self) -> None:
        cs = ClusterSubspace.from_dict(synthetic_CE_binary['cluster_subspace'])
        self.sw = StructureWrangler(cs)
        data = synthetic_CE_binary['data']
        train_ids = np.random.choice(range(len(data)), size=len(data)//5,
                                          replace=False)
        test_ids = np.array(list(set(range(len(data))) - set(train_ids)))
        self.test_structs = [data[i][0] for i in test_ids]
        self.test_energies = np.array([data[i][1] for i in test_ids])
        for i in train_ids:
            struct, energy = data[i]
            self.sw.add_data(struct, {'energy': energy})
        ecis = np.linalg.lstsq(self.sw.feature_matrix,
                               self.sw.get_property_vector('energy', True),
                               rcond=None)[0]
        self.ce = ClusterExpansion(cs, ecis, self.sw.feature_matrix)

    def test_predict_train(self):
        preds = self.ce.predict(self.sw.structures)
        self.assertTrue(np.allclose(preds,
                                    self.sw.get_property_vector('energy')))
        preds = self.ce.predict(self.sw.structures, normalize=True)
        self.assertTrue(np.allclose(preds,
                                    self.sw.get_property_vector('energy', True)))

    def test_predict_test(self):
        preds = self.ce.predict(self.test_structs)
        self.assertTrue(np.allclose(preds, self.test_energies))

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

    def test_prune(self):
        cs = ClusterSubspace.from_dict(synthetic_CE_binary['cluster_subspace'])
        ce = ClusterExpansion(cs, self.ce.ecis.copy(), self.ce.feature_matrix)
        thresh = 8E-3
        ce.prune(threshold=thresh)
        ids = [i for i, eci in enumerate(self.ce.ecis) if abs(eci) >= thresh]
        new_ecis = self.ce.ecis[abs(self.ce.ecis) >= thresh]
        self.assertTrue(len(ce.ecis) == len(new_ecis))
        self.assertTrue(ce.cluster_subspace.n_orbits, len(new_ecis))
        # check new predictions
        self.assertTrue(np.allclose(ce.predict(self.sw.structures, normalize=True),
                                    np.dot(self.sw.feature_matrix[:, ids],
                                           new_ecis)))
        # check the updated feature matrix is correct
        self.assertTrue(np.equal(ce.feature_matrix,
                               self.sw.feature_matrix[:, ids]).all())
        # check that recomputing features produces whats expected
        new_feature_matrix = np.array([cs.corr_from_structure(s)
                                       for s in self.sw.structures])
        self.assertTrue(np.equal(ce.feature_matrix, new_feature_matrix).all())

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
        j = json.dumps(d)
        json.loads(j)