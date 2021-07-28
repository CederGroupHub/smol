import unittest
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from smol.cofe import StructureWrangler, ClusterSubspace, ClusterExpansion, \
    RegressionData
from smol.cofe.extern import EwaldTerm
from tests.data import synthetic_CE_binary, synthetic_CEewald_binary
from tests.utils import assert_msonable


def test_regression_data(cluster_subspace):
    reg = LinearRegression(fit_intercept=False)
    n = np.random.randint(10, 100)
    feat_matrix = np.random.random((n, len(cluster_subspace)))
    prop_vec = np.random.random(n)
    reg_data = RegressionData.from_sklearn(reg, feature_matrix=feat_matrix,
                                           property_vector=prop_vec)
    coeffs = np.random.random(len(cluster_subspace))
    expansion = ClusterExpansion(cluster_subspace, coeffs, reg_data)
    assert reg_data.class_name == reg.__class__.__name__
    assert reg_data.module == reg.__module__
    assert reg_data.parameters == reg.get_params()
    assert_msonable(expansion)


# TODO add tests with synthetic ternary dataset


class TestClusterExpansionBinary(unittest.TestCase):
    """
    Test cluster expansion on a synthetic CE binary dataset
    """

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
        coefs = np.linalg.lstsq(self.sw.feature_matrix,
                                self.sw.get_property_vector('energy', True),
                                rcond=None)[0]
        self.ce = ClusterExpansion(cs, coefs, self.sw.feature_matrix)

    def test_predict_train(self):
        preds = [self.ce.predict(s) for s in self.sw.structures]
        self.assertTrue(np.allclose(preds,
                                    self.sw.get_property_vector('energy', False)))
        preds = [self.ce.predict(s, True) for s in self.sw.structures]
        self.assertTrue(np.allclose(preds,
                                    self.sw.get_property_vector('energy')))

    def test_predict_test(self):
        preds = [self.ce.predict(s) for s in self.test_structs]
        self.assertTrue(np.allclose(preds, self.test_energies))

    def test_convert_eci(self):
        cs = self.ce.cluster_subspace.copy()
        cs.change_site_bases('indicator', orthonormal=True)
        feature_matrix = np.array([cs.corr_from_structure(s)
                                   for s in self.sw.refined_structures])
        eci = self.ce.convert_coefs('indicator',
                                    self.sw.refined_structures,
                                    self.sw.supercell_matrices,
                                    orthonormal=True)
        self.assertTrue(np.allclose(np.dot(self.sw.feature_matrix, self.ce.coefs),
                                    np.dot(feature_matrix, eci)))
        ce = ClusterExpansion(self.ce.cluster_subspace, self.ce.coefs)
        self.assertRaises(AttributeError, ce.convert_coefs, 'indicator',
                          self.sw.refined_structures,
                          self.sw.supercell_matrices,)

    def test_prune(self):
        cs = ClusterSubspace.from_dict(synthetic_CE_binary['cluster_subspace'])
        ce = ClusterExpansion(cs, self.ce.coefs.copy(), self.ce._feat_matrix)
        thresh = 8E-3
        ce.prune(threshold=thresh)
        ids = [i for i, coef in enumerate(self.ce.coefs) if abs(coef) >= thresh]
        new_coefs = self.ce.coefs[abs(self.ce.coefs) >= thresh]
        new_eci = self.ce.eci[ids]
        self.assertEqual(len(ce.coefs), len(new_coefs))
        self.assertTrue(np.array_equal(new_eci, ce.eci))
        self.assertTrue(np.array_equal(new_coefs, ce.coefs))
        self.assertEqual(ce.cluster_subspace.num_orbits, len(new_coefs))
        self.assertEqual(len(ce.eci_orbit_ids), len(new_coefs))
        # check the updated feature matrix is correct
        self.assertTrue(np.array_equal(ce._feat_matrix,
                        self.sw.feature_matrix[:, ids]))
        # check that recomputing features produces whats expected
        new_feature_matrix = np.array([cs.corr_from_structure(s)
                                       for s in self.sw.structures])
        self.assertTrue(np.equal(ce._feat_matrix, new_feature_matrix).all())
        # check new predictions
        preds = [ce.predict(s, normalize=True) for s in self.sw.structures]
        self.assertTrue(np.allclose(preds,
                                    np.dot(self.sw.feature_matrix[:, ids],
                                           new_coefs)))

    def test_print(self):
        _ = str(self.ce)

    def test_msonable(self):
        self.ce.metadata['somethingimportant'] = 75
        d = self.ce.as_dict()
        ce1 = ClusterExpansion.from_dict(d)
        self.assertTrue(np.array_equal(self.ce.coefs, ce1.coefs))
        self.assertIsInstance(self.ce.cluster_subspace, ClusterSubspace)
        self.assertEqual(ce1.metadata, self.ce.metadata)
        j = json.dumps(d)
        json.loads(j)


class TestClusterExpansionEwaldBinary(unittest.TestCase):
    """
    Test cluster expansion on a synthetic CE + Ewald binary dataset
    """

    def setUp(self) -> None:
        self.dataset = synthetic_CEewald_binary
        self.cs = ClusterSubspace.from_dict(self.dataset['cluster_subspace'])

        num_structs = len(self.dataset['data'])
        self.train_ids = np.random.choice(range(num_structs),
                                          size=num_structs//5,
                                          replace=False)
        self.test_ids = np.array(list(set(range(num_structs)) - set(self.train_ids)))

    def test_ewald_only(self):
        data = self.dataset['ewald_data']
        cs = ClusterSubspace.from_cutoffs(self.cs.structure,
                                          cutoffs={2: 0},
                                          basis='sinusoid')
        self.assertEqual(len(cs.orbits), 1)
        cs.add_external_term(EwaldTerm())
        ecis = self._test_predictions(cs, data)
        self.assertAlmostEqual(ecis[-1], 1, places=8)

    def test_ce_ewald(self):
        data = self.dataset['data']
        cs = ClusterSubspace.from_dict(self.dataset['cluster_subspace'])
        for term in EwaldTerm.ewald_term_options:
            cs.add_external_term(EwaldTerm(use_term=term))
            _ = self._test_predictions(cs, data)
            cs._external_terms = []

    def _test_predictions(self, cs, data):
        sw = StructureWrangler(cs)
        for i in self.train_ids:
            struct, energy = data[i]
            sw.add_data(struct, {'energy': energy})
        ecis = np.linalg.lstsq(sw.feature_matrix,
                               sw.get_property_vector('energy', True),
                               rcond=None)[0]
        ce = ClusterExpansion(cs, ecis, sw.feature_matrix)
        test_structs = [data[i][0] for i in self.test_ids]
        test_energies = np.array([data[i][1] for i in self.test_ids])
        preds = [ce.predict(s) for s in sw.structures]
        self.assertTrue(np.allclose(preds,
                                    sw.get_property_vector('energy', False)))
        preds = [ce.predict(s) for s in test_structs]
        self.assertTrue(np.allclose(preds, test_energies))

        return ecis
