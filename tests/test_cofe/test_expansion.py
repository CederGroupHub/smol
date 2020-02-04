import unittest
import numpy as np
from smol.cofe import StructureWrangler, ClusterSubspace, ClusterExpansion
from smol.cofe.configspace import EwaldTerm
from smol.cofe.regression import constrain_dielectric
from smol.cofe.regression.estimator import CVXEstimator, BaseEstimator
from tests.data import lno_prim, lno_data

# Probably should also add fitting to synthetic data.
# i.e. using from pymatgen.transformations.advanced_transformations import
# EnumerateStructureTransformation
class TestClusterExpansion(unittest.TestCase):
    def setUp(self) -> None:
        self.cs = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1},
                                             ltol=0.15, stol=0.2,
                                             angle_tol=5, supercell_size='O2-')
        self.sw = StructureWrangler(self.cs)
        self.sw.add_data(lno_data)

    def test_from_radii(self):
        # Test with numpy because CVXEstimator gives slightly  different ECIs
        # for same parameters
        estimator = BaseEstimator()
        estimator._solve = lambda X, y: np.linalg.lstsq(X, y, rcond=None)[0]
        weights = ['hull', {'temperature': 50}]
        ce = ClusterExpansion.from_radii(lno_prim, {2: 5, 3: 4.1},
                                         ltol=0.15, stol=0.2,
                                         angle_tol=5, supercell_size='O2-',
                                         data=lno_data, estimator=estimator,
                                         weights=weights)
        ce.fit()
        ce1 = ClusterExpansion.from_radii(lno_prim, {2: 5, 3: 4.1},
                                         ltol=0.15, stol=0.2,
                                         angle_tol=5, supercell_size='O2-',
                                         estimator=estimator, data=lno_data,
                                         weights=weights)

        ce1.fit()

        ce2 = ClusterExpansion.from_radii(lno_prim, {2: 5, 3: 4.1},
                                         ltol=0.15, stol=0.2,
                                         angle_tol=5, supercell_size='O2-',
                                         ecis=ce.ecis)
        test_structs = self.sw.structures[:10]

        self.assertTrue(np.allclose(ce.predict(test_structs),
                                    ce1.predict(test_structs)))
        self.assertTrue(np.array_equal(ce.predict(test_structs),
                                       ce2.predict(test_structs)))

        ce3 = ClusterExpansion.from_radii(lno_prim, {2: 5, 3: 4.1},
                                          ltol=0.15, stol=0.2,
                                          angle_tol=5, supercell_size='O2-',
                                          external_terms=[EwaldTerm,
                                                          {'eta': None}],
                                          data=lno_data,
                                          weights=weights)
        ce3.fit()
        self.assertEqual(len(ce3.ecis[:-1]), len(ce.ecis))
        self.assertEqual(len(ce3.predict(test_structs)), len(test_structs))

    def test_cvxestimator(self):
        ce = ClusterExpansion(self.cs, self.sw.refined_structures,
                              self.sw.normalized_properties,
                              estimator=CVXEstimator())
        self.assertRaises(AttributeError, ce.predict, self.sw.structures[:10])
        ce.fit(mu=5)
        self.assertIsNotNone(ce.ecis)
        self.assertEqual(len(ce.ecis), self.cs.n_bit_orderings)
        self.sw._set_weights(self.sw.items, 'hull')
        ce.fit()

    def test_sklearn(self):
        try:
            from sklearn.linear_model import Ridge, LassoCV
        except ImportError:
            return

        ce = ClusterExpansion(self.cs, self.sw.refined_structures,
                              self.sw.normalized_properties, estimator=Ridge())
        self.assertRaises(AttributeError, ce.predict, self.sw.structures[:10])
        ce.fit()
        self.assertIsNotNone(ce.ecis)
        self.assertEqual(len(ce.ecis), self.cs.n_bit_orderings)
        ce.estimator = LassoCV()
        ce.fit()
        self.assertIsNotNone(ce.ecis)
        self.assertEqual(len(ce.ecis), self.cs.n_bit_orderings)

    def test_numpy(self):
        estimator = BaseEstimator()
        estimator._solve = lambda X, y: np.linalg.lstsq(X, y, rcond=None)[0]
        ce = ClusterExpansion(self.cs, self.sw.refined_structures,
                              self.sw.normalized_properties,
                              estimator=estimator)
        self.assertRaises(AttributeError, ce.predict, self.sw.structures[:10])
        ce.fit()
        self.assertIsNotNone(ce.ecis)
        self.assertEqual(len(ce.ecis), self.cs.n_bit_orderings)

    def test_no_estimator(self):
        ecis = np.ones((self.cs.n_bit_orderings))
        ce = ClusterExpansion(self.cs, self.sw.refined_structures,
                              self.sw.normalized_properties, ecis=ecis)
        structs = self.sw.structures[:10]
        p = np.array([sum(self.cs.corr_from_structure(s)) for s in structs])
        self.assertTrue(np.allclose(ce.predict(structs, normalized=True), p))

    def test_convert_eci(self):
        estimator = BaseEstimator()
        estimator._solve = lambda X, y: np.linalg.lstsq(X, y, rcond=None)[0]
        ce = ClusterExpansion(self.cs, self.sw.refined_structures[:15],
                              self.sw.normalized_properties[:15],
                              estimator=estimator)
        ce.fit()
        cs = self.cs.copy()
        cs.change_site_bases('indicator', orthonormal=True)
        feature_matrix = np.array([cs.corr_from_structure(s)
                                   for s in self.sw.refined_structures[15:]])
        eci = ce.convert_eci('indicator', orthonormal=True)

        self.assertTrue(np.allclose(ce.predict(self.sw.refined_structures[15:],
                                               normalized=True),
                                    np.dot(feature_matrix, eci)))

    def test_constrain_dielectric(self):
        self.cs.add_external_term(EwaldTerm)
        ce = ClusterExpansion(self.cs, self.sw.refined_structures,
                              self.sw.normalized_properties,
                              estimator=CVXEstimator())
        ce.fit()
        constrain_dielectric(ce, 5)
        self.assertEqual(ce.ecis[-1], 1/5)

    def test_msonable(self):
        ecis = np.ones((self.cs.n_bit_orderings))
        ce = ClusterExpansion(self.cs, self.sw.refined_structures,
                              self.sw.normalized_properties, ecis=ecis)
        # ce.print_ecis()
        d = ce.as_dict()
        ce1 = ClusterExpansion.from_dict(d)
        self.assertTrue(np.array_equal(ce.ecis, ce1.ecis))
        self.assertIsInstance(ce.subspace, ClusterSubspace)
        self.assertIsInstance(ce.estimator, BaseEstimator)