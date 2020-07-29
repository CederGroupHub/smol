import unittest
import json
import numpy as np
from pymatgen.transformations.standard_transformations import \
    OrderDisorderedStructureTransformation
from smol.moca import CEProcessor, EwaldProcessor, CompositeProcessor
from smol.cofe import ClusterExpansion, StructureWrangler, ClusterSubspace
from smol.cofe.extern import EwaldTerm
from tests.data import lno_prim, lno_data


# TODO test with all synthetic data binary/ternary/electrostatic
# TODO check that delta_corr gives same values for random sympos shuffles of same structure
# TODO check that delta_corr works for all bases and orthonormal combos

RTOL = 0.0  # relative tolerance to check property change functions
# absolute tolerance to check property change functions (eps is approx 2E-16)
ATOL = 3E2 * np.finfo(float).eps
EWALD_ATOL = 3 * ATOL  # ewald term has higher drift
DRIFT_TOL = np.finfo(float).eps  # tolerance of average drift

# Note that for delta_corr_ewald the forward and back check is not strictly
# equal but close to within the above tolerances. If energy drift is ever
# suspected start by looking at the delta_corr_ewald


class TestCEProcessor(unittest.TestCase):
    def setUp(self) -> None:
        cs = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1},
                                         ltol=0.15, stol=0.2,
                                         angle_tol=5,
                                         supercell_size='O2-',
                                         basis='sinusoid',
                                         orthonormal=True,
                                         use_concentration=True)
        self.sw = StructureWrangler(cs)
        for struct, energy in lno_data:
            self.sw.add_data(struct, {'energy': energy})
        coefs = np.linalg.lstsq(self.sw.feature_matrix,
                                self.sw.get_property_vector('energy', True),
                                rcond=None)[0]
        self.ce = ClusterExpansion(cs, coefs, self.sw.feature_matrix)
        scmatrix = np.array([[3, 0, 0],
                             [0, 2, 0],
                             [0, 0, 1]])
        self.pr = CEProcessor(cs, scmatrix, coefficients=coefs)

        # create a test structure
        test_struct = lno_prim.copy()
        test_struct.replace_species({"Li+": {"Li+": 2},
                                     "Ni3+": {"Ni3+": 2},
                                     "Ni4+": {"Ni4+": 0}})
        test_struct.make_supercell(scmatrix)
        ro = {"Li+": {"Li+": 0.5},
              "Ni3+": {"Ni3+": .35, "Ni4+": 1 - .35}}
        test_struct.replace_species(ro)
        order = OrderDisorderedStructureTransformation(algo=2)
        test_struct = order.apply_transformation(test_struct)
        self.test_struct = test_struct
        self.test_occu = cs.occupancy_from_structure(test_struct, scmatrix=scmatrix)
        self.enc_occu = self.pr.occupancy_from_structure(test_struct)
        self.sublattices = []
        for space in self.pr.unique_site_spaces:
            sites = np.array([i for i, b in enumerate(self.pr.allowed_species)
                              if b == list(space.keys())])
            self.sublattices.append({'spaces': list(range(len(space))),
                                     'sites': sites})

    def test_compute_property(self):
        self.assertTrue(np.isclose(self.ce.predict(self.test_struct),
                         self.pr.compute_property(self.enc_occu)))

    def test_compute_correlation(self):
        self.assertTrue(np.allclose(
            self.ce.cluster_subspace.corr_from_structure(self.test_struct, False),
            self.pr.compute_feature_vector(self.enc_occu)))

    def test_compute_property_change(self):
        occu = self.enc_occu.copy()
        for _ in range(50):
            sublatt = np.random.choice(self.sublattices)
            site = np.random.choice(sublatt['sites'])
            new_sp = np.random.choice(sublatt['spaces'])
            new_occu = occu.copy()
            new_occu[site] = new_sp
            prop_f = self.pr.compute_property(new_occu)
            prop_i = self.pr.compute_property(occu)
            dprop = self.pr.compute_property_change(occu, [(site, new_sp)])
            # Check with some tight tolerances.
            self.assertTrue(np.allclose(dprop, prop_f - prop_i,
                                        rtol=RTOL, atol=ATOL))
            # Test reverse matches forward
            old_sp = occu[site]
            rdprop = self.pr.compute_property_change(new_occu, [(site,
                                                                 old_sp)])
            self.assertEqual(dprop, -1*rdprop)

    def test_encode_property(self):
        enc_occu = self.pr.encode_occupancy(self.test_occu)
        self.assertTrue(all(o1 == o2 for o1, o2 in zip(self.enc_occu,
                                                       enc_occu)))

    def test_decode_property(self):
        occu = self.pr.decode_occupancy(self.enc_occu)
        self.assertTrue(all(o1 == o2 for o1, o2 in zip(self.test_occu,
                                                       occu)))

    def test_feature_update(self):
        occu = self.enc_occu.copy()
        for _ in range(50):
            sublatt = np.random.choice(self.sublattices)
            site = np.random.choice(sublatt['sites'])
            new_sp = np.random.choice(sublatt['spaces'])
            new_occu = occu.copy()
            new_occu[site] = new_sp
            # Test forward
            dcorr = self.pr.compute_feature_vector_change(occu, [(site, new_sp)])
            corr_f = self.pr.compute_feature_vector(new_occu)
            corr_i = self.pr.compute_feature_vector(occu)

            self.assertTrue(np.allclose(dcorr, corr_f - corr_i,
                                        rtol=RTOL, atol=ATOL))
            # Test reverse matches forward
            old_sp = occu[site]
            rdcorr = self.pr.compute_feature_vector_change(new_occu, [(site, old_sp)])
            self.assertTrue(np.array_equal(dcorr, -1*rdcorr))

    def test_feature_update_indicator(self):
        cs = self.ce.cluster_subspace.copy()
        cs.change_site_bases('indicator')
        sw = StructureWrangler(cs)
        for struct, energy, matrix in zip(self.sw.structures,
                                  self.sw.get_property_vector('energy'),
                                  self.sw.supercell_matrices):
            sw.add_data(struct, {'energy': energy}, supercell_matrix=matrix)

        coefs = np.linalg.lstsq(sw.feature_matrix,
                               sw.get_property_vector('energy', True),
                               rcond=None)[0]
        pr = CEProcessor(cs, self.pr.supercell_matrix, coefs,
                         optimize_indicator=True)
        occu = self.enc_occu.copy()
        for _ in range(50):
            sublatt = np.random.choice(self.sublattices)
            site = np.random.choice(sublatt['sites'])
            new_sp = np.random.choice(sublatt['spaces'])
            new_occu = occu.copy()
            new_occu[site] = new_sp
            # Test forward
            dcorr = pr.compute_feature_vector_change(occu, [(site, new_sp)])
            corr_f = pr.compute_feature_vector(new_occu)
            corr_i = pr.compute_feature_vector(occu)
            self.assertTrue(np.allclose(dcorr, corr_f - corr_i,
                                        rtol=RTOL, atol=ATOL))
            # Test reverse matches forward
            old_sp = occu[site]
            rdcorr = pr.compute_feature_vector_change(new_occu, [(site, old_sp)])
            self.assertTrue(np.allclose(dcorr, -1 * rdcorr,
                                        rtol=RTOL, atol=ATOL))

    def test_get_average_drift(self):
        forward, reverse = self.pr.get_average_drift()
        self.assertTrue(forward <= DRIFT_TOL and reverse <= DRIFT_TOL)

    def test_structure_from_occupancy(self):
        # The structures do pass as equal by direct == comparison, but as long
        # as the correlation vectors and predicted energy are the same we
        # should be all good.
        test_struct = self.pr.structure_from_occupancy(self.enc_occu)
        self.assertTrue(np.allclose(self.ce.cluster_subspace.corr_from_structure(test_struct),
                                    self.ce.cluster_subspace.corr_from_structure(self.test_struct)))
        self.assertEqual(self.ce.predict(test_struct, normalize=True),
                         self.ce.predict(self.test_struct, normalize=True))

    def test_constructor_except(self):
        coefs = np.random.random(self.pr.n_orbit_functions + 1)
        self.assertRaises(ValueError, CEProcessor, self.pr.cluster_subspace,
                          self.pr.supercell_matrix, coefs)

    def test_msonable(self):
        d = self.pr.as_dict()
        pr = CEProcessor.from_dict(d)
        self.assertEqual(pr.as_dict(), d)
        self.assertEqual(self.pr.compute_property(self.enc_occu),
                         pr.compute_property(self.enc_occu))
        j = json.dumps(d)
        _ = json.loads(j)


class TestEwaldProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cs = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1},
                                            basis='sinusoid',
                                            orthonormal=True,
                                            use_concentration=True,
                                            ltol=0.15, stol=0.2,
                                            angle_tol=5,
                                            supercell_size='O2-')
        cls.sw = StructureWrangler(cls.cs)
        for struct, energy in lno_data:
            cls.sw.add_data(struct, {'energy': energy})

        # create a test structure
        test_struct = lno_prim.copy()
        test_struct.replace_species({"Li+": {"Li+": 2},
                                     "Ni3+": {"Ni3+": 2},
                                     "Ni4+": {"Ni4+": 0}})
        cls.scmatrix = np.array([[3, 0, 0],
                                 [0, 2, 0],
                                 [0, 0, 1]])
        test_struct.make_supercell(cls.scmatrix)
        ro = {"Li+": {"Li+": 0.5},
              "Ni3+": {"Ni3+": .35, "Ni4+": 1 - .35}}
        test_struct.replace_species(ro)
        order = OrderDisorderedStructureTransformation(algo=2)
        cls.test_struct = order.apply_transformation(test_struct)
        cls.test_occu = cls.cs.occupancy_from_structure(cls.test_struct,
                                                        scmatrix=cls.scmatrix)

        # create sublattices
        cls.sublattices = []
        pr = EwaldProcessor(cls.cs, cls.scmatrix, EwaldTerm())
        for doms in pr.unique_site_spaces:
            sites = np.array([i for i, b in enumerate(pr.allowed_species)
                              if b == list(doms.keys())])
            cls.sublattices.append({'spaces': list(range(len(doms))),
                                    'sites': sites})

    def setUp(self):
        self.setUpProcessor()

    def tearDown(self) -> None:
        self.cs._external_terms = []

    def setUpProcessor(self, term='total') -> None:
        ewald_term = EwaldTerm(use_term=term)
        self.cs.add_external_term(ewald_term)
        self.sw.update_features()
        self.pr = EwaldProcessor(self.cs, self.scmatrix, ewald_term)
        self.enc_occu = self.pr.occupancy_from_structure(self.test_struct)

    def test_compute_property_change(self):
        occu = self.enc_occu.copy()
        for _ in range(50):
            sublatt = np.random.choice(self.sublattices)
            site = np.random.choice(sublatt['sites'])
            new_sp = np.random.choice(sublatt['spaces'])
            new_occu = occu.copy()
            new_occu[site] = new_sp
            prop_f = self.pr.compute_property(new_occu)
            prop_i = self.pr.compute_property(occu)
            dprop = self.pr.compute_property_change(occu, [(site, new_sp)])
            self.assertTrue(np.isclose(dprop, prop_f - prop_i,
                                       rtol=RTOL, atol=EWALD_ATOL))
            # Test reverse matches forward
            old_sp = occu[site]
            rdprop = self.pr.compute_property_change(new_occu, [(site,
                                                                 old_sp)])
            self.assertTrue(np.isclose(dprop, -1.0 * rdprop,
                                       rtol=RTOL, atol=EWALD_ATOL))

    def test_get_average_drift(self):
        forward, reverse = self.pr.get_average_drift()
        self.assertTrue(forward <= DRIFT_TOL and reverse <= DRIFT_TOL)

    def test_msonable(self):
        d = self.pr.as_dict()
        pr = EwaldProcessor.from_dict(d)
        self.assertEqual(self.pr.compute_property(self.enc_occu),
                         pr.compute_property(self.enc_occu))
        self.assertTrue(np.array_equal(self.pr.ewald_matrix,
                                       pr.ewald_matrix))
        j = json.dumps(d)
        _ = json.loads(j)


class TestEwaldProcessorReal(TestEwaldProcessor):
    def setUp(self):
        self.setUpProcessor(term='real')

    def tearDown(self) -> None:
        self.cs._external_terms = []


class TestEwaldProcessorRecip(TestEwaldProcessor):
    def setUp(self):
        self.setUpProcessor(term='reciprocal')

    def tearDown(self) -> None:
        self.cs._external_terms = []


class TestEwaldProcessorPoint(TestEwaldProcessor):
    def setUp(self):
        self.setUpProcessor(term='point')

    def tearDown(self) -> None:
        self.cs._external_terms = []


class TestCompositeProcessor(unittest.TestCase):
    def setUp(self) -> None:
        cs = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1},
                                        ltol=0.15, stol=0.2,
                                        angle_tol=5,
                                        supercell_size='O2-',
                                        basis='sinusoid',
                                        orthonormal=True,
                                        use_concentration=True)
        ewald_term = EwaldTerm()
        cs.add_external_term(ewald_term)
        self.sw = StructureWrangler(cs)
        for struct, energy in lno_data:
            self.sw.add_data(struct, {'energy': energy})
        coefs = np.linalg.lstsq(self.sw.feature_matrix,
                                self.sw.get_property_vector('energy', True),
                                rcond=None)[0]
        self.ce = ClusterExpansion(cs, coefs, self.sw.feature_matrix)
        scmatrix = np.array([[3, 0, 0],
                             [0, 2, 0],
                             [0, 0, 1]])
        self.pr = CompositeProcessor(cs, scmatrix)

        self.pr.add_processor(CEProcessor, coefficients=coefs[:-1])
        self.pr.add_processor(EwaldProcessor, coefficient=coefs[-1],
                              ewald_term=ewald_term)

        # create a test structure
        test_struct = lno_prim.copy()
        test_struct.replace_species({"Li+": {"Li+": 2},
                                     "Ni3+": {"Ni3+": 2},
                                     "Ni4+": {"Ni4+": 0}})
        test_struct.make_supercell(scmatrix)
        ro = {"Li+": {"Li+": 0.5},
              "Ni3+": {"Ni3+": .35, "Ni4+": 1 - .35}}
        test_struct.replace_species(ro)
        order = OrderDisorderedStructureTransformation(algo=2)
        test_struct = order.apply_transformation(test_struct)
        self.test_struct = test_struct
        self.test_occu = cs.occupancy_from_structure(test_struct, scmatrix=scmatrix)
        self.enc_occu = self.pr.occupancy_from_structure(test_struct)
        self.sublattices = []
        for space in self.pr.unique_site_spaces:
            sites = np.array([i for i, b in enumerate(self.pr.allowed_species)
                              if b == list(space.keys())])
            self.sublattices.append({'spaces': list(range(len(space))),
                                     'sites': sites})

    def test_compute_property(self):
        self.assertTrue(np.isclose(self.ce.predict(self.test_struct),
                                   self.pr.compute_property(self.enc_occu)))

    def test_compute_property_change(self):
        occu = self.enc_occu.copy()
        for _ in range(50):
            sublatt = np.random.choice(self.sublattices)
            site = np.random.choice(sublatt['sites'])
            new_sp = np.random.choice(sublatt['spaces'])
            new_occu = occu.copy()
            new_occu[site] = new_sp
            prop_f = self.pr.compute_property(new_occu)
            prop_i = self.pr.compute_property(occu)
            dprop = self.pr.compute_property_change(occu, [(site, new_sp)])
            # Check with some tight tolerances.
            self.assertTrue(np.allclose(dprop, prop_f - prop_i,
                                        rtol=RTOL, atol=ATOL))
            # Test reverse matches forward
            old_sp = occu[site]
            rdprop = self.pr.compute_property_change(new_occu, [(site,
                                                                 old_sp)])
            self.assertEqual(dprop, -1 * rdprop)

    def test_feature_update(self):
        occu = self.enc_occu.copy()
        for _ in range(50):
            sublatt = np.random.choice(self.sublattices)
            site = np.random.choice(sublatt['sites'])
            new_sp = np.random.choice(sublatt['spaces'])
            new_occu = occu.copy()
            new_occu[site] = new_sp
            # Test forward
            dcorr = self.pr.compute_feature_vector_change(occu, [(site, new_sp)])
            corr_f = self.pr.compute_feature_vector(new_occu)
            corr_i = self.pr.compute_feature_vector(occu)

            self.assertTrue(np.allclose(dcorr, corr_f - corr_i,
                                        rtol=RTOL, atol=EWALD_ATOL))
            # Test reverse matches forward
            old_sp = occu[site]
            rdcorr = self.pr.compute_feature_vector_change(new_occu, [(site, old_sp)])
            self.assertTrue(np.array_equal(dcorr, -1*rdcorr))

    def test_get_average_drift(self):
        forward, reverse = self.pr.get_average_drift()
        self.assertTrue(forward <= DRIFT_TOL and reverse <= DRIFT_TOL)

    def test_msonable(self):
        d = self.pr.as_dict()
        pr = CompositeProcessor.from_dict(d)
        print(self.pr.__processors)
        self.assertEqual(self.pr.compute_property(self.enc_occu),
                         pr.compute_property(self.enc_occu))
        j = json.dumps(d)
        _ = json.loads(j)
