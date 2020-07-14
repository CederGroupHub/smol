import unittest
import json
import numpy as np
from pymatgen.transformations.standard_transformations import \
    OrderDisorderedStructureTransformation
from smol.moca import CEProcessor, EwaldCEProcessor
from smol.cofe import ClusterExpansion, StructureWrangler, ClusterSubspace
from smol.cofe.extern import EwaldTerm
from tests.data import lno_prim, lno_data


# TODO test with all synthetic data binary/ternary/electrostatic
# TODO check that delta_corr gives same values for random sympos shuffles of same structure
# TODO check that delta_corr works for all bases and orthonormal combos

rtol = 0.0  # relative tolerance to check property change functions
atol = 1E-12  # absolute tolerance to check property change functions

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
        ecis = np.linalg.lstsq(self.sw.feature_matrix,
                               self.sw.get_property_vector('energy', True),
                               rcond=None)[0]
        self.ce = ClusterExpansion(cs, ecis, self.sw.feature_matrix)
        scmatrix = np.array([[3, 0, 0],
                             [0, 2, 0],
                             [0, 0, 1]])
        self.pr = CEProcessor(self.ce, scmatrix)

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
        self.test_occu = self.ce.cluster_subspace.occupancy_from_structure(test_struct,
                                                                           scmatrix=scmatrix)
        self.enc_occu = self.pr.occupancy_from_structure(test_struct)
        self.sublattices = []
        for doms in self.pr.unique_site_spaces:
            sites = np.array([i for i, b in enumerate(self.pr.allowed_species)
                              if b == list(doms.keys())])
            self.sublattices.append({'doms': list(range(len(doms))),
                                     'sites': sites})
        print(self.sublattices)

    def test_compute_property(self):
        self.assertTrue(np.isclose(self.ce.predict(self.test_struct),
                         self.pr.compute_property(self.enc_occu)))

    def test_compute_correlation(self):
        self.assertTrue(np.allclose(self.ce.cluster_subspace.corr_from_structure(self.test_struct),
                                    self.pr.compute_correlation(self.enc_occu)))

    def test_compute_property_change(self):
        occu = self.enc_occu.copy()
        for _ in range(50):
            sublatt = np.random.choice(self.sublattices)
            site = np.random.choice(sublatt['sites'])
            new_sp = np.random.choice(sublatt['doms'])
            new_occu = occu.copy()
            new_occu[site] = new_sp
            prop_f = self.pr.compute_property(new_occu)
            prop_i = self.pr.compute_property(occu)
            dprop = self.pr.compute_property_change(occu, [(site, new_sp)])
            # Check with some tight tolerances.
            self.assertTrue(np.allclose(dprop, prop_f - prop_i,
                                        rtol=rtol, atol=atol))
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

    def test_delta_corr(self):
        occu = self.enc_occu.copy()
        for _ in range(50):
            sublatt = np.random.choice(self.sublattices)
            site = np.random.choice(sublatt['sites'])
            new_sp = np.random.choice(sublatt['doms'])
            new_occu = occu.copy()
            new_occu[site] = new_sp
            # Test forward
            dcorr = self.pr._delta_corr([(site, new_sp)], occu)
            corr_f = self.pr.compute_correlation(new_occu)
            corr_i = self.pr.compute_correlation(occu)

            self.assertTrue(np.allclose(dcorr, corr_f - corr_i,
                                        rtol=rtol, atol=atol))
            # Test reverse matches forward
            old_sp = occu[site]
            rdcorr = self.pr._delta_corr([(site, old_sp)], new_occu)
            self.assertTrue(np.array_equal(dcorr, -1*rdcorr))

    def test_delta_corr_indicator(self):
        cs = self.ce.cluster_subspace.copy()
        cs.change_site_bases('indicator')
        sw = StructureWrangler(cs)
        for struct, energy, matrix in zip(self.sw.structures,
                                  self.sw.get_property_vector('energy'),
                                  self.sw.supercell_matrices):
            sw.add_data(struct, {'energy': energy}, supercell_matrix=matrix)

        ecis = np.linalg.lstsq(sw.feature_matrix,
                               sw.get_property_vector('energy', True),
                               rcond=None)[0]
        ce = ClusterExpansion(cs, ecis, sw.feature_matrix)
        pr = CEProcessor(ce, self.pr.supercell_matrix, optimize_indicator=True)
        occu = self.enc_occu.copy()
        for _ in range(50):
            sublatt = np.random.choice(self.sublattices)
            site = np.random.choice(sublatt['sites'])
            new_sp = np.random.choice(sublatt['doms'])
            new_occu = occu.copy()
            new_occu[site] = new_sp
            # Test forward
            dcorr = pr._delta_corr([(site, new_sp)], occu)
            corr_f = pr.compute_correlation(new_occu)
            corr_i = pr.compute_correlation(occu)
            self.assertTrue(np.allclose(dcorr, corr_f - corr_i,
                                        rtol=rtol, atol=atol))
            # Test reverse matches forward
            old_sp = occu[site]
            rdcorr = pr._delta_corr([(site, old_sp)], new_occu)
            self.assertTrue(np.allclose(dcorr, -1*rdcorr,
                                        rtol=rtol, atol=atol))

    def test_structure_from_occupancy(self):
        # The structures do pass as equal by direct == comparison, but as long
        # as the correlation vectors and predicted energy are the same we
        # should be all good.
        test_struct = self.pr.structure_from_occupancy(self.enc_occu)
        self.assertTrue(np.allclose(self.ce.cluster_subspace.corr_from_structure(test_struct),
                                    self.ce.cluster_subspace.corr_from_structure(self.test_struct)))
        self.assertEqual(self.ce.predict(test_struct, normalize=True),
                         self.ce.predict(self.test_struct, normalize=True))

    def test_msonable(self):
        d = self.pr.as_dict()
        pr = CEProcessor.from_dict(d)
        self.assertEqual(pr.as_dict(), d)
        self.assertEqual(self.pr.compute_property(self.enc_occu),
                         pr.compute_property(self.enc_occu))
        j = json.dumps(d)
        _ = json.loads(j)


class TestEwaldCEProcessor(unittest.TestCase):
    def setUp(self) -> None:
        cs = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1},
                                         ltol=0.15, stol=0.2,
                                         angle_tol=5,
                                         supercell_size='O2-',
                                         basis='sinusoid',
                                         orthonormal=True,
                                         use_concentration=True)
        cs.add_external_term(EwaldTerm())
        self.sw = StructureWrangler(cs)
        for struct, energy in lno_data:
            self.sw.add_data(struct, {'energy': energy})
        ecis = np.linalg.lstsq(self.sw.feature_matrix,
                               self.sw.get_property_vector('energy', True),
                               rcond=None)[0]
        self.ce = ClusterExpansion(cs, ecis, self.sw.feature_matrix)
        scmatrix = np.array([[3, 0, 0],
                             [0, 2, 0],
                             [0, 0, 1]])
        self.pr = EwaldCEProcessor(self.ce, scmatrix)
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
        self.test_occu = self.ce.cluster_subspace.occupancy_from_structure(test_struct,
                                                                           scmatrix=scmatrix)
        self.enc_occu = self.pr.occupancy_from_structure(test_struct)
        self.sublattices = []
        for doms in self.pr.unique_site_spaces:
            sites = np.array([i for i, b in enumerate(self.pr.allowed_species)
                              if b == list(doms.keys())])
            self.sublattices.append({'doms': list(range(len(doms))),
                                     'sites': sites})

    def test_compute_property_change(self):
        occu = self.enc_occu.copy()
        for _ in range(50):
            sublatt = np.random.choice(self.sublattices)
            site = np.random.choice(sublatt['sites'])
            new_sp = np.random.choice(sublatt['doms'])
            new_occu = occu.copy()
            new_occu[site] = new_sp
            prop_f = self.pr.compute_property(new_occu)
            prop_i = self.pr.compute_property(occu)
            dprop = self.pr.compute_property_change(occu, [(site, new_sp)])
            self.assertTrue(np.allclose(dprop, prop_f - prop_i,
                                        rtol=rtol, atol=atol))
            # Test reverse matches forward
            old_sp = occu[site]
            rdprop = self.pr.compute_property_change(new_occu, [(site,
                                                                 old_sp)])
            self.assertTrue(np.isclose(dprop, -1.0*rdprop,
                                       rtol=rtol, atol=atol))

    def test_delta_corr(self):
        occu = self.enc_occu.copy()
        for _ in range(50):
            sublatt = np.random.choice(self.sublattices)
            site = np.random.choice(sublatt['sites'])
            new_sp = np.random.choice(sublatt['doms'])
            new_occu = occu.copy()
            new_occu[site] = new_sp
            # Test forward
            dcorr = self.pr._delta_corr([(site, new_sp)], occu)
            corr_f = self.pr.compute_correlation(new_occu)
            corr_i = self.pr.compute_correlation(occu)
            self.assertTrue(np.allclose(dcorr, corr_f - corr_i,
                                        rtol=rtol, atol=atol))
            # Test reverse matches forward
            old_sp = occu[site]
            rdcorr = self.pr._delta_corr([(site, old_sp)], new_occu)
            self.assertTrue(np.allclose(dcorr, -1*rdcorr,
                                        rtol=rtol, atol=atol))

    def test_msonable(self):
        d = self.pr.as_dict()
        pr = EwaldCEProcessor.from_dict(d)
        self.assertEqual(self.pr.compute_property(self.enc_occu),
                         pr.compute_property(self.enc_occu))
        self.assertTrue(np.array_equal(self.pr.ewald_matrix,
                                       pr.ewald_matrix))
        j = json.dumps(d)
        _ = json.loads(j)