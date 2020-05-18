import unittest
import numpy as np
from pymatgen.transformations.standard_transformations import \
    OrderDisorderedStructureTransformation
from smol.moca import CEProcessor, EwaldCEProcessor
from smol.cofe import ClusterExpansion, StructureWrangler, ClusterSubspace
from smol.cofe.configspace import EwaldTerm
from tests.data import lno_prim, lno_data


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
                                                                           scmatrix)
        self.enc_occu = self.pr.occupancy_from_structure(test_struct)

    def test_compute_property(self):
        self.assertTrue(np.isclose(self.ce.predict(self.test_struct),
                         self.pr.compute_property(self.enc_occu)))

    def test_compute_correlation(self):
        self.assertTrue(np.allclose(self.ce.cluster_subspace.corr_from_structure(self.test_struct),
                                    self.pr.compute_correlation(self.enc_occu)))

    # TODO check that this works for all bases and orthonormal
    def test_compute_property_change(self):
        flips = [(10, 1), (6, 0)]
        new_occu = self.enc_occu.copy()
        new_occu[flips[0][0]] = flips[0][1]
        new_occu[flips[1][0]] = flips[1][1]
        prop_f = self.pr.compute_property(new_occu)
        prop_i = self.pr.compute_property(self.enc_occu)
        dprop = self.pr.compute_property_change(self.enc_occu, flips)
        self.assertTrue(np.allclose(dprop, prop_f - prop_i))

    def test_encode_property(self):
        enc_occu = self.pr.encode_occupancy(self.test_occu)
        self.assertTrue(all(o1 == o2 for o1, o2 in zip(self.enc_occu,
                                                       enc_occu)))

    def test_decode_property(self):
        occu = self.pr.decode_occupancy(self.enc_occu)
        self.assertTrue(all(o1 == o2 for o1, o2 in zip(self.test_occu,
                                                       occu)))

    # TODO check that this works for all bases, and orthonormality
    def test_delta_corr(self):
        flips = [(10, 1), (6, 0)]
        new_occu = self.enc_occu.copy()
        new_occu[flips[0][0]] = flips[0][1]
        new_occu[flips[1][0]] = flips[1][1]
        dcorr = self.pr.delta_corr(flips, self.enc_occu)
        corr_f = self.pr.compute_correlation(new_occu)
        corr_i = self.pr.compute_correlation(self.enc_occu)
        self.assertTrue(np.allclose(dcorr, corr_f - corr_i))

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
        flips = [(10, 1), (6, 0)]
        new_occu = self.enc_occu.copy()
        new_occu[flips[0][0]] = flips[0][1]
        new_occu[flips[1][0]] = flips[1][1]
        dcorr = pr.delta_corr(flips, self.enc_occu)
        corr_f = pr.compute_correlation(new_occu)
        corr_i = pr.compute_correlation(self.enc_occu)
        self.assertTrue(np.allclose(dcorr, corr_f - corr_i))

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
        self.assertTrue(self.pr.bits == pr.bits)
        self.assertTrue(self.pr.structure == pr.structure)
        self.assertEqual(self.pr.compute_property(self.enc_occu),
                         pr.compute_property(self.enc_occu))


class TestEwaldCEProcessor(unittest.TestCase):
    def setUp(self) -> None:
        cs = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1},
                                         ltol=0.15, stol=0.2,
                                         angle_tol=5,
                                         supercell_size='O2-',
                                         basis='sinusoid',
                                         orthonormal=True,
                                         use_concentration=True)
        cs.add_external_term(EwaldTerm)
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
                                                                           scmatrix)
        self.enc_occu = self.pr.occupancy_from_structure(test_struct)

    # TODO check that this works for all bases and orthonormal
    def test_compute_property_change(self):
        flips = [(10, 1), (6, 0)]
        new_occu = self.enc_occu.copy()
        new_occu[flips[0][0]] = flips[0][1]
        new_occu[flips[1][0]] = flips[1][1]
        prop_f = self.pr.compute_property(new_occu)
        prop_i = self.pr.compute_property(self.enc_occu)
        dprop = self.pr.compute_property_change(self.enc_occu, flips)
        self.assertTrue(np.allclose(dprop, prop_f - prop_i))

    # TODO check that this works for all bases, and orthonormality
    def test_delta_corr(self):
        flips = [(10, 1), (6, 0)]
        new_occu = self.enc_occu.copy()
        new_occu[flips[0][0]] = flips[0][1]
        new_occu[flips[1][0]] = flips[1][1]
        dcorr = self.pr.delta_corr(flips, self.enc_occu)
        corr_f = self.pr.compute_correlation(new_occu)
        corr_i = self.pr.compute_correlation(self.enc_occu)
        self.assertTrue(np.allclose(dcorr, corr_f - corr_i))
