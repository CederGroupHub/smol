"""
This is a base test case for all processors. This should never be run as an
actual test, but actual tests can derive from it.
"""

import unittest
import json
import numpy as np
from tests.utils import assert_msonable
from smol.moca.processor.base import Processor

RTOL = 0.0  # relative tolerance to check property change functions
# absolute tolerance to check property change functions (eps is approx 2E-16)
ATOL = 3E2 * np.finfo(float).eps
DRIFT_TOL = np.finfo(float).eps  # tolerance of average drift


class TestProcessor(unittest.TestCase):
    """Class for all processor tests. Inherit from this.
    And write the appropriate setUp or setUpClass"""
    atol = ATOL  # set tolerances, some are loosened for some tests
    drift_tol = DRIFT_TOL

    @classmethod
    def setUpClass(cls):
        # Define these in subclasses
        cls.cs = None
        cls.test_struct = None
        cls.test_occu = None
        cls.enc_occu = None
        cls.pr = None
        cls.sublattices = None

    @staticmethod
    def _create_sublattice_dicts(unique_site_spaces, allowed_species):
        sublattices = []
        for space in unique_site_spaces:
            sites = np.array([i for i, b in enumerate(allowed_species)
                              if b == list(space.keys())])
            sublattices.append({'spaces': list(range(len(space))),
                                'sites': sites})
        return sublattices

    def test_encode_property(self):
        enc_occu = self.pr.encode_occupancy(self.test_occu)
        self.assertTrue(all(o1 == o2 for o1, o2 in zip(self.enc_occu,
                                                       enc_occu)))

    def test_decode_property(self):
        occu = self.pr.decode_occupancy(self.enc_occu)
        self.assertTrue(all(o1 == o2 for o1, o2 in zip(self.test_occu,
                                                       occu)))

    def test_get_average_drift(self):
        forward, reverse = self.pr.compute_average_drift()
        self.assertTrue(forward <= self.drift_tol and reverse <= self.drift_tol)

    # TODO write this
    def test_occupancy_from_structure(self):
        pass

    def test_structure_from_occupancy(self):
        # The structures do pass as equal by direct == comparison, but as long
        # as the correlation vectors and predicted energy are the same we
        # should be all good.
        test_struct = self.pr.structure_from_occupancy(self.enc_occu)
        self.assertTrue(np.allclose(self.cs.corr_from_structure(test_struct),
                                    self.cs.corr_from_structure(self.test_struct)))

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
                                        rtol=RTOL, atol=self.atol))
            # Test reverse matches forward
            old_sp = occu[site]
            rdprop = self.pr.compute_property_change(new_occu, [(site, old_sp)])
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
                                        rtol=RTOL, atol=self.atol))
            # Test reverse matches forward
            old_sp = occu[site]
            rdcorr = self.pr.compute_feature_vector_change(new_occu, [(site, old_sp)])
            self.assertTrue(np.array_equal(dcorr, -1 * rdcorr))

    def test_msonable(self):
        d = self.pr.as_dict()
        pr = Processor.from_dict(d)

        self.assertEqual(self.pr.compute_property(self.enc_occu),
                         pr.compute_property(self.enc_occu))
        # run to see if everything is properly serializable
        # TODO do this when improving tests. Need to fix the old synth datasets
        # assert_msonable(self, self.pr)
