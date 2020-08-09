"""
This is a base test case for all ensembles. This should never be run as an
actual test, but actual tests can derive from it.
"""

import unittest
import numpy as np
import numpy.testing as npt

from smol.constants import kB
from tests.utils import assert_msonable


class _EnsembleTest(unittest.TestCase):
    """Base tests for all ensemble classes."""
    @classmethod
    def setUpClass(cls) -> None:
        # Define these in subclasses
        cls.n_allowed_species = 2  # Binary, ternary, etc
        cls.subspace = None
        cls.expansion = None
        cls.processor = None
        cls.ensemble = None
        cls.ensemble_kwargs = {}
        raise unittest.SkipTest(f'This method is not implemented in class {cls}')

    def setUp(self):
        self.enc_occu = np.random.randint(0, self.n_allowed_species,
                                          size=self.processor.num_sites)
        self.init_occu = self.processor.decode_occupancy(self.enc_occu)

    def test_from_cluster_expansion(self):
        ensemble = self.ensemble.from_cluster_expansion(self.expansion,
                                                        self.processor.supercell_matrix,
                                                        self.ensemble.temperature,
                                                        **self.ensemble_kwargs)
        npt.assert_array_equal(self.ensemble.natural_parameters,
                               ensemble.natural_parameters)
        npt.assert_array_equal(self.ensemble.sublattices[0].sites,
                               ensemble.sublattices[0].sites)
        self.assertEqual(self.processor.compute_property(self.enc_occu),
                         ensemble.processor.compute_property(self.enc_occu))
        npt.assert_array_equal(self.processor.compute_feature_vector(self.enc_occu),
                               ensemble.processor.compute_feature_vector(self.enc_occu))
        for _ in range(10):  # test a few flips
            site = np.random.choice(range(self.processor.num_sites))
            spec = np.random.choice(list(set(range(self.n_allowed_species))-{self.enc_occu[site]}))
            flip = [(site, spec)]
            self.assertEqual(self.processor.compute_property_change(self.enc_occu, flip),
                             ensemble.processor.compute_property_change(self.enc_occu, flip))

    def test_temperature_setter(self):
        self.assertEqual(self.ensemble.beta, 1/(kB*self.ensemble.temperature))
        self.ensemble.temperature = 300
        self.assertEqual(self.ensemble.beta, 1/(kB*300))

    def test_restrict_sites(self):
        sites = np.random.choice(range(self.processor.num_sites), size=5)
        self.ensemble.restrict_sites(sites)
        for sublatt in self.ensemble.sublattices:
            self.assertFalse(any(i in sublatt.active_sites for i in sites))
        self.ensemble.reset_restricted_sites()
        for sublatt in self.ensemble.sublattices:
            npt.assert_array_equal(sublatt.sites, sublatt.active_sites)

    def test_msonable(self):
        # assert_msonable(self, self.ensemble)
        pass  # TODO Broken because of basis in orbit.from_dict as well
