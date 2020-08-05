import unittest
from collections import OrderedDict
import numpy as np

from smol.moca.ensemble.sublattice import Sublattice
from tests.utils import assert_msonable


class TestSublattice(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        site_space = OrderedDict({'A': 0.3, 'B': 0.3, 'C': 0.2, 'D': 0.2})
        sites = np.random.choice(range(100), size=60)
        cls.sublattice = Sublattice(site_space, sites)

    def test_restrict_sites(self):
        sites = np.random.choice(self.sublattice.sites, size=10)
        self.sublattice.restrict_sites(sites)
        self.assertFalse(any(s in self.sublattice.active_sites for s in sites))
        self.assertTrue(all(s in self.sublattice.restricted_sites for s in sites))
        self.assertNotEqual(len(self.sublattice.active_sites),
                            len(self.sublattice.sites))
        self.sublattice.reset_restricted_sites()
        self.assertEqual(len(self.sublattice.active_sites),
                         len(self.sublattice.sites))

    def test_print_repr(self):
        print(self.sublattice)
        repr(self.sublattice)

    def test_msonable(self):
        assert_msonable(self, self.sublattice)
