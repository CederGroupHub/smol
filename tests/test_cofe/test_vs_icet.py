# Run a simple test comparing with the icet tutorial results...
# because icet is HIGH FRKN QUALITY! So I trust it...

import unittest
import numpy as np
from smol.cofe import StructureWrangler, ClusterSubspace, ClusterExpansion
from tests.data import (icet_eci, icet_predictions, icet_fit_structs,
                        icet_test_structs, aupt_prim)

class TestCEvicet(unittest.TestCase):
    def setUp(self) -> None:
        self.cs = ClusterSubspace.from_radii(aupt_prim,
                                             {2: 13.5, 3: 6.0, 4: 5.5},
                                             supercell_size='num_sites',
                                             basis='sinusoid')
        self.sw = StructureWrangler(self.cs)
        # Make this quicker so structure matcher does not slow things down
        self.sw._items = icet_fit_structs
        self.sw.update_features()

    def test_subspace(self):
        self.assertEqual(len(self.cs.orbits_by_size[1]), 1)
        self.assertEqual(len(self.cs.orbits_by_size[2]), 25)
        self.assertEqual(len(self.cs.orbits_by_size[3]), 12)
        self.assertEqual(len(self.cs.orbits_by_size[4]), 16)

    def test_clusterexpansion(self):
        # The eci are not equal for smol and icet since the +1 -1 are assigned
        # exactly opposite. But the first eci should be equal to the average
        # of fit data, and the sum of squares of eci should be the same.
        ecis = np.linalg.lstsq(self.sw.feature_matrix,
                              self.sw.normalized_properties,
                              rcond=None)[0]
        self.assertTrue(np.isclose(icet_eci[0], ecis[0]))
        self.assertTrue(np.isclose(sum(icet_eci**2), sum(ecis**2)))
        # Now test that predictions match
        ce = ClusterExpansion(self.cs, ecis=ecis)
        test_structs = [i['structure'] for i in icet_test_structs]
        self.assertTrue(np.allclose(icet_predictions,
                                    ce.predict(test_structs, normalized=True)))