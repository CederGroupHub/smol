
import unittest
import numpy as np
from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion
from smol.moca import CEProcessor, CanonicalEnsemble
from tests.data import synthetic_CE_binary


class TestCanonicalEnsemble(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cs = ClusterSubspace.from_dict(synthetic_CE_binary['cluster_subspace'])
        sw = StructureWrangler(cs)
        for item in synthetic_CE_binary['data'][:200]:
            sw.add_data(item[0], {'energy': item[1]})
        ecis = np.linalg.lstsq(sw.feature_matrix,
                               sw.get_property_vector('energy', True),
                               rcond=None)[0]
        ce = ClusterExpansion(cs, ecis, sw.feature_matrix)
        sc_matrix = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        pr = CEProcessor(ce, sc_matrix)
        cls.ensemble = CanonicalEnsemble(pr,
                                         temperature=500,
                                         sample_interval=200)

    def setUp(self):
        pass

    def test_run(self):
        pass

    def test_attempt_step(self):
        pass

    def test_get_flips(self):
        pass

    def test_get_current_data(self):
        pass

    def test_dump(self):
        pass

    def test_reset(self):
        pass

    def test_anneal(self):
        pass

    def test_msnonable(self):
        pass
