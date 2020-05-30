import unittest
import numpy as np
from smol.cofe import ClusterSubspace, StructureWrangler
from smol.cofe.extern import EwaldTerm
from smol.learn.utils import constrain_dielectric
from tests.data import lno_prim, lno_data


class TestConstrainDielectric(unittest.TestCase):
    def setUp(self) -> None:
        self.cs = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1},
                                             ltol=0.15, stol=0.2,
                                             angle_tol=5, supercell_size='O2-')
        self.cs.add_external_term(EwaldTerm())
        self.sw = StructureWrangler(self.cs)
        for struct, energy in lno_data:
            self.sw.add_data(struct, {'energy': energy})
        self.ecis = np.linalg.lstsq(self.sw.feature_matrix,
                                    self.sw.get_property_vector('energy',True),
                                    rcond=None)[0]

    def test_runtime(self):
        fit = lambda X, y: np.linalg.lstsq(X, y, rcond=None)[0]

        # Test that it constrains when needed
        max_dielectric = 1 / self.ecis[-1] - 2
        decorated = constrain_dielectric(max_dielectric)(fit)
        ecis = decorated(self.sw.feature_matrix,
                         self.sw.get_property_vector('energy', True))
        self.assertTrue(1/ecis[-1] == max_dielectric)
        # Test that it does not constrain when not
        max_dielectric = 1 / self.ecis[-1] + 2
        decorated = constrain_dielectric(max_dielectric)(fit)
        ecis = decorated(self.sw.feature_matrix,
                         self.sw.get_property_vector('energy', True))
        self.assertTrue(ecis[-1] == self.ecis[-1])

    def test_decorate(self):
        max_dielectric = 1 / self.ecis[-1] - 2

        @constrain_dielectric(max_dielectric)
        def fit(X, y):
            return np.linalg.lstsq(X, y, rcond=None)[0]

        ecis = fit(self.sw.feature_matrix,
                   self.sw.get_property_vector('energy', True))
        self.assertTrue(1 / ecis[-1] == max_dielectric)
