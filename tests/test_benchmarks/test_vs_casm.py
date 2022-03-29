import json
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from monty.json import MontyDecoder

from smol.cofe import ClusterSubspace, StructureWrangler


class TestvsCASM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_dir = (
            Path(__file__).absolute().parent
            / ".."
            / "data"
            / "benchmark_data"
            / "casm_mgcro"
        )

        with open(data_dir / "casm-fit-data.json") as fp:
            cls.fit_data = json.load(fp, cls=MontyDecoder)

        with open(data_dir / "fit-items.json") as fp:
            fit_items = json.load(fp, cls=MontyDecoder)

        # Create the smol ClusterSubspace (takes a while...)
        cutoff_raddi = {int(s): r for s, r in cls.fit_data["cutoff_radii"].items()}
        cls.cs = ClusterSubspace.from_cutoffs(
            cls.fit_data["prim"],
            cutoffs=cutoff_raddi,
            basis="sinusoid",
            supercell_size="volume",
            ltol=0.1,
            stol=0.1,
            angle_tol=5,
        )

        # create the StructureWrangler
        cls.sw = StructureWrangler(cls.cs)
        # cheat adding the data to make things quicker
        cls.sw._items += fit_items

    def test_subspace(self):
        orbits_by_size = self.fit_data["orbits_by_size"]
        self.assertEqual(len(self.fit_data["ecis"]), self.cs.num_corr_functions)
        self.assertEqual(len(self.cs.orbits_by_size[1]), orbits_by_size["1"])
        self.assertEqual(len(self.cs.orbits_by_size[2]), orbits_by_size["2"])
        self.assertEqual(len(self.cs.orbits_by_size[3]), orbits_by_size["3"])
        self.assertEqual(len(self.cs.orbits_by_size[4]), orbits_by_size["4"])

    # Since the CASM fit was done using l1 regularization will use sklearn
    @unittest.skipUnless(find_spec("sklearn"), "sklearn not installed")
    def test_clusterexpansion(self):
        from sklearn.linear_model import LassoCV

        est = LassoCV(fit_intercept=False)
        est.fit(self.sw.feature_matrix, self.sw.get_property_vector("mixing_energy"))
        ecis = est.coef_
        self.assertAlmostEqual(self.fit_data["ecis"][0], ecis[0], places=1)
        self.assertAlmostEqual(
            sum(np.array(self.fit_data["ecis"]) ** 2), sum(ecis**2), places=2
        )
