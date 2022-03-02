"""
Run a simple test comparing with the icet tutorial results...
because icet is HIGH FRKN QUALITY! So I trust it...
"""


import unittest
import numpy as np
from pymatgen.core.periodic_table import get_el_sp
from smol.cofe import StructureWrangler, ClusterSubspace, ClusterExpansion
from smol.moca import SemiGrandEnsemble, Sampler, ClusterExpansionProcessor
from tests.data import (
    icet_eci,
    icet_predictions,
    icet_fit_structs,
    icet_test_structs,
    aupt_prim,
    icet_sgc_run_10000its,
)


class TestCEvicet(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cs = ClusterSubspace.from_cutoffs(
            aupt_prim,
            {2: 13.5, 3: 6.0, 4: 5.5},
            supercell_size="num_sites",
            basis="sinusoid",
        )
        cls.sw = StructureWrangler(cls.cs)
        for item in icet_fit_structs:
            cls.sw.add_data(
                item["structure"], item["properties"], supercell_matrix=item["scmatrix"]
            )
        cls.sw.update_features()
        cls.coefs = np.linalg.lstsq(
            cls.sw.feature_matrix,
            cls.sw.get_property_vector("energy", True),
            rcond=None,
        )[0]

    def test_subspace(self):
        self.assertEqual(self.cs.num_corr_functions, len(icet_eci))
        self.assertEqual(len(self.cs.orbits_by_size[1]), 1)
        self.assertEqual(len(self.cs.orbits_by_size[2]), 25)
        self.assertEqual(len(self.cs.orbits_by_size[3]), 12)
        self.assertEqual(len(self.cs.orbits_by_size[4]), 16)

    def test_clusterexpansion(self):
        # The eci are not equal for smol and icet since the +1 -1 are assigned
        # exactly opposite. But the first eci should be equal to the average
        # of fit data, and the sum of squares of eci should be the same.
        self.assertTrue(np.isclose(icet_eci[0], self.coefs[0]))
        self.assertTrue(np.isclose(sum(icet_eci**2), sum(self.coefs**2)))
        # Now test that predictions match
        ce = ClusterExpansion(self.cs, coefficients=self.coefs)
        test_structs = [i["structure"] for i in icet_test_structs]
        preds = [ce.predict(s, normalize=True) for s in test_structs]
        self.assertTrue(np.allclose(icet_predictions, preds))

    def test_sgc_montecarlo(self):
        sc_matrix = 3 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
        struct = self.cs.structure.copy()
        struct.replace(0, "Ag")
        struct.make_supercell(sc_matrix)
        pr = ClusterExpansionProcessor(self.cs, sc_matrix, self.coefs)
        ens = SemiGrandEnsemble(pr, chemical_potentials={"Ag": 0, "Pd": 0})
        sampler = Sampler.from_ensemble(ens, temperature=300)
        init_occu = pr.occupancy_from_structure(struct)
        iterations = 10000  # icet runs were done with 10000 iterations

        # These still need some tweaking to pass reliably
        # perhaps run the sampling for longer to get better averages
        for temp, dmu_dict in icet_sgc_run_10000its.items():
            icet_avg = np.array([d["average_energy"] for _, d in dmu_dict.items()])
            avg = []
            sampler.mckernel.temperature = float(temp)
            for dmu, data in dmu_dict.items():
                chemical_potentials = {"Ag": 0, "Pd": float(dmu)}
                ens.chemical_potentials = chemical_potentials
                sampler.run(iterations, init_occu, thin_by=len(struct))
                # need to keep these tolerances real slack since very few
                # iterations are run and there is a transition point where the
                # difference is acceptable.
                avg.append(sampler.samples.mean_energy())
                self.assertAlmostEqual(
                    data["acceptance_ratio"],
                    sampler.samples.sampling_efficiency(),
                    places=1,
                )
                for sp, comp in data["composition"].items():
                    self.assertAlmostEqual(
                        comp,
                        sampler.samples.mean_composition()[get_el_sp(sp)],
                        places=1,
                    )
                sampler.clear_samples()

            # test relative error of mean energy norm only since testing
            # individually may need more slack tolerances for point with
            # higher variance, specifically T=300 mu>=0.15
            self.assertAlmostEquals(
                np.linalg.norm(icet_avg - np.array(avg)) / np.linalg.norm(icet_avg),
                0,
                places=1,
            )
