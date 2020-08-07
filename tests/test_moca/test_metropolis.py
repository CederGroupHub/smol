import unittest
import numpy as np
import numpy.testing as npt

from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion
from smol.moca import (CanonicalEnsemble, FuSemiGrandEnsemble,
                       MuSemiGrandEnsemble, MetropolisSampler)
from smol.moca.sampler.mcusher import Swapper, Flipper
from tests.data import synthetic_CE_binary


class TestMetropolisSampler(unittest.TestCase):
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
        sc_matrix = np.array([[3, 1, 1], [1, 3, 1], [1, 1, 3]])
        cls.T = 5000
        cls.censemble = CanonicalEnsemble.from_cluster_expansion(ce,
                                                                 sc_matrix,
                                                                 cls.T)
        cls.msgensemble = MuSemiGrandEnsemble.from_cluster_expansion(ce,
                                                                     sc_matrix,
                                                                     cls.T,
                                chemical_potentials={'Na+': -0.05, 'Cl-': 0})
        cls.fsgensemble = FuSemiGrandEnsemble.from_cluster_expansion(ce,
                                                                     sc_matrix,
                                                                     cls.T)

    def setUp(self):
        self.occu = np.random.randint(0, 2, size=self.censemble.num_sites)
        self.occu_stack = np.vstack([
            np.random.randint(0, 2, size=self.censemble.num_sites)
            for _ in range(5)])

    def test_constructor(self):
        # test that the default mcushers are chosen
        self.assertTrue(isinstance(MetropolisSampler(self.censemble)._usher,
                                   Swapper))
        self.assertTrue(isinstance(MetropolisSampler(self.fsgensemble)._usher,
                                   Flipper))
        self.assertTrue(isinstance(MetropolisSampler(self.msgensemble)._usher,
                                   Flipper))
        # test a bad step type
        self.assertRaises(ValueError, MetropolisSampler, self.censemble,
                          step_type='bloop')

    def test_attempt_step(self):
        # get a sampler with random ensemble
        sampler = MetropolisSampler(np.random.choice([self.censemble,
                                                      self.msgensemble,
                                                      self.fsgensemble]))
        for _ in range(100):
            init_occu = self.occu.copy()
            acc, occu, denth, dfeat = sampler._attempt_step(init_occu)
            if acc:
                self.assertFalse(np.array_equal(occu, self.occu))
            else:
                npt.assert_array_equal(occu, self.occu)

    def test_sample(self):
        # get samplers with random ensemble
        steps = 1000
        sampler = MetropolisSampler(np.random.choice([self.censemble,
                                                      self.msgensemble,
                                                      self.fsgensemble]))
        sampler5 = MetropolisSampler(np.random.choice([self.censemble,
                                                       self.msgensemble,
                                                       self.fsgensemble]),
                                     nwalkers=self.occu_stack.shape[0])
        for t in (1, 10):
            samples = [state for state
                       in sampler.sample(steps, self.occu, thin_by=t)]
            samples5 = [state for state
                        in sampler5.sample(steps, self.occu_stack, thin_by=t)]
            self.assertEqual(len(samples), steps // t)
            self.assertEqual(len(samples5), steps // t)

        it = sampler.sample(43, self.occu, thin_by=7)
        self.assertWarns(RuntimeWarning, next, it)

    def test_run(self):
        steps = 1000
        sampler_c = MetropolisSampler(self.censemble)
        sampler5_c = MetropolisSampler(self.censemble,
                                       nwalkers=self.occu_stack.shape[0])
        sampler_f = MetropolisSampler(self.fsgensemble)
        sampler5_f = MetropolisSampler(self.fsgensemble,
                                       nwalkers=self.occu_stack.shape[0])
        sampler_m = MetropolisSampler(self.msgensemble)
        sampler5_m = MetropolisSampler(self.msgensemble,
                                       nwalkers=self.occu_stack.shape[0])
        for sampler, sampler5 in [(sampler_c, sampler5_c),
                                  (sampler_f, sampler5_f),
                                  (sampler_m, sampler5_m)]:
            for t in (1, 10):
                sampler.run(steps, self.occu, thin_by=t)
                sampler5.run(steps, self.occu_stack, thin_by=t)
                self.assertEqual(len(sampler.samples), steps // t)
                self.assertEqual(len(sampler5.samples), steps // t)
                print(sampler.ensemble, sampler.efficiency(),
                      sum(sampler.samples._accepted))
                print(sampler5.ensemble, sampler5.efficiency(),
                      sum(sampler5.samples._accepted))
                self.assertTrue(0 < sampler.efficiency() <= 1)
                self.assertTrue(0 < sampler5.efficiency() <= 1)
                sampler.clear_samples(), sampler5.clear_samples()

        return
        # TODO finish implementing get_compositions in sample container to
        #  test this
        chem_pots = {'Na+': 100.0, 'Cl-': 0.0}
        self.msgensemble.chemical_potentials = chem_pots
        expected = [1.0, 0.0]
        sampler_m.run(steps)
        npt.assert_array_almost_equal(expected,
                                      sampler5_m.samples.mean_composition())
        chem_pots = {'Na+': -100.0, 'Cl-': 0.0}
        self.msgensemble.chemical_potentials = chem_pots
        expected = [0.0, 0.1]
        sampler_m.run(steps)
        npt.assert_array_almost_equal(expected,
                                      sampler_m.samples.mean_composition())
        sampler_f.run(steps)
        expected = [0.5, 0.5]
        npt.assert_array_almost_equal(expected,
                                      sampler5_f.samples.mean_composition())

    def test_reshape_occu(self):
        sampler = MetropolisSampler(self.censemble)
        self.assertEqual(sampler._reshape_occu(self.occu).shape,
                         (1, len(self.occu)))
