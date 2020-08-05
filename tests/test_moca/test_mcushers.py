import unittest
from collections import OrderedDict
import numpy as np
from smol.moca.ensemble.sublattice import Sublattice
from smol.moca.sampler.mcusher import Swapper, Flipper


class BaseTest:
    """Wrap this up so it is not run as a test."""
    class TestMCMCUsher(unittest.TestCase):
        @classmethod
        def setUpClass(cls) -> None:
            cls.sites = np.arange(100)
            # set up two sublattices
            sites1 = np.random.choice(cls.sites, size=40)
            sites2 = np.setdiff1d(cls.sites, sites1)
            site_space1 = OrderedDict({'A': 0.1, 'B': 0.4, 'C': 0.3, 'D': 0.2})
            site_space2 = OrderedDict({'A': 0.1, 'B': 0.4, 'E': 0.5})
            cls.sublattices = [Sublattice(site_space1, sites1),
                               Sublattice(site_space2, sites2)]
            # create a random test occu
            cls.occu = np.zeros_like(cls.sites)
            for site in cls.sites:
                if site in sites1:
                    cls.occu[site] = np.random.choice(range(len(site_space1)))
                else:
                    cls.occu[site] = np.random.choice(range(len(site_space2)))
            # create an usher here
            cls.mcusher = None

        def test_bad_probabilities(self):
            with self.assertRaises(ValueError):
                self.mcusher.sublattice_probabilities = [0.6, 0.1]
            with self.assertRaises(AttributeError):
                self.mcusher.sublattice_probabilities = [0.5, 0.2, 0.3]

        def test_propose_step(self):
            iterations = 50000
            # test with 50/50 probability
            flipped_sites = []
            count1, count2 = 0, 0
            total = 0
            for i in range(iterations):
                step = self.mcusher.propose_step(self.occu)
                for flip in step:
                    if flip[0] in self.sublattices[0].sites:
                        count1 += 1
                        self.assertTrue(flip[1] in range(len(self.sublattices[0].species)))
                    elif flip[0] in self.sublattices[1].sites:
                        count2 += 1
                        self.assertTrue(flip[1] in range(len(self.sublattices[1].species)))
                    else:
                        raise RuntimeError('Something went wrong in proposing'
                                           f'a step site proposed in {step} is'
                                           ' not in any of the allowed sites')
                    total += 1
                flipped_sites.append(flip[0])

            # check probabilities seem sound
            self.assertAlmostEqual(count1/total, 0.5, places=1)
            self.assertAlmostEqual(count2/total, 0.5, places=1)

            # check that every site was flipped at least once
            self.assertTrue(all(i in flipped_sites for i in self.sites))

            # Now check with a sublattice bias
            self.mcusher.sublattice_probabilities = [0.8, 0.2]
            flipped_sites = []
            count1, count2 = 0, 0
            total = 0
            for i in range(iterations):
                step = self.mcusher.propose_step(self.occu)
                for flip in step:
                    if flip[0] in self.sublattices[0].sites:
                        count1 += 1
                        self.assertTrue(flip[1] in range(len(self.sublattices[0].species)))
                    elif flip[0] in self.sublattices[1].sites:
                        count2 += 1
                        self.assertTrue(flip[1] in range(len(self.sublattices[1].species)))
                    else:
                        raise RuntimeError('Something went wrong in proposing'
                                           f'a step site proposed in {step} is'
                                           ' not in any of the allowed sites')
                    total += 1
                flipped_sites.append(flip[0])

            self.assertAlmostEqual(count1/total, 0.8, places=1)
            self.assertAlmostEqual(count2/total, 0.2, places=1)


class TestFlipper(BaseTest.TestMCMCUsher):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.mcusher = Flipper(cls.sublattices)


class TestSwapper(BaseTest.TestMCMCUsher):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.mcusher = Swapper(cls.sublattices)