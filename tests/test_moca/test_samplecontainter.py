import unittest
from collections import OrderedDict
from itertools import product
import numpy as np
import numpy.testing as npt

from smol.moca.ensemble.sublattice import Sublattice
from smol.moca.sampler import SampleContainer


class TestSamplerContainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # define some face sublattices
        cls.num_sites = 500
        cls.natural_parameters = np.zeros(10)
        cls.natural_parameters[[0, -1]] = -1  # make first and last 1
        cls.num_energy_coefs = 9
        sites = np.random.choice(range(500), size=300, replace=False)
        site_space = OrderedDict({'A': 0.2, 'B': 0.5, 'C': 0.3})
        sublatt1 = Sublattice(site_space, sites)
        site_space = OrderedDict({'A': 0.4, 'D': 0.6})
        sites2 = np.setdiff1d(range(cls.num_sites), sites)
        sublatt2 = Sublattice(site_space, np.array(sites2))
        cls.sublattices = [sublatt1, sublatt2]
        cls.sublatt_comps = [1.0 / 3.0, 1.0 / 2.0]
        
        # generate some fake data
        cls.nsamples, nwalkers = 1000, 5
        cls.occus = np.empty((cls.nsamples, nwalkers, cls.num_sites))
        cls.enths = -5*np.ones((cls.nsamples, nwalkers))
        cls.featblob = np.zeros((cls.nsamples, nwalkers,
                                 len(cls.natural_parameters)))
        cls.accepted = np.random.randint(2, size=(cls.nsamples, nwalkers))
        for i in range(cls.nsamples):
            for j in range(nwalkers):
                # make occupancy compositions (1/3, 1/3, 1/3) & (1/2, 1/2, 1/2)
                s = np.random.choice(sites, size=100, replace=False)
                cls.occus[i, j, s] = 0
                s1 = np.random.choice(np.setdiff1d(sites, s), size=100, replace=False)
                cls.occus[i, j, s1] = 1
                s2 = np.setdiff1d(sites, np.append(s, s1))
                cls.occus[i, j, s2] = 2
                s = np.random.choice(sites2, size=100, replace=False)
                cls.occus[i, j, s] = 0
                cls.occus[i, j, np.setdiff1d(sites2, s)] = 1
                # first and last feature real fake
                cls.featblob[i, j, [0, -1]] = 2.5

        cls.container_1w = SampleContainer(10, cls.num_sites, cls.sublattices,
                                           cls.natural_parameters,
                                           cls.num_energy_coefs)
        cls.container_5w = SampleContainer(10, cls.num_sites, cls.sublattices,
                                           cls.natural_parameters,
                                           cls.num_energy_coefs, nwalkers=5)

    def tearDown(self):
        self.container_1w.clear()
        self.container_5w.clear()

    def setUp(self):
        self._allocate()
        self._add_samples()

    def _allocate(self):
        self.container_1w.allocate(len(self.enths))
        self.container_5w.allocate(len(self.enths))

    def _add_samples(self, thinned_by=1):
        for i in range(self.nsamples):
            self.container_1w.save_sample(self.accepted[i, 0], self.occus[i, 0],
                                          self.enths[i, 0], self.featblob[i, 0],
                                          thinned_by=thinned_by)
            self.container_5w.save_sample(self.accepted[i], self.occus[i],
                                          self.enths[i], self.featblob[i],
                                          thinned_by=thinned_by)

    def test_allocate_and_save(self):
        self.container_1w.clear()
        self.container_5w.clear()

        for container in (self.container_1w, self.container_5w):
            nwalkers = container.shape[0]
            self.assertEqual(len(container), 0)
            self.assertEqual(container._chain.shape,
                             (0, nwalkers, self.num_sites))
            self.assertEqual(container._enthalpy.shape, (0, nwalkers))
            self.assertEqual(container._accepted.shape, (0, nwalkers))

        self._allocate()
        for container in (self.container_1w, self.container_5w):
            nwalkers = container.shape[0]
            self.assertEqual(len(container), 0)
            self.assertEqual(container._chain.shape,
                             (self.nsamples, nwalkers, self.num_sites))
            self.assertEqual(container._enthalpy.shape,
                             (self.nsamples, nwalkers))
            self.assertEqual(container._accepted.shape,
                             (self.nsamples, nwalkers))

        self._add_samples()
        for container in (self.container_1w, self.container_5w):
            self.assertEqual(len(container), self.nsamples)
            self.assertEqual(container.total_mc_steps, self.nsamples)
            container.clear()
        self._allocate()
        thinned = np.random.randint(50)
        self._add_samples(thinned)
        for container in (self.container_1w, self.container_5w):
            self.assertEqual(len(container), self.nsamples)
            self.assertEqual(container.total_mc_steps, thinned*self.nsamples)
            container.clear()

    def test_get_sampled_values(self):
        for container in (self.container_1w, self.container_5w):
            nwalkers = container.shape[0]
            i = container.shape[0]
            for d, t in product((0, 100), (1, 10)):
                # get default flatted values
                nsamples = (self.nsamples - d)//t
                self.assertEqual(container.sampling_efficiency(discard=d),
                                 (self.accepted[d:, :i].sum(axis=0) / (container.total_mc_steps - d)).mean())
                self.assertEqual(container.get_occupancies(discard=d, thin_by=t).shape,
                                 (nsamples*nwalkers, self.num_sites))
                self.assertEqual(container.get_feature_vectors(discard=d, thin_by=t).shape,
                                 (nsamples*nwalkers, len(self.natural_parameters)))
                self.assertEqual(container.get_enthalpies(discard=d, thin_by=t).shape,
                                 (nsamples*nwalkers, ))
                self.assertEqual(container.get_energies(discard=d, thin_by=t).shape,
                                 (nsamples*nwalkers, ))
                for sublattice, comp in zip(self.sublattices, self.sublatt_comps):
                    c = container.get_sublattice_compositions(sublattice, discard=d, thin_by=t)
                    self.assertEqual(c.shape, (nsamples*nwalkers, len(sublattice.species)))
                    npt.assert_array_equal(c, comp*np.ones_like(c))
                # get non flattened values
                npt.assert_array_equal(container.sampling_efficiency(discard=d, flat=False),
                                       (self.accepted[d:, :i].sum(axis=0) / (container.total_mc_steps - d)))
                self.assertEqual(container.get_occupancies(discard=d, thin_by=t, flat=False).shape,
                                 (nsamples, nwalkers, self.num_sites))
                self.assertEqual(container.get_feature_vectors(discard=d, thin_by=t, flat=False).shape,
                                 (nsamples, nwalkers, len(self.natural_parameters)))
                self.assertEqual(container.get_enthalpies(discard=d, thin_by=t, flat=False).shape,
                                 (nsamples, nwalkers, ))
                self.assertEqual(container.get_energies(discard=d, thin_by=t, flat=False).shape,
                                 (nsamples, nwalkers, ))

                for sublattice, comp in zip(self.sublattices, self.sublatt_comps):
                    c = container.get_sublattice_compositions(sublattice, discard=d, thin_by=t, flat=False)
                    self.assertEqual(c.shape, (nsamples, nwalkers, len(sublattice.species)))
                    npt.assert_array_equal(c, comp*np.ones_like(c))

    def test_means_variances(self):
        for container in (self.container_1w, self.container_5w):
            nwalkers = container.shape[0]
            for d, t in product((0, 100), (1, 10)):
                self.assertEqual(container.mean_enthalpy(discard=d, thin_by=t), -5)
                self.assertEqual(container.enthalpy_variance(discard=d, thin_by=t), 0)
                self.assertEqual(container.mean_energy(discard=d, thin_by=t), -2.5)
                self.assertEqual(container.energy_variance(discard=d, thin_by=t), 0)
                npt.assert_array_equal(container.mean_feature_vector(discard=d, thin_by=t),
                                       [2.5] + 8 * [0, ] + [2.5])
                npt.assert_array_equal(container.feature_vector_variance(discard=d, thin_by=t),
                                       10 * [0, ])
                for sublattice, comp in zip(self.sublattices, self.sublatt_comps):
                    npt.assert_array_almost_equal(container.mean_sublattice_composition(sublattice, discard=d, thin_by=t),
                                                  len(sublattice.species)*[comp, ])
                    npt.assert_array_almost_equal(container.sublattice_composition_variance(sublattice, discard=d, thin_by=t),
                                                  len(sublattice.species) * [0, ])

                # without flattening
                npt.assert_array_equal(container.mean_enthalpy(discard=d, thin_by=t,
                                                               flat=False),
                                       nwalkers * [-5])
                npt.assert_array_equal(container.enthalpy_variance(discard=d, thin_by=t,
                                                                   flat=False),
                                       nwalkers * [0])
                npt.assert_array_equal(container.mean_energy(discard=d, thin_by=t,
                                                             flat=False),
                                       nwalkers * [-2.5])
                npt.assert_array_equal(container.energy_variance(discard=d, thin_by=t,
                                                                 flat=False),
                                       nwalkers * [0])
                npt.assert_array_equal(container.mean_feature_vector(discard=d,
                                                                     thin_by=t, flat=False),
                                       nwalkers * [[2.5] + 8 * [0, ] + [2.5]])
                npt.assert_array_equal(container.feature_vector_variance(discard=d,
                                                                         thin_by=t, flat=False),
                                       nwalkers * [10 * [0, ]])
                for sublattice, comp in zip(self.sublattices, self.sublatt_comps):
                    npt.assert_array_almost_equal(container.mean_sublattice_composition(sublattice, discard=d,
                                                                                        thin_by=t, flat=False),
                                                  nwalkers * [len(sublattice.species)*[comp]])
                    npt.assert_array_almost_equal(container.sublattice_composition_variance(sublattice, discard=d,
                                                                                            thin_by=t, flat=False),
                                                  nwalkers * [len(sublattice.species) * [0]])

    def test_get_mins(self):
        # set a fake minimum
        i = np.random.choice(range(self.nsamples))
        for container in (self.container_1w, self.container_5w):
            nwalkers = container.shape[0]
            container._enthalpy[i, :] = -10
            container._feature_blob[i, :, :] = 5.0
            self.assertEqual(container.get_minimum_enthalpy(), -10)
            self.assertEqual(container.get_minimum_energy(), -5)
            npt.assert_array_equal(container.get_minimum_enthalpy(flat=False),
                                   nwalkers * [-10])
            npt.assert_array_equal(container.get_minimum_energy(flat=False),
                                   nwalkers * [-5])
            npt.assert_array_equal(container.get_minimum_enthalpy_occupancy(flat=False),
                                   container._chain[i])
            npt.assert_array_equal(container.get_minimum_energy_occupancy(flat=False),
                                   container._chain[i])

    def test_stream(self):
        pass
