import pytest
import numpy as np
import numpy.testing as npt
from smol.moca import (CanonicalEnsemble, FuSemiGrandEnsemble,
                       MuSemiGrandEnsemble, CEProcessor, Sampler)
from smol.moca.sampler.mcusher import Swapper, Flipper
from smol.moca.sampler.kernel import Metropolis
from tests.utils import gen_random_occupancy

ensembles = [CanonicalEnsemble, MuSemiGrandEnsemble, FuSemiGrandEnsemble]
TEMPERATURE = 5000


@pytest.fixture(params=ensembles)
def ensemble(cluster_subspace, request):
    coefs = np.random.random(cluster_subspace.num_corr_functions)
    proc = CEProcessor(cluster_subspace, 4*np.eye(3), coefs)
    if request.param is MuSemiGrandEnsemble:
        kwargs = {'chemical_potentials':
                  {sp: 0.3 for space in proc.unique_site_spaces
                   for sp in space.keys()}}
    else:
        kwargs = {}
    return request.param(proc, **kwargs)


@pytest.fixture(params=[1, 5])
def sampler(ensemble, request):
    sampler = Sampler.from_ensemble(ensemble, temperature=TEMPERATURE,
                                    nwalkers=request.param)
    # fix this additional attribute to sampler to access in gen occus for tests
    sampler.num_sites = ensemble.num_sites
    return sampler


def test_from_ensemble(sampler):
    if "Canonical" in sampler.samples.metadata["name"]:
        assert isinstance(sampler.mcmckernel._usher, Swapper)
    else:
        assert isinstance(sampler.mcmckernel._usher, Flipper)
    assert isinstance(sampler.mcmckernel, Metropolis)


@pytest.mark.parametrize('thin', (1, 10))
def test_sample(sampler, thin):
    occu = np.vstack([gen_random_occupancy(sampler.mcmckernel._usher.sublattices,
                                           sampler.num_sites)
                      for _ in range(sampler.samples.shape[0])])
    steps = 1000
    samples = [state for state
               in sampler.sample(1000, occu, thin_by=thin)]
    assert len(samples) == steps // thin

    it = sampler.sample(43, occu, thin_by=7)
    with pytest.warns(RuntimeWarning):
        next(it)


@pytest.mark.parametrize('thin', (1, 10))
def test_run(sampler, thin):
    occu = np.vstack([gen_random_occupancy(sampler.mcmckernel._usher.sublattices,
                                           sampler.num_sites)
                      for _ in range(sampler.samples.shape[0])])
    steps = 1000
    sampler.run(steps, occu, thin_by=thin)
    assert len(sampler.samples) == steps // thin
    assert 0 < sampler.efficiency() <= 1
    sampler.clear_samples()


def test_anneal(sampler):
    temperatures = np.linspace(2000, 500, 5)
    occu = np.vstack([gen_random_occupancy(sampler.mcmckernel._usher.sublattices,
                                           sampler.num_sites)
                      for _ in range(sampler.samples.shape[0])])
    steps = 100
    sampler.anneal(temperatures, steps, occu)
    expected = []
    for T in temperatures:
        expected += steps*sampler.samples.shape[0]*[T, ]
    npt.assert_array_equal(sampler.samples.get_temperatures(), expected)
    # test temp error
    with pytest.raises(ValueError):
        sampler.anneal([100, 200], steps)


# TODO test run sgensembles at high temp
"""
# test some high temp high potential values
steps = 10000
chem_pots = {'Na+': 100.0, 'Cl-': 0.0}
self.msgensemble.chemical_potentials = chem_pots
expected = {'Na+': 1.0, 'Cl-': 0.0}
sampler_m.run(steps, self.occu)
comp = sampler_m.samples.mean_composition()
for sp in expected.keys():
    self.assertAlmostEqual(expected[sp], comp[sp], places=2)
sampler_m.clear_samples()

chem_pots = {'Na+': -100.0, 'Cl-': 0.0}
self.msgensemble.chemical_potentials = chem_pots
expected = {'Na+': 0.0, 'Cl-': 1.0}
sampler_m.run(steps, self.occu)
comp = sampler_m.samples.mean_composition()
for sp in expected.keys():
    self.assertAlmostEqual(expected[sp], comp[sp], places=2)
sampler_m.clear_samples()

self.fsgensemble.temperature = 1E9  # go real high to be uniform
sampler_f.run(steps, self.occu)
expected = {'Na+': 0.5, 'Cl-': 0.5}
comp = sampler_f.samples.mean_composition()
for sp in expected.keys():
    self.assertAlmostEqual(expected[sp], comp[sp], places=1)
"""


def test_reshape_occu(ensemble):
    sampler = Sampler.from_ensemble(ensemble, temperature=TEMPERATURE)
    occu = gen_random_occupancy(ensemble.sublattices,
                                ensemble.num_sites)
    assert sampler._reshape_occu(occu).shape == (1, len(occu))

