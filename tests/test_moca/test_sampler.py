import pytest
import numpy as np
import numpy.testing as npt

from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion
from smol.moca import (CanonicalEnsemble, FuSemiGrandEnsemble,
                       MuSemiGrandEnsemble, CEProcessor, MetropolisSampler)
from smol.moca.sampler.mcusher import Swapper, Flipper
from tests.utils import gen_random_occupancy

ensembles = [CanonicalEnsemble, MuSemiGrandEnsemble, FuSemiGrandEnsemble]


@pytest.fixture(params=ensembles, scope='module')
def ensemble(structure, request):
    subspace = ClusterSubspace.from_radii(structure,
                                          radii={2: 6, 3: 5, 4: 4})
    coefs = np.random.random(subspace.n_bit_orderings)
    proc = CEProcessor(subspace, 4*np.eye(3), coefs)
    if request.param is MuSemiGrandEnsemble:
        kwargs = {'chemical_potentials':
                  {sp: 0.3 for space in proc.unique_site_spaces
                   for sp in space.keys()}}
    else:
        kwargs = {}
    return request.param(proc, temperature=5000, **kwargs)


@pytest.fixture(scope='module', params=[1, 5])
def sampler(ensemble, request):
    return MetropolisSampler(ensemble, nwalkers=request.param)


def test_constructor(sampler):
    if isinstance(sampler.ensemble, CanonicalEnsemble):
        assert isinstance(sampler._usher, Swapper)
    else:
        assert isinstance(sampler._usher, Flipper)


def test_attempt_step(sampler):
    occu_ = gen_random_occupancy(sampler.ensemble.sublattices,
                                 sampler.ensemble.num_sites)
    for _ in range(20):
        init_occu = occu_.copy()
        acc, occu, denth, dfeat = sampler._attempt_step(init_occu)
        if acc:
            assert not np.array_equal(occu, occu_)
        else:
            npt.assert_array_equal(occu, occu_)


@pytest.mark.parametrize('thin', (1, 10))
def test_sample(sampler, thin):
    occu = np.vstack([gen_random_occupancy(sampler.ensemble.sublattices,
                                           sampler.ensemble.num_sites)
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
    occu = np.vstack([gen_random_occupancy(sampler.ensemble.sublattices,
                                           sampler.ensemble.num_sites)
                      for _ in range(sampler.samples.shape[0])])
    steps = 1000
    sampler.run(steps, occu, thin_by=thin)
    assert len(sampler.samples) == steps // thin
    assert 0 < sampler.efficiency() <= 1
    sampler.clear_samples()

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
    sampler = MetropolisSampler(ensemble)
    occu = gen_random_occupancy(sampler.ensemble.sublattices,
                                sampler.ensemble.num_sites)
    assert sampler._reshape_occu(occu).shape == (1, len(occu))

