import pytest
import numpy as np
import numpy.testing as npt
from smol.moca import (CanonicalEnsemble, FuSemiGrandEnsemble,
                       MuSemiGrandEnsemble, 
                       DiscChargeNeutralSemiGrandEnsemble,
                       CEProcessor, Sampler)
from smol.moca.sampler.mcusher import (Swapper, Flipper,
                                       Chargeneutralflipper)
from smol.moca.sampler.kernel import Metropolis
from smol.moca.comp_space import CompSpace
from smol.moca.ensemble.sublattice import get_all_sublattices

from tests.utils import (gen_random_occupancy,
                         gen_random_neutral_occupancy)

ensembles = [CanonicalEnsemble, MuSemiGrandEnsemble, FuSemiGrandEnsemble,
             DiscChargeNeutralSemiGrandEnsemble]
TEMPERATURE = 5000

def initialize_occus(sampler):
    if not('Disc' in sampler.samples.metadata["name"]):
        return np.vstack([gen_random_occupancy(sampler.mcmckernel._usher.sublattices,
                                               sampler.samples.num_sites)
                          for _ in range(sampler.samples.shape[0])])
    else:
        return np.vstack([gen_random_neutral_occupancy(sampler.mcmckernel._usher.sublattices,
                                                  sampler.samples.num_sites)
                          for _ in range(sampler.samples.shape[0])])

@pytest.fixture(params=ensembles)
def ensemble(cluster_subspace, request):
    coefs = np.random.random(cluster_subspace.num_corr_functions)
    proc = CEProcessor(cluster_subspace, 4*np.eye(3), coefs)
    if request.param is MuSemiGrandEnsemble:
        kwargs = {'chemical_potentials':
                  {sp: 0.3 for space in proc.unique_site_spaces
                   for sp in space.keys()}}
    elif request.param is DiscChargeNeutralSemiGrandEnsemble:    

        sublattices = get_all_sublattices(proc)
    
        bits = [sl.species for sl in sublattices]
        sl_sizes = [len(sl.sites) for sl in sublattices]
    
        comp_space = CompSpace(bits,sl_sizes)
        kwargs = {'mu':[0.3 for i in range(comp_space.dim)]}

    else:
        kwargs = {}
    return request.param(proc, **kwargs)

@pytest.fixture(params=[1, 5])
def sampler(ensemble, request):
    sampler = Sampler.from_ensemble(ensemble, temperature=TEMPERATURE,
                                    nwalkers=request.param)

    return sampler

def test_from_ensemble(sampler):
    if "Canonical" in sampler.samples.metadata["name"]:
        assert isinstance(sampler.mcmckernel._usher, Swapper)
    elif "Disc" in sampler.samples.metadata["name"]:
        assert isinstance(sampler.mcmckernel._usher, Chargeneutralflipper)
    else:
        assert isinstance(sampler.mcmckernel._usher, Flipper)
    assert isinstance(sampler.mcmckernel, Metropolis)


@pytest.mark.parametrize('thin', (1, 10))
def test_sample(sampler, thin):
    occu = initialize_occus(sampler)
    steps = 1000
    samples = [state for state
               in sampler.sample(1000, occu, thin_by=thin)]
    assert len(samples) == steps // thin

    it = sampler.sample(43, occu, thin_by=7)
    with pytest.warns(RuntimeWarning):
        next(it)


@pytest.mark.parametrize('thin', (1, 10))
def test_run(sampler, thin):
    occu = initialize_occus(sampler)
    steps = 1000
    sampler.run(steps, occu, thin_by=thin)
    assert len(sampler.samples) == steps // thin
    assert 0 < sampler.efficiency() <= 1
    sampler.clear_samples()


def test_anneal(sampler):
    temperatures = np.linspace(2000, 500, 5)
    occu = initialize_occus(sampler)
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
    occu = initialize_occus(sampler)[0]
    assert sampler._reshape_occu(occu).shape == (1, len(occu))

