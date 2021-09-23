import pytest
import numpy as np
import numpy.testing as npt

from pymatgen.core import Structure, Lattice, Composition
from smol.moca import (CanonicalEnsemble, FuSemiGrandEnsemble,
                       MuSemiGrandEnsemble, 
                       DiscChargeNeutralSemiGrandEnsemble,
                       CEProcessor, Sampler)
from smol.moca.sampler.mcusher import (Swapper, Flipper,
                                       Tableflipper)
from smol.moca.sampler.kernel import Metropolis
from smol.moca.comp_space import CompSpace
from smol.moca.ensemble.sublattice import get_all_sublattices, Sublattice
from smol.cofe import ClusterSubspace, ClusterExpansion
from smol.cofe.space.domain import SiteSpace

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
        return np.vstack([gen_random_neutral_occupancy(sampler.mcmckernel._usher.all_sublattices,
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
        assert isinstance(sampler.mcmckernel._usher, Tableflipper)
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

def test_partial():
    space1 = SiteSpace(Composition({'Li+':0.5,'Mn3+':0.3333333,'Ti4+':0.1666667}))
    space2 = SiteSpace(Composition({'O2-':0.8333333,'P3-':0.1666667}))
    sl1 = Sublattice(space1,np.array([0,1,2]))
    sl2 = Sublattice(space2,np.array([3,4,5]))
    sl2.restrict_sites([5])

    lat = Lattice.from_parameters(1,1,1,60,60,60)
    prim = Structure(lat, [{'Li+':0.5,'Mn3+':0.3333333,'Ti4+':0.1666667}, {'O2-':0.8333333,'P3-':0.1666667}], [[0,0,0],[0.5,0.5,0.5]])
    cutoffs = {2: 2.0, 3: 1.2, 4:1.2}
    cspace = ClusterSubspace.from_cutoffs(prim, cutoffs=cutoffs)
    coefs = np.zeros(cspace.num_corr_functions)
    ce = ClusterExpansion(cspace, coefs)

    ca_ens = CanonicalEnsemble.from_cluster_expansion(ce, [[3,0,0],[0,1,0],[0,0,1]], optimize_indicator=True, sublattices=[sl1, sl2])
    #print("Sublattices:", [sl1, sl2])
    #print("Ensemble sublattices:", ca_ens.all_sublattices)
    for s in ca_ens.all_sublattices:
        if sl1.species == s.species:
            assert np.all(sl1.active_sites == s.active_sites)
            assert np.all(sl1.sites == s.sites)
        elif sl2.species == s.species:
            assert np.all(sl2.active_sites == s.active_sites)
            assert np.all(sl2.sites == s.sites)
        else:
            raise ValueError("A sublattice {} in ensemble not found in original sublattices!".format(s))

    chempots = {'Li+':0, 'Mn3+':0, 'Ti4+':0, 'P3-':0, 'O2-':0}
    gc_ens = MuSemiGrandEnsemble.from_cluster_expansion(ce, [[3,0,0],[0,1,0],[0,0,1]], optimize_indicator=True,
                                                        chemical_potentials=chempots, sublattices=[sl1, sl2])

    ca_sampler = Sampler.from_ensemble(ca_ens, temperature=4000)
    gc_sampler = Sampler.from_ensemble(gc_ens, temperature=4000, step_type='table-flip', swap_weight=0)
    ca_actives = []
    gc_actives = []
    for s in ca_sampler._kernel._usher.sublattices:
        ca_actives.extend(list(s.active_sites))
    for s in gc_sampler._kernel._usher.sublattices:
        gc_actives.extend(list(s.active_sites))
    assert set(ca_actives) == set([0, 1, 2, 3, 4])
    assert set(gc_actives) == set([0, 1, 2, 3, 4])

    def count_species(o):
        return np.array([np.sum(o[:3]==0), np.sum(o[:3]==1), np.sum(o[:3]==2),
                         np.sum(o[3:]==0), np.sum(o[3:]==1)])
    ca_sampler.run(1000, initial_occupancies=np.array([[0, 1, 2, 0, 1, 0]]))
    gc_sampler.run(1000, initial_occupancies=np.array([[0, 1, 2, 0, 1, 0]]))

    ca_occus = np.array(ca_sampler.samples.get_occupancies(), dtype=int)
    gc_occus = np.array(gc_sampler.samples.get_occupancies(), dtype=int)

    gc_accept = gc_sampler.samples._accepted[:,0]
    for o in ca_occus:
        assert np.all(count_species(ca_occus[0]) == count_species(o))
        assert o[5] == 0

    assert np.sum(gc_accept) > 0
    for oid, o in enumerate(gc_occus):
        if oid>0:
            if np.all(count_species(gc_occus[oid]) == count_species(gc_occus[oid-1])):
                assert gc_accept[oid] == 0
            assert o[5] == 0
