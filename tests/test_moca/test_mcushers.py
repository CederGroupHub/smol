import pytest
import numpy as np
from pymatgen import Composition,Specie
from smol.cofe.space.domain import SiteSpace
from smol.moca.ensemble.sublattice import Sublattice
from smol.moca.sampler.mcusher import (Swapper, Flipper, 
                                       Combinedflipper,
                                       Chargeneutralflipper)
from smol.moca.comp_space import CompSpace
from smol.moca.utils.occu_utils import direction_from_step

mcmcusher_classes = [Flipper, Swapper]
num_sites = 100


@pytest.fixture
def sublattices():
    # generate two tests sublattices
    sites = np.arange(num_sites)
    sites1 = np.random.choice(sites, size=40)
    sites2 = np.setdiff1d(sites, sites1)
    site_space1 = SiteSpace(
        Composition({'A': 0.1, 'B': 0.4, 'C': 0.3, 'D': 0.2}))
    site_space2 = SiteSpace(
        Composition({'A': 0.1, 'B': 0.4, 'E': 0.5}))
    sublattices = [Sublattice(site_space1, sites1),
                   Sublattice(site_space2, sites2)]
    return sublattices

@pytest.fixture
def sublattices_neutral():
    space1 = SiteSpace(Composition({'Li+':0.5,'Mn3+':0.3333333,'Ti4+':0.1666667}))
    space2 = SiteSpace(Composition({'O2-':0.8333333,'P3-':0.1666667}))
    sl1 = Sublattice(space1,np.array([0,1,2,3,4,5]))
    sl2 = Sublattice(space2,np.array([6,7,8,9,10,11]))
    return [sl1,sl2]

@pytest.fixture
def flip_combinations():
    li = Specie.from_string('Li+')
    mn = Specie.from_string('Mn3+')
    ti = Specie.from_string('Ti4+')
    p = Specie.from_string('P3-')
    o = Specie.from_string('O2-')
    bits = [[li,mn,ti],[p,o]]
    return CompSpace(bits).min_flips


@pytest.fixture
def rand_occu(sublattices):
    # generate a random occupancy according to the sublattices
    occu = np.zeros(sum(len(s.sites) for s in sublattices), dtype=int)
    for site in range(len(occu)):
        for sublattice in sublattices:
            if site in sublattice.sites:
                occu[site] = np.random.choice(range(len(sublattice.site_space)))
    return occu


@pytest.fixture(params=mcmcusher_classes)
def mcmcusher(request, sublattices):
    # instantiate mcmcushers to test
    return request.param(sublattices)


def test_bad_propabilities(mcmcusher):
    with pytest.raises(ValueError):
        mcmcusher.sublattice_probabilities = [0.6, 0.1]
    with pytest.raises(AttributeError):
        mcmcusher.sublattice_probabilities = [0.5, 0.2, 0.3]


def test_propose_step(mcmcusher, rand_occu):
    iterations = 50000
    # test with 50/50 probability
    flipped_sites = []
    count1, count2 = 0, 0
    total = 0
    for i in range(iterations):
        step = mcmcusher.propose_step(rand_occu)
        for flip in step:
            if flip[0] in mcmcusher.sublattices[0].sites:
                count1 += 1
                assert flip[1] in range(len(mcmcusher.sublattices[0].site_space))
            elif flip[0] in mcmcusher.sublattices[1].sites:
                count2 += 1
                assert flip[1] in range(len(mcmcusher.sublattices[1].site_space))
            else:
                raise RuntimeError('Something went wrong in proposing'
                                   f'a step site proposed in {step} is'
                                   ' not in any of the allowed sites')
            total += 1
            flipped_sites.append(flip[0])

    # check probabilities seem sound
    assert count1 / total == pytest.approx(0.5, abs=1E-2)
    assert count2 / total == pytest.approx(0.5, abs=1E-2)

    # check that every site was flipped at least once
    assert all(i in flipped_sites for i in np.arange(num_sites))

    # Now check with a sublattice bias
    mcmcusher.sublattice_probabilities = [0.8, 0.2]
    flipped_sites = []
    count1, count2 = 0, 0
    total = 0
    for i in range(iterations):
        step = mcmcusher.propose_step(rand_occu)
        for flip in step:
            if flip[0] in mcmcusher.sublattices[0].sites:
                count1 += 1
                assert flip[1] in range(len(mcmcusher.sublattices[0].site_space))
            elif flip[0] in mcmcusher.sublattices[1].sites:
                count2 += 1
                assert flip[1] in range(len(mcmcusher.sublattices[1].site_space))
            else:
                raise RuntimeError('Something went wrong in proposing'
                                   f'a step site proposed in {step} is'
                                   ' not in any of the allowed sites')
            total += 1
            flipped_sites.append(flip[0])
    assert count1 / total == pytest.approx(0.8, abs=1E-2)
    assert count2 / total == pytest.approx(0.2, abs=1E-2)


def test_combined_flip(sublattices_neutral,flip_combinations):
    tf = Combinedflipper(sublattices_neutral,flip_combinations)
    occu = np.array([0,2,0,1,1,0,1,1,1,0,1,1])
    
    count_directions = {-1:0,1:0,-2:0,2:0}
    for i in range(28000):
        step = tf.propose_step(occu)
        #print('Proposal number {}:'.format(i),step)
        direction = direction_from_step(step,tf.sl_list,tf.flip_combs)
        count_directions[direction]+=1
    
    true_directions = {-1:10000,1:1000,-2:2000,2:15000}
    
    for d in count_directions:
        assert(abs(true_directions[d]-count_directions[d])/true_directions[d]<0.1)

def test_charge_neutral_flip(sublattices_neutral):
    cf = Chargeneutralflipper(sublattices_neutral)
    assert cf.n_links == 180
