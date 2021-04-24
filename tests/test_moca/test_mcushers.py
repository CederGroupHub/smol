import pytest
import numpy as np
import random
from copy import deepcopy
from collections import Counter

from scipy.stats import norm

from pymatgen import Composition,Specie
from smol.cofe.space.domain import SiteSpace
from smol.moca.ensemble.sublattice import Sublattice
from smol.moca.sampler.mcusher import (Swapper, Flipper,
                                       Tableflipper,
                                       Subchainwalker)
from smol.moca.comp_space import CompSpace
from smol.moca.utils.occu_utils import (delta_ccoords_from_step,
                                        occu_to_species_stat)
from smol.moca.utils.math_utils import GCD_list

from tests.utils import gen_random_neutral_occupancy

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
def rand_occu(sublattices):
    # generate a random occupancy according to the sublattices
    occu = np.zeros(sum(len(s.sites) for s in sublattices), dtype=int)
    for site in range(len(occu)):
        for sublattice in sublattices:
            if site in sublattice.sites:
                occu[site] = np.random.choice(range(len(sublattice.site_space)))
    return occu

@pytest.fixture
def rand_occu_neutral(sublattices_neutral):
    return gen_random_neutral_occupancy(sublattices_neutral, 12)

@pytest.fixture(params=mcmcusher_classes)
def mcmcusher(request, sublattices):
    # instantiate mcmcushers to test
    return request.param(sublattices)

@pytest.fixture
def tableflipper(sublattices_neutral):
    return Tableflipper(sublattices_neutral, add_swap = False)

@pytest.fixture
def subchainwalker(sublattices_neutral):
    return Subchainwalker(sublattices_neutral, sub_bias_type='square-charge',
                          minimize_swap=True, add_swap=False)


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


def get_oxi_state(sp):
    if 'oxi_state' in dir(sp):
        return sp.oxi_state
    else:
        return 0

def is_neutral(occu,bits,sublat_list):
    charge = 0
    for sl_id,sl in enumerate(sublat_list):
        charge += sum([get_oxi_state(bits[sl_id][sp_id]) for sp_id in occu[sl]])
    return charge == 0

def test_flip_neutral(tableflipper, rand_occu_neutral):

    occu = deepcopy(rand_occu_neutral)
    sl_list = [[0,1,2,3,4,5],[6,7,8,9,10,11]]
    bits = tableflipper.bits

    for i in range(30000):
        assert is_neutral(occu,bits,sl_list)
        step = tableflipper.propose_step(occu)
        #print('step:',step)
        for s_id, sp_id in step:
            occu[s_id] = sp_id

def is_canonical(occu,step,sublat_list):
    if len(step)==0:
        return True
    if len(step)!=2:
        return False
    if (occu[step[0][0]] == step[1][1] and
        occu[step[1][0]] == step[0][1]):
        sl_id_0 = None
        sl_id_1 = None
        for sl_id, sl in enumerate(sublat_list):
            if step[0][0] in sl:
                sl_id_0 = sl_id
            if step[1][0] in sl:
                sl_id_1 = sl_id
        if sl_id_0 == sl_id_1:
            return True
    return False

def compstat_to_random_occu(compstat, sl_sizes, sc_size):
    random_occu = []
    for sl_comp, sl_size in zip(compstat, sl_sizes):
        sl_occu = np.zeros(sl_size * sc_size, dtype=np.int64)
        if sum(sl_comp) != sl_size * sc_size:
            raise ValueError("Number of sublattice species does\
                              not sum to sublattice size")
        shuffled_ids = list(range(sl_size*sc_size))
        random.shuffle(shuffled_ids)
        n_assigned = 0
        for sp, n in enumerate(sl_comp):
            sl_occu[shuffled_ids[n_assigned: n_assigned + n]] = sp
            n_assigned += n
        random_occu.append(sl_occu)
    return np.array(random_occu, dtype=np.int64).flatten()

def hash_occu(occu):
    hashed = ''
    for o in occu:
        hashed += str(round(o))
    return hashed

def dehash_occu(hashed):
    occu = []
    for i in hashed:
        occu.append(int(i))
    return np.array(occu, dtype=np.int64)

# This takes a long time.
def test_neutral_probabilities(tableflipper, sublattices_neutral):

    sl_list = [[0,1,2,3,4,5],[6,7,8,9,10,11]]
    bits = tableflipper.bits
    sl_sizes = [1,1]
    comp_space = CompSpace(bits, sl_sizes)
    compstats = comp_space.int_grids(sc_size=6, form='compstat')
    counters = []
    for comp in compstats:
        sub_counters = []
        for _ in range(3):
            occu = compstat_to_random_occu(comp, sl_sizes, 6)
            next_occus = []
            for _ in range(72000):
                next_occu = occu.copy()
                step = tableflipper.propose_step(occu)
                for i, sp in step:
                    next_occu[i] = sp
                next_occus.append(hash_occu(next_occu))
            sub_counters.append(Counter(next_occus))
        counters.append(sub_counters)

    for comp, sub_counters in zip(compstats, counters):
        direction_1_bias = []
        direction_2_bias = []
        for counter in sub_counters:
            sub_1_bias = []
            sub_2_bias = []
            for hashed_occu in counter:
                print('start_comp:', comp)
                print('hashed_occu:', hashed_occu)
                count = counter[hashed_occu]
                print('count:', count)
                occu = dehash_occu(hashed_occu)
                stat = occu_to_species_stat(occu, bits, sl_list)
                ccoords_next = comp_space.translate_format(stat, from_format='compstat',
                                                           to_format='constr',
                                                           sc_size=6)
                ccoords_prev = comp_space.translate_format(comp, from_format='compstat',
                                                           to_format='constr',
                                                           sc_size=6)
                if np.allclose(np.abs(ccoords_next-ccoords_prev), [1,0]):
                    # Mn + O -> Ti + P
                    print('direction: 1')
                    direction_1_bias.append((count/72000 - 0.5/(sum(comp[0][1:])*6))/(0.5/(sum(comp[0][1:])*6)))
                    sub_1_bias.append((count/72000 - 0.5/(sum(comp[0][1:])*6))/(0.5/(sum(comp[0][1:])*6)))
                elif np.allclose(np.abs(ccoords_next-ccoords_prev), [0,1]):
                    print('direction: 2')
                    direction_2_bias.append((count/72000 - 0.5/180)/(0.5/180))
                    sub_2_bias.append((count/72000 - 0.5/180)/(0.5/180))

            if len(sub_1_bias)==0:
                sub_1_bias = [0]
            if len(sub_2_bias)==0:
                sub_2_bias = [0]

            assert np.sqrt(np.var(sub_1_bias))<0.1  #No occupation bias difference
            assert np.sqrt(np.var(sub_2_bias))<0.15 

        if len(direction_1_bias) == 0:
            direction_1_bias = [0]
        if len(direction_2_bias) == 0:
            direction_2_bias = [0]

        assert abs(np.average(direction_1_bias)) < 0.1
        assert abs(np.average(direction_2_bias)) < 0.15
        assert not(np.all(np.array(direction_1_bias)<0)) and not(np.all(np.array(direction_1_bias)>0))
        assert not(np.all(np.array(direction_2_bias)<0)) and not(np.all(np.array(direction_2_bias)>0))


def test_subchain_walk(subchainwalker, rand_occu_neutral):
    comp_change_count = 0
    occu = rand_occu_neutral.copy()

    bits = [s.species for s in subchainwalker.all_sublattices]
    sl_list = [[0,1,2,3,4,5],[6,7,8,9,10,11]]

    for _ in range(100):
        step = subchainwalker.propose_step(occu)
        assert is_neutral(occu, bits, sl_list)

        if not is_canonical(occu, step, sl_list):
            comp_change_count += 1
        for i, sp in step:
            occu[i] = sp

    assert comp_change_count >= 80
