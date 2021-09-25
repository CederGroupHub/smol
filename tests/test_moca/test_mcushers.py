import pytest
import numpy as np
import random
from copy import deepcopy
from collections import Counter

from scipy.stats import norm

from pymatgen.core import Composition, Species

from smol.cofe.space.domain import SiteSpace
from smol.moca.ensemble.sublattice import Sublattice
from smol.moca.sampler.mcusher import (Swapper, Flipper,
                                       Tableflipper,
                                       Subchainwalker)
from smol.moca.comp_space import CompSpace
from smol.moca.utils.occu_utils import (delta_ccoords_from_step,
                                        occu_to_species_stat,
                                        flip_weights_mask)
from smol.moca.utils.math_utils import GCD_list

from tests.utils import gen_random_neutral_occupancy
from itertools import permutations

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
def sublattices_partial():
    space1 = SiteSpace(Composition({'Li+':0.5,'Mn3+':0.3333333,'Ti4+':0.1666667}))
    space2 = SiteSpace(Composition({'O2-':0.8333333,'P3-':0.1666667}))
    sl1 = Sublattice(space1,np.array([0,1,2,3,4,5]))
    sl1.restrict_sites([0, 3]) # Ti not on these sites.
    sl2 = Sublattice(space2,np.array([6,7,8,9,10,11]))
    sl2.restrict_sites([6, 9]) # P not on these sites.
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

@pytest.fixture
def rand_occu_partial():
    return np.array([0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 0], dtype=int)

@pytest.fixture(params=mcmcusher_classes)
def mcmcusher(request, sublattices):
    # instantiate mcmcushers to test
    return request.param(sublattices)

@pytest.fixture
def tableflipper(sublattices_neutral):
    return Tableflipper(sublattices_neutral, swap_weight = 0)

@pytest.fixture
def partialflipper(sublattices_partial):
    return Tableflipper(sublattices_partial, swap_weight = 0)

@pytest.fixture
def subchainwalker(sublattices_neutral):
    return Subchainwalker(sublattices_neutral, sub_bias_type='square-charge',
                          minimize_swap=True, add_swap=False)

@pytest.fixture
def partialwalker(sublattices_partial):
    return Subchainwalker(sublattices_partial, sub_bias_type='square-charge',
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
        # With weights mask, we should not propose null steps any more.
        assert len(step) > 0
        #print('step:',step)
        for s_id, sp_id in step:
            occu[s_id] = sp_id

def test_flip_partial(partialflipper, rand_occu_partial):

    occu = deepcopy(rand_occu_partial)
    sl_list = [[0,1,2,3,4,5],[6,7,8,9,10,11]]
    bits = partialflipper.bits
    print("Bits:", bits)

    for i in range(30000):
        assert is_neutral(occu,bits,sl_list)
        step = partialflipper.propose_step(occu)
        # With weights mask, we should not propose null steps any more.
        assert len(step) > 0
        assert not any(i in [0, 3, 6, 9] for i, s in step)
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
        hashed += str(int(round(o)))
    return hashed

def dehash_occu(hashed):
    occu = []
    for i in hashed:
        occu.append(int(i))
    return np.array(occu, dtype=np.int64)

# Run MC on a null hamiltonian to check correctness of detail balance.
# This will take some time.
def get_all_neutral_occus(compspace):
    compstats = comp_space.int_grids(sc_size=3, form='compstat')
    occus = []
    for c in compstats:
        occus_an = []
        occus_ca = []
        for perm in permutations(range(3)):
            o = np.zeros(3, dtype=int)
            o[:] = 2
            o[:perm[c[0][0]]] = 0
            o[perm[c[0][0]:(c[0][0]+c[0][1])]] = 1
            o_hash = hash_occu(o)
            if o_hash not in occus_ca:
                occus_ca.append(o_hash)
        for perm in permutations(range(3)):
            o = np.zeros(3, dtype=int)
            o[:] = 1
            o[:perm[c[1][0]]] = 0
            o_hash = hash_occu(o)
            if o_hash not in occus_an:
                occus_an.append(o_hash)
        for o_an in occus_an:
            for o_ca in occus_ca:
                occus.append(o_ca+o_an)
    assert len(occus) == 34
    return occus


def test_mask():
    bits = [[Species('Li',1), Species('Mn',3), Species('Ti',4)],[Species('P',-3), Species('O',-2)]]

    space = CompSpace(bits)
    # Flip 1: Ti+P -> Mn+O
    # Flip 2: Li+Ti+O -> 2Mn+P
    table = space.min_flip_table
    comp_stats = [[[4,0,2],[0,6]],
                  [[3,3,0],[0,6]],
                  [[3,1,2],[2,4]],
                  [[2,4,0],[2,4]],
                  [[3,0,3],[3,3]],
                  [[2,3,1],[3,3]],
                  [[2,0,4],[6,0]],
                  [[1,3,2],[6,0]],
                  [[0,6,0],[6,0]]]
    masks = [[0,0,1,0],
             [0,1,0,0],
             [1,1,1,0],
             [0,1,0,1],
             [1,0,1,0],
             [1,1,1,1],
             [1,0,0,0],
             [1,0,0,1],
             [0,0,0,1]]
    test_masks = [flip_weights_mask(table, c) for c in comp_stats]
    assert np.allclose(masks, test_masks)


def test_neutral_probabilities():

    space1 = SiteSpace(Composition({'Li+':0.5,'Mn3+':0.3333333,'Ti4+':0.1666667}))
    space2 = SiteSpace(Composition({'O2-':0.8333333,'P3-':0.1666667}))
    sl1 = Sublattice(space1,np.array([0,1,2]))
    sl2 = Sublattice(space2,np.array([3,4,5]))

    tableflipper = Tableflipper([sl1, sl2])
    comp_space = tableflipper._compspace
    compstats = comp_space.int_grids(sc_size=3, form='compstat')
    state_counter = {}
    comp = random.choice(compstats)
    occu = compstat_to_random_occu(comp,sl_sizes=[1,1], sc_size=3)
    print("Initial occupancy:", occu)
    for i in range(1000000):
        occu_str = hash_occu(occu)
        if i > 100000:
            if occu_str not in state_counter:
                state_counter[occu_str] = 1
            else:
                state_counter[occu_str] += 1
        step = tableflipper.propose_step(occu)
        factor = tableflipper.compute_a_priori_factor(occu, step)
        #factor = 1
        if np.log(random.random()) < np.log(factor):
            # Accept step
            for sid, sp_to in step:
                occu[sid] = sp_to
    N = sum(list(state_counter.values()))
    sorted_counter = sorted(list(state_counter.items()), key=lambda x: x[1])
    print("Total number of states:", len(state_counter))
    print("Max frequency:", sorted_counter[-1][1])
    print("Max frequency occu:", sorted_counter[-1][0])
    print("Min frequency:", sorted_counter[0][1])
    print("Min frequency occu:", sorted_counter[0][0])
    print("Average frequency:", np.average(list(state_counter.values())))
 
    # Test with zero hamiltonian, assert equal frequency.
    assert len(state_counter) <= 34  # Total 34 possible states.
    assert len(state_counter) > 31
    assert (max(state_counter.values())-min(state_counter.values()))/np.average(list(state_counter.values())) < 0.1

def test_partial_probabilities():

    space1 = SiteSpace(Composition({'Li+':0.5,'Mn3+':0.3333333,'Ti4+':0.1666667}))
    space2 = SiteSpace(Composition({'O2-':0.8333333,'P3-':0.1666667}))
    sl1 = Sublattice(space1,np.array([0,1,2]))
    sl2 = Sublattice(space2,np.array([3,4,5]))
    sl2.restrict_sites([5])

    tableflipper = Tableflipper([sl1, sl2])
    comp_space = tableflipper._compspace
    state_counter = {}
    print("Bits:", tableflipper.bits)

    occu = np.array([2, 2, 2, 0, 0, 0])
    print("Initial occupancy:", occu)
    for i in range(500000):
        occu_str = hash_occu(occu)
        if i > 50000:
            if occu_str not in state_counter:
                state_counter[occu_str] = 1
            else:
                state_counter[occu_str] += 1
        step = tableflipper.propose_step(occu)
        factor = tableflipper.compute_a_priori_factor(occu, step)
        #factor = 1
        if np.log(random.random()) < np.log(factor):
            # Accept step
            for sid, sp_to in step:
                occu[sid] = sp_to
    N = sum(list(state_counter.values()))
    sorted_counter = sorted(list(state_counter.items()), key=lambda x: x[1])
    print("Total number of states:", len(state_counter))
    print("Max frequency:", sorted_counter[-1][1])
    print("Max frequency occu:", sorted_counter[-1][0])
    print("Min frequency:", sorted_counter[0][1])
    print("Min frequency occu:", sorted_counter[0][0])
    print("Average frequency:", np.average(list(state_counter.values())))
 
    # Test with zero hamiltonian, assert equal frequency.
    assert len(state_counter) <= 19  # Total 19 possible states.
    assert len(state_counter) > 17
    assert (max(state_counter.values())-min(state_counter.values()))/np.average(list(state_counter.values())) < 0.1

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

def test_partial_walk(partialwalker, rand_occu_partial):
    comp_change_count = 0
    occu = rand_occu_partial.copy()

    bits = [s.species for s in partialwalker.all_sublattices]
    sl_list = [[0,1,2,3,4,5],[6,7,8,9,10,11]]

    for _ in range(100):
        step = partialwalker.propose_step(occu)
        assert is_neutral(occu, bits, sl_list)
        assert not any(i in [0, 3, 6, 9] for i, s in step)

        if not is_canonical(occu, step, sl_list):
            comp_change_count += 1
        for i, sp in step:
            occu[i] = sp

    assert comp_change_count >= 80
