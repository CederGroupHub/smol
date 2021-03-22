import pytest
import numpy as np
import random
from copy import deepcopy
from pymatgen import Composition,Specie
from smol.cofe.space.domain import SiteSpace
from smol.moca.ensemble.sublattice import Sublattice
from smol.moca.sampler.mcusher import (Swapper, Flipper,
                                       Chargeneutralflipper)
from smol.moca.comp_space import CompSpace
from smol.moca.utils.occu_utils import delta_ccoords_from_step
from smol.moca.utils.math_utils import GCD_list

mcmcusher_classes = [Flipper, Swapper]
num_sites = 100

def gen_random_neutral_occupancy(sublattices, num_sites):
    """Generate charge neutral occupancies according to a list of sublattices.
       Occupancies are encoded.

    Args:
        sublattices (Sequence of Sublattice):
           A sequence of sublattices
        num_sites (int):
           Total number of sites

    Returns:
        ndarray: encoded occupancy, charge neutral guaranteed.
    """
    rand_occu = np.zeros(num_sites, dtype=int)
    bits = [sl.species for sl in sublattices]
    sl_sizes = [len(sl.sites) for sl in sublattices]
    sc_size = GCD_list(sl_sizes)

    sl_sizes_prim = np.array(sl_sizes)//sc_size
    comp_space = CompSpace(bits,sl_sizes_prim)

    random_comp = random.choice(comp_space.int_grids(sc_size=sc_size,form='compstat'))

    sites = []
    assignments = []
    for sl,sl_comp in zip(sublattices,random_comp):
        sl_sites = list(sl.sites)
        random.shuffle(sl_sites)
        sites.extend(sl_sites)
        for sp_id,sp_n in enumerate(sl_comp):
            assignments.extend([sp_id for i in range(sp_n)])

    rand_occu[sites] = assignments
    return rand_occu

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
def cnflipper(sublattices_neutral):
    return Chargeneutralflipper(sublattices_neutral)

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

def test_flip_neutral(cnflipper, rand_occu_neutral):
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

    occu = deepcopy(rand_occu_neutral)
    sl_list = [[0,1,2,3,4,5],[6,7,8,9,10,11]]
    bits = cnflipper.bits

    for i in range(30000):
        assert is_neutral(occu,bits,sl_list)
        step = cnflipper.propose_step(occu)
        #print('step:',step)
        for s_id, sp_id in step:
            occu[s_id] = sp_id

def test_equal_a_priori_neutral(cnflipper, rand_occu_neutral):
    def is_canonical(occu,step,sublat_list):
        if len(step)==0:
            return False
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

    occu = deepcopy(rand_occu_neutral)
    sl_list = [[0,1,2,3,4,5],[6,7,8,9,10,11]]
    bits = cnflipper.bits
    bias_values = []
    for i in range(5):
        print('Occu:',occu)

        proposed_steps = [cnflipper.propose_step(occu) for i in range(80000)]
        while True:
            chosen_step = random.choice(proposed_steps)
            if not is_canonical(occu,chosen_step,sl_list):
                break
        reverse_step = [(s_id,occu[s_id]) for s_id,sp_id in chosen_step]
        chosen_count = 0
        for step in proposed_steps:
            if set(step) == set(chosen_step):
                chosen_count += 1
        
        print('chosen_step:',chosen_step)

        next_occu = deepcopy(occu)
        for s_id, sp_id in chosen_step:
            next_occu[s_id] = sp_id

        proposed_rev_steps = [cnflipper.propose_step(next_occu) for i in range(80000)]
        reverse_count = 0
        for step in proposed_rev_steps:
            if set(step) == set(reverse_step):
                reverse_count += 1

        assert abs(chosen_count-reverse_count)/chosen_count <= 0.15
      
        occu = next_occu # Move forward

        bias_values.append((chosen_count-reverse_count)/chosen_count)

    assert not(np.all(np.array(bias_values)>0)) and not(np.all(np.array(bias_values)<0))
