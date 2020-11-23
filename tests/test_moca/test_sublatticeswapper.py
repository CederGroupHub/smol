import pytest
import json
import numpy as np
from smol.moca.ensemble.sublattice import Sublattice
from smol.moca.sampler.mcusher import Sublatticeswapper
from smol.cofe.space.domain import SiteSpace
from pymatgen.core.composition import Composition

mcmcusher_classes = [Sublatticeswapper]
num_sites = 100

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

@pytest.fixture
def sublattices():
    # generate two tests sublattices
    sites = np.arange(num_sites)
    sites1 = np.random.choice(sites, size=40)
    sites2 = np.setdiff1d(sites, sites1)
    site_space1 = SiteSpace(Composition({'A': 0.1, 'B': 0.4, 'C': 0.3, 'D': 0.2}))
    site_space2 = SiteSpace(Composition({'A': 0.1, 'B': 0.4, 'E': 0.5}))
    sublattices = [Sublattice(site_space1, sites1),
                   Sublattice(site_space2, sites2)]
    return sublattices


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
    # test with 6:3 probability because site_space1 has 3+2+1=6 swaps and
    # site_space2 has 2+1 = 3 swaps. This amounts to probability=6/9 and 3/9
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
    assert count1 / total == pytest.approx(6/9., abs=1E-2)
    assert count2 / total == pytest.approx(3/9., abs=1E-2)

    # check that every site was flipped at least once
    assert all(i in flipped_sites for i in np.arange(num_sites))

    # Now check with a table input with biased input
    counts = []
    for swapType in mcmcusher.swap_table:
        mcmcusher.swap_table[swapType] = np.random.random()
    mcmcusher._normalize_swap_table()
    counts = dict()
    total = 0
    for i in range(iterations):
        mcmcusher.propose_step(rand_occu)
        [sp2,site1, sp1,site2] = mcmcusher.current_flip_info[:4]
        swapA = (sp2,site1)
        swapB = (sp1,site2)
        # put in the swaps now, determine order in the table
        if (swapA,swapB) in mcmcusher.swap_table.keys():
            if (swapA,swapB) not in counts: counts[(swapA,swapB)] = 0
            counts[(swapA,swapB)] += 1
        elif (swapB,swapA) in mcmcusher.swap_table.keys():
            if (swapB,swapA) not in counts: counts[swapB,swapA] = 0
            counts[(swapB, swapA)] += 1
        else:
            raise RuntimeError("Swap type proposed not in swap table")
        total += 1

    # check table-assigned probabilities are sound
    for swapType in counts:
        assert (counts[swapType] / total == pytest.approx(mcmcusher.swap_table[swapType],
                                                          abs = 1E-2))

def generate_rand_occu(sublattices):
    # generate a random occupancy according to the sublattices
    occu = np.zeros(sum(len(s.sites) for s in sublattices), dtype=int)
    for site in range(len(occu)):
        for sublattice in sublattices:
            if site in sublattice.sites:
                occu[site] = np.random.choice(range(len(sublattice.site_space)))
    return occu


def generate_mcmcusher_w_Mn(sublattMn):
    return Sublatticeswapper(sublattMn,
                            allow_crossover = True,
                            swap_table = None,
                            Mn_swap_probability = 0.75)

def test_update_step(mcmcusher, sublattices):
    #check that the occupancy is being updated
    iterations = 50000
    occu = generate_rand_occu(sublattices)
    for _ in range(iterations):
        flip = mcmcusher.propose_step(occu)
        if flip != tuple():
            mcmcusher.update_aux_state(flip)
            (site1, sp2i), (site2, sp1i) = flip
            sublatt1 = mcmcusher._sites_to_sublattice[site1]
            sublatt2 = mcmcusher._sites_to_sublattice[site2]
            sp1 = list(sublatt1)[sp1i]
            sp2 = list(sublatt2)[sp2i]
            assert site1 not in mcmcusher._site_table[sp1][sublatt1]
            assert site1 in mcmcusher._site_table[sp2][sublatt2]
            assert site2 not in mcmcusher._site_table[sp2][sublatt2]
            assert site2 in mcmcusher._site_table[sp1][sublatt1]
            for f in flip:
                occu[f[0]] = f[1]
            assert occu[site1] != sp1i
            assert occu[site2] != sp2i


def test_Mn():
    with open('sublattices.json', 'r') as fin:
        sublattMn = json.load(fin)
    sublattMn = [Sublattice.from_dict(i) for i in sublattMn]
    occu = np.load('occu.npy')
    mcmcusher = generate_mcmcusher_w_Mn(sublattMn)
    iterations = 50000
    for _ in range(iterations):
        flip = mcmcusher.propose_step(occu)
        if flip != tuple():
            (site1, sp2i), (site2, sp1i) = flip
            for f in flip:
                occu[f[0]] = f[1]
            mcmcusher.update_aux_state(flip)

            # now test the flip is valid
            sublatt1 = mcmcusher._sites_to_sublattice[site1]
            sublatt2 = mcmcusher._sites_to_sublattice[site2]
            old_sp1 = mcmcusher.old_sp1
            old_sp2 = mcmcusher.old_sp2
            sp1 = list(sublatt1)[sp1i]
            sp2 = list(sublatt1)[sp2i]
            assert site1 not in mcmcusher._site_table[old_sp1][sublatt1]
            assert site1 in mcmcusher._site_table[sp2][sublatt2]
            assert site2 not in mcmcusher._site_table[old_sp2][sublatt2]
            assert site2 in mcmcusher._site_table[sp1][sublatt1]



