import pytest
import numpy as np
from pymatgen import Composition
from smol.cofe.space.domain import SiteSpace
from smol.moca.ensemble.sublattice import Sublattice
from smol.moca.sampler.mcusher import Swapper, Flipper

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
