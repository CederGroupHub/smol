import numpy as np
import pytest
from pymatgen.core import Composition

from smol.cofe.space.domain import SiteSpace
from smol.moca.sampler.mcusher import Flip, Swap
from smol.moca.sublattice import InactiveSublattice, Sublattice

mcmcusher_classes = [Flip, Swap]
num_sites = 100


@pytest.fixture
def all_sublattices():
    # generate two tests sublattices
    sites = np.arange(num_sites)
    rng = np.random.default_rng()
    sites1 = rng.choice(sites, size=num_sites // 3)
    sites2 = rng.choice(np.setdiff1d(sites, sites1), size=num_sites // 4)
    sites3 = np.setdiff1d(sites, np.concatenate((sites1, sites2)))
    site_space1 = SiteSpace(Composition({"A": 0.1, "B": 0.4, "C": 0.3, "D": 0.2}))
    site_space2 = SiteSpace(Composition({"A": 0.1, "B": 0.4, "E": 0.5}))
    site_space3 = SiteSpace(Composition({"G": 1}))
    sublattices = [Sublattice(site_space1, sites1), Sublattice(site_space2, sites2)]
    inactive_sublattices = [
        InactiveSublattice(site_space3, sites3),
    ]
    return sublattices, inactive_sublattices


@pytest.fixture
def rand_occu(all_sublattices):
    # generate a random occupancy according to the sublattices
    rng = np.random.default_rng()
    occu = np.zeros(
        sum(len(s.sites) for sublat in all_sublattices for s in sublat), dtype=int
    )
    for site in range(len(occu)):
        for sublattice in all_sublattices[0]:
            if site in sublattice.sites:
                occu[site] = rng.choice(range(len(sublattice.site_space)))
    return occu, all_sublattices[1][0].sites  # return indices of fixed sites


@pytest.fixture(params=mcmcusher_classes)
def mcmcusher(request, all_sublattices):
    # instantiate mcmcushers to test
    return request.param(*all_sublattices)


def test_bad_propabilities(mcmcusher):
    with pytest.raises(ValueError):
        mcmcusher.sublattice_probabilities = [0.6, 0.1]
    with pytest.raises(AttributeError):
        mcmcusher.sublattice_probabilities = [0.5, 0.2, 0.3]


def test_propose_step(mcmcusher, rand_occu):
    occu, fixed_sites = rand_occu
    iterations = 50000
    # test with 50/50 probability
    flipped_sites = []
    count1, count2 = 0, 0
    total = 0
    for i in range(iterations):
        step = mcmcusher.propose_step(occu)
        for flip in step:
            if flip[0] in mcmcusher.sublattices[0].sites:
                count1 += 1
                assert flip[1] in range(len(mcmcusher.sublattices[0].site_space))
            elif flip[0] in mcmcusher.sublattices[1].sites:
                count2 += 1
                assert flip[1] in range(len(mcmcusher.sublattices[1].site_space))
            else:
                raise RuntimeError(
                    "Something went wrong in proposing"
                    f"a step site proposed in {step} is"
                    " not in any of the allowed sites"
                )
            total += 1
            flipped_sites.append(flip[0])

    # check probabilities seem sound
    assert count1 / total == pytest.approx(0.5, abs=1e-2)
    assert count2 / total == pytest.approx(0.5, abs=1e-2)

    # check that every site was flipped at least once
    assert all(
        i in flipped_sites for i in np.setdiff1d(np.arange(num_sites), fixed_sites)
    )

    # make sure fixed sites remain the same
    assert all(i not in fixed_sites for i in flipped_sites)

    # Now check with a sublattice bias
    mcmcusher.sublattice_probabilities = [0.8, 0.2]
    flipped_sites = []
    count1, count2 = 0, 0
    total = 0
    for i in range(iterations):
        step = mcmcusher.propose_step(occu)
        for flip in step:
            if flip[0] in mcmcusher.sublattices[0].sites:
                count1 += 1
                assert flip[1] in range(len(mcmcusher.sublattices[0].site_space))
            elif flip[0] in mcmcusher.sublattices[1].sites:
                count2 += 1
                assert flip[1] in range(len(mcmcusher.sublattices[1].site_space))
            else:
                raise RuntimeError(
                    "Something went wrong in proposing"
                    f"a step site proposed in {step} is"
                    " not in any of the allowed sites"
                )
            total += 1
            flipped_sites.append(flip[0])
    assert count1 / total == pytest.approx(0.8, abs=1e-2)
    assert count2 / total == pytest.approx(0.2, abs=1e-2)
