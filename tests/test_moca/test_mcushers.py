import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.core import Composition
from collections import Counter

from smol.cofe.space.domain import SiteSpace
from smol.moca.sampler.mcusher import Flip, Swap, Tableflip
from smol.moca.sublattice import Sublattice
from smol.moca.sampler.bias import SquarechargeBias
from smol.moca.utils.math_utils import comb

from tests.utils import gen_random_occupancy, gen_random_neutral_occupancy

mcmcusher_classes = [Flip, Swap, Tableflip]
num_sites = 100


@pytest.fixture
def all_sublattices(rng):
    # generate two tests sublattices
    sites = np.arange(num_sites)
    sites1 = rng.choice(sites, size=num_sites // 3)
    sites2 = rng.choice(np.setdiff1d(sites, sites1), size=num_sites // 4)
    sites3 = np.setdiff1d(sites, np.concatenate((sites1, sites2)))
    site_space1 = SiteSpace(Composition({"A": 0.1, "B": 0.4, "C": 0.3, "D": 0.2}))
    site_space2 = SiteSpace(Composition({"A": 0.1, "B": 0.4, "E": 0.5}))
    site_space3 = SiteSpace(Composition({"G": 1}))
    active_sublattices = [
        Sublattice(site_space1, sites1),
        Sublattice(site_space2, sites2),
    ]
    inactive_sublattices = [Sublattice(site_space3, sites3)]
    return active_sublattices, inactive_sublattices


@pytest.fixture
def all_sublattices_lmtpo():  # Do a test on sampling probabilities.
    # generate two tests sublattices
    sites = np.arange(6, dtype=int)
    sites1 = np.random.choice(sites, size=3, replace=False)
    sites2 = np.setdiff1d(sites, sites1)
    site_space1 = SiteSpace(Composition({"Li+": 2 / 3, "Zr4+": 1 / 6, "Mn3+": 1 / 6}))
    site_space2 = SiteSpace(Composition({"O2-": 5 / 6, "F-": 1 / 6}))
    active_sublattices = [Sublattice(site_space1, sites1), Sublattice(site_space2, sites2)]
    inactive_sublattices = []
    return active_sublattices, inactive_sublattices


@pytest.fixture
def rand_occu(all_sublattices):
    # generate a random occupancy according to the sublattices
    occu = gen_random_occupancy(all_sublattices[0] + all_sublattices[1])
    return occu, all_sublattices[1][0].sites  # return indices of fixed sites


@pytest.fixture
def rand_occu_lmtpo(all_sublattices_lmtpo):
    # generate a random occupancy according to the sublattices
    occu = gen_random_neutral_occupancy(all_sublattices_lmtpo[0] + all_sublattices_lmtpo[1])
    return occu, []  # return indices of fixed sites


@pytest.fixture(params=mcmcusher_classes)
def mcmcusher(request, all_sublattices):
    # instantiate mcmcushers to test
    if request.param == Tableflip:
        return request.param(all_sublattices[0] + all_sublattices[1], swap_weight=0)
    return request.param(all_sublattices[0] + all_sublattices[1])


@pytest.fixture
def table_flip(all_sublattices_lmtpo):
    return Tableflip(all_sublattices_lmtpo[0]
                     + all_sublattices_lmtpo[1],
                     optimize_basis=True,
                     table_ergodic=True,
                     swap_weight=0.2)


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
        assert len(step) == len(set([s for s, c in step]))
        # No duplicate site allowed!
        for flip in step:
            assert flip[1] != occu[flip[0]]
            if flip[0] in mcmcusher.active_sublattices[0].active_sites:
                count1 += 1
                assert flip[1] in mcmcusher.active_sublattices[0].encoding
            elif flip[0] in mcmcusher.active_sublattices[1].active_sites:
                count2 += 1
                assert flip[1] in mcmcusher.active_sublattices[1].encoding
            else:
                raise RuntimeError(
                    "Something went wrong in proposing"
                    f"a step site proposed in {step} is"
                    " not in any of the allowed sites"
                )
            total += 1
            flipped_sites.append(flip[0])

    # check probabilities seem sound
    if not isinstance(mcmcusher, Tableflip):
        assert count1 / total == pytest.approx(0.5, abs=1e-2)
        assert count2 / total == pytest.approx(0.5, abs=1e-2)
    else:
        # Because Table flip is equal per-direction.
        assert count1 / total == pytest.approx(0.6, abs=5e-2)
        assert count2 / total == pytest.approx(0.4, abs=5e-2)

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
            if flip[0] in mcmcusher.active_sublattices[0].active_sites:
                count1 += 1
                assert flip[1] in mcmcusher.active_sublattices[0].encoding
            elif flip[0] in mcmcusher.active_sublattices[1].sites:
                count2 += 1
                assert flip[1] in mcmcusher.active_sublattices[1].encoding
            else:
                raise RuntimeError(
                    "Something went wrong in proposing"
                    f"a step site proposed in {step} is"
                    " not in any of the allowed sites"
                )
            total += 1
            flipped_sites.append(flip[0])
    if not isinstance(mcmcusher, Tableflip):
        assert count1 / total == pytest.approx(0.8, abs=1e-2)
        assert count2 / total == pytest.approx(0.2, abs=1e-2)


def test_table_flip_factors():
    sites1 = np.array([0, 1, 2])
    sites2 = np.array([3, 4, 5])
    site_space1 = SiteSpace(Composition({"Li+": 2 / 3, "Zr4+": 1 / 6, "Mn3+": 1 / 6}))
    site_space2 = SiteSpace(Composition({"O2-": 5 / 6, "F-": 1 / 6}))
    sublattices = [Sublattice(site_space1, sites1), Sublattice(site_space2, sites2)]

    tf = Tableflip(sublattices,
                   optimize_basis=True,
                   table_ergodic=True)
    # Case 1:
    occu1 = np.array([0, 0, 1, 0, 0, 0])
    step1 = [(2, 2), (4, 1)]
    assert np.isclose(tf.compute_log_priori_factor(occu1, step1), np.log(3 / 2))  # forth p=1/3, back p=1/2
    # Case 2:
    occu2 = np.array([0, 0, 2, 1, 0, 0])
    step2 = [(2, 1), (3, 0)]
    assert np.isclose(tf.compute_log_priori_factor(occu2, step2), np.log(2 / 3))  # forth p=1/2, back p=1/3
    # Case 3:
    occu3 = np.array([0, 0, 2, 1, 0, 0])
    step3 = [(2, 0), (4, 1), (5, 1)]
    assert np.isclose(tf.compute_log_priori_factor(occu3, step3), np.log(2 / 9))  # forth p=1/2, back p=1/9
    # Case 4:
    occu4 = np.array([0, 0, 0, 1, 1, 1])
    step4 = [(0, 2), (4, 0), (5, 0)]
    assert np.isclose(tf.compute_log_priori_factor(occu4, step4), np.log(9 / 2))  # forth p=1/9, back p=1/2
    # Case 5:
    occu5 = np.array([0, 0, 2, 1, 0, 0])
    step5 = [(2, 0), (0, 2)]
    assert np.isclose(tf.compute_log_priori_factor(occu5, step5), 0)  # forth p=back p.


def test_table_flip(table_flip, rand_occu_lmtpo):

    def get_n(occu, sublattices):
        sl1, sl2 = sublattices
        n = np.array([(occu[sl1.sites] == 0).sum(),
                      (occu[sl1.sites] == 1).sum(),
                      (occu[sl1.sites] == 2).sum(),
                      (occu[sl2.sites] == 0).sum(),
                      (occu[sl2.sites] == 1).sum()],
                     dtype=int
                     )
        return n

    def get_hash(a):
        return tuple(a.tolist())

    def get_n_states(n):
        assert n[:3].sum() == 3
        assert n[3:].sum() == 3
        return comb(3, n[0]) * comb(3 - n[0], n[1]) * comb(3, n[3])

    occu = rand_occu_lmtpo[0].copy()
    bias = SquarechargeBias(table_flip.sublattices)
    o_counter = Counter()
    n_counter = Counter()

    # Uniformly random kernel.
    # print("Sublattices:", table_flip.sublattices)
    # print("flip table:", table_flip.flip_table)
    l = 100000
    for i in range(l):
        assert bias.compute_bias(occu) == 0
        step = table_flip.propose_step(occu)
        n = get_n(occu, table_flip.sublattices)
        # print("occu:", occu)
        # print("n:", n)
        # print("step:", step)
        flip_id, direction = table_flip._get_flip_id(occu, step)
        occu_next = occu.copy()
        for s_id, code in step:
            occu_next[s_id] = code
        # print("occu_next:", occu_next)
        # print("n_next:", get_n(occu_next))
        dn = get_n(occu_next, table_flip.sublattices) - n
        # Check dn is always correct.
        if flip_id == -1:
            assert direction == 0
            npt.assert_array_equal(dn, 0)
            if len(step) == 2:
                assert np.any(n != 6)
                assert np.all(n >= 0)
            else:
                assert len(step) == 0
        else:
            dd = - 2 * direction + 1
            npt.assert_array_equal(dd * table_flip.flip_table[flip_id, :],
                                   dn)

        n_counter[get_hash(n)] += 1
        o_counter[get_hash(occu)] += 1

        log_priori = table_flip.compute_log_priori_factor(occu, step)
        # Null step might still exist.
        # assert len(step) > 0
        if log_priori >= 0 or log_priori >= np.log(np.random.rand()):
            # Accepted.
            occu = occu_next.copy()

    # When finished, see if distribution is correct.
    assert len(n_counter) == 3
    n_occus = []
    for n_hash in n_counter.keys():
        n = np.array(n_hash, dtype=int)
        n_occus.append(get_n_states(n))
    n_occus = np.array(n_occus)
    assert len(o_counter) == sum(n_occus)
    o_count_av = l / sum(n_occus)
    npt.assert_allclose(np.array(list(o_counter.values())) / o_count_av,
                        1, atol=0.1)
    n_counts = np.array(list(n_counter.values()))
    r_counts = n_counts / n_counts.sum()
    r_occus = n_occus / n_occus.sum()
    npt.assert_allclose(r_counts, r_occus, atol=0.1)
    # print("r_counts:", r_counts)
    # print("r_occus:", r_occus)
    # print("occupancies:", o_counter)
    # assert False
    # Read numerical values, they are acceptable.
