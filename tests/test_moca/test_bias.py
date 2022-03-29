"""Test all bias terms."""

from smol.moca.sampler.bias import (Nullbias, Squarechargebias,
                                    Squarecompconstraintbias,
                                    FugacityBias,
                                    mcbias_factory)
from smol.moca.ensemble.sublattice import get_all_sublattices
from smol.moca.processor import CEProcessor
import numpy as np
import random
import pytest
from smol.moca.sampler.bias import mcbias_factory, FugacityBias
from tests.utils import gen_random_occupancy

bias_classes = [FugacityBias, Squarechargebias, Squarecompconstraintbias]


@pytest.fixture(scope="module")
def all_sublattices(ce_processor):
    return ce_processor.get_sublattices()


@pytest.fixture(params=bias_classes)
def mcbias(all_sublattices, request):
    return request.param(all_sublattices)

def get_charge(occupancy, sublattices):
    n_cols = max(len(s.species) for s in sublattices)
    n_rows = len(occupancy)
    oxi_table = np.zeros((n_rows, n_cols))
    for s in sublattices:
        for j, sp in enumerate(s.species):
            oxi_table[s.sites, j] = get_oxi_state(sp)
    return np.sum([oxi_table[i, o] for i, o in enumerate(occupancy)])

def get_ucoords(occupancy, sublattices):
    bits = [s.species for s in sublattices]
    sl_list = [s.sites for s in sublattices]

    occu = np.array(occupancy)
    compstat = [[(occu[sl_list[sl_id]] == sp_id).sum()
                for sp_id, sp in enumerate(sl)]
                for sl_id, sl in enumerate(bits)]

    ucoords = []
    for sl in compstat:
        ucoords.extend(sl[:-1])
    return ucoords

def test_compute_bias(mcmcbias, rand_occu):
    if mcmcbias.__class__.__name__ == 'Nullbias':
        assert mcmcbias.compute_bias(rand_occu) == 0
    if mcmcbias.__class__.__name__ == 'Squarechargebias':
        assert (mcmcbias.compute_bias(rand_occu) ==
                0.5 * get_charge(rand_occu, mcmcbias.sublattices)**2)
    if mcmcbias.__class__.__name__ == 'Squarecompconstraintbias':
        x = get_ucoords(rand_occu, mcmcbias.sublattices)
        C = mcmcbias.C
        b = mcmcbias.b
        assert (mcmcbias.compute_bias(rand_occu) ==
                0.5 * np.sum((C@x-b)**2))

def test_compute_bias_change(mcmcbias, rand_occu):
    step = []
    occu = gen_random_occupancy(mcbias.sublattices)
    new_occu = occu.copy()
    for _ in range(50):
        s = random.choice(list(range(len(mcbias.active_sublattices))))
        i = random.choice(mcbias.active_sublattices[s].sites)
        sp = random.choice(list(range(len(mcbias.active_sublattices[s].species))))
        step.append((i, sp))
        if i == 81:
            raise (ValueError, "81!!!!")
        new_occu[i] = sp
    print(mcbias.sublattices)
    print(step)
    assert mcbias.compute_bias_change(occu, step) == pytest.approx(
        mcbias.compute_bias(new_occu) - mcbias.compute_bias(occu)
    )


def test_mcbias_factory(all_sublattices):
    for bias in bias_classes:
        assert isinstance(mcbias_factory(bias.__name__, all_sublattices), bias)


# Tests for FugacityBias
# Tests for FuSemiGrandEnsemble
@pytest.fixture(scope="module")
def fugacity_bias(all_sublattices):
    return FugacityBias(all_sublattices)


def test_bad_fugacity_fractions(fugacity_bias):
    fug_fracs = deepcopy(fugacity_bias.fugacity_fractions)
    with pytest.raises(ValueError):
        fug_fracs[0] = {s: v for s, v in list(fug_fracs[0].items())[:-1]}
        fugacity_bias.fugacity_fractions = fug_fracs
    with pytest.raises(ValueError):
        fug_fracs[0] = {sp: 1.1 for sp in fug_fracs[0].keys()}
        fugacity_bias.fugacity_fractions = fug_fracs
    with pytest.raises(ValueError):
        fug_fracs[0] = {"A": 0.5, "D": 0.6}
        fugacity_bias.fugacity_fractions = fug_fracs
    with pytest.raises(ValueError):
        fug_fracs[0]["foo"] = 0.4
        FugacityBias(
            fugacity_bias.sublattices,
            fugacity_fractions=fug_fracs,
        )
    with pytest.raises(ValueError):
        del fug_fracs[0]["foo"]
        FugacityBias(
            fugacity_bias.sublattices,
            fugacity_fractions=fug_fracs,
        )
    with pytest.raises(ValueError):
        fug_fracs[0][str(list(fug_fracs[0].keys())[0])] = 0.0
        fugacity_bias.fugacity_fractions = fug_fracs


def test_build_fu_table(fugacity_bias):
    table = fugacity_bias._build_fu_table(fugacity_bias.fugacity_fractions)
    for sublatt in fugacity_bias.active_sublattices:
        for fus in fugacity_bias.fugacity_fractions:
            if list(sublatt.site_space.keys()) == list(fus.keys()):
                fugacity_fractions = fus
        for i in sublatt.sites:
            for j, species in zip(sublatt.encoding, sublatt.site_space):
                assert fugacity_fractions[species] == table[i, j]
