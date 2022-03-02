"""Test all bias terms."""

import pytest
from copy import deepcopy
import random
from smol.moca.sampler.bias import mcbias_factory, FugacityBias
from tests.utils import gen_random_occupancy


bias_classes = [FugacityBias]


@pytest.fixture(scope="module")
def all_sublattices(ce_processor):
    return ce_processor.get_sublattices(), ce_processor.get_inactive_sublattices()


@pytest.fixture(params=bias_classes)
def mcbias(all_sublattices, request):
    return request.param(*all_sublattices)


def test_compute_bias_change(mcbias):
    step = []
    occu = gen_random_occupancy(mcbias.sublattices, mcbias.inactive_sublattices)
    new_occu = occu.copy()
    for _ in range(50):
        s = random.choice(list(range(len(mcbias.sublattices))))
        i = random.choice(mcbias.sublattices[s].sites)
        sp = random.choice(list(range(len(mcbias.sublattices[s].species))))
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
        assert isinstance(mcbias_factory(bias.__name__, *all_sublattices), bias)


# Tests for FugacityBias
# Tests for FuSemiGrandEnsemble
@pytest.fixture(scope="module")
def fugacity_bias(all_sublattices):
    return FugacityBias(*all_sublattices)


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
            fugacity_bias.inactive_sublattices,
            fugacity_fractions=fug_fracs,
        )
    with pytest.raises(ValueError):
        del fug_fracs[0]["foo"]
        FugacityBias(
            fugacity_bias.sublattices,
            fugacity_bias.inactive_sublattices,
            fugacity_fractions=fug_fracs,
        )
    with pytest.raises(ValueError):
        fug_fracs[0][str(list(fug_fracs[0].keys())[0])] = 0.0
        fugacity_bias.fugacity_fractions = fug_fracs


def test_build_fu_table(fugacity_bias):
    table = fugacity_bias._build_fu_table(fugacity_bias.fugacity_fractions)
    for sublatt in fugacity_bias.sublattices:
        for fus in fugacity_bias.fugacity_fractions:
            if list(sublatt.site_space.keys()) == list(fus.keys()):
                fugacity_fractions = fus
        for i in sublatt.sites:
            for j, species in enumerate(sublatt.site_space):
                assert fugacity_fractions[species] == table[i, j]
