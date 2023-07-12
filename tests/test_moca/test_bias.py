"""Test all bias terms."""

from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pytest

from smol.capp.generate.random import _gen_unconstrained_ordered_occu
from smol.moca.composition import get_oxi_state
from smol.moca.kernel.bias import (
    FugacityBias,
    SquareChargeBias,
    SquareHyperplaneBias,
    mcbias_factory,
)
from tests.utils import assert_pickles

bias_classes = [FugacityBias, SquareChargeBias, SquareHyperplaneBias]


@pytest.fixture(scope="module")
def all_sublattices(ce_processor):
    return ce_processor.get_sublattices()


@pytest.fixture(params=bias_classes)
def mcbias(all_sublattices, request):
    if request.param == SquareHyperplaneBias:
        n_dims = sum(len(sublatt.species) for sublatt in all_sublattices)
        n_cons = max(n_dims - 1, 1)
        a = np.random.randint(low=-10, high=10, size=(n_cons, n_dims))
        b = np.random.randint(low=-10, high=10, size=n_cons)
        return request.param(all_sublattices, a, b)
    return request.param(all_sublattices)


def test_compute_bias_change(mcbias, rng):
    step = []
    occu = _gen_unconstrained_ordered_occu(mcbias.sublattices, rng=rng)
    new_occu = occu.copy()
    rng = np.random.default_rng()
    for _ in range(50):
        s = rng.choice(list(range(len(mcbias.active_sublattices))))
        i = rng.choice(mcbias.active_sublattices[s].sites)
        sp = rng.choice(list(range(len(mcbias.active_sublattices[s].species))))
        step.append((i, sp))
        new_occu[i] = sp
    assert mcbias.compute_bias_change(occu, step) == pytest.approx(
        mcbias.compute_bias(new_occu) - mcbias.compute_bias(occu)
    )


def test_mcbias_factory(all_sublattices):
    for bias in bias_classes:
        if bias == SquareHyperplaneBias:
            n_dims = sum(len(sublatt.species) for sublatt in all_sublattices)
            n_cons = max(n_dims - 1, 1)
            a = np.random.randint(low=-10, high=10, size=(n_cons, n_dims))
            b = np.random.randint(low=-10, high=10, size=n_cons)
            kwargs = {"hyperplane_normals": a, "hyperplane_intercepts": b}
        else:
            kwargs = {}
        assert isinstance(
            mcbias_factory(bias.__name__, all_sublattices, **kwargs), bias
        )


# Tests for FugacityBias
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


@pytest.fixture(scope="module")
def square_charge_bias(all_sublattices):
    return SquareChargeBias(all_sublattices)


def test_charge_bias(square_charge_bias, rng):
    table = square_charge_bias._c_table
    n_species = max(max(s.encoding) for s in square_charge_bias.sublattices) + 1
    n_sites = sum(len(s.sites) for s in square_charge_bias.sublattices)
    assert table.shape == (n_sites, n_species)
    # All sites on all sublattices must be included in table
    for sublatt in square_charge_bias.sublattices:
        charges = np.array([get_oxi_state(sp) for sp in sublatt.species])
        npt.assert_array_equal(
            table[sublatt.sites[:, None], sublatt.encoding] - charges[None, :], 0
        )
    # Bias should be implemented as negative.
    for _ in range(100):
        occu = _gen_unconstrained_ordered_occu(square_charge_bias.sublattices, rng=rng)
        assert square_charge_bias.compute_bias(occu) <= 1e-6


@pytest.fixture(scope="module")
def square_comp_bias(all_sublattices):
    n_dims = sum(len(sublatt.species) for sublatt in all_sublattices)
    n_cons = max(n_dims - 1, 1)
    a = np.random.randint(low=-10, high=10, size=(n_cons, n_dims))
    b = np.random.randint(low=-10, high=10, size=n_cons)
    return SquareHyperplaneBias(all_sublattices, a, b)


def test_comp_bias(square_comp_bias, rng):
    # Bias should be implemented as negative.
    for _ in range(100):
        occu = _gen_unconstrained_ordered_occu(square_comp_bias.sublattices, rng=rng)
        assert square_comp_bias.compute_bias(occu) <= 1e-6


def test_pickles(mcbias):
    assert_pickles(mcbias)
