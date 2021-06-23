"""Test all bias terms."""

from smol.moca.sampler.bias import (Nullbias, Squarechargebias,
                                    Squarecompconstraintbias,
                                    mcbias_factory)
from smol.moca.ensemble.sublattice import get_all_sublattices
from smol.moca.processor import CEProcessor
import numpy as np
import random
import pytest

from tests.utils import gen_random_neutral_occupancy

bias_classes = [Nullbias, Squarechargebias, Squarecompconstraintbias]

@pytest.fixture
def composite_processor(cluster_subspace):
    coefs = 2 * np.random.random(cluster_subspace.num_corr_functions)
    scmatrix = 2 * np.eye(3)
    return CEProcessor(cluster_subspace, scmatrix, coefs)

@pytest.fixture
def all_sublattices(composite_processor):
    return get_all_sublattices(composite_processor)

@pytest.fixture
def rand_occu(all_sublattices):
    num_sites = sum(len(s.sites) for s in all_sublattices)
    return gen_random_neutral_occupancy(all_sublattices, num_sites)

@pytest.fixture(params=bias_classes)
def mcmcbias(all_sublattices, request):
    if request.param == Squarecompconstraintbias:
        bits = [s.species for s in all_sublattices]
        D = sum(len(sl)-1 for sl in bits)
        C = np.random.random((4,D))
        b = np.random.random(4)
        return request.param(all_sublattices, C=C, b=b)
    return request.param(all_sublattices)

def get_oxi_state(sp):
    if 'oxi_state' in dir(sp):
        return sp.oxi_state
    else:
        return 0

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
    occu = rand_occu.copy()
    for _ in range(50):
        s = random.choice(list(range(len(mcmcbias.sublattices))))
        i = random.choice(mcmcbias.sublattices[s].sites)
        sp = random.choice(list(range(len(mcmcbias.sublattices[s].species))))
        step.append((i,sp))
        occu[i] = sp

    assert (mcmcbias.compute_bias_change(rand_occu, step) ==
            (mcmcbias.compute_bias(occu) -
             mcmcbias.compute_bias(rand_occu)))

def test_mcbias_factory(all_sublattices):
    for bias in bias_classes:
        kwargs = {}
        if bias.__name__ == 'Squarecompconstraintbias':
            kwargs['C'] = [[1,2,3],[4,5,6]]
            kwargs['b'] = [1,2]
        assert isinstance(mcbias_factory(bias.__name__, all_sublattices,
                                         **kwargs),
                          bias)
