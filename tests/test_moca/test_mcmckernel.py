import pytest
import numpy as np
import numpy.testing as npt
from smol.constants import kB
from smol.moca import (CanonicalEnsemble, FuSemiGrandEnsemble,
                       MuSemiGrandEnsemble, 
                       DiscChargeNeutralSemiGrandEnsemble,
                       CEProcessor)
from smol.moca.sampler.mcusher import (Swapper, Flipper,
                                       Tableflipper,
                                       Subchainwalker)
from smol.moca.sampler.kernel import Metropolis
from smol.moca.comp_space import CompSpace
from smol.moca.ensemble.sublattice import get_all_sublattices
from smol.moca.utils.math_utils import GCD_list

from tests.utils import (gen_random_occupancy,
                         gen_random_neutral_occupancy)

ensembles = [CanonicalEnsemble, MuSemiGrandEnsemble, FuSemiGrandEnsemble]

# TODO this could be moved to a conftest.py file since same fixture also used
#  in test_sampler
@pytest.fixture(params=ensembles)
def ensemble(cluster_subspace, request):
    coefs = np.random.random(cluster_subspace.num_corr_functions)
    proc = CEProcessor(cluster_subspace, 4*np.eye(3), coefs)
    if request.param is MuSemiGrandEnsemble:
        kwargs = {'chemical_potentials':
                  {sp: 0.3 for space in proc.unique_site_spaces
                   for sp in space.keys()}}
    else:
        kwargs = {}
    return request.param(proc, **kwargs)

@pytest.fixture
def disc_ensemble(cluster_subspace):
    coefs = np.random.random(cluster_subspace.num_corr_functions)
    proc = CEProcessor(cluster_subspace, 4*np.eye(3), coefs)

    sublattices = get_all_sublattices(proc)

    bits = [sl.species for sl in sublattices]
    sl_sizes = [len(sl.sites) for sl in sublattices]
    sc_size = GCD_list(sl_sizes)
    sl_sizes = [sz//sc_size for sz in sl_sizes]

    comp_space = CompSpace(bits,sl_sizes)
    mu = [0.3 for i in range(comp_space.dim)]
    return DiscChargeNeutralSemiGrandEnsemble(proc,mu)


@pytest.fixture(params=['swap', 'flip'])
def metropolis_kernel(ensemble, request):
    mkernel = Metropolis(ensemble, temperature=5000, step_type=request.param,
                         bias_type='null')
    # fix num_sites to gen random occu
    mkernel.num_sites = ensemble.num_sites
    return mkernel

@pytest.fixture(params=['table-flip', 'subchain-walk'])
def metropolis_kernel_neutral(disc_ensemble, request):
    kwargs = {}
    if request.param == 'subchain-walk':
        kwargs['sub_bias_type'] = 'square-charge'

    mkernel = Metropolis(disc_ensemble, temperature=5000, step_type=request.param,
                         bias_type='null', swap_weight=0.0, **kwargs)

    mkernel.num_sites = disc_ensemble.num_sites
    mkernel.all_sublattices = disc_ensemble.all_sublattices
    return mkernel


@pytest.mark.parametrize("step_type, mcusher",
                         [("swap", Swapper), ("flip", Flipper)])
def test_constructor(ensemble, step_type, mcusher):
    assert isinstance(Metropolis(ensemble, 5000, step_type, 
                      bias_type='null')._usher, mcusher)


def test_constructor_neutral(disc_ensemble):
    assert isinstance(Metropolis(disc_ensemble, 5000,
                      'table-flip', bias_type='null')._usher,
                      Tableflipper)
    assert isinstance(Metropolis(disc_ensemble, 5000,
                      'subchain-walk', bias_type='null',
                      sub_bias_type='square-charge',
                      lam=0.1)._usher,
                      Subchainwalker)


def test_single_step(metropolis_kernel):

    occu_ = gen_random_occupancy(metropolis_kernel._usher.sublattices,
                                 metropolis_kernel.num_sites)

    for _ in range(20):
        init_occu = occu_.copy()
        acc, occu, _, _, denth, dfeat = metropolis_kernel.single_step(init_occu)
        if acc:
            assert not np.array_equal(occu, occu_)
        else:
            npt.assert_array_equal(occu, occu_)

def test_single_step_neutral(metropolis_kernel_neutral):
    occu_ = gen_random_neutral_occupancy(metropolis_kernel_neutral.all_sublattices,
                                         metropolis_kernel_neutral.num_sites)

    bit_charges = [[sp.oxi_state for sp in s.species] for s in metropolis_kernel_neutral.all_sublattices]
    print("Bit charges:", bit_charges)
    all_sublattices = metropolis_kernel_neutral.all_sublattices
    for _ in range(20):
        init_occu = occu_.copy()
        acc, occu, _, _, denth, dfeat = metropolis_kernel_neutral.single_step(init_occu)
        c = 0
        for sl_id, sl in enumerate(all_sublattices):
            for sp_id, sp_c in enumerate(bit_charges[sl_id]):
                c += (occu[sl.sites] == sp_id).sum() * sp_c
            
        assert c == 0
        if acc:
            assert not np.array_equal(occu, occu_)
        else:
            npt.assert_array_equal(occu, occu_)

def test_temperature_setter(metropolis_kernel):
    assert metropolis_kernel.beta == 1/(kB*metropolis_kernel.temperature)
    metropolis_kernel.temperature = 500
    assert metropolis_kernel.beta == 1 / (kB * 500)
