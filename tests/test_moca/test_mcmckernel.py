import pytest
import numpy as np
import numpy.testing as npt
from smol.constants import kB
from smol.moca import (CanonicalEnsemble, FuSemiGrandEnsemble,
                       MuSemiGrandEnsemble, 
                       DiscChargeNeutralSemiGrandEnsemble,
                       CEProcessor)
from smol.moca.sampler.mcusher import Swapper, Flipper,
                                      Chargeneutralflipper
from smol.moca.sampler.kernel import Metropolis
from smol.moca.comp_space import CompSpace

from tests.utils import gen_random_occupancy,
                        gen_random_occupancy_cn

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

    ca_ensemble = CanoncialEnsemble(proc)
    sublattices = ca_ensemble.sublattices

    bits = [sl.species for sl in self.sublattices]
    sl_sizes = [len(sl.sites) for sl in self.sublattices]

    comp_space = CompSpace(bits,sl_sizes)
    mu = [0.3 for i in range(comp_space.dim)]
    return DiscChargeNeutralSemiGrandEnsemble(proc,mu)

@pytest.fixture(params=['swap', 'flip'])
def metropolis_kernel(ensemble, request):
    mkernel = Metropolis(ensemble, temperature=5000, step_type=request.param)
    # fix num_sites to gen random occu
    mkernel.num_sites = ensemble.num_sites
    return mkernel

@pytest.fixture
def metropolis_kernel_cn(disc_ensemble):
    mkernel = Metropolis(ensemble, temperature=5000, step_type='charge-neutral-flip')
    mkernel.num_sites = disc_ensemble.num_sites
    return mkernel

@pytest.mark.parametrize("step_type, mcusher",
                         [("swap", Swapper), ("flip", Flipper)])
def test_constructor(ensemble, step_type, mcusher):
    assert isinstance(Metropolis(ensemble, 5000, step_type)._usher, mcusher)


def test_constructor_cn(disc_ensemble):
    assert isinstance(Metropolis(disc_ensemble, 5000, 'charge-neutral-flip')._usher,
                      Chargeneutralflipper)

def test_single_step(metropolis_kernel):

    occu_ = gen_random_occupancy(metropolis_kernel._usher.sublattices,
                                 metropolis_kernel.num_sites)

    for _ in range(20):
        init_occu = occu_.copy()
        acc, occu, denth, dfeat = metropolis_kernel.single_step(init_occu)
        if acc:
            assert not np.array_equal(occu, occu_)
        else:
            npt.assert_array_equal(occu, occu_)

def test_single_step_cn(metropolis_kernel_cn):
    occu_ = gen_random_occupancy_cn(metropolis_kernel_cn._usher.sublattices,
                                    metropolis_kernel_cn.num_sites)

    for _ in range(20):
        init_occu = occu_.copy()
        acc, occu, denth, dfeat = metropolis_kernel_cn.single_step(init_occu)
        if acc:
            assert not np.array_equal(occu, occu_)
        else:
            npt.assert_array_equal(occu, occu_)


def test_temperature_setter(metropolis_kernel):
    assert metropolis_kernel.beta == 1/(kB*metropolis_kernel.temperature)
    metropolis_kernel.temperature = 500
    assert metropolis_kernel.beta == 1 / (kB * 500)
