import pytest
from itertools import product
import numpy as np
import numpy.testing as npt
from smol.constants import kB
from smol.moca.sampler.mcusher import Swap, Flip
from smol.moca.sampler.kernel import Metropolis, UniformlyRandom, \
    ThermalKernel, ALL_MCUSHERS
from tests.utils import gen_random_occupancy

kernels = [UniformlyRandom, Metropolis]
ushers = ALL_MCUSHERS


@pytest.fixture(params=product(kernels, ushers.keys()))
def mckernel(ensemble, request):
    kwargs = {}
    kernel_class, step_type = request.param
    if issubclass(kernel_class, ThermalKernel):
        kwargs['temperature'] = 5000
    kernel = kernel_class(ensemble, step_type=step_type, **kwargs)
    # fix num_sites to gen random occu
    kernel.num_sites = ensemble.num_sites
    return kernel


@pytest.mark.parametrize(
    "step_type, mcusher", [("swap", Swap), ("flip", Flip)])
def test_constructor(ensemble, step_type, mcusher):
    assert isinstance(
        Metropolis(ensemble, step_type=step_type, temperature=500)._usher,
        mcusher)


def test_single_step(mckernel):
    occu_ = gen_random_occupancy(mckernel._usher.sublattices,
                                 mckernel.num_sites)
    for _ in range(20):
        init_occu = occu_.copy()
        acc, occu, denth, dfeat = mckernel.single_step(init_occu)
        if acc:
            assert not np.array_equal(occu, occu_)
        else:
            npt.assert_array_equal(occu, occu_)


def test_temperature_setter(single_canonical_ensemble):
    metropolis_kernel = Metropolis(single_canonical_ensemble, step_type="flip",
                                   temperature=500)
    assert metropolis_kernel.beta == 1/(kB*metropolis_kernel.temperature)
    metropolis_kernel.temperature = 500
    assert metropolis_kernel.beta == 1 / (kB * 500)
