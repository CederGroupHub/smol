import pytest
from itertools import product
import numpy as np
import numpy.testing as npt
from smol.constants import kB
from smol.moca.sampler.mcusher import Swap, Flip
from smol.moca.sampler.kernel import Metropolis, UniformlyRandom, \
    ThermalKernel, Trace, StepTrace, ALL_MCUSHERS
from smol.moca.sampler.bias import FugacityBias
from tests.utils import gen_random_occupancy

kernels = [UniformlyRandom, Metropolis]
ushers = ALL_MCUSHERS


@pytest.fixture(params=product(kernels, ushers), scope='module')
def mckernel(ensemble, request):
    kwargs = {}
    kernel_class, step_type = request.param
    if issubclass(kernel_class, ThermalKernel):
        kwargs['temperature'] = 5000
    kernel = kernel_class(ensemble, step_type=step_type, **kwargs)
    return kernel


@pytest.fixture(params=product(kernels, ushers), scope='module')
def mckernel_bias(ensemble, request):
    kwargs = {}
    kernel_class, step_type = request.param
    if issubclass(kernel_class, ThermalKernel):
        kwargs['temperature'] = 5000
    kernel = kernel_class(ensemble, step_type=step_type, **kwargs)
    kernel.bias = FugacityBias(kernel.mcusher.sublattices,
                               kernel.mcusher.inactive_sublattices)
    return kernel


@pytest.mark.parametrize(
    "step_type, mcusher", [("swap", Swap), ("flip", Flip)])
def test_constructor(ensemble, step_type, mcusher):
    kernel = Metropolis(ensemble, step_type=step_type, temperature=500)
    assert isinstance(kernel._usher, mcusher)
    assert isinstance(kernel.trace, StepTrace)
    assert 'temperature' in kernel.trace.field_names
    kernel.bias = FugacityBias(kernel.mcusher.sublattices,
                               kernel.mcuser.inactive_sublattices)
    assert 'bias' in kernel.trace.delta.field_names


def test_trace():
    trace = Trace(first=np.ones(10), second=np.zeros(10),
                  trace1=Trace(inner=np.zeros(10)))
    assert all(isinstance(val, np.ndarray) or isinstance(val, Trace)
               for val in trace.__dict__.values())
    trace.third = np.random.random(10)
    trace.trace2 = Trace(inner=np.ones(10))
    assert all(isinstance(val, np.ndarray) or isinstance(val, Trace)
               for val in trace.__dict__.values())
    fields = ['first', 'second', 'third', 'trace1', 'trace2']
    assert all(field in fields for field in trace.field_names)

    with pytest.raises(TypeError):
        trace.fourth = 'blabla'
        trace2 = Trace(one=np.zeros(40), two=66)
    
    steptrace = StepTrace(one=np.zeros(10))
    assert isinstance(steptrace.delta, Trace)


def test_single_step(mckernel):
    occu_ = gen_random_occupancy(mckernel._usher.sublattices,
                                 mckernel._usher.inactive_sublattices)
    for _ in range(20):
        init_occu = occu_.copy()
        acc, occu, denth, dfeat = mckernel.single_step(init_occu)
        if acc:
            assert not np.array_equal(occu, occu_)
        else:
            npt.assert_array_equal(occu, occu_)


def test_single_step_bias(mckernel_bias):
    occu_ = gen_random_occupancy(mckernel_bias._usher.sublattices,
                                 mckernel_bias._usher.inactive_sublattices)
    for _ in range(20):
        init_occu = occu_.copy()
        acc, occu, denth, dfeat = mckernel_bias.single_step(init_occu)
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
