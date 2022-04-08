from itertools import product

import numpy as np
import numpy.testing as npt
import pytest

from smol.constants import kB
from smol.moca.sampler.bias import FugacityBias
from smol.moca.sampler.kernel import (
    ALL_MCUSHERS,
    Metropolis,
    StepTrace,
    ThermalKernel,
    Trace,
    UniformlyRandom,
)
from smol.moca.sampler.mcusher import Flip, Swap, Tableflip
from tests.utils import gen_random_occupancy

kernels = [UniformlyRandom, Metropolis]
ushers = ALL_MCUSHERS


@pytest.fixture(params=product(kernels, ushers), scope="module")
def mckernel(ensemble, request):
    kwargs = {}
    kernel_class, step_type = request.param
    if issubclass(kernel_class, ThermalKernel):
        kwargs["temperature"] = 5000
    kernel = kernel_class(ensemble, step_type=step_type, **kwargs)
    return kernel


@pytest.fixture(params=product(kernels, ushers), scope="module")
def mckernel_bias(ensemble, request):
    kwargs = {}
    kernel_class, step_type = request.param
    if issubclass(kernel_class, ThermalKernel):
        kwargs["temperature"] = 5000
    kernel = kernel_class(ensemble, step_type=step_type, **kwargs)
    kernel.bias = FugacityBias(kernel.mcusher.sublattices)
    return kernel


@pytest.mark.parametrize("step_type, mcusher", [("swap", Swap), ("flip", Flip),
                                                ("tableflip", Tableflip)])
def test_constructor(ensemble, step_type, mcusher):
    kernel = Metropolis(ensemble, step_type=step_type, temperature=500)
    assert isinstance(kernel._usher, mcusher)
    assert isinstance(kernel.trace, StepTrace)
    assert "temperature" in kernel.trace.names
    kernel.bias = FugacityBias(kernel.mcusher.sublattices)
    assert "bias" in kernel.trace.delta_trace.names


def test_trace(rng):
    trace = Trace(first=np.ones(10), second=np.zeros(10))
    assert all(isinstance(val, np.ndarray) for _, val in trace.items())

    trace.third = rng.random(10)
    assert all(isinstance(val, np.ndarray) for _, val in trace.items())
    names = ["first", "second", "third"]
    assert all(name in names for name in trace.names)

    with pytest.raises(TypeError):
        trace.fourth = "blabla"
        Trace(one=np.zeros(40), two=66)

    steptrace = StepTrace(one=np.zeros(10))
    assert isinstance(steptrace.delta_trace, Trace)
    with pytest.raises(ValueError):
        steptrace.delta_trace = np.ones(10)
        StepTrace(delta_trace=np.ones(10))

    # check saving
    assert trace.as_dict() == trace.__dict__
    steptrace_d = steptrace.__dict__.copy()
    steptrace_d["delta_trace"] = steptrace_d["delta_trace"].__dict__.copy()
    assert steptrace.as_dict() == steptrace_d


def test_single_step(mckernel):
    occu_ = gen_random_occupancy(mckernel._usher.sublattices)
    for _ in range(20):
        trace = mckernel.single_step(occu_.copy())
        if trace.accepted:
            assert not np.array_equal(trace.occupancy, occu_)
        else:
            npt.assert_array_equal(trace.occupancy, occu_)


def test_single_step_bias(mckernel_bias):
    occu = gen_random_occupancy(mckernel_bias._usher.sublattices)
    for _ in range(20):
        trace = mckernel_bias.single_step(occu.copy())
        # assert delta bias is there and recorded
        assert isinstance(trace.delta_trace.bias, np.ndarray)
        print(trace.delta_trace.bias)
        assert len(trace.delta_trace.bias.shape) == 0  # 0 dimensional
        if trace.accepted:
            assert not np.array_equal(trace.occupancy, occu)
        else:
            npt.assert_array_equal(trace.occupancy, occu)


def test_temperature_setter(single_canonical_ensemble):
    metropolis_kernel = Metropolis(
        single_canonical_ensemble, step_type="flip", temperature=500
    )
    assert metropolis_kernel.beta == 1 / (kB * metropolis_kernel.temperature)
    metropolis_kernel.temperature = 500
    assert metropolis_kernel.beta == 1 / (kB * 500)
