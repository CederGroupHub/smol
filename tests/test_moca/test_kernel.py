from itertools import product
from random import choice, choices

import numpy as np
import numpy.testing as npt
import pytest

from smol.capp.generate.random import _gen_unconstrained_ordered_occu
from smol.constants import kB
from smol.moca.kernel import (
    Metropolis,
    MulticellMetropolis,
    UniformlyRandom,
    WangLandau,
)
from smol.moca.kernel.base import ALL_MCUSHERS, ThermalKernelMixin
from smol.moca.kernel.bias import FugacityBias
from smol.moca.kernel.mcusher import Flip, Swap, TableFlip
from smol.moca.trace import StepTrace, Trace
from tests.utils import assert_pickles

kernels_with_bias = [UniformlyRandom, Metropolis]
kernels_no_bias = [MulticellMetropolis, WangLandau]
kernels = kernels_with_bias + kernels_no_bias
ushers = ALL_MCUSHERS


@pytest.fixture(params=product(kernels, ushers), scope="module")
def mckernel(ensemble, rng, request):
    kwargs = {}
    kernel_class, step_type = request.param

    if issubclass(kernel_class, ThermalKernelMixin):
        kwargs["temperature"] = 5000

    if kernel_class == WangLandau:
        kwargs["min_enthalpy"] = -5000
        kwargs["max_enthalpy"] = 5000
        kwargs["bin_size"] = 10
        kwargs["check_period"] = 5

    if step_type == "MultiStep":
        kwargs["mcusher"] = choice(
            [c for c in ushers if c not in ("MultiStep", "Composite")]
        )
        kwargs["step_lengths"] = 4
    elif step_type == "Composite":
        kwargs["mcushers"] = choices(
            [c for c in ushers if c not in ("MultiStep", "Composite")], k=2
        )
    # TODO create ensemble with different cell shapes
    if kernel_class == MulticellMetropolis:
        kernels = [
            Metropolis(ensemble, step_type=step_type, **kwargs) for _ in range(10)
        ]
        kernel = kernel_class(
            kernels, kwargs["temperature"]
        )  # , kernel_hop_periods=101)
        kernel.set_aux_state(
            np.vstack(
                [
                    _gen_unconstrained_ordered_occu(kernel.mcusher.sublattices, rng=rng)
                    for _ in range(len(kernel.mckernels))
                ]
            )
        )
    else:
        kernel = kernel_class(ensemble, step_type=step_type, **kwargs)
        kernel.set_aux_state(
            _gen_unconstrained_ordered_occu(kernel.mcusher.sublattices, rng=rng)
        )
    return kernel


@pytest.fixture(params=product(kernels_with_bias, ushers), scope="module")
def mckernel_bias(ensemble, request):
    kwargs = {}
    kernel_class, step_type = request.param
    if issubclass(kernel_class, ThermalKernelMixin):
        kwargs["temperature"] = 5000

    if step_type == "MultiStep":
        kwargs["mcusher"] = choice(
            [c for c in ushers if c not in ("MultiStep", "Composite")]
        )
        kwargs["step_lengths"] = 4
    elif step_type == "Composite":
        kwargs["mcushers"] = choices(
            [c for c in ushers if c not in ("MultiStep", "Composite")], k=2
        )

    kernel = kernel_class(ensemble, step_type=step_type, **kwargs)
    kernel.bias = FugacityBias(kernel.mcusher.sublattices)
    return kernel


@pytest.mark.parametrize(
    "step_type, mcusher", [("swap", Swap), ("flip", Flip), ("table-flip", TableFlip)]
)
def test_constructor(ensemble, step_type, mcusher):
    kernel = Metropolis(ensemble, step_type=step_type, temperature=500)
    assert isinstance(kernel.mcusher, mcusher)
    assert isinstance(kernel.trace, StepTrace)
    assert "temperature" in kernel.trace.names
    kernel.bias = FugacityBias(kernel.mcusher.sublattices)
    assert "bias" in kernel.trace.delta_trace.names


def test_single_step(mckernel, rng):
    occu = _gen_unconstrained_ordered_occu(mckernel.mcusher.sublattices, rng=rng)
    mckernel.set_aux_state(occu)
    prev_kernel_index = 0  # for multicell metropolis
    prev_occu = occu.copy()

    for _ in range(100):
        trace = mckernel.single_step(occu)
        curr_features = mckernel.ensemble.compute_feature_vector(occu)

        if trace.accepted:
            npt.assert_array_equal(trace.occupancy, occu)
            if (
                isinstance(mckernel, MulticellMetropolis)
                and prev_kernel_index != trace.kernel_index
            ):
                prev_features = mckernel._features[prev_kernel_index]
                assert np.allclose(
                    mckernel.mckernels[
                        prev_kernel_index
                    ].ensemble.compute_feature_vector(prev_occu),
                    prev_features,
                )
                prev_kernel_index = trace.kernel_index
            else:
                prev_features = mckernel.ensemble.compute_feature_vector(prev_occu)

            npt.assert_allclose(
                curr_features - prev_features,
                trace.delta_trace.features,
                rtol=1e-5,
                atol=1e-8,
            )
        else:
            npt.assert_array_equal(trace.occupancy, prev_occu)

        if isinstance(mckernel, MulticellMetropolis):
            npt.assert_allclose(
                mckernel._features[trace.kernel_index],
                curr_features,
                rtol=1e-5,
                atol=1e-8,
            )
            assert mckernel.trace.kernel_index == mckernel._current_kernel_index
            assert (
                mckernel.trace
                is mckernel.mckernels[mckernel._current_kernel_index].trace
            )
        elif isinstance(mckernel, WangLandau):
            assert mckernel.bin_size > 0
            assert isinstance(mckernel.levels, np.ndarray)
            assert isinstance(mckernel.entropy, np.ndarray)
            assert isinstance(mckernel.histogram, np.ndarray)

            assert "histogram" in trace.names
            assert "occurrences" in trace.names
            assert "entropy" in trace.names
            assert "cumulative_mean_features" in trace.names
            assert "mod_factor" in trace.names

        # save previous occu
        prev_occu[:] = occu


def test_single_step_bias(mckernel_bias, rng):
    mckernel_bias.set_aux_state(
        _gen_unconstrained_ordered_occu(mckernel_bias.mcusher.sublattices)
    )
    for _ in range(100):
        occu = _gen_unconstrained_ordered_occu(
            mckernel_bias.mcusher.sublattices, rng=rng
        )
        trace = mckernel_bias.single_step(occu.copy())
        # assert delta bias is there and recorded
        assert isinstance(trace.delta_trace.bias, np.ndarray)
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

    with pytest.raises(TypeError):
        steptrace.fourth = "blabla"
        StepTrace(one=np.zeros(40), two=66)

    # check saving
    assert trace.as_dict() == trace.__dict__
    steptrace_d = steptrace.__dict__.copy()
    steptrace_d["delta_trace"] = steptrace_d["delta_trace"].__dict__.copy()
    assert steptrace.as_dict() == steptrace_d


def test_pickles(mckernel):
    assert_pickles(mckernel)


# TODO add direct multicell tests, especially to check proper updates of traces
#  before/after cell jumps
