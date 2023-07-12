import os

import numpy as np
import numpy.testing as npt
import pytest

from smol.capp.generate.random import _gen_unconstrained_ordered_occu
from smol.moca import SampleContainer, Sampler
from smol.moca.kernel import Metropolis
from smol.moca.kernel.mcusher import Flip, Swap
from tests.utils import assert_pickles

TEMPERATURE = 5000
# Correlations are within ATOL 1E-14, but ewald energies sometimes need more slack
ATOL = 5e-13  # this is still enough precision anyway


@pytest.fixture(params=[1, 5])
def sampler(ensemble, rng, request):
    sampler = Sampler.from_ensemble(
        ensemble,
        temperature=TEMPERATURE,
        seeds=[rng for _ in range(request.param)],
        nwalkers=request.param,
    )
    # fix this additional attribute to sampler to access in gen occus for tests
    sampler.num_sites = ensemble.num_sites
    return sampler


def test_from_ensemble(sampler):
    if hasattr(sampler.samples.metadata, "chemical_potentials"):
        assert isinstance(sampler.mckernels[0]._usher, Flip)
    else:
        assert isinstance(sampler.mckernels[0]._usher, Swap)
    assert isinstance(sampler.mckernels[0], Metropolis)


@pytest.mark.parametrize("thin", (1, 10))
def test_sample(sampler, thin, rng):
    occu = np.vstack(
        [
            _gen_unconstrained_ordered_occu(
                sampler.mckernels[0]._usher.sublattices, rng=rng
            )
            for _ in range(sampler.samples.shape[0])
        ]
    )
    steps = 1000
    samples = [state for state in sampler.sample(1000, occu, thin_by=thin)]
    assert len(samples) == steps // thin

    it = sampler.sample(43, occu, thin_by=7)
    with pytest.warns(RuntimeWarning):
        next(it)


@pytest.mark.parametrize("thin", (1, 10))
def test_run(sampler, thin, rng):
    occu = np.vstack(
        [
            _gen_unconstrained_ordered_occu(kernel._usher.sublattices, rng=rng)
            for kernel in sampler.mckernels
        ]
    )
    steps = 1000
    sampler.run(steps, occu, thin_by=thin)
    assert len(sampler.samples) == steps // thin
    assert 0 <= sampler.efficiency() <= 1

    # pick some random samples and check recorded traces match!
    for i in rng.choice(range(sampler.samples.num_samples), size=50):
        npt.assert_allclose(
            sampler.samples.get_feature_vectors(flat=False)[i],
            np.vstack(
                list(
                    map(
                        sampler.mckernels[0].ensemble.compute_feature_vector,
                        sampler.samples.get_occupancies(flat=False)[i],
                    )
                )
            ),
            atol=ATOL,
        )

    sampler.clear_samples()


def test_anneal(sampler, rng, tmpdir):
    temperatures = np.linspace(2000, 500, 5)
    occu = np.vstack(
        [
            _gen_unconstrained_ordered_occu(
                sampler.mckernels[0]._usher.sublattices, rng=rng
            )
            for _ in range(sampler.samples.shape[0])
        ]
    )
    steps = 100
    sampler.anneal(temperatures, steps, occu)
    expected = []
    for T in temperatures:
        expected += (
            steps
            * sampler.samples.shape[0]
            * [
                T,
            ]
        )
    npt.assert_array_equal(sampler.samples.get_trace_value("temperature"), expected)
    assert sampler.samples.num_samples == len(temperatures) * steps

    # test streaming anneal
    new_container = SampleContainer.from_dict(sampler.samples.as_dict())
    new_container.clear()
    new_sampler = Sampler(sampler.mckernels, new_container)
    file_path = os.path.join(tmpdir, "test.h5")

    new_sampler.anneal(
        temperatures, steps, occu, stream_chunk=10, stream_file=file_path
    )
    assert new_sampler.samples.num_samples == 0
    samples = SampleContainer.from_hdf5(file_path)
    assert samples.num_samples == len(temperatures) * steps
    npt.assert_array_equal(samples.get_trace_value("temperature"), expected)
    os.remove(file_path)
    # test temp error
    with pytest.raises(ValueError):
        sampler.anneal([100, 200], steps)


def test_reshape_occu(ensemble, rng):
    sampler = Sampler.from_ensemble(ensemble, temperature=TEMPERATURE)
    occu = _gen_unconstrained_ordered_occu(ensemble.sublattices, rng=rng)
    assert sampler._reshape_occu(occu).shape == (1, len(occu))


def test_pickles(sampler):
    assert_pickles(sampler)
