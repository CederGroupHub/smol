import os

import numpy as np
import numpy.testing as npt
import pytest

from smol.moca import SampleContainer, Sampler
from smol.moca.sampler.kernel import Metropolis
from smol.moca.sampler.mcusher import Flip, Swap
from tests.utils import gen_random_occupancy

TEMPERATURE = 5000
ATOL = 1e-14


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
def test_sample(sampler, thin):
    occu = np.vstack(
        [
            gen_random_occupancy(sampler.mckernels[0]._usher.sublattices)
            for _ in range(sampler.samples.shape[0])
        ]
    )
    steps = 1000
    samples = [state for state in sampler.sample(1000, occu, thin_by=thin)]
    assert len(samples) == steps // thin

    it = sampler.sample(43, occu, thin_by=7)
    with pytest.warns(RuntimeWarning):
        next(it)


# TODO efficiency is sometimes =0 and so fails
@pytest.mark.parametrize("thin", (1, 10))
def test_run(sampler, thin, rng):
    occu = np.vstack(
        [
            gen_random_occupancy(kernel._usher.sublattices)
            for kernel in sampler.mckernels
        ]
    )
    steps = 1000
    sampler.run(steps, occu, thin_by=thin)
    assert len(sampler.samples) == steps // thin
    assert 0 <= sampler.efficiency() <= 1

    # pick some random samples and check recorded traces match!
    for i in rng.choice(range(sampler.samples.num_samples), size=10):
        npt.assert_allclose(
            sampler.samples.get_feature_vectors(flat=False)[i],
            np.vstack(
                list(
                    map(
                        sampler.mckernels[0]._compute_features,
                        sampler.samples.get_occupancies(flat=False)[i],
                    )
                )
            ),
            atol=ATOL,
        )

    sampler.clear_samples()


def test_anneal(sampler, tmpdir):
    temperatures = np.linspace(2000, 500, 5)
    occu = np.vstack(
        [
            gen_random_occupancy(sampler.mckernels[0]._usher.sublattices)
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


# TODO test run sgensembles at high temp
"""
# test some high temp high potential values
steps = 10000
chem_pots = {'Na+': 100.0, 'Cl-': 0.0}
self.msgensemble.chemical_potentials = chem_pots
expected = {'Na+': 1.0, 'Cl-': 0.0}
sampler_m.run(steps, self.occu)
comp = sampler_m.samples.mean_composition()
for sp in expected.keys():
    self.assertAlmostEqual(expected[sp], comp[sp], places=2)
sampler_m.clear_samples()

chem_pots = {'Na+': -100.0, 'Cl-': 0.0}
self.msgensemble.chemical_potentials = chem_pots
expected = {'Na+': 0.0, 'Cl-': 1.0}
sampler_m.run(steps, self.occu)
comp = sampler_m.samples.mean_composition()
for sp in expected.keys():
    self.assertAlmostEqual(expected[sp], comp[sp], places=2)
sampler_m.clear_samples()

self.fsgensemble.temperature = 1E9  # go real high to be uniform
sampler_f.run(steps, self.occu)
expected = {'Na+': 0.5, 'Cl-': 0.5}
comp = sampler_f.samples.mean_composition()
for sp in expected.keys():
    self.assertAlmostEqual(expected[sp], comp[sp], places=1)
"""


def test_reshape_occu(ensemble):
    sampler = Sampler.from_ensemble(ensemble, temperature=TEMPERATURE)
    occu = gen_random_occupancy(ensemble.sublattices)
    assert sampler._reshape_occu(occu).shape == (1, len(occu))
