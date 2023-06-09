import os
from itertools import product

import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.core.composition import Composition

from smol.cofe.space.domain import SiteSpace
from smol.moca import SampleContainer
from smol.moca.sublattice import Sublattice
from smol.moca.trace import Trace
from tests.utils import assert_msonable

NSAMPLES = 100
# compositions will depend on number of sites and num species in the
# sublattices created in the fixture, if changing make sure it works out.
SUBLATTICE_COMPOSITIONS = [1.0 / 3.0, 1.0 / 2.0]


@pytest.fixture(params=[1, 5])
def container(request, single_sgc_ensemble, rng):
    trace = Trace(
        occupancy=np.zeros(
            (0, request.param, single_sgc_ensemble.num_sites), dtype=int
        ),
        features=rng.random(
            (0, request.param, len(single_sgc_ensemble.natural_parameters))
        ),
        enthalpy=np.ones((0, request.param, 1)),
        temperature=np.zeros((0, request.param, 1)),
        accepted=np.zeros((0, request.param, 1), dtype=bool),
    )
    sample_container = SampleContainer(
        single_sgc_ensemble,
        sample_trace=trace,
        sampling_metadata=single_sgc_ensemble.thermo_boundaries,
    )
    yield sample_container
    sample_container.clear()


@pytest.fixture
def fake_traces(container, rng):
    nwalkers = container.shape[0]
    occus = np.empty((NSAMPLES, nwalkers, container.shape[-1]))
    enths = -5 * np.ones((NSAMPLES, nwalkers, 1))
    temps = -300.56 * np.ones((NSAMPLES, nwalkers, 1))
    feats = np.zeros((NSAMPLES, nwalkers, len(container.natural_parameters)))
    accepted = rng.integers(2, size=(NSAMPLES, nwalkers, 1))
    sites1 = container.sublattices[0].sites
    sites2 = container.sublattices[1].sites
    for i in range(NSAMPLES):
        for j in range(nwalkers):
            # make occupancy compositions (1/3, 1/3, 1/3) & (1/2, 1/2)
            size = int(SUBLATTICE_COMPOSITIONS[0] * len(sites1))
            s = rng.choice(sites1, size=size, replace=False)
            occus[i, j, s] = 0
            s1 = rng.choice(np.setdiff1d(sites1, s), size=size, replace=False)
            occus[i, j, s1] = 1
            s2 = np.setdiff1d(sites1, np.append(s, s1))
            occus[i, j, s2] = 2
            size = int(SUBLATTICE_COMPOSITIONS[1] * len(sites2))
            s = rng.choice(sites2, size=size, replace=False)
            occus[i, j, s] = 0
            occus[i, j, np.setdiff1d(sites2, s)] = 1
            # first and last feature real fake
            feats[i, j, [0, -1]] = 2.5
    traces = [
        Trace(
            occupancy=occus[i],
            features=feats[i],
            enthalpy=enths[i],
            accepted=accepted[i],
            temperature=temps[i],
        )
        for i in range(len(accepted))
    ]
    return traces


def add_samples(sample_container, fake_traces, thinned_by=1):
    sample_container.allocate(len(fake_traces))
    for trace in fake_traces:
        sample_container.save_sampled_trace(trace, thinned_by=thinned_by)


def test_allocate_and_save(container, fake_traces, rng):
    nwalkers = container.shape[0]
    nsites = container.shape[-1]
    assert len(container) == 0
    assert container._trace.occupancy.shape == (0, nwalkers, nsites)
    assert container._trace.enthalpy.shape == (0, nwalkers, 1)
    assert container._trace.accepted.shape == (0, nwalkers, 1)

    container.allocate(NSAMPLES)
    assert len(container) == 0
    assert container._trace.occupancy.shape == (NSAMPLES, nwalkers, nsites)
    assert container._trace.enthalpy.shape == (NSAMPLES, nwalkers, 1)
    assert container._trace.accepted.shape == (NSAMPLES, nwalkers, 1)
    container.clear()

    add_samples(container, fake_traces)
    assert len(container) == NSAMPLES
    assert container._total_steps == NSAMPLES
    container.clear()

    thinned = rng.integers(50)
    add_samples(container, fake_traces, thinned_by=thinned)
    assert len(container) == NSAMPLES
    assert container._total_steps == thinned * NSAMPLES
    container.clear()


@pytest.mark.parametrize("discard, thin", product((0, 10), (1, 5)))
def test_get_sampled_values(container, fake_traces, discard, thin):
    nsites = container.shape[-1]
    add_samples(container, fake_traces)
    nat_params = container.natural_parameters
    sublattices = container.sublattices
    nw = container.shape[0]
    # get default flatted values
    accepted = container._trace.accepted
    nsamples = (NSAMPLES - discard) // thin
    expected = (
        accepted[discard:].sum(axis=0) / (container._total_steps - discard)
    ).mean()
    assert container.sampling_efficiency(discard=discard) == expected
    assert container.get_occupancies(discard=discard, thin_by=thin).shape == (
        nsamples * nw,
        nsites,
    )
    assert container.get_feature_vectors(discard=discard, thin_by=thin).shape == (
        nsamples * nw,
        len(nat_params),
    )
    assert container.get_enthalpies(discard=discard, thin_by=thin).shape == (
        nsamples * nw,
    )
    assert container.get_energies(discard=discard, thin_by=thin).shape == (
        nsamples * nw,
    )
    for sublattice, comp in zip(sublattices, SUBLATTICE_COMPOSITIONS):
        c = container.get_sublattice_compositions(
            sublattice, discard=discard, thin_by=thin
        )
        if len(sublattice.species) == 1:  # single species sublattice
            assert c.shape == (nsamples * nw,)
        else:
            assert c.shape == (nsamples * nw, len(sublattice.species))
        npt.assert_array_equal(c, comp * np.ones_like(c))

    # get non flattened values
    npt.assert_array_equal(
        container.sampling_efficiency(discard=discard, flat=False),
        (accepted[discard:,].sum(axis=0) / (container._total_steps - discard)),
    )
    assert container.get_occupancies(
        discard=discard, thin_by=thin, flat=False
    ).shape == (nsamples, nw, nsites)
    assert container.get_feature_vectors(
        discard=discard, thin_by=thin, flat=False
    ).shape == (nsamples, nw, len(nat_params))
    assert container.get_enthalpies(
        discard=discard, thin_by=thin, flat=False
    ).shape == (nsamples, nw, 1)
    assert container.get_energies(discard=discard, thin_by=thin, flat=False).shape == (
        nsamples,
        nw,
        1,
    )

    for sublattice, comp in zip(sublattices, SUBLATTICE_COMPOSITIONS):
        c = container.get_sublattice_compositions(
            sublattice, discard=discard, thin_by=thin, flat=False
        )
        assert c.shape == (nsamples, nw, len(sublattice.species))
        npt.assert_array_equal(c, comp * np.ones_like(c))

    with pytest.raises(ValueError):
        sublattice = Sublattice(SiteSpace(Composition({"foo": 7})), np.zeros(10))
        container.get_sublattice_compositions(sublattice)


@pytest.mark.parametrize("discard, thin", product((0, 10), (1, 5)))
def test_means_and_variances(container, fake_traces, discard, thin):
    add_samples(container, fake_traces)
    sublattices = container.sublattices
    nw = container.shape[0]
    assert container.mean_enthalpy(discard=discard, thin_by=thin) == -5.0
    assert container.enthalpy_variance(discard=discard, thin_by=thin) == 0.0
    assert container.mean_energy(discard=discard, thin_by=thin) == -2.5
    assert container.energy_variance(discard=discard, thin_by=thin) == 0.0
    npt.assert_array_equal(
        container.mean_feature_vector(discard=discard, thin_by=thin),
        [2.5]
        + (len(container.natural_parameters) - 2)
        * [
            0,
        ]
        + [2.5],
    )
    npt.assert_array_equal(
        container.feature_vector_variance(discard=discard, thin_by=thin),
        len(container.natural_parameters)
        * [
            0,
        ],
    )
    for sublattice, comp in zip(sublattices, SUBLATTICE_COMPOSITIONS):
        npt.assert_array_almost_equal(
            container.mean_sublattice_composition(
                sublattice, discard=discard, thin_by=thin
            ),
            len(sublattice.species)
            * [
                comp,
            ],
        )
        npt.assert_array_almost_equal(
            container.sublattice_composition_variance(
                sublattice, discard=discard, thin_by=thin
            ),
            len(sublattice.species)
            * [
                0,
            ],
        )

    # without flattening
    npt.assert_array_equal(
        container.mean_enthalpy(discard=discard, thin_by=thin, flat=False),
        np.array([nw * [-5.0]]).T,
    )
    npt.assert_array_equal(
        container.enthalpy_variance(discard=discard, thin_by=thin, flat=False),
        np.array([nw * [0.0]]).T,
    )
    npt.assert_array_equal(
        container.mean_energy(discard=discard, thin_by=thin, flat=False),
        np.array([nw * [-2.5]]).T,
    )
    npt.assert_array_equal(
        container.energy_variance(discard=discard, thin_by=thin, flat=False),
        np.array([nw * [0.0]]).T,
    )
    npt.assert_array_equal(
        container.mean_feature_vector(discard=discard, thin_by=thin, flat=False),
        nw
        * [
            [2.5]
            + (len(container.natural_parameters) - 2)
            * [
                0.0,
            ]
            + [2.5]
        ],
    )
    npt.assert_array_equal(
        container.feature_vector_variance(discard=discard, thin_by=thin, flat=False),
        nw
        * [
            len(container.natural_parameters)
            * [
                0.0,
            ]
        ],
    )
    for sublattice, comp in zip(sublattices, SUBLATTICE_COMPOSITIONS):
        npt.assert_array_almost_equal(
            container.mean_sublattice_composition(
                sublattice, discard=discard, thin_by=thin, flat=False
            ),
            nw * [len(sublattice.species) * [comp]],
        )
        npt.assert_array_almost_equal(
            container.sublattice_composition_variance(
                sublattice, discard=discard, thin_by=thin, flat=False
            ),
            nw * [len(sublattice.species) * [0]],
        )


def test_get_mins(container, fake_traces, rng):
    add_samples(container, fake_traces)
    i = rng.choice(range(NSAMPLES))
    nwalkers = container.shape[0]
    container._trace.enthalpy[i, :] = -11
    container._trace.features[i, :, 0] = 5.0
    assert container.get_minimum_enthalpy() == -11.0
    assert container.get_minimum_energy() == -5.0
    npt.assert_array_equal(
        container.get_minimum_enthalpy(flat=False), np.array([nwalkers * [-11.0]]).T
    )
    npt.assert_array_equal(
        container.get_minimum_energy(flat=False), np.array([nwalkers * [-5.0]]).T
    )
    npt.assert_array_equal(
        container.get_minimum_enthalpy_occupancy(flat=False),
        container._trace.occupancy[i],
    )
    npt.assert_array_equal(
        container.get_minimum_energy_occupancy(flat=False),
        container._trace.occupancy[i],
    )


def test_msonable(container, fake_traces):
    # fails for empty container with nwalkers > 1
    # since the _chain.tolist turns the empty array to an empty list and so
    # the shape is lost, but dont think anyone really cares about an empty
    # container....
    # d = container.as_dict()
    # cntr = container.from_dict(d)
    # assert cntr.shape == container.shape

    add_samples(container, fake_traces)
    d = container.as_dict()
    cntr = container.from_dict(d)
    assert cntr.shape == container.shape
    npt.assert_array_equal(container.get_occupancies(), cntr.get_occupancies())
    assert_msonable(cntr)


def test_hdf5(container, fake_traces, tmpdir):
    add_samples(container, fake_traces)
    file_path = os.path.join(tmpdir, "test.h5")
    container.to_hdf5(file_path)
    cntr = SampleContainer.from_hdf5(file_path)
    npt.assert_array_equal(container.get_occupancies(), cntr.get_occupancies())
    npt.assert_array_equal(container.get_energies(), cntr.get_energies())
    os.remove(file_path)


@pytest.mark.parametrize("mode", [False, True])
def test_flush_to_hdf5(container, fake_traces, mode, tmpdir):
    # deepcopy does not work with Cython extensions with nontrivial __cinit__
    # and creating a new container from dict of an empty one loses shape information
    # so we add samples first and then create the flushed container, and clear it
    add_samples(container, fake_traces)
    flushed_container = SampleContainer.from_dict(container.as_dict())
    flushed_container.clear()

    file_path = os.path.join(tmpdir, "test.h5")
    chunk = len(fake_traces) // 4
    flushed_container.allocate(chunk)
    backend = flushed_container.get_backend(file_path, len(fake_traces), swmr_mode=mode)

    start = 0
    for _ in range(4):
        for i in range(start, start + chunk):
            flushed_container.save_sampled_trace(fake_traces[i], thinned_by=1)
        assert flushed_container._trace.occupancy.shape[0] == chunk
        flushed_container.flush_to_backend(backend)
        start += chunk

    if mode is False:
        backend.close()

    assert flushed_container._trace.occupancy.shape[0] == chunk
    assert flushed_container.num_samples == 0
    cntr = SampleContainer.from_hdf5(file_path)
    assert cntr.num_samples == container.num_samples
    assert cntr.total_mc_steps == container.total_mc_steps
    npt.assert_array_equal(container.get_occupancies(), cntr.get_occupancies())
    npt.assert_array_equal(container.get_energies(), cntr.get_energies())

    if mode is True:
        backend.close()

    # now get it again, and make sure more space is allocated
    add_samples(container, fake_traces)
    backend = flushed_container.get_backend(file_path, len(fake_traces))
    assert backend["trace"].attrs["nsamples"] == len(fake_traces)
    assert len(backend["trace"]["occupancy"]) == 2 * len(fake_traces)

    start = 0
    for _ in range(4):
        for i in range(start, start + chunk):
            flushed_container.save_sampled_trace(fake_traces[i], thinned_by=1)
        assert flushed_container._trace.occupancy.shape[0] == chunk
        flushed_container.flush_to_backend(backend)
        start += chunk

    assert flushed_container._trace.occupancy.shape[0] == chunk
    assert flushed_container.num_samples == 0
    cntr = SampleContainer.from_hdf5(file_path)
    assert cntr.num_samples == container.num_samples
    assert cntr.total_mc_steps == container.total_mc_steps
    npt.assert_array_equal(container.get_occupancies(), cntr.get_occupancies())
    npt.assert_array_equal(container.get_energies(), cntr.get_energies())

    backend.close()

    os.remove(file_path)


def test_get_sampled_structures(container, fake_traces, rng):
    # not the best test here, but it will do for now...
    add_samples(container, fake_traces)
    ids = rng.choice(range(container.num_samples), size=5)

    # check for flattened chains
    structures = container.get_sampled_structures(ids, flat=True)
    for i, struct in zip(ids, structures):
        occu = container.get_occupancies()[i]
        assert container.ensemble.processor.structure_from_occupancy(occu) == struct

    # check for unflattened chains
    structures_list = container.get_sampled_structures(ids, flat=False)
    for i, structures in zip(ids, structures_list):
        occus = container.get_occupancies(flat=False)[i]
        for occu, struct in zip(occus, structures):
            assert container.ensemble.processor.structure_from_occupancy(occu) == struct


def test_allocate_vacuum(container, fake_traces):
    add_samples(container, fake_traces)
    n_samples = len(container)

    container.allocate(100)
    assert len(container) == n_samples
    assert container._trace.occupancy.shape[0] == n_samples + 100

    container.vacuum()
    assert len(container) == n_samples
    assert container._trace.occupancy.shape[0] == n_samples
