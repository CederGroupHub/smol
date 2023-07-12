import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.capp.generate.random import _gen_unconstrained_ordered_occu
from smol.cofe import ClusterExpansion
from smol.cofe.extern import EwaldTerm
from smol.cofe.space.domain import Vacancy, get_allowed_species
from smol.moca.processor import (
    ClusterDecompositionProcessor,
    ClusterExpansionProcessor,
    CompositeProcessor,
    EwaldProcessor,
)
from smol.moca.processor.base import Processor
from smol.moca.processor.distance import (
    ClusterInteractionDistanceProcessor,
    CorrelationDistanceProcessor,
)
from smol.utils._openmp_helpers import _openmp_effective_numthreads
from smol.utils.cluster.numthreads import DEFAULT_NUM_THREADS
from tests.utils import assert_msonable, assert_pickles, gen_random_ordered_structure

pytestmark = pytest.mark.filterwarnings("ignore:All bit combos have been removed")

RTOL = 1e-12  # relative tolerance to check property change functions
# absolute tolerance to check property change functions (eps is approx 2E-16)
ATOL = 2e4 * np.finfo(float).eps
DRIFT_TOL = 10 * np.finfo(float).eps  # tolerance of average drift


@pytest.fixture
def ce_processor(cluster_subspace, rng):
    coefs = 2 * rng.random(cluster_subspace.num_corr_functions)
    scmatrix = 3 * np.eye(3)
    return ClusterExpansionProcessor(
        cluster_subspace, supercell_matrix=scmatrix, coefficients=coefs
    )


@pytest.fixture(params=["real", "reciprocal", "point"])
def ewald_processor(cluster_subspace, rng, request):
    coef = rng.random(1)
    scmatrix = 3 * np.eye(3)
    ewald_term = EwaldTerm(use_term=request.param)
    return EwaldProcessor(
        cluster_subspace,
        supercell_matrix=scmatrix,
        coefficient=coef,
        ewald_term=ewald_term,
    )


@pytest.fixture(params=["correlation", "interaction"])
def processor_distance_processor(cluster_subspace, rng, request):
    # return a processor and the corresponding distance processor
    scmatrix = 3 * np.eye(3)
    coefs = np.ones(len(cluster_subspace))
    if request.param == "correlation":
        target_weights = rng.random(len(cluster_subspace) - 1)
        procs = (
            ClusterExpansionProcessor(cluster_subspace, scmatrix, coefs),
            CorrelationDistanceProcessor(
                cluster_subspace, scmatrix, target_weights=target_weights
            ),
        )
    else:
        expansion = ClusterExpansion(cluster_subspace, coefs)
        target_weights = rng.random(cluster_subspace.num_orbits - 1)
        procs = (
            ClusterDecompositionProcessor(
                cluster_subspace, scmatrix, expansion.cluster_interaction_tensors
            ),
            ClusterInteractionDistanceProcessor(
                cluster_subspace,
                scmatrix,
                expansion.cluster_interaction_tensors,
                target_weights=target_weights,
            ),
        )
    return procs


def test_encode_decode_property(ce_processor, rng):
    occu = _gen_unconstrained_ordered_occu(ce_processor.get_sublattices(), rng=rng)
    decoccu = ce_processor.decode_occupancy(occu)
    for species, space in zip(decoccu, ce_processor.allowed_species):
        assert species in space
    npt.assert_equal(occu, ce_processor.encode_occupancy(decoccu))


def test_site_spaces(ce_processor):
    assert all(
        sp in ce_processor.structure.composition
        for space in ce_processor.unique_site_spaces
        for sp in space
        if sp != Vacancy()
    )
    assert all(
        sp in ce_processor._subspace.expansion_structure.composition
        for space in ce_processor.active_site_spaces
        for sp in space
        if sp != Vacancy()
    )


def test_sublattice(ce_processor):
    sublattices = ce_processor.get_sublattices()
    # These are default initialized, not split.
    site_species = get_allowed_species(ce_processor.structure)
    for sublatt, site_space in zip(sublattices, ce_processor.unique_site_spaces):
        assert sublatt.site_space == site_space
        for site in sublatt.sites:
            assert site_species[site] == list(site_space.keys())
    assert sum(len(sublatt.sites) for sublatt in sublattices) == len(
        ce_processor.structure
    )


def test_structure_occupancy_conversion(ce_processor, rng):
    sm = StructureMatcher()
    for _ in range(10):
        s_init = gen_random_ordered_structure(
            ce_processor.cluster_subspace.structure,
            size=ce_processor.supercell_matrix,
            rng=rng,
        )
        s_init = s_init.get_sorted_structure()
        occu_init = ce_processor.occupancy_from_structure(s_init)

        s_conv = ce_processor.structure_from_occupancy(occu_init)
        s_conv = s_conv.get_sorted_structure()

        # occu_conv = ce_processor.occupancy_from_structure(s_conv)

        # For symmetrically equivalent structures, StructureMatcher might generate
        # different structure_site_mappings
        # (see cluster_subspace.structure_site_mappings), therefore we may get
        # different occupancy strings with occupancy_from_structure, and
        # occu1 -> structure -> occu2 conversion cycle does not guarantee that
        # occu1 == occu2. In most use cases, it is not necessary to enforce that
        # occu1 == occu2. If you have to do so, you'll need to deeply modify the code of
        # StructureMatcher, which might not be a trivial task. Here we will only test
        # whether occu1 -> str1 and occu2 -> str2 are symmetrically equivalent.
        # This should be enough in our application. We notify the users about this
        # mismatch in the documentations.
        assert sm.fit(s_init, s_conv)


# only test composite_processor predicting an energy value
def test_compute_energy(composite_processor, rng):
    occu = _gen_unconstrained_ordered_occu(
        composite_processor.get_sublattices(), rng=rng
    )
    struct = composite_processor.structure_from_occupancy(occu)
    pred = np.dot(
        composite_processor.raw_coefs,
        composite_processor.cluster_subspace.corr_from_structure(struct, False),
    )
    assert composite_processor.compute_property(occu) == pytest.approx(pred, abs=ATOL)


# General tests for all processors
# Currently being done only on composites because I can not for the life of
# me figure out a clean way to parametrize with parametrized fixtures or use a
# fixture union from pytest_cases that works.


def test_get_average_drift(processor):
    forward, reverse = processor.compute_average_drift()
    assert forward <= DRIFT_TOL and reverse <= DRIFT_TOL


def test_compute_property_change(processor, rng):
    sublattices = processor.get_sublattices()
    occu = _gen_unconstrained_ordered_occu(sublattices, rng=rng)
    active_sublattices = [sublatt for sublatt in sublattices if sublatt.is_active]

    for _ in range(100):
        sublatt = rng.choice(active_sublattices)
        site = rng.choice(sublatt.sites)
        new_sp = rng.choice(sublatt.encoding)
        new_occu = occu.copy()
        new_occu[site] = new_sp
        prop_f = processor.compute_property(new_occu)
        prop_i = processor.compute_property(occu)
        dprop = processor.compute_property_change(occu, [(site, new_sp)])
        # Check with some tight tolerances.
        npt.assert_allclose(dprop, prop_f - prop_i, rtol=RTOL, atol=ATOL)
        # Test reverse matches forward
        old_sp = occu[site]
        rdprop = processor.compute_property_change(new_occu, [(site, old_sp)])
        npt.assert_allclose(dprop, -1 * rdprop, rtol=RTOL, atol=ATOL)


def test_compute_feature_change(processor, rng):
    sublattices = processor.get_sublattices()
    occu = _gen_unconstrained_ordered_occu(sublattices, rng=rng)
    active_sublattices = [sublatt for sublatt in sublattices if sublatt.is_active]
    processor.cluster_subspace.change_site_bases("indicator")

    for _ in range(100):
        sublatt = rng.choice(active_sublattices)
        site = rng.choice(sublatt.sites)
        new_sp = rng.choice(sublatt.encoding)
        new_occu = occu.copy()
        new_occu[site] = new_sp
        feat_f = processor.compute_feature_vector(new_occu)
        feat_i = processor.compute_feature_vector(occu)
        dfeat = processor.compute_feature_vector_change(occu, [(site, new_sp)])
        # Check with some tight tolerances.
        npt.assert_allclose(dfeat, feat_f - feat_i, rtol=RTOL, atol=ATOL)
        # Test reverse matches forward
        old_sp = occu[site]
        rdfeat = processor.compute_feature_vector_change(new_occu, [(site, old_sp)])
        npt.assert_allclose(dfeat, -1 * rdfeat, rtol=RTOL, atol=ATOL)


def test_msonable(processor, rng):
    occu = _gen_unconstrained_ordered_occu(processor.get_sublattices(), rng=rng)
    d = processor.as_dict()
    pr = Processor.from_dict(d)
    assert processor.compute_property(occu) == pr.compute_property(occu)
    npt.assert_array_equal(processor.coefs, pr.coefs)
    # send in pr bc composite_processor is scoped for function and new random
    # coefficients will be created.
    assert_msonable(pr)


def test_pickles(processor, rng):
    assert_pickles(processor)


# ClusterExpansionProcessor only tests
def test_compute_correlation_vector(ce_processor, rng):
    occu = _gen_unconstrained_ordered_occu(ce_processor.get_sublattices(), rng=rng)
    struct = ce_processor.structure_from_occupancy(occu)
    # same as normalize=False in corr_from_structure
    npt.assert_allclose(
        ce_processor.compute_feature_vector(occu) / ce_processor.size,
        ce_processor.cluster_subspace.corr_from_structure(struct),
    )

    npt.assert_allclose(
        ce_processor.compute_feature_vector(occu) / ce_processor.size,
        ce_processor.cluster_subspace.corr_from_structure(struct),
    )


# cluster decomposition processor
def test_compute_cluster_interactions(cluster_subspace, rng):
    coefs = 2 * np.random.random(cluster_subspace.num_corr_functions)
    scmatrix = 3 * np.eye(3)
    expansion = ClusterExpansion(cluster_subspace, coefs)
    processor = ClusterDecompositionProcessor(
        cluster_subspace, scmatrix, expansion.cluster_interaction_tensors
    )

    occu = _gen_unconstrained_ordered_occu(processor.get_sublattices(), rng=rng)
    struct = processor.structure_from_occupancy(occu)
    # same as normalize=False in corr_from_structure
    proc_interactions = processor.compute_feature_vector(occu) / processor.size
    exp_interactions = expansion.cluster_interactions_from_structure(struct)
    npt.assert_allclose(proc_interactions, exp_interactions)
    pred_energy = expansion.predict(struct, normalized=True)
    assert sum(
        cluster_subspace.orbit_multiplicities * proc_interactions
    ) == pytest.approx(pred_energy, abs=ATOL)


def test_bad_coef_length(cluster_subspace, rng):
    coefs = rng.random(cluster_subspace.num_corr_functions - 1)
    with pytest.raises(ValueError):
        ClusterExpansionProcessor(cluster_subspace, 5 * np.eye(3), coefficients=coefs)


def test_bad_composite(cluster_subspace, rng):
    coefs = 2 * rng.random(cluster_subspace.num_corr_functions)
    scmatrix = 3 * np.eye(3)
    proc = CompositeProcessor(cluster_subspace, supercell_matrix=scmatrix)
    with pytest.raises(AttributeError):
        proc.add_processor(
            CompositeProcessor(cluster_subspace, supercell_matrix=scmatrix)
        )
    with pytest.raises(ValueError):
        proc.add_processor(
            ClusterExpansionProcessor(
                cluster_subspace, 2 * scmatrix, coefficients=coefs
            )
        )
    with pytest.raises(ValueError):
        new_cs = cluster_subspace.copy()
        ids = range(1, new_cs.num_corr_functions)
        new_cs.remove_corr_functions(rng.choice(ids, size=10))
        proc.add_processor(
            ClusterExpansionProcessor(new_cs, scmatrix, coefficients=coefs)
        )


def test_distance_processor(processor_distance_processor, rng):
    # test distance processor results vs compute directly from the corresponding
    # processor
    processor, distance_processor = processor_distance_processor
    for _ in range(5):
        occu = _gen_unconstrained_ordered_occu(processor.get_sublattices(), rng=rng)

        # remove first entry since it is the "exact match diameter" in the distance metric
        expected = abs(
            processor.compute_feature_vector(occu)[1:] / processor.size
            - distance_processor.target_vector[1:]
        )
        actual = distance_processor.compute_feature_vector(occu)
        npt.assert_allclose(actual[1:], expected)

        diameter = distance_processor.exact_match_max_diameter(actual)
        expected = np.dot(
            distance_processor.coefs, np.concatenate([[diameter], expected])
        )
        actual = distance_processor.compute_property(occu)
        assert actual == pytest.approx(expected)

        for _ in range(100):
            active_sublattices = [
                sublatt for sublatt in processor.get_sublattices() if sublatt.is_active
            ]
            sublatt = rng.choice(active_sublattices)
            site = rng.choice(sublatt.sites)
            new_sp = rng.choice(sublatt.encoding)
            new_occu = occu.copy()
            new_occu[site] = new_sp
            flips = [(site, new_sp)]

            # remove first element since it is the "exact match diameter" in the distance metric
            dist_f = abs(
                processor.compute_feature_vector(new_occu)[1:] / processor.size
                - distance_processor.target_vector[1:]
            )
            dist_i = abs(
                processor.compute_feature_vector(occu)[1:] / processor.size
                - distance_processor.target_vector[1:]
            )

            distances = distance_processor.compute_feature_vector_distances(occu, flips)
            npt.assert_allclose(distances[0][1:], dist_i, rtol=RTOL, atol=ATOL)
            npt.assert_allclose(distances[1][1:], dist_f, rtol=RTOL, atol=ATOL)

            diameter_i = distance_processor.exact_match_max_diameter(distances[0])
            diameter_f = distance_processor.exact_match_max_diameter(distances[1])

            expected = np.dot(
                distance_processor.coefs,
                np.concatenate([[diameter_f - diameter_i], dist_f - dist_i]),
            )
            actual = distance_processor.compute_property_change(occu, flips)
            assert actual == pytest.approx(expected)


def test_exact_match_max_diameter(processor_distance_processor, rng):
    processor, distance_processor = processor_distance_processor

    # check a vector with all zeros returns the max diameter
    distance_vector = np.zeros(len(processor.coefs))
    max_diameter = max(processor.cluster_subspace.orbits_by_diameter.keys())
    assert distance_processor.exact_match_max_diameter(distance_vector) == max_diameter

    # check a random orbit in between
    # exclude zero diameter, and smallest diameter orbit
    diameter = rng.choice(
        list(processor.cluster_subspace.orbits_by_diameter.keys())[2:]
    )
    orbit = rng.choice(processor.cluster_subspace.orbits_by_diameter[diameter])

    if isinstance(distance_processor, CorrelationDistanceProcessor):
        # this is correlation based
        index = rng.choice(range(orbit.bit_id, orbit.bit_id + len(orbit)))
    else:
        index = orbit.id

    distance_vector[index] = 2 * distance_processor.match_tol
    assert 0 < distance_processor.exact_match_max_diameter(distance_vector) < diameter

    # check a vector exceeding match_tol for first point orbit
    distance_vector[1] = 2 * distance_processor.match_tol
    assert distance_processor.exact_match_max_diameter(distance_vector) == 0.0


def test_bad_distance_processor(single_subspace, rng):
    scm = 3 * np.eye(3)
    subspace = single_subspace.copy()

    with pytest.raises(ValueError):
        subspace.add_external_term(EwaldTerm())
        proc = CorrelationDistanceProcessor(subspace, scm)

    with pytest.raises(ValueError):
        proc = CorrelationDistanceProcessor(single_subspace, scm, match_weight=-1)

    with pytest.raises(ValueError):
        target_weights = np.ones(len(single_subspace) - 4)
        proc = CorrelationDistanceProcessor(
            single_subspace, scm, target_weights=target_weights
        )


def test_set_threads(single_subspace):
    coefs = 2 * np.random.random(single_subspace.num_corr_functions)
    scmatrix = 3 * np.eye(3)
    expansion = ClusterExpansion(single_subspace, coefs)

    ceproc = ClusterExpansionProcessor(single_subspace, scmatrix, coefs)
    cdproc = ClusterDecompositionProcessor(
        single_subspace,
        scmatrix,
        expansion.cluster_interaction_tensors,
    )

    # assert defaults
    assert ceproc.num_threads == DEFAULT_NUM_THREADS
    assert ceproc._evaluator.num_threads == ceproc.num_threads
    assert ceproc.num_threads_full == DEFAULT_NUM_THREADS

    for eval_data in ceproc._eval_data_by_sites.values():
        assert eval_data.evaluator.num_threads == ceproc.num_threads

    assert cdproc.num_threads == DEFAULT_NUM_THREADS
    assert ceproc.num_threads_full == DEFAULT_NUM_THREADS

    for eval_data in cdproc._eval_data_by_sites.values():
        assert eval_data.evaluator.num_threads == cdproc.num_threads

    # assert setting thread values
    ceproc.num_threads = 1
    cdproc.num_threads = 1

    ceproc.num_threads_full = 1
    cdproc.num_threads_full = 1

    assert ceproc.num_threads == 1
    assert ceproc.num_threads_full == 1

    for eval_data in ceproc._eval_data_by_sites.values():
        assert eval_data.evaluator.num_threads == 1

    assert cdproc.num_threads == 1
    assert ceproc.num_threads_full == 1

    for eval_data in ceproc._eval_data_by_sites.values():
        assert eval_data.evaluator.num_threads == 1

    # assert -1 gives max number of threads
    ceproc.num_threads = -1
    assert ceproc.num_threads == _openmp_effective_numthreads(n_threads=-1)

    # assert setting more complains
    with pytest.raises(ValueError):
        ceproc.num_threads = _openmp_effective_numthreads() + 1
