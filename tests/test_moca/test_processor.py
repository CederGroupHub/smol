import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.cofe.extern import EwaldTerm
from smol.cofe.space.domain import Vacancy, get_allowed_species
from smol.moca.processor import (
    ClusterExpansionProcessor,
    CompositeProcessor,
    EwaldProcessor,
)
from smol.moca.processor.base import Processor
from tests.utils import assert_msonable, gen_random_occupancy, gen_random_structure

pytestmark = pytest.mark.filterwarnings("ignore:All bit combos have been removed")

RTOL = 0.0  # relative tolerance to check property change functions
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


# General tests for all processors
# Currently being done only on composites because I can not for the life of
# me figure out a clean way to parametrize with parametrized fixtures or use a
# fixture union from pytest_cases that works.
def test_encode_decode_property(composite_processor):
    occu = gen_random_occupancy(composite_processor.get_sublattices())
    decoccu = composite_processor.decode_occupancy(occu)
    for species, space in zip(decoccu, composite_processor.allowed_species):
        assert species in space
    npt.assert_equal(occu, composite_processor.encode_occupancy(decoccu))


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


def test_get_average_drift(composite_processor):
    forward, reverse = composite_processor.compute_average_drift()
    assert forward <= DRIFT_TOL and reverse <= DRIFT_TOL


def test_compute_property_change(composite_processor, rng):
    sublattices = composite_processor.get_sublattices()
    occu = gen_random_occupancy(sublattices)
    active_sublattices = [sublatt for sublatt in sublattices if sublatt.is_active]

    for _ in range(100):
        sublatt = rng.choice(active_sublattices)
        site = rng.choice(sublatt.sites)
        new_sp = rng.choice(sublatt.encoding)
        new_occu = occu.copy()
        new_occu[site] = new_sp
        prop_f = composite_processor.compute_property(new_occu)
        prop_i = composite_processor.compute_property(occu)
        dprop = composite_processor.compute_property_change(occu, [(site, new_sp)])
        # Check with some tight tolerances.
        npt.assert_allclose(dprop, prop_f - prop_i, rtol=RTOL, atol=ATOL)
        # Test reverse matches forward
        old_sp = occu[site]
        rdprop = composite_processor.compute_property_change(new_occu, [(site, old_sp)])
        assert dprop == -1 * rdprop


def test_structure_occupancy_conversion(ce_processor):
    sm = StructureMatcher()
    for _ in range(10):
        s_init = gen_random_structure(
            ce_processor.cluster_subspace.structure, size=ce_processor.supercell_matrix
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


def test_compute_feature_change(composite_processor, rng):
    sublattices = composite_processor.get_sublattices()
    occu = gen_random_occupancy(sublattices)
    active_sublattices = [sublatt for sublatt in sublattices if sublatt.is_active]
    composite_processor.cluster_subspace.change_site_bases("indicator")

    for _ in range(100):
        sublatt = rng.choice(active_sublattices)
        site = rng.choice(sublatt.sites)
        new_sp = rng.choice(sublatt.encoding)
        new_occu = occu.copy()
        new_occu[site] = new_sp
        prop_f = composite_processor.compute_property(new_occu)
        prop_i = composite_processor.compute_property(occu)
        dprop = composite_processor.compute_property_change(occu, [(site, new_sp)])
        # Check with some tight tolerances.
        npt.assert_allclose(dprop, prop_f - prop_i, rtol=RTOL, atol=ATOL)
        # Test reverse matches forward
        old_sp = occu[site]
        rdprop = composite_processor.compute_property_change(new_occu, [(site, old_sp)])
        assert dprop == -1 * rdprop


def test_compute_property(composite_processor):
    occu = gen_random_occupancy(composite_processor.get_sublattices())
    struct = composite_processor.structure_from_occupancy(occu)
    pred = np.dot(
        composite_processor.coefs,
        composite_processor.cluster_subspace.corr_from_structure(struct, False),
    )
    assert composite_processor.compute_property(occu) == pytest.approx(pred, abs=ATOL)


def test_msonable(composite_processor):
    occu = gen_random_occupancy(composite_processor.get_sublattices())
    d = composite_processor.as_dict()
    pr = Processor.from_dict(d)
    assert composite_processor.compute_property(occu) == pr.compute_property(occu)
    npt.assert_array_equal(composite_processor.coefs, pr.coefs)
    # send in pr bc composite_processor is scoped for function and new random
    # coefficients will be created.
    assert_msonable(pr)


# ClusterExpansionProcessor only tests
def test_compute_feature_vector(ce_processor):
    occu = gen_random_occupancy(ce_processor.get_sublattices())
    struct = ce_processor.structure_from_occupancy(occu)
    # same as normalize=False in corr_from_structure
    npt.assert_allclose(
        ce_processor.compute_feature_vector(occu) / ce_processor.size,
        ce_processor.cluster_subspace.corr_from_structure(struct),
    )


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
