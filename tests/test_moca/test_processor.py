import pytest
import numpy as np
import numpy.testing as npt
from tests.utils import assert_msonable
from smol.cofe.extern import EwaldTerm
from smol.moca import CEProcessor, EwaldProcessor, CompositeProcessor
from smol.moca.processor.base import Processor

RTOL = 0.0  # relative tolerance to check property change functions
# absolute tolerance to check property change functions (eps is approx 2E-16)
ATOL = 2E4 * np.finfo(float).eps
DRIFT_TOL = 10 * np.finfo(float).eps  # tolerance of average drift


# helper functions to generate test data
def gen_sublattices(proc):
    # sublattices as simple dicts
    sublattices = []
    for space in proc.unique_site_spaces:
        sites = np.array([i for i, b in enumerate(proc.allowed_species)
                          if b == list(space.keys())])
        sublattices.append({'codes': list(range(len(space))),
                            'space': space,
                            'sites': sites})
    return sublattices


def gen_occupancy(sublattice_dicts):
    rand_occu = np.zeros(sum(len(s['sites']) for s in sublattice_dicts), dtype=int)
    for sublatt in sublattice_dicts:
        rand_occu[sublatt['sites']] = np.random.choice(sublatt['codes'],
                                                       size=len(sublatt['sites']),
                                                       replace=True)
    return rand_occu


# Fixtures to run tests with
@pytest.fixture(scope='module')
def ceprocessor_data(ce_system):
    subspace, dataset = ce_system
    scmatrix = 5 * np.eye(3)
    proc = CEProcessor(subspace, scmatrix, coefficients=dataset['coefs'])
    sublattices = gen_sublattices(proc)
    rand_occu = gen_occupancy(sublattices)
    return proc, sublattices, rand_occu


@pytest.fixture(scope='module')
def comprocessor_data(composite_system):
    subspace, dataset = composite_system
    scmatrix = 5 * np.eye(3)
    ewald_term = EwaldTerm()
    subspace.add_external_term(ewald_term)
    proc = CompositeProcessor(subspace, scmatrix)
    coeffs = dataset['extern']['coefs']
    proc.add_processor(CEProcessor, coefficients=coeffs[:-1])
    proc.add_processor(EwaldProcessor, coefficient=coeffs[-1],
                       ewald_term=ewald_term)
    sublattices = gen_sublattices(proc)
    rand_occu = gen_occupancy(sublattices)
    return proc, sublattices, rand_occu


@pytest.fixture(scope='module', params=['real', 'reciprocal', 'point'])
def ewaldprocessor_data(electrostatic_system, request):
    subspace, dataset = electrostatic_system
    ewald_term = EwaldTerm(use_term=request.param)
    scmatrix = 5 * np.eye(3)
    proc = EwaldProcessor(subspace, scmatrix, ewald_term=ewald_term)
    sublattices = gen_sublattices(proc)
    rand_occu = gen_occupancy(sublattices)
    return proc, sublattices, rand_occu


# General tests for all processors
# Currently being done only on composites because I can not for the life of
# me figure out a clean way to parametrize with parametrized fixtures or use a
# fixture union from pytest_cases that works.
def test_encode_decode_property(comprocessor_data):
    processor, sublatts, occu = comprocessor_data
    decoccu = processor.decode_occupancy(occu)
    for species, space in zip(decoccu, processor.allowed_species):
        assert species in space
    npt.assert_equal(occu, processor.encode_occupancy(decoccu))


def test_get_average_drift(comprocessor_data):
    processor, _, _ = comprocessor_data
    forward, reverse = processor.compute_average_drift()
    assert forward <= DRIFT_TOL and reverse <= DRIFT_TOL


def test_compute_property_change(comprocessor_data):
    processor, sublatts, occu = comprocessor_data
    for _ in range(100):
        sublatt = np.random.choice(sublatts)
        site = np.random.choice(sublatt['sites'])
        new_sp = np.random.choice(sublatt['codes'])
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
        assert dprop == -1 * rdprop


# TODO implement these
def test_structure_from_occupancy():
    pass


def test_occupancy_from_structure():
    pass


def test_compute_feature_change(comprocessor_data):
    processor, sublatts, occu = comprocessor_data
    for _ in range(100):
        sublatt = np.random.choice(sublatts)
        site = np.random.choice(sublatt['sites'])
        new_sp = np.random.choice(sublatt['codes'])
        new_occu = occu.copy()
        new_occu[site] = new_sp
        # Test forward
        dcorr = processor.compute_feature_vector_change(occu, [(site, new_sp)])
        corr_f = processor.compute_feature_vector(new_occu)
        corr_i = processor.compute_feature_vector(occu)

        npt.assert_allclose(dcorr, corr_f - corr_i, rtol=RTOL, atol=ATOL)
        # Test reverse matches forward
        old_sp = occu[site]
        rdcorr = processor.compute_feature_vector_change(new_occu, [(site, old_sp)])
        npt.assert_array_equal(dcorr, -1 * rdcorr)


def test_compute_property(comprocessor_data):
    processor, _, occu = comprocessor_data
    struct = processor.structure_from_occupancy(occu)
    pred = np.dot(processor.coefs,
                  processor.cluster_subspace.corr_from_structure(struct, False))
    assert processor.compute_property(occu) == pytest.approx(pred, abs=ATOL)


def test_msonable(comprocessor_data):
    processor, _, occu = comprocessor_data
    d = processor.as_dict()
    pr = Processor.from_dict(d)
    assert processor.compute_property(occu) == pr.compute_property(occu)
    # TODO error in constructor of basis functions
    #assert_msonable(ceprocessor)


# CEProcessor only tests
def test_compute_feature_vector(ceprocessor_data):
    processor, _, occu = ceprocessor_data
    struct = processor.structure_from_occupancy(occu)
    # same as normalize=False in corr_from_structure
    npt.assert_allclose(processor.compute_feature_vector(occu) / processor.size,
                        processor.cluster_subspace.corr_from_structure(struct))


def test_feature_change_indictator(ceprocessor_data):
    processor, sublatts, occu = ceprocessor_data
    processor.cluster_subspace.change_site_bases('indicator')
    pr = CEProcessor(processor.cluster_subspace,
                     processor.supercell_matrix, processor.coefs,
                     optimize_indicator=True)
    for _ in range(100):
        sublatt = np.random.choice(sublatts)
        site = np.random.choice(sublatt['sites'])
        new_sp = np.random.choice(sublatt['codes'])
        new_occu = occu.copy()
        new_occu[site] = new_sp
        # Test forward
        dcorr = pr.compute_feature_vector_change(occu, [(site, new_sp)])
        corr_f = pr.compute_feature_vector(new_occu)
        corr_i = pr.compute_feature_vector(occu)
        npt.assert_allclose(dcorr, corr_f - corr_i, rtol=RTOL, atol=ATOL)
        # Test reverse matches forward
        old_sp = occu[site]
        rdcorr = pr.compute_feature_vector_change(new_occu, [(site, old_sp)])
        npt.assert_allclose(dcorr, -1 * rdcorr, rtol=RTOL, atol=ATOL)


def test_bad_coef_length(ce_system):
    subspace, dataset = ce_system
    with pytest.raises(ValueError):
        CEProcessor(subspace, 5*np.eye(3), coefficients=dataset['coefs'][:-1])


# Ewald only tests, these are basically copy and paste from above
# read comment on parametrizing :(
def test_get_average_drift(ewaldprocessor_data):
    processor, _, _ = ewaldprocessor_data
    forward, reverse = processor.compute_average_drift()
    assert forward <= DRIFT_TOL and reverse <= DRIFT_TOL


def test_compute_property_change(ewaldprocessor_data):
    processor, sublatts, occu = ewaldprocessor_data
    for _ in range(100):
        sublatt = np.random.choice(sublatts)
        site = np.random.choice(sublatt['sites'])
        new_sp = np.random.choice(sublatt['codes'])
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
        assert dprop == -1 * rdprop
