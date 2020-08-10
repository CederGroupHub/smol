import pytest
from pytest_cases import fixture_plus, fixture_union
import numpy as np
import numpy.testing as npt
from tests.utils import assert_msonable
from smol.moca import CEProcessor, EwaldProcessor, CompositeProcessor
from smol.moca.processor.base import Processor

RTOL = 0.0  # relative tolerance to check property change functions
# absolute tolerance to check property change functions (eps is approx 2E-16)
ATOL = 2E4 * np.finfo(float).eps
DRIFT_TOL = 10 * np.finfo(float).eps  # tolerance of average drift


def sublattice_dicts(processor):
    sublattices = []
    for space in processor.unique_site_spaces:
        sites = np.array([i for i, b in enumerate(processor.allowed_species)
                          if b == list(space.keys())])
        sublattices.append({'codes': list(range(len(space))),
                            'space': space,
                            'sites': sites})
    return sublattices


def occupancy(sublattice_dicts):
    occu = np.zeros(sum(len(s['sites']) for s in sublattice_dicts), dtype=int)
    for sublatt in sublattice_dicts:
        occu[sublatt['sites']] = np.random.choice(sublatt['codes'],
                                                  size=len(sublatt['sites']),
                                                  replace=True)
    return occu


ce_data = fixture_union(name='ce_data',
                        fixtures=['synthetic_data', 'real_data'],
                        scope='module')


@pytest.fixture(scope='module')
def ceprocessor(ce_data):
    subspace, dataset = ce_data
    scmatrix = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])
    return CEProcessor(subspace, scmatrix, coefficients=dataset['coefs'])


def test_encode_decode_property(ceprocessor):
    sublattices = sublattice_dicts(ceprocessor)
    encoccu = occupancy(sublattices)
    occu = ceprocessor.decode_occupancy(encoccu)
    for species, space in zip(occu, ceprocessor.allowed_species):
        assert species in space
    npt.assert_equal(encoccu, ceprocessor.encode_occupancy(occu))


def test_get_average_drift(ceprocessor):
    forward, reverse = ceprocessor.compute_average_drift()
    assert forward <= DRIFT_TOL and reverse <= DRIFT_TOL


def test_compute_property_change(ceprocessor):
    sublattices = sublattice_dicts(ceprocessor)
    occu = occupancy(sublattices)
    for _ in range(50):
        sublatt = np.random.choice(sublattices)
        site = np.random.choice(sublatt['sites'])
        new_sp = np.random.choice(sublatt['codes'])
        new_occu = occu.copy()
        new_occu[site] = new_sp
        prop_f = ceprocessor.compute_property(new_occu)
        prop_i = ceprocessor.compute_property(occu)
        dprop = ceprocessor.compute_property_change(occu, [(site, new_sp)])
        # Check with some tight tolerances.
        npt.assert_allclose(dprop, prop_f - prop_i, rtol=RTOL, atol=ATOL)
        # Test reverse matches forward
        old_sp = occu[site]
        rdprop = ceprocessor.compute_property_change(new_occu, [(site, old_sp)])
        assert dprop == -1 * rdprop


def test_structure_from_occupancy():
    pass


def test_occupancy_from_structure():
    pass


def test_compute_feature_change(ceprocessor):
    sublattices = sublattice_dicts(ceprocessor)
    occu = occupancy(sublattices)
    for _ in range(50):
        sublatt = np.random.choice(sublattices)
        site = np.random.choice(sublatt['sites'])
        new_sp = np.random.choice(sublatt['codes'])
        new_occu = occu.copy()
        new_occu[site] = new_sp
        # Test forward
        dcorr = ceprocessor.compute_feature_vector_change(occu, [(site, new_sp)])
        corr_f = ceprocessor.compute_feature_vector(new_occu)
        corr_i = ceprocessor.compute_feature_vector(occu)

        npt.assert_allclose(dcorr, corr_f - corr_i, rtol=RTOL, atol=ATOL)
        # Test reverse matches forward
        old_sp = occu[site]
        rdcorr = ceprocessor.compute_feature_vector_change(new_occu, [(site, old_sp)])
        npt.assert_array_equal(dcorr, -1 * rdcorr)


def test_msonable(ceprocessor):
    sublattices = sublattice_dicts(ceprocessor)
    occu = occupancy(sublattices)
    d = ceprocessor.as_dict()
    pr = Processor.from_dict(d)
    assert ceprocessor.compute_property(occu) == pr.compute_property(occu)
    # TODO error in constructor of basis functions
    #assert_msonable(ceprocessor)


def test_compute_property(ceprocessor):
    sublattices = sublattice_dicts(ceprocessor)
    occu = occupancy(sublattices)
    struct = ceprocessor.structure_from_occupancy(occu)
    pred = np.dot(ceprocessor.coefs,
                  ceprocessor.cluster_subspace.corr_from_structure(struct, False))
    assert ceprocessor.compute_property(occu) == pytest.approx(pred, abs=ATOL)


def test_compute_feature_vector(ceprocessor):
    sublattices = sublattice_dicts(ceprocessor)
    occu = occupancy(sublattices)
    struct = ceprocessor.structure_from_occupancy(occu)
    # same as normalize=False in corr_from_structure
    npt.assert_allclose(ceprocessor.compute_feature_vector(occu)/ceprocessor.size,
                        ceprocessor.cluster_subspace.corr_from_structure(struct))


def test_feature_change_indictator(ceprocessor):
    sublattices = sublattice_dicts(ceprocessor)
    occu = occupancy(sublattices)
    ceprocessor.cluster_subspace.change_site_bases('indicator')
    pr = CEProcessor(ceprocessor.cluster_subspace,
                     ceprocessor.supercell_matrix, ceprocessor.coefs,
                     optimize_indicator=True)
    for _ in range(50):
        sublatt = np.random.choice(sublattices)
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
