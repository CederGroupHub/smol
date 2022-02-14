import pytest
import numpy.testing as npt
import numpy as np

from smol.cofe import ClusterExpansion, RegressionData
from smol.moca import CanonicalEnsemble, MuSemiGrandEnsemble, \
    CompositeProcessor, CEProcessor, EwaldProcessor
from tests.utils import assert_msonable, gen_random_occupancy

ensembles = [CanonicalEnsemble, MuSemiGrandEnsemble]


@pytest.fixture
def canonical_ensemble(composite_processor):
    return CanonicalEnsemble(composite_processor)


@pytest.fixture
def mugrand_ensemble(composite_processor):
    chemical_potentials = {sp: 0.3 for space in composite_processor.unique_site_spaces
                           for sp in space.keys()}
    return MuSemiGrandEnsemble(composite_processor,
                               chemical_potentials=chemical_potentials)


# Test with composite processors to cover more ground
@pytest.mark.parametrize('ensemble_cls', ensembles)
def test_from_cluster_expansion(cluster_subspace_ewald, ensemble_cls):
    coefs = np.random.random(cluster_subspace_ewald.num_corr_functions + 1)
    scmatrix = 3 * np.eye(3)
    proc = CompositeProcessor(cluster_subspace_ewald, scmatrix)
    proc.add_processor(CEProcessor(cluster_subspace_ewald, scmatrix,
                                   coefficients=coefs[:-1]))
    proc.add_processor(EwaldProcessor(cluster_subspace_ewald, scmatrix,
                       cluster_subspace_ewald.external_terms[0],
                                      coefficient=coefs[-1]))
    reg_data = RegressionData(
        module='fake.module', estimator_name='Estimator',
        feature_matrix=np.random.random((5, len(coefs))),
        property_vector=np.random.random(len(coefs)),
        parameters={'foo': 'bar'})
    expansion = ClusterExpansion(cluster_subspace_ewald, coefs, reg_data)

    if ensemble_cls is MuSemiGrandEnsemble:
        kwargs = {'chemical_potentials':
                  {sp: 0.3 for space in proc.unique_site_spaces
                   for sp in space.keys()}}
    else:
        kwargs = {}
    ensemble = ensemble_cls.from_cluster_expansion(expansion,
                                                   supercell_matrix=scmatrix,
                                                   **kwargs)
    npt.assert_array_equal(ensemble.natural_parameters[:ensemble.num_energy_coefs],
                           coefs)
    occu = np.zeros(ensemble.num_sites, dtype=int)
    for _ in range(50):  # test a few flips
        sublatt = np.random.choice(ensemble.sublattices)
        site = np.random.choice(sublatt.sites)
        spec = np.random.choice(range(len(sublatt.site_space)))
        flip = [(site, spec)]
        assert proc.compute_property_change(occu, flip) == ensemble.processor.compute_property_change(occu, flip)
        assert proc.compute_property(occu) == ensemble.processor.compute_property(occu)
        occu[site] = spec


def test_restrict_sites(ensemble):
    sites = np.random.choice(range(ensemble.processor.num_sites), size=5)
    ensemble.restrict_sites(sites)
    for sublatt in ensemble.sublattices:
        assert not any(i in sublatt.active_sites for i in sites)
    ensemble.reset_restricted_sites()
    for sublatt in ensemble.sublattices:
        npt.assert_array_equal(sublatt.sites, sublatt.active_sites)


def test_msonable(ensemble):
    assert_msonable(ensemble)


# Canonical Ensemble tests
def test_compute_feature_vector_canonical(canonical_ensemble):
    processor = canonical_ensemble.processor
    occu = gen_random_occupancy(canonical_ensemble.sublattices,
                                canonical_ensemble.inactive_sublattices)
    assert (np.dot(canonical_ensemble.natural_parameters,
                   canonical_ensemble.compute_feature_vector(occu))
            == pytest.approx(processor.compute_property(occu)))
    npt.assert_array_equal(canonical_ensemble.compute_feature_vector(occu),
                           canonical_ensemble.processor.compute_feature_vector(occu))
    for _ in range(50):  # test a few flips
        sublatt = np.random.choice(canonical_ensemble.sublattices)
        site = np.random.choice(sublatt.sites)
        spec = np.random.choice(range(len(sublatt.site_space)))
        flip = [(site, spec)]
        assert (np.dot(canonical_ensemble.natural_parameters,
                       canonical_ensemble.compute_feature_vector_change(occu, flip))
                == pytest.approx(processor.compute_property_change(occu, flip)))
        npt.assert_array_equal(canonical_ensemble.compute_feature_vector_change(occu, flip),
                               processor.compute_feature_vector_change(occu, flip))


# MuSemigrandEnsemble Tests
def test_compute_feature_vector_sgc(mugrand_ensemble):
    proc = mugrand_ensemble.processor
    occu = gen_random_occupancy(mugrand_ensemble.sublattices,
                                mugrand_ensemble.inactive_sublattices)
    assert (np.dot(mugrand_ensemble.natural_parameters,
                   mugrand_ensemble.compute_feature_vector(occu))
            == pytest.approx(proc.compute_property(occu) - mugrand_ensemble.compute_chemical_work(occu)))
    npt.assert_array_equal(mugrand_ensemble.compute_feature_vector(occu)[:-1],
                           mugrand_ensemble.processor.compute_feature_vector(occu))
    for _ in range(50):  # test a few flips
        sublatt = np.random.choice(mugrand_ensemble.sublattices)
        site = np.random.choice(sublatt.sites)
        spec = np.random.choice(range(len(sublatt.site_space)))
        flip = [(site, spec)]
        dmu = mugrand_ensemble._mu_table[site][spec] - mugrand_ensemble._mu_table[site][occu[site]]
        assert (np.dot(mugrand_ensemble.natural_parameters,
                       mugrand_ensemble.compute_feature_vector_change(occu, flip))
                == pytest.approx(proc.compute_property_change(occu, flip) - dmu))
        npt.assert_array_equal(mugrand_ensemble.compute_feature_vector_change(occu, flip),
                               np.append(proc.compute_feature_vector_change(occu, flip), dmu))


def test_bad_chemical_potentials(mugrand_ensemble):
    proc = mugrand_ensemble.processor
    chem_pots = mugrand_ensemble.chemical_potentials
    print(chem_pots)
    with pytest.raises(ValueError):
        items = list(chem_pots.items())
        mugrand_ensemble.chemical_potentials = {items[0][0]: items[0][1]}
    with pytest.raises(ValueError):
        mugrand_ensemble.chemical_potentials = {'A': 0.5, 'D': 0.6}
    with pytest.raises(ValueError):
        chem_pots['foo'] = 0.4
        MuSemiGrandEnsemble(proc, chemical_potentials=chem_pots)
    with pytest.raises(ValueError):
        del chem_pots['foo']
        chem_pots.pop(items[0][0])
        MuSemiGrandEnsemble(proc, chemical_potentials=chem_pots)
    with pytest.raises(ValueError):
        chem_pots[str(list(chem_pots.keys()))[0]] = 0.0
        mugrand_ensemble.chemical_potentials = chem_pots


def test_build_mu_table(mugrand_ensemble):
    proc = mugrand_ensemble.processor
    table = mugrand_ensemble._build_mu_table(mugrand_ensemble.chemical_potentials)
    for space, row in zip(proc.allowed_species, table):
        if len(space) == 1:  # skip inactive sites
            continue
        for i, species in enumerate(space):
            assert mugrand_ensemble.chemical_potentials[species] == row[i]
