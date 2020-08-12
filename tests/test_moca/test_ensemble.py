import pytest
from copy import deepcopy
import numpy.testing as npt
import numpy as np

from smol.constants import kB
from smol.cofe import ClusterExpansion
from smol.cofe.extern import EwaldTerm
from smol.moca import (CanonicalEnsemble, MuSemiGrandEnsemble,
                       FuSemiGrandEnsemble, CompositeProcessor,
                       CEProcessor, EwaldProcessor)
from tests.utils import assert_msonable, gen_random_occupancy

ensembles = [CanonicalEnsemble, MuSemiGrandEnsemble, FuSemiGrandEnsemble]
TEMPERATURE = 1000


@pytest.fixture
def composite_processor(cluster_subspace):
    coefs = 2 * np.random.random(cluster_subspace.n_bit_orderings)
    scmatrix = 4 * np.eye(3)
    return CEProcessor(cluster_subspace, scmatrix, coefs)


@pytest.fixture(params=ensembles)
def ensemble(composite_processor, request):
    if request.param is MuSemiGrandEnsemble:
        kwargs = {'chemical_potentials':
                  {sp: 0.3 for space in composite_processor.unique_site_spaces
                   for sp in space.keys()}}
    else:
        kwargs = {}
    return request.param(composite_processor, temperature=TEMPERATURE, **kwargs)


@pytest.fixture
def canonical_ensemble(composite_processor):
    return CanonicalEnsemble(composite_processor, temperature=TEMPERATURE)


@pytest.fixture
def mugrand_ensemble(composite_processor):
    chemical_potentials= {sp: 0.3 for space in composite_processor.unique_site_spaces
                          for sp in space.keys()}
    return MuSemiGrandEnsemble(composite_processor, temperature=TEMPERATURE,
                               chemical_potentials=chemical_potentials)


@pytest.fixture
def fugrand_ensemble(composite_processor):
    return FuSemiGrandEnsemble(composite_processor, temperature=TEMPERATURE)


# Test with composite processors to cover more ground
@pytest.mark.parametrize('ensemble_cls', ensembles)
def test_from_cluster_expansion(cluster_subspace, ensemble_cls):
    cluster_subspace.add_external_term(EwaldTerm())
    coefs = np.random.random(cluster_subspace.n_bit_orderings + 1)
    scmatrix = 3 * np.eye(3)
    proc = CompositeProcessor(cluster_subspace, scmatrix)
    proc.add_processor(CEProcessor, coefficients=coefs[:-1])
    proc.add_processor(EwaldProcessor, coefficient=coefs[-1],
                       ewald_term=cluster_subspace.external_terms[0])
    fake_feature_matrix = np.random.random((5, len(coefs)))
    expansion = ClusterExpansion(cluster_subspace, coefs, fake_feature_matrix)
    if ensemble_cls is MuSemiGrandEnsemble:
        kwargs = {'chemical_potentials':
                  {sp: 0.3 for space in proc.unique_site_spaces
                   for sp in space.keys()}}
    else:
        kwargs = {}
    ensemble = ensemble_cls.from_cluster_expansion(expansion,
                                                   supercell_matrix=scmatrix,
                                                   temperature=500,
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


def test_temperature_setter(ensemble):
    assert ensemble.beta == 1/(kB*ensemble.temperature)
    ensemble.temperature = 500
    assert ensemble.beta == 1 / (kB * 500)


def test_restrict_sites(ensemble):
    sites = np.random.choice(range(ensemble.processor.num_sites), size=5)
    ensemble.restrict_sites(sites)
    for sublatt in ensemble.sublattices:
        assert not any(i in sublatt.active_sites for i in sites)
    ensemble.reset_restricted_sites()
    for sublatt in ensemble.sublattices:
        npt.assert_array_equal(sublatt.sites, sublatt.active_sites)


def test_msonable(ensemble):
    # assert_msonable(ensemble)
    pass


# Canonical Ensemble tests
def test_compute_feature_vector(canonical_ensemble):
    processor = canonical_ensemble.processor
    occu = gen_random_occupancy(canonical_ensemble.sublattices,
                                canonical_ensemble.num_sites)
    assert (np.dot(canonical_ensemble.natural_parameters,
                   canonical_ensemble.compute_feature_vector(occu))
            == processor.compute_property(occu))
    npt.assert_array_equal(canonical_ensemble.compute_feature_vector(occu),
                           canonical_ensemble.processor.compute_feature_vector(occu))
    for _ in range(50):  # test a few flips
        sublatt = np.random.choice(canonical_ensemble.sublattices)
        site = np.random.choice(sublatt.sites)
        spec = np.random.choice(range(len(sublatt.site_space)))
        flip = [(site, spec)]
        assert (np.dot(canonical_ensemble.natural_parameters,
                       canonical_ensemble.compute_feature_vector_change(occu, flip))
                == processor.compute_property_change(occu, flip))
        npt.assert_array_equal(canonical_ensemble.compute_feature_vector_change(occu, flip),
                               processor.compute_feature_vector_change(occu, flip))


# MuSemigrandEnsemble Tests
def test_compute_feature_vector(mugrand_ensemble):
    proc = mugrand_ensemble.processor
    occu = gen_random_occupancy(mugrand_ensemble.sublattices,
                                mugrand_ensemble.num_sites)
    assert (np.dot(mugrand_ensemble.natural_parameters,
                   mugrand_ensemble.compute_feature_vector(occu))
            == proc.compute_property(occu) - mugrand_ensemble.compute_chemical_work(occu))
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
                == proc.compute_property_change(occu, flip) - dmu)
        npt.assert_array_equal(mugrand_ensemble.compute_feature_vector_change(occu, flip),
                               np.append(proc.compute_feature_vector_change(occu, flip), dmu))


def test_bad_chemical_potentials(mugrand_ensemble):
    proc = mugrand_ensemble.processor
    chem_pots = mugrand_ensemble.chemical_potentials
    with pytest.raises(ValueError):
        items = list(chem_pots.items())
        mugrand_ensemble.chemical_potentials = {items[0][0]: items[0][1]}
    with pytest.raises(ValueError):
        mugrand_ensemble.chemical_potentials = {'A': 0.5, 'D': 0.6}
    with pytest.raises(ValueError):
        chem_pots['foo'] = 0.4
        MuSemiGrandEnsemble(proc, 500, chemical_potentials=chem_pots)
    with pytest.raises(ValueError):
        del chem_pots['foo']
        chem_pots.pop(items[0][0])
        MuSemiGrandEnsemble(proc, 500, chemical_potentials=chem_pots)
    

def test_build_mu_table(mugrand_ensemble):
    proc = mugrand_ensemble.processor
    table = mugrand_ensemble._build_mu_table(mugrand_ensemble.chemical_potentials)
    for space, row in zip(proc.allowed_species, table):
        if len(space) == 1:  # skip inactive sites
            continue
        for i, species in enumerate(space):
            assert mugrand_ensemble.chemical_potentials[species] == row[i]


# Tests for FuSemiGrandEnsemble
def test_compute_feature_vector(fugrand_ensemble):
    proc = fugrand_ensemble.processor
    occu = gen_random_occupancy(fugrand_ensemble.sublattices,
                                fugrand_ensemble.num_sites)
    assert (np.dot(fugrand_ensemble.natural_parameters,
                   fugrand_ensemble.compute_feature_vector(occu))
            == proc.compute_property(occu) - fugrand_ensemble.compute_chemical_work(occu))
    npt.assert_array_equal(fugrand_ensemble.compute_feature_vector(occu)[:-1],
                           fugrand_ensemble.processor.compute_feature_vector(occu))
    for _ in range(50):  # test a few flips
        sublatt = np.random.choice(fugrand_ensemble.sublattices)
        site = np.random.choice(sublatt.sites)
        spec = np.random.choice(range(len(sublatt.site_space)))
        flip = [(site, spec)]
        dfu = np.log(fugrand_ensemble._fu_table[site][spec]/fugrand_ensemble._fu_table[site][occu[site]])
        assert (np.dot(fugrand_ensemble.natural_parameters,
                       fugrand_ensemble.compute_feature_vector_change(occu, flip))
                == proc.compute_property_change(occu, flip) - dfu)
        npt.assert_array_equal(fugrand_ensemble.compute_feature_vector_change(occu, flip),
                               np.append(proc.compute_feature_vector_change(occu, flip), dfu))


def test_bad_fugacity_fractions(fugrand_ensemble):
    proc = fugrand_ensemble.processor
    fug_fracs = deepcopy(fugrand_ensemble.fugacity_fractions)
    with pytest.raises(ValueError):
        fug_fracs[0] = {s: v for s, v in list(fug_fracs[0].items())[:-1]}
        fugrand_ensemble.fugacity_fractions = fug_fracs
    with pytest.raises(ValueError):
        fug_fracs[0] = {sp: 1.1 for sp in fug_fracs[0].keys()}
        fugrand_ensemble.fugacity_fractions = fug_fracs
    with pytest.raises(ValueError):
        fug_fracs[0] = {'A': 0.5, 'D': 0.6}
        fugrand_ensemble.fugacity_fractions = fug_fracs
    with pytest.raises(ValueError):
        fug_fracs[0]['foo'] = 0.4
        FuSemiGrandEnsemble(proc, 500, fugacity_fractions=fug_fracs)
    with pytest.raises(ValueError):
        del fug_fracs[0]['foo']
        FuSemiGrandEnsemble(proc, 500, fugacity_fractions=fug_fracs)


def test_build_fu_table(fugrand_ensemble):
    table = fugrand_ensemble._build_fu_table(fugrand_ensemble.fugacity_fractions)
    proc = fugrand_ensemble.processor
    for space, row in zip(proc.allowed_species, table):
        fugacity_fractions = None
        if len(space) == 1: # skip inactive sites
            continue
        for fus in fugrand_ensemble.fugacity_fractions:
            if space == list(fus.keys()):
                fugacity_fractions = fus
        for i, species in enumerate(space):
            assert fugacity_fractions[species] == row[i]
