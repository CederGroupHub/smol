import pytest
from copy import deepcopy
import numpy.testing as npt
import numpy as np

from smol.cofe import ClusterExpansion
from smol.cofe.extern import EwaldTerm
from smol.moca import (CanonicalEnsemble, MuSemiGrandEnsemble,
                       FuSemiGrandEnsemble, CompositeProcessor,
                       DiscChargeNeutralSemiGrandEnsemble,
                       CEProcessor, EwaldProcessor)

from smol.moca.ensemble.sublattice import get_all_sublattices

from smol.moca.comp_space import CompSpace
from smol.moca.sampler.mcusher import Tableflipper
from smol.moca.utils.math_utils import GCD_list

from tests.utils import (assert_msonable, gen_random_occupancy,
                         gen_random_neutral_occupancy)

ensembles = [CanonicalEnsemble, MuSemiGrandEnsemble, FuSemiGrandEnsemble,
             DiscChargeNeutralSemiGrandEnsemble]


@pytest.fixture
def composite_processor(cluster_subspace):
    coefs = 2 * np.random.random(cluster_subspace.num_corr_functions)
    scmatrix = 4 * np.eye(3)
    return CEProcessor(cluster_subspace, scmatrix, coefs)


@pytest.fixture(params=ensembles)
def ensemble(composite_processor, request):
    if request.param is MuSemiGrandEnsemble:
        kwargs = {'chemical_potentials':
                  {sp: 0.3 for space in composite_processor.unique_site_spaces
                   for sp in space.keys()}}

    elif request.param is DiscChargeNeutralSemiGrandEnsemble:

        sublattices = get_all_sublattices(composite_processor)
    
        bits = [sl.species for sl in sublattices]
        sl_sizes = [len(sl.sites) for sl in sublattices]
        sc_size = GCD_list(sl_sizes)
        sl_sizes = [sz//sc_size for sz in sl_sizes]
    
        comp_space = CompSpace(bits,sl_sizes)
        mu = [0.3+i*0.01 for i in range(comp_space.dim)]
        kwargs = {'mu':mu}

    else:
        kwargs = {}
    return request.param(composite_processor, **kwargs)


@pytest.fixture
def canonical_ensemble(composite_processor):
    return CanonicalEnsemble(composite_processor)


@pytest.fixture
def mugrand_ensemble(composite_processor):
    chemical_potentials = {sp: 0.3 for space in composite_processor.unique_site_spaces
                           for sp in space.keys()}
    return MuSemiGrandEnsemble(composite_processor,
                               chemical_potentials=chemical_potentials)


@pytest.fixture
def fugrand_ensemble(composite_processor):
    return FuSemiGrandEnsemble(composite_processor)

@pytest.fixture
def disc_ensemble(composite_processor):

    sublattices = get_all_sublattices(composite_processor)

    bits = [sl.species for sl in sublattices]
    sl_sizes = [len(sl.sites) for sl in sublattices]
    sc_size = GCD_list(sl_sizes)     
    sl_sizes = [sz//sc_size for sz in sl_sizes]

    comp_space = CompSpace(bits,sl_sizes)
    mu = [0.3+0.01*i for i in range(comp_space.dim)]
    return DiscChargeNeutralSemiGrandEnsemble(composite_processor,mu)


# Test with composite processors to cover more ground
@pytest.mark.parametrize('ensemble_cls', ensembles)
def test_from_cluster_expansion(cluster_subspace, ensemble_cls):
    cluster_subspace.add_external_term(EwaldTerm())
    coefs = np.random.random(cluster_subspace.num_corr_functions + 1)
    scmatrix = 3 * np.eye(3)
    proc = CompositeProcessor(cluster_subspace, scmatrix)
    proc.add_processor(CEProcessor(cluster_subspace, scmatrix,
                                   coefficients=coefs[:-1]))
    proc.add_processor(EwaldProcessor(cluster_subspace, scmatrix,
                       cluster_subspace.external_terms[0],
                                      coefficient=coefs[-1]))
    fake_feature_matrix = np.random.random((5, len(coefs)))
    expansion = ClusterExpansion(cluster_subspace, coefs, fake_feature_matrix)
    if ensemble_cls is MuSemiGrandEnsemble:
        kwargs = {'chemical_potentials':
                  {sp: 0.3 for space in proc.unique_site_spaces
                   for sp in space.keys()}}

    elif ensemble_cls is DiscChargeNeutralSemiGrandEnsemble:
        
        sublattices = get_all_sublattices(proc)
    
        bits = [sl.species for sl in sublattices]
        sl_sizes = [len(sl.sites) for sl in sublattices]
        sc_size = GCD_list(sl_sizes)
        sl_sizes = [sz//sc_size for sz in sl_sizes]
    
        comp_space = CompSpace(bits,sl_sizes)
        mu = [0.3+i*0.01 for i in range(comp_space.dim)]
        kwargs = {'mu':mu}

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


# Tests for FuSemiGrandEnsemble
def test_compute_feature_vector(fugrand_ensemble):
    proc = fugrand_ensemble.processor
    occu = gen_random_occupancy(fugrand_ensemble.sublattices,
                                fugrand_ensemble.num_sites)
    assert (np.dot(fugrand_ensemble.natural_parameters,
                   fugrand_ensemble.compute_feature_vector(occu))
            == pytest.approx(proc.compute_property(occu) - fugrand_ensemble.compute_chemical_work(occu),
                             abs=1E-12))
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
                == pytest.approx(proc.compute_property_change(occu, flip) - dfu, abs=1E-13))
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
        FuSemiGrandEnsemble(proc, fugacity_fractions=fug_fracs)
    with pytest.raises(ValueError):
        del fug_fracs[0]['foo']
        FuSemiGrandEnsemble(proc, fugacity_fractions=fug_fracs)
    with pytest.raises(ValueError):
        fug_fracs[0][str(list(fug_fracs[0].keys())[0])] = 0.0
        fugrand_ensemble.fugacity_fractions = fug_fracs


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

#Tests for Discriminative charge neutral semigrand ensemble.
def test_compute_feature_vector(disc_ensemble):
    proc = disc_ensemble.processor
    dmus = []

    usher = Tableflipper(disc_ensemble.all_sublattices)
    for i in range(10):
        occu = gen_random_neutral_occupancy(disc_ensemble.all_sublattices,
                                            disc_ensemble.num_sites)
        assert np.dot(disc_ensemble.natural_parameters,
                       disc_ensemble.compute_feature_vector(occu))\
                == pytest.approx(proc.compute_property(occu) - disc_ensemble.compute_chemical_work(occu),
                                 abs=1E-7)
        npt.assert_array_equal(disc_ensemble.compute_feature_vector(occu)[:-1],
                               disc_ensemble.processor.compute_feature_vector(occu))
    
        # Using minimum flip table.
        for _ in range(500):  # test a few flips
            flip = usher.propose_step(occu)
            dmu = disc_ensemble.compute_feature_vector_change(occu,flip)[-1]
            if dmu != 0:
                dmus.append(dmu)
            assert (np.any(np.isclose(np.append(np.array(disc_ensemble.mu),0),dmu)) or
                    np.any(np.isclose(np.append(np.array(disc_ensemble.mu),0),-dmu)))

    assert not(np.all(np.array(dmus)==dmus[0]))
