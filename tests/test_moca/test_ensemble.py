import numpy as np
import numpy.testing as npt
import pytest
from copy import deepcopy

from smol.cofe import ClusterExpansion, RegressionData
from smol.moca import (
    CanonicalEnsemble,
    ClusterExpansionProcessor,
    CompositeProcessor,
    EwaldProcessor,
    SemiGrandEnsemble,
)
from tests.utils import assert_msonable, gen_random_occupancy

ensembles = [CanonicalEnsemble, SemiGrandEnsemble]


@pytest.fixture
def canonical_ensemble(composite_processor):
    return CanonicalEnsemble(composite_processor)


@pytest.fixture
def mugrand_ensemble(composite_processor):
    species = {sp for space in
               composite_processor.active_site_spaces
               for sp in space.keys()}
    chemical_potentials = {sp: 0.3 for sp in species}
    return SemiGrandEnsemble(
        composite_processor, chemical_potentials=chemical_potentials
    )


# Test with composite processors to cover more ground
@pytest.mark.parametrize("ensemble_cls", ensembles)
def test_from_cluster_expansion(cluster_subspace_ewald, ensemble_cls):
    rng = np.random.default_rng()
    coefs = rng.random(cluster_subspace_ewald.num_corr_functions + 1)
    scmatrix = 3 * np.eye(3)
    proc = CompositeProcessor(cluster_subspace_ewald, scmatrix)
    proc.add_processor(
        ClusterExpansionProcessor(
            cluster_subspace_ewald, scmatrix, coefficients=coefs[:-1]
        )
    )
    proc.add_processor(
        EwaldProcessor(
            cluster_subspace_ewald,
            scmatrix,
            cluster_subspace_ewald.external_terms[0],
            coefficient=coefs[-1],
        )
    )
    reg_data = RegressionData(
        module="fake.module",
        estimator_name="Estimator",
        feature_matrix=rng.random((5, len(coefs))),
        property_vector=rng.random(len(coefs)),
        parameters={"foo": "bar"},
    )
    expansion = ClusterExpansion(cluster_subspace_ewald, coefs, reg_data)

    if ensemble_cls is SemiGrandEnsemble:
        species = {sp for space in
                   proc.active_site_spaces
                   for sp in space.keys()}
        chemical_potentials = {sp: 0.3 for sp in species}
        kwargs = {"chemical_potentials": chemical_potentials}
    else:
        kwargs = {}
    ensemble = ensemble_cls.from_cluster_expansion(
        expansion, supercell_matrix=scmatrix, **kwargs
    )
    npt.assert_array_equal(
        ensemble.natural_parameters[: ensemble.num_energy_coefs], coefs
    )
    occu = np.zeros(ensemble.num_sites, dtype=int)
    for _ in range(50):  # test a few flips
        sublatt = rng.choice(ensemble.active_sublattices)
        site = rng.choice(sublatt.sites)
        spec = rng.choice(sublatt.encoding)
        flip = [(site, spec)]
        assert proc.compute_property_change(
            occu, flip
        ) == ensemble.processor.compute_property_change(occu, flip)
        assert proc.compute_property(occu) == ensemble.processor.compute_property(occu)
        occu[site] = spec


def test_restrict_sites(ensemble):
    rng = np.random.default_rng()
    sites = rng.choice(range(ensemble.processor.num_sites), size=5)
    ensemble.restrict_sites(sites)
    for sublatt in ensemble.sublattices:
        assert not any(i in sublatt.active_sites for i in sites)
    ensemble.reset_restricted_sites()
    for sublatt in ensemble.sublattices:
        if len(sublatt.site_space) > 1:
            npt.assert_array_equal(sublatt.sites, sublatt.active_sites)
        else:
            assert len(sublatt.active_sites) == 0


def test_msonable(ensemble):
    assert_msonable(ensemble)


def test_split_ensemble(ensemble):
    occu = gen_random_occupancy(ensemble.sublattices)
    for sublattice in ensemble.sublattices:
        npt.assert_array_equal(np.arange(len(sublattice.species)),
                               sublattice.encoding)
        # ensemble must have been initialized from default.
    while len(ensemble.active_sublattices) > 0:
        is_active = [s.is_active for s in ensemble.sublattices]
        sl_id = np.random.choice(np.arange(len(is_active),
                                           dtype=int)[is_active])
        sublattice = ensemble.sublattices[sl_id]
        S = len(sublattice.species)
        old_sublattices = deepcopy(ensemble.sublattices)
        old_species = deepcopy(ensemble.species)
        if isinstance(ensemble, SemiGrandEnsemble):
            old_chemical_potentials = deepcopy(ensemble.chemical_potentials)
            old_mu_table = deepcopy(ensemble._mu_table)
        split_encodings = [sublattice.encoding[: S // 2],
                           sublattice.encoding[S // 2:]]
        ensemble.split_sublattice_by_species(sl_id, occu,
                                             split_encodings)
        assert len(ensemble.sublattices) == len(old_sublattices) + 1
        for i in range(len(old_sublattices)):
            if i != sl_id:
                if i < sl_id:
                    new_sublattice = ensemble.sublattices[i]
                    old_sublattice = old_sublattices[i]
                else:
                    new_sublattice = ensemble.sublattices[i + 1]
                    old_sublattice = old_sublattices[i]
                assert new_sublattice.site_space == old_sublattice.site_space
                npt.assert_array_equal(new_sublattice.sites,
                                       old_sublattice.sites)
                npt.assert_array_equal(new_sublattice.active_sites,
                                       old_sublattice.active_sites)
                npt.assert_array_equal(new_sublattice.encoding,
                                       old_sublattice.encoding)
            else:
                old_sublattice = old_sublattices[i]
                new1 = ensemble.sublattices[i]
                new2 = ensemble.sublattices[i + 1]
                npt.assert_array_equal(np.sort(old_sublattice.sites),
                                       np.sort(np.concatenate((new1.sites, new2.sites)))
                                       )
                if new1.is_active and new2.is_active:
                    npt.assert_array_equal(np.sort(old_sublattice.active_sites),
                                           np.sort(np.concatenate((new1.active_sites,
                                                                   new2.active_sites)))
                                           )
                npt.assert_array_equal(np.sort(old_sublattice.encoding),
                                       np.sort(np.concatenate((new1.encoding,
                                                               new2.encoding)))
                                       )
                assert len(new1.encoding) == len(new1.species)
                assert len(new2.encoding) == len(new2.species)
        if isinstance(ensemble, SemiGrandEnsemble):
            assert (set(ensemble.chemical_potentials.keys())
                    == set(ensemble.species))
            assert ensemble._mu_table.shape == old_mu_table.shape
            for sp in ensemble.species:
                assert sp in old_species
                assert ensemble.chemical_potentials[sp] == old_chemical_potentials[sp]
                for sublattice in ensemble.active_sublattices:
                    if sp in sublattice.species:
                        code = sublattice.encoding[sublattice.species.index(sp)]
                        npt.assert_array_equal(ensemble._mu_table[sublattice.sites, code],
                                               ensemble.chemical_potentials[sp])
            for sp in set(old_species) - set(ensemble.species):
                for sublattice in old_sublattices:
                    if sp in sublattice.species:
                        code = sublattice.encoding[sublattice.species.index(sp)]
                        npt.assert_array_equal(ensemble._mu_table[sublattice.sites, code],
                                               0)
            for sublattice in ensemble.sublattices:
                if not sublattice.is_active:
                    npt.assert_array_equal(ensemble._mu_table[sublattice.sites, :],
                                           0)


# Canonical Ensemble tests
def test_compute_feature_vector_canonical(canonical_ensemble):
    processor = canonical_ensemble.processor
    occu = gen_random_occupancy(canonical_ensemble.sublattices)
    assert np.dot(
        canonical_ensemble.natural_parameters,
        canonical_ensemble.compute_feature_vector(occu),
    ) == pytest.approx(processor.compute_property(occu))
    npt.assert_array_equal(
        canonical_ensemble.compute_feature_vector(occu),
        canonical_ensemble.processor.compute_feature_vector(occu),
    )
    rng = np.random.default_rng()
    for _ in range(50):  # test a few flips
        sublatt = np.random.choice(canonical_ensemble.active_sublattices)
        site = rng.choice(sublatt.sites)
        spec = rng.choice(range(len(sublatt.site_space)))
        flip = [(site, spec)]
        assert np.dot(
            canonical_ensemble.natural_parameters,
            canonical_ensemble.compute_feature_vector_change(occu, flip),
        ) == pytest.approx(processor.compute_property_change(occu, flip))
        npt.assert_array_equal(
            canonical_ensemble.compute_feature_vector_change(occu, flip),
            processor.compute_feature_vector_change(occu, flip),
        )

    # Can still work normally with processor after splitting.
    is_active = [s.is_active for s in canonical_ensemble.sublattices]
    sl_id = np.random.choice(np.arange(len(canonical_ensemble.sublattices),
                                       dtype=int)[is_active])
    encoding = canonical_ensemble.sublattices[sl_id].encoding
    S = len(encoding)
    split = [encoding[: S // 2], encoding[S // 2:]]
    canonical_ensemble.split_sublattice_by_species(sl_id, occu, split)
    if len(canonical_ensemble.active_sublattices) > 0:
        assert np.dot(
            canonical_ensemble.natural_parameters,
            canonical_ensemble.compute_feature_vector(occu),
        ) == pytest.approx(processor.compute_property(occu))
        npt.assert_array_equal(
            canonical_ensemble.compute_feature_vector(occu),
            canonical_ensemble.processor.compute_feature_vector(occu),
        )
        for _ in range(50):  # test a few flips
            sublatt = np.random.choice(canonical_ensemble.active_sublattices)
            site = np.random.choice(sublatt.sites)
            spec = np.random.choice(range(len(sublatt.site_space)))
            flip = [(site, spec)]
            assert np.dot(
                canonical_ensemble.natural_parameters,
                canonical_ensemble.compute_feature_vector_change(occu, flip),
            ) == pytest.approx(processor.compute_property_change(occu, flip))
            npt.assert_array_equal(
                canonical_ensemble.compute_feature_vector_change(occu, flip),
                processor.compute_feature_vector_change(occu, flip),
            )


# MuSemigrandEnsemble Tests
def test_compute_feature_vector_sgc(mugrand_ensemble):
    proc = mugrand_ensemble.processor
    occu = gen_random_occupancy(mugrand_ensemble.sublattices)
    assert np.dot(
        mugrand_ensemble.natural_parameters,
        mugrand_ensemble.compute_feature_vector(occu),
    ) == pytest.approx(
        proc.compute_property(occu) - mugrand_ensemble.compute_chemical_work(occu)
    )
    npt.assert_array_equal(
        mugrand_ensemble.compute_feature_vector(occu)[:-1],
        mugrand_ensemble.processor.compute_feature_vector(occu),
    )
    rng = np.random.default_rng()
    for _ in range(50):  # test a few flips
        sublatt = rng.choice(mugrand_ensemble.active_sublattices)
        site = rng.choice(sublatt.sites)
        spec = rng.choice(sublatt.encoding)
        flip = [(site, spec)]
        dmu = (mugrand_ensemble._mu_table[site][spec]
               - mugrand_ensemble._mu_table[site][occu[site]]
               )
        assert np.dot(
            mugrand_ensemble.natural_parameters,
            mugrand_ensemble.compute_feature_vector_change(occu, flip),
        ) == pytest.approx(proc.compute_property_change(occu, flip) - dmu)
        npt.assert_array_equal(
            mugrand_ensemble.compute_feature_vector_change(occu, flip),
            np.append(proc.compute_feature_vector_change(occu, flip), dmu),
        )
    # Can still work normally with processor after splitting.
    is_active = [s.is_active for s in mugrand_ensemble.sublattices]
    sl_id = np.random.choice(np.arange(len(mugrand_ensemble.sublattices),
                                       dtype=int)[is_active])
    encoding = mugrand_ensemble.sublattices[sl_id].encoding
    S = len(encoding)
    split = [encoding[: S // 2], encoding[S // 2:]]
    mugrand_ensemble.split_sublattice_by_species(sl_id, occu, split)
    if len(mugrand_ensemble.active_sublattices) > 0:
        assert np.dot(
            mugrand_ensemble.natural_parameters,
            mugrand_ensemble.compute_feature_vector(occu),
        ) == pytest.approx(
            proc.compute_property(occu) - mugrand_ensemble.compute_chemical_work(occu)
        )
        npt.assert_array_equal(
            mugrand_ensemble.compute_feature_vector(occu)[:-1],
            mugrand_ensemble.processor.compute_feature_vector(occu),
        )
        for _ in range(50):  # test a few flips
            sublatt = np.random.choice(mugrand_ensemble.active_sublattices)
            site = np.random.choice(sublatt.sites)
            spec = np.random.choice(sublatt.encoding)
            flip = [(site, spec)]
            dmu = (mugrand_ensemble._mu_table[site][spec]
                   - mugrand_ensemble._mu_table[site][occu[site]]
                   )
            assert np.dot(
                mugrand_ensemble.natural_parameters,
                mugrand_ensemble.compute_feature_vector_change(occu, flip),
            ) == pytest.approx(proc.compute_property_change(occu, flip) - dmu)
            npt.assert_array_equal(
                mugrand_ensemble.compute_feature_vector_change(occu, flip),
                np.append(proc.compute_feature_vector_change(occu, flip), dmu),
            )


def test_bad_chemical_potentials(mugrand_ensemble):
    proc = mugrand_ensemble.processor
    chem_pots = mugrand_ensemble.chemical_potentials
    print(chem_pots)
    with pytest.raises(ValueError):
        items = list(chem_pots.items())
        mugrand_ensemble.chemical_potentials = {items[0][0]: items[0][1]}
    with pytest.raises(ValueError):
        mugrand_ensemble.chemical_potentials = {"A": 0.5, "D": 0.6}
    with pytest.raises(ValueError):
        chem_pots["foo"] = 0.4
        SemiGrandEnsemble(proc, chemical_potentials=chem_pots)
    with pytest.raises(ValueError):
        del chem_pots["foo"]
        chem_pots.pop(items[0][0])
        SemiGrandEnsemble(proc, chemical_potentials=chem_pots)
    with pytest.raises(ValueError):
        chem_pots[str(list(chem_pots.keys()))[0]] = 0.0
        mugrand_ensemble.chemical_potentials = chem_pots


def test_build_mu_table(mugrand_ensemble):
    # Do not use processor sub-lattices in these tests.
    table = mugrand_ensemble._build_mu_table(mugrand_ensemble.chemical_potentials)
    for sublatt, row in zip(mugrand_ensemble.active_sublattices, table):
        if len(sublatt.species) == 1:  # skip inactive sites
            continue
        for i, species in zip(sublatt.encoding, sublatt.species):
            assert mugrand_ensemble.chemical_potentials[species] == row[i]
