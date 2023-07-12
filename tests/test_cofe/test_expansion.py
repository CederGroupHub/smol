import numpy as np
import numpy.testing as npt
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from smol.cofe import ClusterExpansion, RegressionData
from tests.utils import assert_msonable, assert_pickles, gen_random_ordered_structure


@pytest.fixture(scope="module")
def cluster_expansion(cluster_subspace, rng):
    reg = Ridge(alpha=1e-8, fit_intercept=False)
    n = rng.integers(50, 100)
    feat_matrix = np.empty((n, len(cluster_subspace)))
    structures, scmatrices = [], []
    for i in range(n):
        scmatrix = np.eye(3, dtype=int) * rng.integers(2, 4)
        scmatrix[0, 1] = 2  # Intentionally made less symmetric
        scmatrix[1, 2] = 1
        structure = gen_random_ordered_structure(
            cluster_subspace.structure, size=scmatrix, rng=rng
        )
        structures.append(structure)
        scmatrices.append(scmatrix)
        feat_matrix[i] = cluster_subspace.corr_from_structure(
            structure, scmatrix=scmatrix
        )

    prop_vec = -5 * rng.random(n)
    reg.fit(feat_matrix, prop_vec)
    reg_data = RegressionData.from_sklearn(
        reg, feature_matrix=feat_matrix, property_vector=prop_vec
    )
    expansion = ClusterExpansion(cluster_subspace, reg.coef_, reg_data)
    # bind the structures and matrices to the expansion willy nilly
    expansion.structures = structures
    expansion.scmatrices = scmatrices

    return expansion


def test_regression_data(cluster_subspace, rng):
    reg = LinearRegression(fit_intercept=False)
    n = rng.integers(10, 100)
    feat_matrix = rng.random((n, len(cluster_subspace)))
    prop_vec = rng.random(n)
    reg_data = RegressionData.from_sklearn(
        reg, feature_matrix=feat_matrix, property_vector=prop_vec
    )
    coeffs = rng.random(len(cluster_subspace))
    expansion = ClusterExpansion(cluster_subspace, coeffs, reg_data)
    assert reg_data.estimator_name == reg.__class__.__name__
    assert reg_data.module == reg.__module__
    assert reg_data.parameters == reg.get_params()
    assert_msonable(expansion)

    # test bad feature matrix shape
    reg_data = RegressionData.from_sklearn(
        reg, feature_matrix=feat_matrix[:, :-1], property_vector=prop_vec
    )
    with pytest.raises(AttributeError):
        expansion = ClusterExpansion(cluster_subspace, coeffs, reg_data)
    # test bad coeff length
    with pytest.raises(AttributeError):
        expansion = ClusterExpansion(cluster_subspace, coeffs[:-1], reg_data)


def test_predict(cluster_expansion, rng):
    subspace = cluster_expansion.cluster_subspace

    prim = cluster_expansion.structure
    scmatrix = np.eye(3, dtype=int) * 3
    scmatrix[0, 1] = 2  # Intentionally made less symmetric
    scmatrix[1, 2] = 1
    N = np.abs(np.linalg.det(scmatrix))
    pool = [gen_random_ordered_structure(prim, scmatrix, rng=rng) for _ in range(100)]
    feature_matrix = np.array(
        [
            subspace.corr_from_structure(s, scmatrix=scmatrix, normalized=True)
            for s in pool
        ]
    )

    comps = [s.composition for s in pool]
    all_species = list({b for c in comps for b in c.keys()})
    mus = rng.random(len(all_species))

    def get_energy(structure, species, chempots):
        return np.dot(chempots, [structure.composition[sp] for sp in species])

    energies = np.array([get_energy(s, all_species, mus) for s in pool]) / N
    reg = LinearRegression(fit_intercept=False)
    reg.fit(feature_matrix, energies)
    coefs = reg.coef_
    expansion_new = ClusterExpansion(subspace, coefs)
    # Why don't we add a "scmatrix" option into ClusterExpansion.predict?
    # This will make it safer to structure skew, because pymatgen can't seem
    # to figure out highly skewed supercell matrix correctly.
    energies_pred = np.array(
        [expansion_new.predict(s, scmatrix=scmatrix, normalized=True) for s in pool]
    )
    np.testing.assert_almost_equal(energies, energies_pred, decimal=6)


def test_prune(cluster_expansion):
    expansion = cluster_expansion.copy()

    thresh = 1e-2
    expansion.prune(threshold=thresh)
    ids = [i for i, coef in enumerate(cluster_expansion.coefs) if abs(coef) >= thresh]
    new_coefs = cluster_expansion.coefs[ids]
    new_eci = cluster_expansion.eci[ids]

    assert len(expansion.coefs) == len(new_coefs)
    npt.assert_array_equal(new_eci, expansion.eci)
    npt.assert_array_equal(new_coefs, expansion.coefs)
    assert len(expansion.cluster_subspace) == len(new_coefs)
    assert len(expansion.eci_orbit_ids) == len(new_coefs)
    assert (
        len(expansion.cluster_interaction_tensors)
        == expansion.cluster_subspace.num_orbits
    )
    assert (
        len(expansion.cluster_interaction_tensors)
        == len(expansion.cluster_subspace.orbits) + 1
    )

    pruned_feat_matrix = cluster_expansion._feat_matrix[:, ids]
    npt.assert_array_equal(expansion._feat_matrix, pruned_feat_matrix)
    # check that recomputing features produces what's expected
    new_feature_matrix = np.array(
        [
            expansion.cluster_subspace.corr_from_structure(s, scmatrix=m)
            for s, m in zip(cluster_expansion.structures, cluster_expansion.scmatrices)
        ]
    )
    npt.assert_array_equal(new_feature_matrix, pruned_feat_matrix)
    # check new predictions
    preds = [
        expansion.predict(s, normalized=True, scmatrix=m)
        for s, m in zip(cluster_expansion.structures, cluster_expansion.scmatrices)
    ]
    npt.assert_allclose(preds, np.dot(pruned_feat_matrix, new_coefs))

    # check energy computed with interactions also matches
    ints = np.array(
        [
            expansion.cluster_interactions_from_structure(s, scmatrix=m)
            for s, m in zip(cluster_expansion.structures, cluster_expansion.scmatrices)
        ]
    )
    preds = np.sum(
        cluster_expansion.cluster_subspace.orbit_multiplicities * ints, axis=1
    )
    npt.assert_allclose(preds, np.dot(pruned_feat_matrix, new_coefs), atol=1e-5)


def test_msonable(cluster_expansion):
    _ = repr(cluster_expansion)
    _ = str(cluster_expansion)
    _ = str(cluster_expansion)
    d = cluster_expansion.as_dict()
    ce1 = ClusterExpansion.from_dict(d)
    npt.assert_array_equal(cluster_expansion.coefs, ce1.coefs)
    # change this to just use assert_msonable
    assert_msonable(cluster_expansion)


def test_pickles(cluster_expansion):
    assert_pickles(cluster_expansion)
