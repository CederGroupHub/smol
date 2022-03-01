import pytest
import numpy as np
import numpy.testing as npt
from sklearn.linear_model import LinearRegression, Ridge
from smol.cofe import ClusterExpansion, RegressionData
from tests.utils import assert_msonable, gen_random_structure


@pytest.fixture(scope='module')
def cluster_expansion(cluster_subspace):
    reg = Ridge(alpha=1E-8, fit_intercept=False)
    n = np.random.randint(50, 100)
    feat_matrix = np.empty((n, len(cluster_subspace)))
    structures = []
    for i in range(n):
        structure = gen_random_structure(
            cluster_subspace.structure, size=np.random.randint(2, 5))
        structures.append(structure)
        feat_matrix[i] = cluster_subspace.corr_from_structure(structure)

    prop_vec = -5 * np.random.random(n)
    reg.fit(feat_matrix, prop_vec)
    reg_data = RegressionData.from_sklearn(
        reg, feature_matrix=feat_matrix, property_vector=prop_vec)
    expansion = ClusterExpansion(cluster_subspace, reg.coef_, reg_data)
    # bind the structures to the expansion willy nilly
    expansion.structures = structures
    return expansion


def test_regression_data(cluster_subspace):
    reg = LinearRegression(fit_intercept=False)
    n = np.random.randint(10, 100)
    feat_matrix = np.random.random((n, len(cluster_subspace)))
    prop_vec = np.random.random(n)
    reg_data = RegressionData.from_sklearn(
        reg, feature_matrix=feat_matrix, property_vector=prop_vec)
    coeffs = np.random.random(len(cluster_subspace))
    expansion = ClusterExpansion(cluster_subspace, coeffs, reg_data)
    assert reg_data.estimator_name == reg.__class__.__name__
    assert reg_data.module == reg.__module__
    assert reg_data.parameters == reg.get_params()
    assert_msonable(expansion)


# TODO need a realiable test, maybe create energy from a simple species concentration model
def test_predict(cluster_expansion):
    """
    n = len(cluster_expansion.structures)
    inds = np.random.choice(range(n), size=n//2)
    structures = [cluster_expansion.structures[i] for i in inds]
    preds = [cluster_expansion.predict(s, normalize=True) for s in structures]
    npt.assert_allclose(
        preds, cluster_expansion.regression_data.property_vector[inds],
        atol=1E-4
    )
    """ 
    pass

def test_prune(cluster_expansion):
    expansion = cluster_expansion.copy()
    thresh = 1E-2
    expansion.prune(threshold=thresh)
    ids = [i for i, coef in enumerate(cluster_expansion.coefs) if abs(coef) >= thresh]
    new_coefs = cluster_expansion.coefs[ids]
    new_eci = cluster_expansion.eci[ids]
    assert len(expansion.coefs) == len(new_coefs)
    npt.assert_array_equal(new_eci, expansion.eci)
    npt.assert_array_equal(new_coefs, expansion.coefs)
    assert len(expansion.cluster_subspace) == len(new_coefs)
    assert len(expansion.eci_orbit_ids) == len(new_coefs)
    pruned_feat_matrix = cluster_expansion._feat_matrix[:, ids]
    npt.assert_array_equal(
        expansion._feat_matrix, pruned_feat_matrix)
    # check that recomputing features produces whats expected
    new_feature_matrix = np.array(
        [expansion.cluster_subspace.corr_from_structure(s)
         for s in cluster_expansion.structures])
    npt.assert_array_equal(new_feature_matrix, pruned_feat_matrix)
    # check new predictions
    preds = [expansion.predict(s, normalize=True) for s in cluster_expansion.structures]
    npt.assert_allclose(
        preds, np.dot(pruned_feat_matrix, new_coefs))


def test_msonable(cluster_expansion):
    _ = str(cluster_expansion)
    d = cluster_expansion.as_dict()
    ce1 = ClusterExpansion.from_dict(d)
    npt.assert_array_equal(cluster_expansion.coefs, ce1.coefs)
    # change this to just use assert_msonable
    assert_msonable(cluster_expansion)

