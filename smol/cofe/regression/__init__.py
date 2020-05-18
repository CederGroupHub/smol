"""
This module contains functionality necessary for fitting Cluster Expansions
and testing the performance of the fit
"""

from .estimator import CVXEstimator, BaseEstimator

__all__ = ['CVXEstimator', 'BaseEstimator', 'constrain_dielectric']


# TODO fix and add unittest for this
def constrain_dielectric(ce, max_dielectric, e_ind=-1, *fitargs, **fitkwargs):
    """
    Refit a cluster expansion that contains an EwaldTerm to constrain the
    dielectric constant to be positive and below the supplied value
    (note that this is also affected by whether the primitive cell is the
    correct size)

    Args:
        ce (ClusterExpansion):
            Fitted cluster expansion.
        max_dielectric (float):
            Value of maximum dielectric constant to constrain by.
        e_ind (int):
            Index of ewald "ECI" in the expansions ECI vector
        fitargs:
            arguments to be passed to the estimators's fit method
        fitkwargs:
            keyword arguments to be passed to the estimator's fit method
    """
    ext_terms = [term.__name__ for term, _, _
                 in ce.cluster_subspace.external_terms]
    if 'EwaldTerm' not in ext_terms:
        raise RuntimeError('This ClusterExpansion does not have an Ewald term')

    A_in = ce.feature_matrix.copy()
    y_in = ce.property_vector.copy()

    if ce.ecis[e_ind] < 1.0 / max_dielectric:
        y_in -= A_in[:, e_ind] / max_dielectric
        A_in[:, e_ind] = 0
        ce.estimator.fit(A_in, y_in, *fitargs, **fitkwargs)
        ce.ecis[e_ind] = 1.0 / max_dielectric
