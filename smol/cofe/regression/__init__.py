# TODO think of adding a metrics module


def constrain_dielectric(ce, max_dielectric, e_ind=-1, *fitargs, **fitkwargs):
    """
    Refit a cluster expansion that contains an EwaldTerm to Constrain the
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
        fitkwards:
            keyword arguments to be passed to the estimator's fit method
    """
    ext_terms = [term.__name__ for term, _, _ in ce.wrangler.cs.external_terms]
    if 'EwaldTerm' not in ext_terms:
        raise RuntimeError('This ClusterExpansion does not have an Ewald term')
    elif ce.ecis is None:
        raise RuntimeError('This ClusterExpansion does not have ECIs.'
                           'Perhaps it has not been fitted yet?')

    A_in = ce.wrangler.feature_matrix.copy()
    y_in = ce.wrangler.normalized_properties.copy()

    if ce.ecis[e_ind] < 1.0 / max_dielectric:
        y_in -= A_in[:, e_ind] / max_dielectric
        A_in[:, e_ind] = 0
        ce.estimator.fit(A_in, y_in, *fitargs, **fitkwargs)
        ce.ecis[e_ind] = 1.0 / max_dielectric
