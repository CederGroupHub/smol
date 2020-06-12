"""Some utility functions to polish Cluster Expansion fits."""

__author__ = "Luis Barroso-Luque"
__credits__ = "William Davidson Richard"

from functools import wraps


def constrain_dielectric(max_dielectric, ewald_ind=-1):
    """Constrain a fit method to keep dieletric 0<= e < max_dielectric.

    Decorator to enforce that a fit method fitting a cluster expansion that
    contains an EwaldTerm to constrain the dielectric constant to be positive
    and below the supplied value.

    If the dielectric (inverse of the Ewald eci) is negative or above the max
    dielectric, the decorator will force the given fit_method to refit to the
    target vector with the Ewald interactions times the max dielectric
    subtracted out.

    Use this as a standard decorator with parameters:
    - At runtime: ecis = constrain_dielectric(max_dielectric)(fit_method)(X, y)
    - In fit_method definitions: @constrain_dielectric(max_dielectric)
                                 def your_fit_method(X, y):

    Args:
        max_dielectric (float):
            Value of maximum dielectric constant to constrain by.
        ewald_ind (int):
            Index of column of Ewald interaction features in the feature matrix
    """
    def decorate_fit_method(fit_method):
        """Decorate a fit method to constrain "dielectric constant".

        Args:
            fit_method (callable):
                the fit_method you will use to fit your cluster expansion.
                Must take the feature matrix X and target vector y as first
                arguments. (i.e. fit_method(X, y, *args, **kwargs)
        """
        @wraps(fit_method)
        def wrapped(X, y, *args, **kwargs):
            ecis = fit_method(X, y, *args, **kwargs)
            if ecis[ewald_ind] < 1.0 / max_dielectric:
                X_, y_ = X.copy(), y.copy()
                y_ -= X_[:, ewald_ind] / max_dielectric
                X_[:, ewald_ind] = 0
                ecis = fit_method(X_, y_, *args, **kwargs)
                ecis[ewald_ind] = 1.0 / max_dielectric
            return ecis
        return wrapped
    return decorate_fit_method
