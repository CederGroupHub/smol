"""
Solvers aka functions that fit a linear model and are used to define the fit method of the Estimator
to be used for a ClusterExpansion

If your solver is simple enough then just write the function here, otherwise make a seperate file and import
the function (ie see solve_gs_preserve)
"""

import numpy as np
from .gspreserve import solve_gs_preserve

def solve_bregman(self, X, y, mu):
    return split_bregman(X, y, MaxIt=1e5, tol=1e-7, mu=mu, l=1, quiet=True)

def solve_cvxopt(X, y, mu):
    """
    X and y should already have been adjusted to account for weighting
    """
    from .l1regls import l1regls, solvers
    solvers.options['show_progress'] = False
    from cvxopt import matrix

    X1 = matrix(X)
    b = matrix(y * mu)
    return (np.array(l1regls(X1, b)) / mu).flatten()

