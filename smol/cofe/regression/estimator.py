"""
Solvers aka functions that fit a linear model and are used to define the fit method of the Estimator
to be used for a ClusterExpansion

If your solver is simple enough then just write the Subclass here, otherwise make a seperate file and import
the Estimator in the __init__.py file (ie see solve_gs_preserve)
"""
import numpy as np
import logging
import warnings
from ..utils import NotFittedError

class BaseEstimator():
    """
    A simple estimator class to use different 'in-house'  solvers to fit a cluster-expansion
    This should be used to create specific estimator classes by inheriting. New classes simple need to implement
    the solve method.
    The methods have the same signatures as most Scikit-learn regressors, such that those can be directly used
    instead of this to fit a cluster-expansion
    The base estimator does not fit. It only has a predict function for Expansions where the user supplies the
    ecis
    """

    def __init__(self):
        self.coef_ = None
        self.mus = None
        self.cvs = None

    def _solve(self, X, y, *args, **kwargs):
        '''This needs to be overloaded in derived classes'''
        raise AttributeError(f'No solve method specified: self._solve: {self._solve}')

    def fit(self, X, y, sample_weight=None, *args, **kwargs):
        if sample_weight is not None:
            X = X * sample_weight[:, None] ** 0.5
            y = y * sample_weight ** 0.5

        self.coef_ = self._solve(X, y, *args, **kwargs)

    def predict(self, X):
        if self.coef_ is None:
            raise NotFittedError
        return np.dot(X, self.coef_)

    def _calc_cv_score(self, mu, X, y, weights, k=5):
        """
        Args:
            mu: weight of error in bregman
            X: sensing matrix (scaled appropriately)
            y: data to fit (scaled appropriately)
            k: number of partitions

        Partition the sample into k partitions, calculate out-of-sample
        variance for each of these partitions, and add them together
        """
        if weights is None:
            weights = np.ones(len(X[:,0]))
        logging.info('starting cv score calculations for mu: {}, k: {}'.format(mu, k))
        # generate random partitions
        partitions = np.tile(np.arange(k), len(y) // k + 1)
        np.random.shuffle(partitions)
        partitions = partitions[:len(y)]

        ssr = 0
        ssr_uw = 0
        for i in range(k):
            ins = (partitions != i)  # in the sample for this iteration
            oos = (partitions == i)  # out of the sample for this iteration

            self.fit(X[ins], y[ins], weights[ins], mu)
            res = (np.dot(X[oos], self.coef_) - y[oos]) ** 2
            ssr += np.sum(res * weights[oos]) / np.average(weights[oos])
            ssr_uw += np.sum(res)

        logging.info(
            'cv rms_error: {} (weighted) {} (unweighted)'.format(np.sqrt(ssr / len(y)), np.sqrt(ssr_uw / len(y))))
        cv = 1 - ssr / np.sum((y - np.average(y)) ** 2)
        return cv


class CVXEstimator(BaseEstimator):
    """
    Estimator implementing the written l1regs cvx based solver
    """

    def __init__(self):
        super().__init__()

    def _solve(self, X, y, mu):
        """
        X and y should already have been adjusted to account for weighting
        """

        # Maybe its cleaner to use importlib at the top to try and import these?
        from .l1regls import l1regls, solvers
        solvers.options['show_progress'] = False
        from cvxopt import matrix

        X1 = matrix(X)
        b = matrix(y * mu)
        return (np.array(l1regls(X1, b)) / mu).flatten()

    def _get_optimum_mu(self, X, y, weights, k=5, min_mu=0.1, max_mu=6):
        """
        Finds the value of mu that maximizes the cv score
        """
        mus = list(np.logspace(min_mu, max_mu, 10))
        cvs = [self._calc_cv_score(mu, X, y, weights, k) for mu in mus]

        for _ in range(2):
            i = np.nanargmax(cvs)
            if i == len(mus) - 1:
                warnings.warn('Largest mu chosen. You should probably increase the basis set')
                break

            mu = (mus[i] * mus[i + 1]) ** 0.5
            mus[i + 1:i + 1] = [mu]
            cvs[i + 1:i + 1] = [self._calc_cv_score(mu, X, y, weights, k)]

            mu = (mus[i - 1] * mus[i]) ** 0.5
            mus[i:i] = [mu]
            cvs[i:i] = [self._calc_cv_score(mu, X, y, weights, k)]

        self.mus = mus
        self.cvs = cvs
        logging.info('best cv score: {}'.format(np.nanmax(self.cvs)))
        return mus[np.nanargmax(cvs)]

    def fit(self, X, y, sample_weight=None, mu=None, *args, **kwargs):
        if sample_weight is not None:
            X = X * sample_weight[:, None] ** 0.5
            y = y * sample_weight ** 0.5

        if mu is None:
            mu = self._get_optimum_mu(X, y, sample_weight)
        self.coef_ = self._solve(X, y, mu, *args, **kwargs)