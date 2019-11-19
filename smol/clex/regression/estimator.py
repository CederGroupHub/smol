import numpy as np
import logging
import warnings

from ..utils import NotFittedError
from ..regression import solvers

class Estimator(object):
    """
    A simple estimator class to use different 'in-house'  solvers to fit a cluster-expansion
    The methods have the same signatures as most Scikit-learn regressors, such that those can be directly used
    instead of this to fit a cluster-expansion
    """

    def __init__(self, solver=solvers.solve_cvxopt):
        self._solver = solver
        self.coef_ = None
        self.mus = None
        self.cvs = None

    def fit(self, X, y, sample_weights=None, mu=None, *args, **kwargs):
        if self._solver is None:
            raise AttributeError(f'No solver specified: self._solver: {self._solver}')
        if sample_weights is not None:
            X = X * sample_weights[:, None] ** 0.5
            y = y * sample_weights ** 0.5

        if mu is None:
            mu = self._get_optimum_mu(X, y, sample_weights)
        self.coef_ = self._solver(X, y, mu, *args, **kwargs)

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

    def __str__(self):
        return f'Estimator, solver: {str(self._solver)}'
