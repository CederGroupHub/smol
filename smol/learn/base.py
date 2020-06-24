
__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod
import numpy as np
from smol.exceptions import NotFittedError


class BaseEstimator(ABC):
    """
    A simple estimator class to use different 'in-house'  solvers to fit a
    cluster-expansion. This should be used to create specific estimator classes
    by inheriting. New classes simple need to implement the solve method.
    The methods have the same signatures as most scikit-learn regressors, such
    that those can be directly used instead of this to fit a cluster-expansion
    The base estimator does not fit. It only has a predict function for
    Expansions where the user supplies the ecis.
    """

    def __init__(self):
        self.coef_ = None

    def fit(self, feature_matrix, target_vector, sample_weight=None,
            *args, **kwargs):
        """Prepare fit input then fit."""
        if sample_weight is not None:
            feature_matrix = feature_matrix * sample_weight[:, None] ** 0.5
            target_vector = target_vector * sample_weight ** 0.5

        self.coef_ = self._solve(feature_matrix, target_vector,
                                 *args, **kwargs)

    def predict(self, feature_matrix):
        """Predict a new value based on fit"""
        if self.coef_ is None:
            raise NotFittedError('This estimator has not been fitted.')
        return np.dot(feature_matrix, self.coef_)

    @abstractmethod
    def _solve(self, feature_matrix, target_vector, *args, **kwargs):
        """Solve for the learn coefficients."""
        return
