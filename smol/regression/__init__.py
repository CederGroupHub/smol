"""
This module contains functionality necessary for fitting Cluster Expansions
and testing the performance of the fit
"""

from .estimator import CVXEstimator, BaseEstimator
from .utils import constrain_dielectric

__all__ = ['CVXEstimator', 'BaseEstimator', 'constrain_dielectric']
