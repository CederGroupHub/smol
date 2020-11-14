"""Classes for external terms that can be added to a cluster subspace.

Contains classes for external terms to be added to a cluster subspace
representing additional features to be fitted in a cluster expansion.

Currently only a base class for calculating pairwise interactions and a child class for Ewald electrostatic interaction term exists. 
"""

from __future__ import division
from .ewald import EwaldTerm
from .base import PairwiseTerms

__all__ = ['EwaldTerm','PairwiseTerms']
