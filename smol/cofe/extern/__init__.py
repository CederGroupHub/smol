"""
Contains classes for external terms to be added to a cluster subspace
representing additional features to be fitted in a cluster expansion.
Currently only an Ewald electrostatic interaction term exists. Maybe it will
be the only term ever needed, but alas abstraction?
"""

from __future__ import division
from .ewald import EwaldTerm

__all__ = ['EwaldTerm']
