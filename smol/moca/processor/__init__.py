"""
Implementation of base processor classes for a fixed size super cell.

Processor classes are used to represent a configuration domain for a fixed
sized supercell and should implement a "fast" way to compute the property
they represent or changes in said property from site flips. Things necessary
to run Monte Carlo sampling.
"""

from .composite import CompositeProcessor
from .expansion import ClusterExpansionProcessor
from .ewald import EwaldProcessor

__all__ = ['CompositeProcessor', 'ClusterExpansionProcessor', 'EwaldProcessor']
