"""Classes to define subspaces of functions over crystalline configurations.

Contains classes and functions necessary to represent the configurational space
of a crystalline system and the space of functions over it.
"""

from .basis import basis_factory
from .cluster import Cluster
from .domain import Vacancy, get_allowed_species, get_site_spaces, get_species
from .orbit import Orbit

__all__ = [
    "Cluster",
    "Orbit",
    "Vacancy",
    "get_species",
    "get_site_spaces",
    "get_allowed_species",
    "basis_factory",
]
