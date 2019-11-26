"""
Definitions for site functions spaces.
These include the basis functions and measure that defines the inner product
"""

from abc import ABC, abstractmethod
import numpy as np


class SiteBasis(ABC):
    """
    Class that represents the site function space using a specified basis. This abstract class must
    by derived from for specific bases to represent a site function space.
    Note that all SiteFunction Spaces have the first basis function = 1, but this should not be
    defined since it is handled implicitly when computing bit_combos using total no. species - 1
    in the Orbit class
    """

    def __init__(self, species, orthogonalize=False):
        """

        Parameters
        ----------
        species : tuple or dict
            tuple or dict of species. If dict, the species should be the keys and the value should
            should correspond to the probability measure associated to that specie. If a tuple is given
            a uniform probability is assumed.
        """

        if not isinstance(species, dict):
            self._measure = {specie: 1 / len(species) for specie in species}
        else:
            self._measure = species

    @property
    def species(self):
        return self._measure.keys()

    def measure(self, *species):
        """
        Parameters
        ----------
        species : str
            species names or single species names

        Returns
        -------
        numpy array or float
            represents the associated measure with the give species
        """
        if isinstance(species, list) or isinstance(species, tuple):
            return np.array([self._measure(s) for s in species])
        else:
            return self._measure[species]

    @property
    @abstractmethod
    def functions(self):
        """This must be overloaded by subclasses and must return a tuple of basis functions"""
        if self._functions is not None:
            return self._functions

    def encode(self, *species):
        """
        Possible mapping from species to another set (i.e. species names to set of integers)

        Parameters
        ----------
        species: str
            species names or single species names

        Returns
        -------
        np array or float
        """
        #encoding = {s: i for i, s in enumerate(self.species)}
        return species

    def eval(self, bits, species): #nned to thinkg of this a bit more, how to evaluate with disticnt sites?
        """
        Evaluates the site basis function for the given species.

        Parameters
        ----------
        species : tuple, list, str
            list of species names or single species names
        bits : list
            list of bits indexing the basis function number to be used for each species

        Returns
        -------
        np array or float
        """
        return np.array([self.functions[b](s) for b, s in zip(bits, self.encode(species))])


class IndicatorBasis(SiteBasis):
    """

    """

    def __init__(self, species):
        super().__init__(species)
        self._functions = tuple(lambda s: 1 if s == sp else 0 for sp in self.species[:-1])

    @property
    def functions(self):
        return self._functions

class SinusoidBasis(SiteBasis):
    """

    """

    def __init__(self, species):
        super().__init__(species)


class ChebyshevBasis(SiteBasis):
    """

    """

    def __init__(self, species):
        super().__init__(species)
