"""
Definitions for site functions spaces.
These include the basis functions and measure that defines the inner product
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.legendre import legval

class BasisNotImplemented(NotImplementedError):
    pass


class SiteBasis(ABC):
    """
    Class that represents the site function space using a specified basis. This abstract class must
    by derived from for specific bases to represent a site function space.

    Name all derived classes NameBasis. See implementations below

    Note that all SiteBasis in theory have the first basis function = 1, but this should not be
    defined since it is handled implicitly when computing bit_combos using total no. species - 1
    in the Orbit class
    """

    def __init__(self, species):
        """

        Args:
            species (tuple/dict): Species. If dict, the species should be the keys and
                the value should should correspond to the probability measure associated to that
                specie. If a tuple is given a uniform probability is assumed.
            orthonormal (bool): Whether the site basis functions should be orthonormalized according
                to the supplied measure (concentration).
        """

        if not isinstance(species, dict):
            self._measure = {specie: 1 / len(species) for specie in species}
        else:
            self._measure = species

    @property
    def species(self):
        return list(self._measure.keys())

    def measure(self, *species):
        """

        Args:
            *species (str): species names or single species names

        Returns:
            numpy array or float: represents the associated measure with the give species
        """

        if isinstance(species, list) or isinstance(species, tuple):
            return np.array([self._measure(s) for s in species])
        else:
            return self._measure[species]

    @property
    @abstractmethod
    def functions(self):
        """This must be overloaded by subclasses and must return a tuple of basis functions"""
        pass

    def encode(self, specie):
        """
        Possible mapping from species to another set (i.e. species names to set of integers)

        Args:
            species (str): specie name

        Returns:
            float/str encoding
        """
        return specie

    def eval(self, fun_ind, specie): #nned to thinkg of this a bit more, how to evaluate with disticnt sites?
        """
        Evaluates the site basis function for the given species.

        Args:
            fun_ind (int): basis function index
            specie (str): specie name. must be in self.species

        Returns: float

        """
        return self.functions[fun_ind](self.encode(specie))

    def orthonormalize(self):
        pass


class IndicatorBasis(SiteBasis):
    """
    Indicator Basis
    """

    def __init__(self, species):
        super().__init__(species)
        self._functions = tuple(lambda s: 1.0 if s == sp else 0 for sp in self.species[:-1])

    @property
    def functions(self):
        return self._functions


class SinusoidBasis(SiteBasis):
    """
    Sinusoid (Sine/cosine basis) as proposed by A.VdW.
    """

    def __init__(self, species):
        super().__init__(species)
        M = len(species)
        m = M//2
        enc = range(-m, m)
        self._encoding = {s: i for (s, i) in zip(species, enc)}
        if m % 2 == 0:
            self._functions = tuple(lambda m: np.sin(np.pi * n * m / M) for n in range(1, M))
        else:
            self._functions = tuple(lambda m: np.cos(np.pi*(n+1)*m/M) for n in range(1, M))

    def encode(self, specie):
        return self._encoding[specie]

    @property
    def functions(self):
        return self._functions


class NumpyPolyBasis(SiteBasis):
    """
    Abstract class to quickly write polynomial basis included in Numpy
    """
    def __init__(self, species, poly_fun):
        super().__init__(species)
        M = len(species)
        enc = np.linspace(-1, 1, M)
        self._encoding = {s: i for (s, i) in zip(species, enc)}
        funcs, coeffs = [], [1]
        for i in range(M-1):
            coeffs.append(0)
            funcs.append(lambda x: poly_fun(x, list(reversed(coeffs))))
        self._functions = tuple(funcs)

    def encode(self, specie):
        return self._encoding[specie]


class ChebyshevBasis(NumpyPolyBasis):
    """
    Chebyshev Polynomial Basis
    """

    def __init__(self, species):
        super().__init__(species, chebval)

    @property
    def functions(self):
        return self._functions


class LegendreBasis(NumpyPolyBasis):
    """
    Legendre Polynomial Basis
    """

    def __init__(self, species):
        super().__init__(species, legval)

    @property
    def functions(self):
        return self._functions
