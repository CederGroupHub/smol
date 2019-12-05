"""
Definitions for site functions spaces.
These include the basis functions and measure that defines the inner product
"""

from abc import ABC, abstractmethod
from copy import deepcopy
import warnings
from functools import partial
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
            if not np.allclose(sum(species.values()), 1):
                warnings.warn('The measure given does not sum to 1. Are you sure this is what you want?',
                              RuntimeWarning)
            self._measure = species

        self._functions = None

    @property
    def species(self):
        return list(self._measure.keys())

    def measure(self, species):
        """

        Args:
            species (str): species names or single species names

        Returns:
            float: represents the associated measure with the give species
        """

        return self._measure[species]

    def inner_prod(self, f, g):
        res = sum([self.measure(s)*f(self.encode(s))*g(self.encode(s)) for s in self.species])
        if abs(res) < 5E-16:
            res = 0.0
        return res

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

    def eval(self, fun_ind, specie):
        """
        Evaluates the site basis function for the given species.

        Args:
            fun_ind (int): basis function index
            specie (str): specie name. must be in self.species

        Returns: float

        """
        return self.functions[fun_ind](self.encode(specie))

    def orthonormalize(self):
        """
        Returns an orthonormal basis function set based on the measure given
        (basis functions are also orthogonal to phi_0 = 1)

        Its sort of black magic, there may be a better way to write this...
        """

        on_funs = [lambda s: 1.0]
        for f in self._functions:
            def g_factory(f, on_funs):

                def g_0(s):
                    return f(s) - sum(self.inner_prod(f, g)*g(s) for g in on_funs)
                norm = np.sqrt(self.inner_prod(g_0, g_0))

                def g_norm(s):
                    return g_0(s)/norm
                return g_norm

            g = g_factory(f, deepcopy(on_funs))
            on_funs.append(g)
        on_funs.pop(0)
        self._functions = on_funs


class IndicatorBasis(SiteBasis):
    """
    Indicator Basis
    """

    def __init__(self, species):
        super().__init__(species)

        def indicator(s, sp):
            return int(s == sp)

        self._functions = tuple(partial(indicator, sp=sp) for sp in self.species[:-1])

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

        def fun(s, n):
            a = -(-n//2)  #ceiling division
            if n % 2 == 0:
                return -np.sin(2*np.pi*a*s/M)
            else:
                return -np.cos(2*np.pi*a*s/M)

        self._functions = tuple(partial(fun, n=n) for n in range(1, M))
        self._encoding = {s: i for (i, s) in enumerate(species)}

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
            funcs.append(partial(poly_fun, c=list(reversed(coeffs))))
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
