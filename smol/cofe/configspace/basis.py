"""
Definitions for site functions spaces. The product of single site functions
make up a cluster/orbit function used to obtain correlation vectors.

Site function spaces include the basis functions and measure that defines the
inner product for a single site. Most commonly a uniform measure, but this
can be changed to use "concentration dependent" bases.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import combinations
import inspect
import warnings
from collections import OrderedDict
from functools import partial
import numpy as np
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.legendre import legval


class SiteBasis(ABC):
    """
    Class that represents the site function space using a specified basis.
    This abstract class must by derived from for specific bases to represent a
    site function space.

    Name all derived classes NameBasis. See implementations below

    Note that all SiteBasis in theory have the first basis function = 1, but
    this should not be defined since it is handled implicitly when computing
    bit_combos using total no. species - 1 in the Orbit class.

    Derived classes must define a list of functions self._functions used to
    compute the function array
    """

    def __init__(self, species):
        """
        Args:
            species (tuple/dict):
                Species. If dict, the species should be the keys and
                the value should should correspond to the probability measure
                associated to that specie. If a tuple is given a uniform
                probability is assumed.
        """
        if not isinstance(species, dict):
            species = {specie: 1 / len(species) for specie in species}
        else:
            if not np.allclose(sum(species.values()), 1):
                warnings.warn('The measure given does not sum to 1.'
                              'Are you sure this is what you want?',
                              RuntimeWarning)

        self._domain = OrderedDict(species)
        self._functions = None  # derived classes must properly define this.
        self._func_arr = None
        self._r_array = None  # array from QR in basis orthonormalization

    @property
    def function_array(self):
        if self._func_arr is None:
            self._func_arr = np.array([[f(self._encode(s))
                                        for s in self.species]
                                       for f in [lambda x: 1.0] +
                                       self._functions])
        return self._func_arr[1:]

    @property
    def measure_array(self):
        return np.diag(list(self._domain.values()))

    @property
    def measure_vector(self):
        return np.array(list(self._domain.values()))

    @property
    def orthonormalization_array(self):
        return self._r_array

    @property
    def species(self):
        return list(self._domain.keys())

    @property
    def site_space(self):
        """
        The site space refers to the probability space represented by the
        allowed species and their respective probabilities (concentration)
        over which the site functions are defined.
        """
        return self._domain

    def measure(self, specie):
        """
        Args:
            specie (str): specie name

        Returns:
            float: represents the associated measure with the give species
        """
        return self._domain[specie]

    def inner_prod(self, f, g):
        """
        Compute the inner product of two functions over probability the space
        spanned by basis
        Args:
            f: function
            g: function

        Returns: float
            inner product result
        """
        res = sum([self.measure(s) * f(self._encode(s)) * g(self._encode(s))
                   for s in self.species])
        if abs(res) < 5E-15:  # Not sure what's causing these numerical issues
            res = 0.0
        return res

    @property
    def is_orthogonal(self):
        """ Test if the basis is orthogonal """
        # add the implicit 0th function
        prods = np.dot(np.dot(self.measure_array, self._func_arr.T).T,
                       self._func_arr.T)
        d_terms = all(not np.isclose(prods[i, i], 0)
                      for i in range(prods.shape[0]))
        x_terms = all(np.isclose(prods[i, j], 0)
                      for i in range(prods.shape[0])
                      for j in range(prods.shape[1])
                      if i != j)
        return x_terms and d_terms

    @property
    def is_orthonormal(self):
        """Test if the basis is orthonormal"""
        prods = np.dot(np.dot(self.measure_array, self._func_arr.T).T,
                       self._func_arr.T)
        I = np.eye(*prods.shape)
        return np.allclose(I, prods)

    def orthonormalize(self):
        """
        Computes an orthonormal basis function set based on the measure given
        (basis functions are also orthogonal to phi_0 = 1)

        Modified GS-QR factorization of function array (here we are using
        row vectors as opposed to the correct way of doing QR using columns.
        Due to how the func_arr is saved (rows are vectors/functions) this
        allows us to not sprinkle so many transposes.
        """
        Q = np.zeros_like(self._func_arr)
        R = np.zeros_like(self._func_arr)
        V = self._func_arr.copy()
        n = V.shape[0]
        for i, phi in enumerate(V):
            R[i, i] = np.sqrt(np.dot(self.measure_vector*phi, phi))
            Q[i] = phi/R[i, i]
            R[i, i+1:n] = np.dot(V[i+1:n], self.measure_vector*Q[i])
            V[i+1:n] = V[i+1:n] - np.outer(R[i, i+1:n], Q[i])

        self._r_array = R
        self._func_arr = Q

    def _encode(self, specie):
        """
        Possible mapping from species to another set (i.e. species names to
        set of integers)
        """
        return specie


class IndicatorBasis(SiteBasis):
    """
    Indicator Basis. This basis as defined is not orthogonal.
    """

    def __init__(self, species):
        super().__init__(species)

        def indicator(s, sp):
            return int(s == sp)

        self._functions = [partial(indicator, sp=sp)
                           for sp in self.species[:-1]]
        _ = self.function_array  # initialize function array (hacky sorry)


class SinusoidBasis(SiteBasis):
    """
    Sinusoid (Sine/cosine basis) as proposed by A.VdW.
    """

    def __init__(self, species):
        super().__init__(species)
        m = len(species)

        def fun_factory(n):
            a = -(-n//2)  # ceiling division
            if n % 2 == 0:
                return lambda s: -np.sin(2 * np.pi * a * s / m)
            else:
                return lambda s: -np.cos(2 * np.pi * a * s / m)

        self._functions = [fun_factory(n) for n in range(1, m)]
        self.__encoding = {s: i for (i, s) in enumerate(species)}
        _ = self.function_array  # initialize function array (hacky sorry)

    def _encode(self, specie):
        return self.__encoding[specie]


class NumpyPolyBasis(SiteBasis, ABC):
    """
    Abstract class to quickly write polynomial basis included in Numpy
    """
    def __init__(self, species, poly_fun):
        super().__init__(species)
        m = len(species)
        enc = np.linspace(-1, 1, m)
        self.__encoding = {s: i for (s, i) in zip(species, enc)}
        funcs, coeffs = [], [1]
        for i in range(1, m):
            coeffs.append(0)
            funcs.append(partial(poly_fun, c=coeffs[::-1]))
        self._functions = funcs
        _ = self.function_array  # initialize function array (hacky sorry)

    def _encode(self, specie):
        return self.__encoding[specie]


class ChebyshevBasis(NumpyPolyBasis):
    """
    Chebyshev Polynomial Basis
    """

    def __init__(self, species):
        super().__init__(species, chebval)


class LegendreBasis(NumpyPolyBasis):
    """
    Legendre Polynomial Basis
    """

    def __init__(self, species):
        super().__init__(species, legval)


def basis_factory(basis_name, *args, **kwargs):
    """Tries to return an instance of a Basis class defined in basis.py"""
    try:
        class_name = basis_name.capitalize() + 'Basis'
        basis_class = globals()[class_name]
        instance = basis_class(*args, **kwargs)
    except KeyError:
        available = _get_subclasses(SiteBasis)
        raise NotImplementedError(f'{basis_name} is not implemented. '
                                  f'Choose one of {available}')
    return instance


def _get_subclasses(base_class):
    """
    Gets all non-abstract classes that inherit from the given base class in
    a module This is used to obtain all the available basis functions.
    """
    sub_classes = []
    for c in base_class.__subclasses__():
        if inspect.isabstract(c):
            sub_classes += _get_subclasses(c)
        else:
            sub_classes.append(c.__name__[:-5].lower())
    return sub_classes
