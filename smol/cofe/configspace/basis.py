"""Definitions for site functions spaces.

The product of single site functions make up a cluster/orbit function used to
obtain correlation vectors. Site function spaces include the basis functions
and measure that defines the inner product for a single site. Most commonly a
uniform measure, but this can be changed to use "concentration" biased bases.
"""

__author__ = "Luis Barroso-Luque"

# from typing import Callable
from abc import ABC
import inspect
import warnings
from collections import OrderedDict
from functools import partial
import numpy as np
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.legendre import legval


class SiteBasis(ABC):
    """Abstract base class to represent the basis for a site function space.

    This abstract class must by derived from for specific bases to represent a
    site function space.

    Name all derived classes NameBasis. See implementations below.

    Note that all SiteBasis in theory have the first basis function = 1, but
    this should not be defined since it is handled implicitly when computing
    bit_combos using total no. species - 1 in the Orbit class.

    Derived classes must define a list of functions self._functions used to
    compute the function array
    """

    def __init__(self, species):
        """Initialize a SiteBasis.

        Args:
            species (tuple or dict):
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
        """Get array with the non-constant site functions as rows."""
        if self._func_arr is None:
            self._func_arr = np.array([[f(self._encode(s))
                                        for s in self.species]
                                       for f in [lambda x: 1.0] +
                                       self._functions])
        return self._func_arr[1:]

    @property
    def measure_array(self):
        """Get diagonal array with site species measures."""
        return np.diag(list(self._domain.values()))

    @property
    def measure_vector(self):
        """Get vector of site species measures."""
        return np.array(list(self._domain.values()))

    @property
    def orthonormalization_array(self):
        """Get R array from QR factorization."""
        return self._r_array

    @property
    def species(self):
        """Get list of allowed site species."""
        return list(self._domain.keys())

    @property
    def site_space(self):
        """Get dict of the site probability space.

        The site space refers to the probability space represented by the
        allowed species and their respective probabilities (concentration)
        over which the site functions are defined.
        """
        return self._domain

    def measure(self, species):
        """Get the site probability measure of a species.

        Args:
            specie (str): specie name

        Returns:
            float: represents the associated measure with the give species
        """
        return self._domain[species]

    def inner_prod(self, f, g):
        """Compute inner product of two functions in space.

        Compute the inner product of two functions over probability the space
        spanned by basis.

        Args:
            f (Callable): function
            g (Callable): function

        Returns:
            float: inner product result
        """
        res = sum([self.measure(s) * f(self._encode(s)) * g(self._encode(s))
                   for s in self.species])
        if abs(res) < 5E-15:  # Not sure what's causing these numerical issues
            res = 0.0
        return res

    @property
    def is_orthogonal(self):
        """Test if the basis is orthogonal."""
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
        """Test if the basis is orthonormal."""
        prods = np.dot(np.dot(self.measure_array, self._func_arr.T).T,
                       self._func_arr.T)
        identity = np.eye(*prods.shape)
        return np.allclose(identity, prods)

    def orthonormalize(self):
        """Orthonormalizes basis function set based on initial basis set.

        Functions are orthonormal w.r.t the measure given.
        (basis functions are also orthogonal to phi_0 = 1).

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
        """Encode species to another set (i.e. species names to integers)."""
        return specie


# This is not defined within the class so the class can be pickled.
def indicator(s, sp):
    """Check if s == p.

    singleton indicator function for elementary events.
    """
    return int(s == sp)


class IndicatorBasis(SiteBasis):
    """Cluster Indicator Site Basis.

    This basis as defined is not orthogonal for any number of species.
    """

    def __init__(self, species):
        """Intialize an indicator basis set.

        Args:
            species (tuple or dict):
                Species. If dict, the species should be the keys and
                the value should should correspond to the probability measure
                associated to that specie. If a tuple is given a uniform
                probability is assumed.
        """
        super().__init__(species)

        self._functions = [partial(indicator, sp=sp)
                           for sp in self.species[:-1]]
        _ = self.function_array  # initialize function array (hacky sorry)


# Same reasoning. Defined at module level to make pickling happy.
def sinusoid(n, m):
    """Sine or cosine based on AVvW sinusoid site basis."""
    a = -(-n // 2)  # ceiling division
    if n % 2 == 0:
        return partial(sin_f, a=a, m=m)
    else:
        return partial(cos_f, a=a, m=m)


def sin_f(s, a, m):
    """Return basis function for even indices."""
    return -np.sin(2 * np.pi * a * s / m)


def cos_f(s, a, m):
    """Return basis function for odd indices."""
    return -np.cos(2 * np.pi * a * s / m)


class SinusoidBasis(SiteBasis):
    """Sinusoid (Sine/cosine basis) as proposed by A. van de Walle.

    A. van de Walle, Calphad. 33, 266–278 (2009).

    This basis is properly orthogonal for any number of allowed species out of
    the box.
    """

    def __init__(self, species):
        """Initialize a sinusoid basis set.

        Args:
            species (tuple or dict):
                Species. If dict, the species should be the keys and
                the value should should correspond to the probability measure
                associated to that specie. If a tuple is given a uniform
                probability is assumed.
        """
        super().__init__(species)
        m = len(species)

        self._functions = [sinusoid(n, m) for n in range(1, m)]
        self.__encoding = {s: i for (i, s) in enumerate(species)}
        _ = self.function_array  # initialize function array (hacky sorry)

    def _encode(self, specie):
        return self.__encoding[specie]


class NumpyPolyBasis(SiteBasis, ABC):
    """Abstract class to quickly write polynomial basis included in Numpy.

    Inherit from this class for quick polynomial basis sets. See below.
    """

    def __init__(self, species, poly_fun):
        """Initialize Basis.

        Args:
            species (tuple or dict):
                Species. If dict, the species should be the keys and
                the value should should correspond to the probability measure
                associated to that specie. If a tuple is given a uniform
                probability is assumed.
            poly_fun (Callable):
                A numpy polynomial eval function (i.e. chebval)
        """
        super().__init__(species)
        m = len(species)
        enc = np.linspace(-1, 1, m)
        self.__encoding = {s: i for (s, i) in zip(species, enc)}
        funcs, coeffs = [], [1]
        for _ in range(1, m):
            coeffs.append(0)
            funcs.append(partial(poly_fun, c=coeffs[::-1]))
        self._functions = funcs
        _ = self.function_array  # initialize function array (hacky sorry)

    def _encode(self, specie):
        return self.__encoding[specie]


class ChebyshevBasis(NumpyPolyBasis):
    """Chebyshev Polynomial Site Basis.

    The actual implementation here differs from the one proposed originally
    in J. M. Sanchez, et al., Physica A. 128, 334–350 (1984). Which is properly
    orthonormal.

    As implemented here, this basis will not be orthogonal for more
    than 2 species. But can be orthonormalized calling the orthonormalize
    method, then it will be equivalent to the originally proposed one.
    """

    def __init__(self, species):
        """Initialize ChebyshevBasis.

        Args:
            species (tuple or dict):
                Species. If dict, the species should be the keys and
                the value should should correspond to the probability measure
                associated to that specie. If a tuple is given a uniform
                probability is assumed.
        """
        super().__init__(species, chebval)


class LegendreBasis(NumpyPolyBasis):
    """Legendre Polynomial Site Basis.

    Note that as implemented here, this basis will not be orthogonal for more
    than 2 species. But can be orthonormalized calling the orthonormalize
    method.
    """

    def __init__(self, species):
        """Initialize LegendreBasis.

        Args:
            species (tuple or dict):
                Species. If dict, the species should be the keys and
                the value should should correspond to the probability measure
                associated to that specie. If a tuple is given a uniform
                probability is assumed.
        """
        super().__init__(species, legval)


def basis_factory(basis_name, *args, **kwargs):
    """Try to return an instance of a Basis class defined in basis.py."""
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
    """Get all non-abstract subclasses of a class.

    Gets all non-abstract classes that inherit from the given base class in
    a module. This is used to obtain all the available basis functions.
    """
    sub_classes = []
    for c in base_class.__subclasses__():
        if inspect.isabstract(c):
            sub_classes += _get_subclasses(c)
        else:
            sub_classes.append(c.__name__[:-5].lower())
    return sub_classes
