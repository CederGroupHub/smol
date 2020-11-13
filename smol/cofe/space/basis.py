"""Definitions for basis functions over a site function space.

The product of single site functions make up a cluster/orbit function used to
obtain correlation vectors. The domain of a site function is a site space,
which is defined by the allowed species at the site and their measures, which
is concentration of the species in the random structure)
"""

import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator
from functools import partial, wraps
import numpy as np
from numpy.polynomial.polynomial import polyval
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.legendre import legval
from monty.json import MSONable

from .domain import SiteSpace
from smol.utils import derived_class_factory, get_subclasses


__author__ = "Luis Barroso-Luque"


class DiscreteBasis(MSONable, metaclass=ABCMeta):
    """Abstract class to represent a basis set over a discrete finite domain.

    In our case the domain is a site space which can take on values of the
    allowed species.
    """

    def __init__(self, site_space, basis_functions):
        """Initialize a StandardBasis.

        Currently also accepts an OrderedDict but if you find yourself
        creating one like so for use in production and not debuging know that
        it will break MSONable methods in classes that use these, and at any
        point I could change this to not allow OrderedDicts.

        Args:
            site_space (OrderedDict or SiteSpace):
                Dict representing site space (Specie, measure) or a SiteSpace
                object.
            basis_functions (BasisIterator):
                A BasisIterator for the nonconstant basis functions. Must take
                the values of species in the site space as input.
        """
        if isinstance(site_space, OrderedDict):
            if not np.allclose(sum(site_space.values()), 1):
                warnings.warn('The measure given does not sum to 1.'
                              'Are you sure this is what you want?',
                              RuntimeWarning)
        elif not isinstance(site_space, SiteSpace):
            raise TypeError('site_space argument must be a SiteSpaces or an '
                            'OrderedDict.')

        self.flavor = basis_functions.flavor
        self._domain = site_space

        if set(site_space) != set(basis_functions.species):
            raise ValueError("Basis function iterator provided does not "
                             f"contain all species {site_space} in the site "
                             "space provided.")

        self._f_array = self._construct_function_array(basis_functions)

    @abstractmethod
    def _construct_function_array(self, basis_functions):
        """Construct function array with basis functions as rows."""
        return

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

    @property
    def function_array(self):
        """Get function array with site functions as rows."""
        return self._f_array

    @property
    def measure_array(self):
        """Get diagonal array with site species measures."""
        return np.diag(list(self._domain.values()))

    @property
    def measure_vector(self):
        """Get vector of site species measures."""
        return np.array(list(self._domain.values()))

    def as_dict(self) -> dict:
        """Get MSONable dict representation of a DiscreteBasis."""
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "site_space": self._domain.as_dict(),
             "flavor": self.flavor}
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a DiscreteBasis from dict representation.

        Args:
            d (dict):
                MSONable dict representation
        Returns:
            DiscreteBasis: A subclass of DiscreteBasis
        """
        try:
            subclass = get_subclasses(cls)[d['@class']]
        except KeyError:
            if d['@class'] == 'SiteBasis':
                warnings.warn(
                    "The object you have loaded was saved with an older "
                    "version of smol.\n Please save this object again to "
                    "prevent these warnings.", FutureWarning)
                subclass = StandardBasis
            else:
                raise NameError(f"{d['@class']} is not implemented or is not "
                                f"a subclass of {cls}.")
        return subclass.from_dict(d)


class StandardBasis(DiscreteBasis):
    r"""Class that represents the basis for a site function space.

    Note that all StandardBasis in theory have the first basis function
    :math:`\phi_0 = 1`, but this should not be defined since it is handled
    implicitly when computing bit_combos using total no. species - 1 in the
    Orbit class. As such a StandardBasis as implemented here represents a
    Standard and/or Fourier site basis (the standard basis using indicator
    functions is not a Fourier basis but can be used as "cluster site basis")

    The particular basis set is set by giving an iterable of basis functions.
    See BasisIterator classes for details.
    """

    def __init__(self, site_space, basis_functions):
        """Initialize a StandardBasis.

        Currently also accepts an OrderedDict but if you find yourself creating
        one like so for use in production and not debuging know that it will
        break MSONable methods in classes that use these, and at any point I
        could change this to not allow OrderedDicts.

        Args:
            site_space (OrderedDict or SiteSpace):
                Dict representing site space (Specie, measure) or a SiteSpace
                object.
            basis_functions (BasisIterator):
                A BasisIterator for the nonconstant basis functions. Must take
                the values of species in the site space as input.
        """
        super().__init__(site_space, basis_functions)
        self._r_array = None  # array from QR in basis orthonormalization

    def _construct_function_array(self, basis_functions):
        """Construct function array with basis functions as rows."""
        # exclude the last basis function since the constant phi_0 will
        # take its place
        nconst_functions = [function for function in basis_functions][:-1]
        func_arr = np.array([[function(sp) for sp in self.species]
                             for function in nconst_functions])
        # stack the constant basis function on there for proper normalization
        return np.vstack((np.ones_like(func_arr[0]), func_arr))

    @property
    def function_array(self):
        """Get array with the non-constant site functions as rows."""
        return self._f_array[1:]

    @property
    def orthonormalization_array(self):
        """Get R array from QR factorization."""
        return self._r_array

    @property
    def is_orthogonal(self):
        """Test if the basis is orthogonal."""
        # add the implicit 0th function
        prods = np.dot(np.dot(self.measure_array, self._f_array.T).T,
                       self._f_array.T)
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
        prods = np.dot(np.dot(self.measure_array, self._f_array.T).T,
                       self._f_array.T)
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
        Q = np.zeros_like(self._f_array)
        R = np.zeros_like(self._f_array)
        V = self._f_array.copy()
        n = V.shape[0]
        for i, phi in enumerate(V):
            R[i, i] = np.sqrt(np.dot(self.measure_vector*phi, phi))
            Q[i] = phi/R[i, i]
            R[i, i+1:n] = np.dot(V[i+1:n], self.measure_vector*Q[i])
            V[i+1:n] = V[i+1:n] - np.outer(R[i, i+1:n], Q[i])

        self._r_array = R
        self._f_array = Q

    def as_dict(self) -> dict:
        """Get MSONable dict representation."""
        d = super().as_dict()
        d["func_array"] = self._f_array.tolist()
        d["orthonorm_array"] = None if self._r_array is None \
            else self._r_array.tolist()
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a StandardBasis from dict representation.

        Args:
            d (dict):
                MSONable dict representation
        Returns:
            StandardBasis
        """
        site_space = SiteSpace.from_dict(d['site_space'])
        site_basis = basis_factory(d['flavor'], site_space)
        # restore arrays
        site_basis._f_array = np.array(d['func_array'])
        site_basis._r_array = np.array(d['orthonorm_array'])
        return site_basis


class IndicatorBasis(DiscreteBasis, MSONable):
    """Class that represents a full indicator basis for a site space.

    This class represents the "trivial" indicator basis, wich includes an
    indicator function for every species in the site space, and does NOT
    include a contant function.
    NOT to be confuse with a cluster indicator basis used for a Cluster
    Expansion (that is represented in smol by a StandardBasis with a
    IndicatorIterator).

    @lbluque takes full responsibility for the confusing terminilogy...
    """

    def __init__(self, site_space):
        """Initialize an indicator basis for give site space.

        Args:
            site_space (OrderedDict or SiteSpace):
                dict representing site space (Specie, measure) or a SiteSpace
                object.
        """
        super().__init__(site_space,
                         IndicatorIterator(tuple(site_space.keys())))

    def _construct_function_array(self, basis_functions):
        func_array = np.array([[function(sp) for sp in self.species]
                               for function in basis_functions])
        return func_array

    @classmethod
    def from_dict(cls, d):
        """Create a SiteSpace from dict representation.

        Args:
            d (dict):
                MSONable dict representation
        Returns:
            StandardBasis
        """
        return cls(SiteSpace.from_dict(d['site_space']))


class BasisIterator(Iterator):
    r"""Abstract basis iterator class.

    A basis iterator iterates through all non-constant site basis functions.
    i.e. for basis :math:`\phi_0 = 1, \phi_1, ..., \phi_{n-1}`,
    the iterator will just iterate through :math:`\phi_1, ..., \phi_{n-1}`

    Attributes:
        flavor (str):
            Name specifying the type of basis that is generated.
    """

    flavor = 'abstract'

    def __init__(self, species):
        """Initialize a BasisIterator.

        Args:
            species (tuple):
                tuple of allowed species in site spaces
        """
        self.species_iter = iter(species)
        self.species = species

    def __len__(self):
        """Get length of sequence."""
        return len(self.species)


class IndicatorIterator(BasisIterator):
    """Iterator for cluster indicator site basis functions.

    The basis generated as defined is not orthogonal for any number of species.
    """

    flavor = 'indicator'

    def __next__(self):
        """Generate the next basis function."""
        func = partial(indicator, sp=next(self.species_iter))
        return func


class SinusoidIterator(BasisIterator):
    """Iterator for sinusoid (trig basis) as proposed by A. van de Walle.

    A. van de Walle, Calphad. 33, 266–278 (2009).

    This basis is properly orthogonal for any number of allowed species out of
    the box, but it is not orthonormal for allowed species > 2.
    """

    flavor = 'sinusoid'

    def __init__(self, species):
        """Initialize a SinusoidIterator.

        Args:
            species (tuple):
                tuple of allowed species in site spaces
        """
        super().__init__(species)
        self.encoding = {s: i for (i, s) in enumerate(self.species)}

    def __next__(self):
        """Generate the next basis function."""
        n = self.encoding[next(self.species_iter)] + 1
        func = encode_domain(self.encoding)(sinusoid_factory(n, len(self.species)))  # noqa
        return func


class NumpyPolyIterator(BasisIterator):
    """Class to quickly implement polynomial basis sets included in numpy."""

    def __init__(self, species, low=-1, high=1):
        """Initialize a NumpyPolyIterator.

        Args:
            species (tuple):
                tuple of allowed species in site spaces
            low (float): optional
                lower limit of interval for encoding
            high (float): optional
                higher limit of interval for encoding
        """
        super().__init__(species)
        enc = np.linspace(low, high, len(self.species))
        self.encoding = {s: i for (s, i) in zip(species, enc)}

    @property
    @abstractmethod
    def polyval(self):
        """Return a numpy polyval function."""
        return

    def __next__(self):
        """Generate the next basis function."""
        n = self.species.index(next(self.species_iter)) + 1
        coeffs = n*[0, ] + [1]
        func = encode_domain(self.encoding)(partial(self.polyval, c=coeffs))
        return func


class PolynomialIterator(NumpyPolyIterator):
    """A standard polynomial basis set iterator."""

    @property
    def polyval(self):
        """Return numpy Chebyshev polynomial eval."""
        return polyval


class ChebyshevIterator(NumpyPolyIterator):
    """Chebyshev polynomial basis set iterator."""

    @property
    def polyval(self):
        """Return numpy Chebyshev polynomial eval."""
        return chebval


class LegendreIterator(NumpyPolyIterator):
    """Legendre polynomial basis set iterator."""

    @property
    def polyval(self):
        """Return numpy Legendre polynomial eval."""
        return legval


# The actual definitions of the functions used as basis functions.
# These functions should simply define a univariate injective function for
# a finite set of species. If the function requires an encoding for the species
# simply use the encode_domain decorator.
# Definitions must be done outside of classes to prevent pickling problems.


def indicator(s, sp):
    """Singleton indicator function for elementary events."""
    return float(s == sp)


def sinusoid_factory(n, m):
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


def encode_domain(encoding):
    """Decorate a function with an encoding for its domain.

    Args:
        encoding (dict):
            dictionary for encoding.
    """
    def decorate_func(func):
        @wraps(func)
        def encoded(s, *args, **kwargs):
            return func(encoding[s], *args, **kwargs)
        return encoded
    return decorate_func


def basis_factory(basis_name, site_space):
    """Create a site basis for the given basis name.

    Args:
        basis_name (str):
            Name of the basis.
        site_space (Sequence or OrderedDict):
            Site space over which the basis set is defined.

    Returns:
        StandardBasis
    """
    if isinstance(site_space, OrderedDict):
        species = tuple(site_space.keys())
    else:
        species = tuple(site_space)
    iterator_name = basis_name.capitalize() + 'Iterator'
    basis_funcs = derived_class_factory(iterator_name, BasisIterator, species)
    return StandardBasis(site_space, basis_funcs)
