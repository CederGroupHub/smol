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
    bit_combos using total no. species - 1 in the Orbit class
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
            self._domain = {specie: 1 / len(species) for specie in species}
        else:
            if not np.allclose(sum(species.values()), 1):
                warnings.warn('The measure given does not sum to 1.'
                              'Are you sure this is what you want?',
                              RuntimeWarning)
            self._domain = species

        self._functions = None
        self._func_arr = None

    @property
    @abstractmethod
    def functions(self):
        """
        This must be overloaded by subclasses and must return a tuple
        of basis functions
        """
        pass

    @property
    def function_array(self):
        if self._func_arr is None:
            self._func_arr = np.array([[f(self.encode(s))
                                        for s in self.species]
                                       for f in self._functions])
        return self._func_arr

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
        res = sum([self.measure(s)*f(self.encode(s))*g(self.encode(s))
                   for s in self.species])
        if abs(res) < 5E-15:  # Not sure what's causing these numerical issues
            res = 0.0
        return res

    def encode(self, specie):
        """
        Possible mapping from species to another set (i.e. species names to
        set of integers)

        Args:
            specie (str): specie name

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
        if specie not in self.species:
            raise ValueError(f'{specie} is not in valid species: '
                             f'{self.species}')

        return self.functions[fun_ind](self.encode(specie))

    @property
    def is_orthogonal(self):
        """ Test if the basis is orthogonal """
        # add the implicit 0th function
        functions = [lambda x: 1, *self.functions]
        x_terms = all(self.inner_prod(f, g) == 0
                      for f, g in combinations(functions, 2))
        d_terms = all(self.inner_prod(f, f) != 0 for f in self.functions)
        return x_terms and d_terms

    @property
    def is_orthonormal(self):
        """Test if the basis is orthonormal"""
        d_terms = all(np.isclose(self.inner_prod(f, f), 1)
                      for f in self.functions)
        return d_terms and self.is_orthogonal

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
                    projs = sum(self.inner_prod(f, g)*g(s) for g in on_funs)
                    return f(s) - projs

                norm = np.sqrt(self.inner_prod(g_0, g_0))

                def g_norm(s):
                    return g_0(s)/norm
                return g_norm

            g = g_factory(f, deepcopy(on_funs))
            on_funs.append(g)
        on_funs.pop(0)  # remove phi_0 = 1, since it is handled implicitly

        self._func_arr = None # reset this
        self._functions = on_funs


class IndicatorBasis(SiteBasis):
    """
    Indicator Basis. This basis as defined is not orthogonal.
    """

    def __init__(self, species):
        super().__init__(species)

        def indicator(s, sp):
            return int(s == sp)

        self._functions = tuple(partial(indicator, sp=sp)
                                for sp in self.species[:-1])

    @property
    def functions(self):
        return self._functions


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

        self._functions = tuple(fun_factory(n) for n in range(1, m))
        self._encoding = {s: i for (i, s) in enumerate(species)}

    def encode(self, specie):
        return self._encoding[specie]

    @property
    def functions(self):
        return self._functions


class NumpyPolyBasis(SiteBasis, ABC):
    """
    Abstract class to quickly write polynomial basis included in Numpy
    """
    def __init__(self, species, poly_fun):
        super().__init__(species)
        m = len(species)
        enc = np.linspace(-1, 1, m)
        self._encoding = {s: i for (s, i) in zip(species, enc)}
        funcs, coeffs = [], [1]
        for i in range(1, m):
            coeffs.append(0)
            funcs.append(partial(poly_fun, c=coeffs[::-1]))
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
