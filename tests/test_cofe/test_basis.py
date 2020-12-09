import unittest
from collections import OrderedDict
import numpy as np
import numpy.testing as npt
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.legendre import legval
from pymatgen import Composition
from smol.cofe.space import domain
from smol.cofe.space import basis
from tests.utils import assert_msonable


# TODO change all tests to use SiteSpaces only and not OrderedDicts
# TODO add test for out-of-box basis not orthogonal with non-uniform measure
class TestBasis(unittest.TestCase):
    def setUp(self) -> None:
        comp = Composition((('A', 0.25), ('B', 0.25),
                           ('C', 0.25), ('D', 0.25)))
        self.site_space = domain.SiteSpace(comp)

    def _test_basis_measure_raise(self, basis_iterator):
        species = tuple(self.site_space.keys())
        comp = Composition((('A', 0.25), ('B', 0.25),
                            ('C', 0.8), ('D', 0.25)))

        with self.assertRaises(ValueError):
            site_space = domain.SiteSpace(comp)
            b = basis.StandardBasis(site_space, basis_iterator(species))

    def _test_measure(self, basis_iterator):
        species = tuple(self.site_space.keys())
        b = basis.StandardBasis(self.site_space, basis_iterator(species))
        measure = list(b.measure_vector)
        self.assertEqual(measure, list(self.site_space.values()))

    def test_indicator_basis(self):
        species = tuple(self.site_space.keys())
        b = basis.StandardBasis(self.site_space, basis.IndicatorIterator(species))
        self.assertFalse(b.is_orthogonal)

        # test evaluation of basis functions
        n = len(self.site_space) - 1
        for i in range(n):
            self.assertEqual(b.function_array[i, i], 1)

        self._test_basis_measure_raise(basis.IndicatorIterator)
        self._test_measure(basis.IndicatorIterator)

        original = b._f_array.copy()
        b.orthonormalize()
        self.assertTrue(b.is_orthonormal)
        self.assertTrue(np.allclose(b._r_array @ b._f_array, original))

    def test_sinusoid_basis(self):
        species = tuple(self.site_space.keys())
        b = basis.StandardBasis(self.site_space, basis.SinusoidIterator(species))
        self.assertTrue(b.is_orthogonal)

        # test evaluation of basis functions
        m = len(self.site_space)
        for n in range(1, len(self.site_space)):
            a = -(-n//2)
            f = lambda s: -np.sin(2*np.pi*a*s/m) if n % 2 == 0 else -np.cos(2*np.pi*a*s/m)
            for i, _ in enumerate(self.site_space):
                self.assertTrue(np.isclose(b.function_array[n-1, i], f(i)))

        self._test_basis_measure_raise(basis.SinusoidIterator)
        self._test_measure(basis.SinusoidIterator)
        b = basis.StandardBasis(self.site_space, basis.SinusoidIterator(species))
        self.assertTrue(b.is_orthogonal)
        original = b._f_array.copy()
        b.orthonormalize()
        self.assertTrue(b.is_orthonormal)
        self.assertTrue(np.allclose(b._r_array @ b._f_array, original))

    def test_chebyshev_basis(self):
        species = tuple(self.site_space.keys())
        site_space = OrderedDict(tuple(self.site_space.items())[:2])
        b = basis.StandardBasis(site_space, basis.ChebyshevIterator(species[:2]))
        self.assertTrue(b.is_orthogonal)  # orthogonal only for 2 species
        b = basis.StandardBasis(self.site_space,
                                basis.ChebyshevIterator(tuple(self.site_space.keys())))
        self.assertFalse(b.is_orthogonal)

        # test evaluation of basis functions
        m, coeffs = len(self.site_space), [1]
        fun_range = np.linspace(-1, 1, m)
        for n in range(m-1):
            coeffs.append(0)
            for sp, x in enumerate(fun_range):
                self.assertEqual(b.function_array[n, sp],
                                 chebval(x, c=coeffs[::-1]))

        self._test_basis_measure_raise(basis.ChebyshevIterator)
        self._test_measure(basis.ChebyshevIterator)
        b = basis.StandardBasis(self.site_space, basis.ChebyshevIterator(species))
        self.assertFalse(b.is_orthogonal)
        b.orthonormalize()
        self.assertTrue(b.is_orthonormal)

    def test_legendre_basis(self):
        species = tuple(self.site_space.keys())
        site_space = OrderedDict(tuple(self.site_space.items())[:2])
        b = basis.StandardBasis(site_space, basis.LegendreIterator(species[:2]))
        self.assertTrue(b.is_orthogonal)  # orthogonal only for 2 species
        b = basis.StandardBasis(self.site_space,
                                basis.LegendreIterator(species))
        self.assertFalse(b.is_orthogonal)

        # test evaluation of basis functions
        m, coeffs = len(self.site_space), [1]
        fun_range = np.linspace(-1, 1, m)
        for n in range(m-1):
            coeffs.append(0)
            for sp, x in enumerate(fun_range):
                self.assertEqual(b.function_array[n, sp],
                                 legval(x, c=coeffs[::-1]))

        self._test_basis_measure_raise(basis.LegendreIterator)
        self._test_measure(basis.LegendreIterator)

        b = basis.StandardBasis(self.site_space, basis.LegendreIterator(species))
        self.assertFalse(b.is_orthogonal)
        b.orthonormalize()
        self.assertTrue(b.is_orthonormal)

    def test_constructor(self):
        species = OrderedDict({'A': 1, 'B': 2, 'C': 1})
        basis_iter = basis.SinusoidIterator(tuple(species.keys()))
        self.assertWarns(RuntimeWarning, basis.StandardBasis, species, basis_iter)
        species = OrderedDict({'A': .1, 'B': .1, 'C': .1})
        basis_iter = basis.SinusoidIterator(tuple(species.keys()))
        self.assertWarns(RuntimeWarning, basis.StandardBasis, species, basis_iter)
        basis_iter = basis.SinusoidIterator(tuple(species.keys())[:-1])
        self.assertRaises(ValueError, basis.StandardBasis, species, basis_iter)
        species = {'A': .1, 'B': .1, 'C': .1}
        basis_iter = basis.SinusoidIterator(tuple(species.keys()))
        self.assertRaises(TypeError, basis.StandardBasis, species, basis_iter)

    def test_msonable(self):
        species = tuple(self.site_space.keys())
        b = basis.StandardBasis(self.site_space, basis.SinusoidIterator(species))
        self.assertTrue(b.is_orthogonal)
        b.orthonormalize()
        self.assertTrue(b.is_orthonormal)
        assert_msonable(b)
        b1 = basis.StandardBasis.from_dict(b.as_dict())
        self.assertTrue(b1.is_orthonormal)
        npt.assert_array_equal(b.function_array, b1.function_array)
        npt.assert_array_equal(b.orthonormalization_array, b1.orthonormalization_array)
