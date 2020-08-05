import unittest
from collections import OrderedDict
import numpy as np
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.legendre import legval
from smol.cofe.configspace import basis


class TestBasis(unittest.TestCase):
    def setUp(self) -> None:
        self.species = OrderedDict((('Li+', 0.5), ('Mn2+', 0.2),
                                   ('Vacancy', 0.3)))

    def _test_basis_uniform_measure(self, basis_iterator):
        species = tuple(self.species.keys())
        b = basis.SiteBasis(species, basis_iterator(species))
        measure = measure = list(b.measure_vector)
        self.assertEqual(measure, len(self.species)*[1/len(self.species), ])

    def _test_measure(self, basis_iterator):
        species = tuple(self.species.keys())
        b = basis.SiteBasis(self.species, basis_iterator(species))
        measure = list(b.measure_vector)
        self.assertEqual(measure, list(self.species.values()))

    def test_indicator_basis(self):
        species = tuple(self.species.keys())
        b = basis.SiteBasis(species, basis.IndicatorIterator(species))
        self.assertFalse(b.is_orthogonal)

        # test evaluation of basis functions
        n = len(self.species) - 1
        for i in range(n):
            self.assertEqual(b.function_array[i, i], 1)

        self._test_basis_uniform_measure(basis.IndicatorIterator)
        self._test_measure(basis.IndicatorIterator)

        b.orthonormalize()
        self.assertTrue(b.is_orthonormal)

    def test_sinusoid_basis(self):
        species = tuple(self.species.keys())
        b = basis.SiteBasis(species, basis.SinusoidIterator(species))
        self.assertTrue(b.is_orthogonal)

        # test evaluation of basis functions
        m = len(self.species)
        for n in range(1, len(self.species)):
            a = -(-n//2)
            f = lambda s: -np.sin(2*np.pi*a*s/m) if n % 2 == 0 else -np.cos(2*np.pi*a*s/m)
            for i, _ in enumerate(self.species):
                self.assertEqual(b.function_array[n-1, i], f(i))

        self._test_basis_uniform_measure(basis.SinusoidIterator)
        self._test_measure(basis.SinusoidIterator)
        b = basis.SiteBasis(self.species, basis.SinusoidIterator(species))
        self.assertFalse(b.is_orthogonal)
        b.orthonormalize()
        self.assertTrue(b.is_orthonormal)

    def test_chebyshev_basis(self):
        species = tuple(self.species.keys())
        b = basis.SiteBasis(species[:2], basis.ChebyshevIterator(species[:2]))
        self.assertTrue(b.is_orthogonal)  # orthogonal only for 2 species
        b = basis.SiteBasis(species, basis.ChebyshevIterator(species))
        self.assertFalse(b.is_orthogonal)

        # test evaluation of basis functions
        m, coeffs = len(self.species), [1]
        fun_range = np.linspace(-1, 1, m)
        for n in range(m-1):
            coeffs.append(0)
            for sp, x in enumerate(fun_range):
                self.assertEqual(b.function_array[n, sp],
                                 chebval(x, c=coeffs[::-1]))

        self._test_basis_uniform_measure(basis.ChebyshevIterator)
        self._test_measure(basis.ChebyshevIterator)
        b = basis.SiteBasis(self.species, basis.ChebyshevIterator(species))
        self.assertFalse(b.is_orthogonal)
        b.orthonormalize()
        self.assertTrue(b.is_orthonormal)

    def test_legendre_basis(self):
        species = tuple(self.species.keys())
        b = basis.SiteBasis(species[:2], basis.LegendreIterator(species[:2]))
        self.assertTrue(b.is_orthogonal)  # orthogonal only for 2 species
        b = basis.SiteBasis(species, basis.LegendreIterator(species))
        self.assertFalse(b.is_orthogonal)

        # test evaluation of basis functions
        m, coeffs = len(self.species), [1]
        fun_range = np.linspace(-1, 1, m)
        for n in range(m-1):
            coeffs.append(0)
            for sp, x in enumerate(fun_range):
                self.assertEqual(b.function_array[n, sp],
                                 legval(x, c=coeffs[::-1]))

        self._test_basis_uniform_measure(basis.LegendreIterator)
        self._test_measure(basis.LegendreIterator)

        b = basis.SiteBasis(self.species, basis.LegendreIterator(species))
        self.assertFalse(b.is_orthogonal)
        b.orthonormalize()
        self.assertTrue(b.is_orthonormal)

    def test_constructor(self):
        species = OrderedDict({'A': 1, 'B': 2, 'C': 1})
        basis_iter = basis.SinusoidIterator(tuple(species.keys()))
        self.assertWarns(RuntimeWarning, basis.SiteBasis, species, basis_iter)
        species = OrderedDict({'A': .1, 'B': .1, 'C': .1})
        basis_iter = basis.SinusoidIterator(tuple(species.keys()))
        self.assertWarns(RuntimeWarning, basis.SiteBasis, species, basis_iter)
        basis_iter = basis.SinusoidIterator(tuple(species.keys())[:-1])
        self.assertRaises(ValueError, basis.SiteBasis, species, basis_iter)
        species = {'A': .1, 'B': .1, 'C': .1}
        basis_iter = basis.SinusoidIterator(tuple(species.keys()))
        self.assertRaises(TypeError, basis.SiteBasis, species, basis_iter)
