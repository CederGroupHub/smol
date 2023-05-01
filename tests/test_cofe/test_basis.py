from collections import OrderedDict

import numpy as np
import numpy.testing as npt
import pytest
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.legendre import legval
from pymatgen.core import Composition

from smol.cofe.space import basis, domain
from smol.utils.class_utils import get_subclasses
from tests.utils import assert_msonable

pytestmark = pytest.mark.filterwarnings("ignore:The measure given does not sum to 1.")
basis_iterators = list(get_subclasses(basis.BasisIterator).values())


@pytest.fixture(params=[True, False], scope="module")
def site_space(expansion_structure, request):
    spaces = domain.get_site_spaces(expansion_structure, include_measure=request.param)
    return spaces[0]


@pytest.fixture(params=basis_iterators)
def standard_basis(site_space, request):
    species = tuple(site_space.keys())
    return basis.StandardBasis(site_space, request.param(species))


@pytest.mark.parametrize("basis_iterator_cls", basis_iterators)
def test_basis_measure_raise(basis_iterator_cls):
    comp = Composition((("A", 0.25), ("B", 0.25), ("C", 0.8), ("D", 0.25)))
    with pytest.raises(ValueError):
        space = domain.SiteSpace(comp)
        species = tuple(space.keys())
        basis.StandardBasis(space, basis_iterator_cls(species))


@pytest.mark.parametrize("basis_iterator_cls", basis_iterators)
def test_constructor(basis_iterator_cls):
    species = OrderedDict({"A": 1, "B": 2, "C": 1})
    basis_iter = basis_iterator_cls(tuple(species.keys()))
    with pytest.warns(RuntimeWarning):
        basis.StandardBasis(species, basis_iter)

    species = OrderedDict({"A": 0.1, "B": 0.1, "C": 0.1})
    basis_iter = basis.SinusoidIterator(tuple(species.keys()))

    with pytest.warns(RuntimeWarning):
        basis.StandardBasis(species, basis_iter)
    basis_iter = basis_iterator_cls(tuple(species.keys())[:-1])

    with pytest.raises(ValueError):
        basis.StandardBasis(species, basis_iter)


@pytest.mark.parametrize("basis_iterator_cls", basis_iterators)
def test_msonable(site_space, basis_iterator_cls):
    species = tuple(site_space.keys())
    b = basis.StandardBasis(site_space, basis_iterator_cls(species))
    assert_msonable(b)
    b1 = basis.StandardBasis.from_dict(b.as_dict())
    assert b.is_orthogonal == b1.is_orthogonal
    assert b.is_orthonormal == b1.is_orthonormal
    npt.assert_array_equal(b.function_array, b1.function_array)
    npt.assert_array_equal(b.orthonormalization_array, b1.orthonormalization_array)

    b.orthonormalize()
    b1 = basis.StandardBasis.from_dict(b.as_dict())
    assert b1.is_orthogonal
    assert b1.is_orthonormal
    npt.assert_array_equal(b.function_array, b1.function_array)
    npt.assert_array_equal(b.orthonormalization_array, b1.orthonormalization_array)


def test_standard_basis(standard_basis):
    site_space = standard_basis.site_space
    measure = list(standard_basis.measure_vector)
    assert measure == list(site_space.values())

    original = standard_basis._f_array.copy()
    standard_basis.orthonormalize()
    assert standard_basis.is_orthonormal
    assert np.allclose(standard_basis._r_array @ standard_basis._f_array, original)


def test_rotate(standard_basis, rng):
    f_array = standard_basis._f_array.copy()
    theta = np.pi / rng.integers(2, 10)

    # catch and test non-uniform measures
    if not np.allclose(standard_basis.measure_vector[0], standard_basis.measure_vector):
        with pytest.warns(UserWarning):
            standard_basis.rotate(theta)
        return

    if standard_basis.is_orthogonal:
        standard_basis.rotate(theta)
        assert standard_basis.is_orthogonal
        # if orthogonal all inner products should match!
        npt.assert_array_almost_equal(
            f_array @ f_array.T, standard_basis._f_array @ standard_basis._f_array.T
        )
        npt.assert_array_almost_equal(
            standard_basis._f_array[0], np.ones(len(standard_basis.site_space))
        )
        if len(standard_basis.site_space) == 2:  # rotation is * -1
            npt.assert_array_almost_equal(standard_basis._f_array[1], -1 * f_array[1])
        else:
            npt.assert_almost_equal(
                np.arccos(
                    np.dot(standard_basis._f_array[1], f_array[1])
                    / (
                        np.linalg.norm(standard_basis._f_array[1])
                        * np.linalg.norm(f_array[1])
                    )
                ),
                theta,
            )
        standard_basis.rotate(-theta)
        npt.assert_array_almost_equal(f_array, standard_basis._f_array)
        standard_basis.orthonormalize()
        assert standard_basis.is_orthonormal
        standard_basis.rotate(theta)
        assert standard_basis.is_orthonormal

        if len(standard_basis.site_space) > 2:
            with pytest.raises(ValueError):
                standard_basis.rotate(theta, 0, 0)

            with pytest.raises(ValueError):
                standard_basis.rotate(theta, len(standard_basis.site_space))
            with pytest.raises(ValueError):
                standard_basis.rotate(theta, 0, len(standard_basis.site_space))

        comp = Composition((("A", 0.2), ("B", 0.2), ("C", 0.3), ("D", 0.3)))
        b = basis.basis_factory("sinusoid", domain.SiteSpace(comp))
        with pytest.warns(UserWarning):
            b.rotate(theta)
    else:
        with pytest.raises(RuntimeError):
            standard_basis.rotate(theta)


# basis specific tests
def test_indicator_basis(site_space):
    species = tuple(site_space.keys())
    b = basis.StandardBasis(site_space, basis.IndicatorIterator(species))
    assert not b.is_orthogonal

    # test evaluation of basis functions
    n = len(site_space) - 1
    for i in range(n):
        assert b.function_array[i, i] == 1

    original = b._f_array.copy()
    b.orthonormalize()
    assert b.is_orthonormal
    npt.assert_allclose(b._r_array @ b._f_array, original, atol=5e-16)


def test_sinusoid_basis(site_space):
    species = tuple(site_space.keys())
    b = basis.StandardBasis(site_space, basis.SinusoidIterator(species))

    # ortho only holds for uniform measure
    if np.allclose(b.measure_vector[0], b.measure_vector):
        assert b.is_orthogonal

        if len(site_space) == 2:
            assert b.is_orthonormal
        else:
            assert not b.is_orthonormal

    # test evaluation of basis functions
    m = len(site_space)
    for n in range(1, len(site_space)):
        a = -(-n // 2)
        f = (
            lambda s: -np.sin(2 * np.pi * a * s / m)
            if n % 2 == 0
            else -np.cos(2 * np.pi * a * s / m)
        )
        for i, _ in enumerate(site_space):
            npt.assert_almost_equal(b.function_array[n - 1, i], f(i))

    original = b._f_array.copy()
    b.orthonormalize()
    assert b.is_orthonormal
    npt.assert_allclose(b._r_array @ b._f_array, original, atol=1e-15)


def test_chebyshev_basis(site_space):
    species = tuple(site_space.keys())
    b = basis.StandardBasis(site_space, basis.ChebyshevIterator(species))
    if len(site_space) == 2:
        assert b.is_orthogonal
    else:
        assert not b.is_orthogonal

    # test evaluation of basis functions
    m, coeffs = len(site_space), [1]
    fun_range = np.linspace(-1, 1, m)
    for n in range(m - 1):
        coeffs.append(0)
        for sp, x in enumerate(fun_range):
            assert b.function_array[n, sp] == chebval(x, c=coeffs[::-1])
    b.orthonormalize()
    assert b.is_orthonormal


def test_legendre_basis(site_space):
    species = tuple(site_space.keys())
    b = basis.StandardBasis(site_space, basis.LegendreIterator(species))
    if len(site_space) == 2:
        assert b.is_orthogonal
    else:
        assert not b.is_orthogonal

    # test evaluation of basis functions
    m, coeffs = len(site_space), [1]
    fun_range = np.linspace(-1, 1, m)
    for n in range(m - 1):
        coeffs.append(0)
        for sp, x in enumerate(fun_range):
            assert b.function_array[n, sp] == legval(x, c=coeffs[::-1])

    b.orthonormalize()
    assert b.is_orthonormal
