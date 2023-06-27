"""Test smol.moca.utils.math_utils."""
from collections import Counter
from fractions import Fraction

import numpy as np
import numpy.testing as npt
import polytope as pc
import pytest

from smol.utils.math import (
    NUM_TOL,
    choose_section_from_partition,
    compute_snf,
    connectivity,
    count_row_matches,
    flip_size,
    flip_weights_mask,
    get_ergodic_vectors,
    get_natural_centroid,
    get_natural_solutions,
    get_nonneg_float_vertices,
    get_one_dim_solutions,
    get_optimal_basis,
    integerize_multiple,
    integerize_vector,
    rationalize_number,
    solve_diophantines,
    yield_hermite_normal_forms,
)
from tests.utils import assert_table_set_equal

a1 = [[1, 3, 4, -3, -2], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]
b1 = [0, 1, 1]  # LMTPO, prim

a2 = [
    [1, 2, 3, 4, -2, -1],
    [1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 1, -1, 0, 0, 0],
    [0, 0, 1, -1, 0, 0],
]
b2 = [0, 1, 1, 0, 0]  # LNMTOF, prim

a3 = [
    [1, 2, 3, 4, 5, 0, -2, -1, -1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
]
b3 = [0, 3, 2, 1, 5, 1]  # L M(234) Nb Vac O(-2-1) F, topotactic, prim.

a4 = [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]
b4 = [0, 1, 2]  # Some random charge neutral alloy of 2 sub-lattices, prim.


all_ab = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]


@pytest.fixture(scope="module", params=all_ab)
def a(request):
    # The matrix A in snf.
    return np.array(request.param[0], dtype=int)


@pytest.fixture(scope="module", params=all_ab)
def ab(request):
    # The matrix A in snf.
    return (
        np.array(request.param[0], dtype=int),
        np.array(request.param[1], dtype=int),
    )


def test_rational():
    for _ in range(500):
        rand_num, rand_den = np.random.randint(low=-1000, high=1000, size=2).tolist()
        g = np.gcd(rand_num, rand_den)
        rand_num = rand_num // g
        rand_den = rand_den // g
        if rand_den < 0:
            rand_den = -rand_den
        if rand_den == 0:
            rand_den = 1
        perturbation = np.random.rand() * NUM_TOL * 0.5
        x = float(rand_num) / rand_den + perturbation
        num, den = rationalize_number(x, max_denominator=1000, dtol=NUM_TOL)
        if rand_num != 0:
            assert num == rand_num
            assert den == rand_den

        rand_num, rand_den = 0, np.random.randint(1000)
        if rand_den == 0:
            rand_den = 1
        perturbation = np.random.rand() * NUM_TOL * 0.5
        x = float(rand_num) / rand_den + perturbation
        num, den = rationalize_number(x, max_denominator=1000, dtol=NUM_TOL)
        assert num == 0
        assert den == 1

    # Test bad cases.
    for _ in range(500):
        rand_num, rand_den = np.random.randint(low=-1000, high=1000, size=2).tolist()
        if rand_den < 0:
            rand_den = -rand_den
        if rand_den == 0:
            rand_den = 1
        perturbation = np.random.rand() * NUM_TOL * 10
        x = float(rand_num) / rand_den + perturbation
        f = Fraction.from_float(x)
        f2 = Fraction.from_float(x).limit_denominator(1000)
        num = f.numerator
        den = f.denominator
        num2 = f2.numerator
        den2 = f2.denominator
        if abs(num2 / den2 - num / den) > NUM_TOL:
            with pytest.raises(ValueError):
                _, _ = rationalize_number(x, max_denominator=1000, dtol=NUM_TOL)


def test_integerize():
    for _ in range(100):
        a = np.random.randint(low=-10000, high=10000, size=100)
        aa = a.reshape((10, 10))
        den1, den2 = np.random.randint(1000, size=2).tolist()
        if den1 == 0:
            den1 = 1
        if den2 == 0:
            den2 = 1
        g1 = np.gcd(np.gcd.reduce(a), den1)
        g2 = np.gcd(np.gcd.reduce(a), den2)
        a = a // g1
        den1 = den1 // g1
        aa = aa // g2
        den2 = den2 // g2

        pert = np.random.rand(100) * NUM_TOL * 0.5
        a_t = a / den1 + pert
        aa_t = aa / den2 + pert.reshape((10, 10))

        a_r, den1_r = integerize_vector(a_t, max_denominator=1000, dtol=NUM_TOL)
        aa_r, den2_r = integerize_multiple(aa_t, max_denominator=1000, dtol=NUM_TOL)

        npt.assert_array_equal(a_r, a)
        npt.assert_array_equal(aa_r, aa)
        assert den1 == den1_r
        assert den2 == den2_r

    a = np.zeros(100, dtype=int)
    a_r, den_r = integerize_vector(a, max_denominator=1000, dtol=NUM_TOL)
    npt.assert_array_equal(a_r, a)
    assert den_r == 1


def test_snf(a):
    rank = np.linalg.matrix_rank(a)
    s, m, t = compute_snf(a)
    assert m.shape == a.shape
    npt.assert_array_equal(m, s @ a @ t)
    m_diag = m.diagonal().copy()
    m_diag = m_diag[m_diag != 0]
    assert len(m_diag) == rank
    npt.assert_array_equal(*np.nonzero(m))

    n, d = a.shape
    for _ in range(100):
        x = np.random.randint(low=-10, high=10, size=(d, d - rank))
        npt.assert_array_equal(a @ (t[:, rank:] @ x.T), 0)


def test_snf_specific():
    # Specific test from wikipedia
    a = np.array([[2, 4, 4], [-6, 6, 12], [10, 4, 16]])
    s, m, t = compute_snf(a)
    npt.assert_array_equal(m.diagonal(), [2, 2, 156])
    npt.assert_array_equal(m, s @ a @ t)

    # Zeros
    a = np.zeros((50, 100), dtype=int)
    s, m, t = compute_snf(a)
    npt.assert_array_equal(m, 0)
    assert_table_set_equal(s, np.eye(50, dtype=int))
    assert_table_set_equal(t, np.eye(100, dtype=int))


def test_snf_rand():
    for _ in range(10):
        a = np.random.randint(low=-8, high=8, size=(10, 20))
        rank = np.linalg.matrix_rank(a)
        s, m, t = compute_snf(a)
        assert m.shape == a.shape
        npt.assert_array_equal(m, s @ a @ t)
        m_diag = m.diagonal().copy()
        m_diag = m_diag[m_diag != 0]
        assert len(m_diag) == rank
        npt.assert_array_equal(*np.nonzero(m))

        n, d = a.shape
        for _ in range(10):
            x = np.random.randint(low=-100, high=100, size=(d, d - rank))
            npt.assert_array_equal(a @ (t[:, rank:] @ x.T), 0)


# This can cause some issues when ran on GitHub tests. Not fully clear what happened.
def test_solve_diop_rand():
    fail_counts = 0
    for _ in range(500):
        a = np.random.randint(low=-8, high=8, size=(10, 20))
        rank = np.linalg.matrix_rank(a)
        m, d = a.shape
        nn = np.random.randint(low=0, high=10, size=20)
        b = a @ nn
        # TODO: maybe find what causes numerical issues someday.
        try:  # Overwrite numerical issues.
            n0, vs = solve_diophantines(a, b)
            assert vs.shape == (d - rank, d)
            x = np.random.randint(low=-100, high=100, size=(100, d - rank))
            ns = x @ vs + n0
            npt.assert_array_equal(a @ ns.T - b[:, None], 0)
        except ValueError:
            fail_counts += 1
    assert fail_counts < 500


def test_solve_diop_specific(ab):
    a, b = ab
    rank = np.linalg.matrix_rank(a)
    m, d = a.shape
    n0, vs = solve_diophantines(a, b)
    assert vs.shape == (d - rank, d)
    x = np.random.randint(low=-100, high=100, size=(100, d - rank))
    ns = x @ vs + n0
    npt.assert_array_equal(a @ ns.T - b[:, None], 0)


def test_float_verts_specific(ab):
    # Can not do fully random test because of some weird
    # Numerical issues.
    a, b = ab
    verts = get_nonneg_float_vertices(a, b)
    n, d = a.shape
    assert verts.shape[1] == d
    assert np.all(np.any(np.isclose(verts, 0, atol=NUM_TOL), axis=-1))
    npt.assert_almost_equal(a @ verts.T - b[:, None], 0, decimal=6)
    assert np.all(verts >= 0 - NUM_TOL)
    # Also checked numbers, they are reasonable.


def test_centroid_specific(ab):
    # Can not do fully random test because random polytopes might not
    # be full dimensional, neither might it be bounded.
    a, b = ab
    b = b * 10

    n0, vs = solve_diophantines(a, b)
    x_cent = get_natural_centroid(n0, vs, 10)
    n_cent = x_cent @ vs + n0
    npt.assert_array_equal(a @ n_cent, b)
    assert np.all(n_cent >= 0)
    # Can not test for optimality.


def test_centroid_conditioned():
    a = np.array([[1, 3, 4, -3, -2], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]])
    b = np.array([0, 1, 1]) * 12  # LMTPO, sc_size = 12

    a_leq = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]])
    b_leq = np.array([5 / 6, 5 / 6])

    a_geq = np.array([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]])
    b_geq = np.array([1 / 6, 1 / 6])

    n0, vs = solve_diophantines(a, b)
    poly = pc.Polytope(-vs.transpose(), n0)
    centroid = np.average(pc.extreme(poly), axis=0)
    x_cent = get_natural_centroid(n0, vs, 12)
    x_cent2 = get_natural_centroid(n0, vs, 12, a_leq, b_leq, a_geq, b_geq)
    assert np.isclose(
        np.linalg.norm(x_cent - centroid), np.linalg.norm(x_cent2 - centroid)
    )
    # Will allow some degeneracy.

    # When setting 4 / 6 in ECOS_BB, this test can not work. In guorbi 4/6 works.
    # Very strange indeed.
    b_geq2 = np.array([1 / 6, 3.5 / 6])  # In per prim unit.
    x_cent3 = get_natural_centroid(n0, vs, 12, a_leq, b_leq, a_geq, b_geq2)
    assert np.linalg.norm(x_cent - centroid) < np.linalg.norm(x_cent3 - centroid)


def test_one_dim_solution():
    for _ in range(10):
        n = np.random.randint(low=0, high=1000, size=20)
        v = np.random.randint(low=-1000, high=1000, size=20)
        x0 = np.random.randint(low=-10, high=10)
        n0 = n - x0 * v
        # Now this is going to have at least one solution.

        xs = get_one_dim_solutions(n0, v)
        assert len(xs) > 0
        assert np.all(np.outer(xs, v) + n0 >= 0)
        npt.assert_array_equal(xs, np.arange(min(xs), max(xs) + 1, dtype=int))
        # Check that the bound is tightly filled.
        xs_oob = np.array([min(xs) - 1, max(xs) + 1], dtype=int)
        assert np.all(np.any(np.outer(xs_oob, v) + n0 < 0, axis=-1))

    # Test 3 bad cases with no solution.
    # Case 1: when bounds create an empty set.
    v = np.array([-2, 1, -1, 1, -1, 1])
    n0 = np.array([1, 1, 0, 2, 0, -1])
    xs = get_one_dim_solutions(n0, v)
    assert len(xs) == 0
    assert len(xs.shape) == 1
    # Case 2: actually a special case 1, when a v=0 while n0<0.
    n0 = np.array([1, -1, 0])
    v = np.array([0, 1, -5])
    xs = get_one_dim_solutions(n0, v)
    assert len(xs) == 0
    # Counter example of case 2, and unbounded.
    n0 = np.array([1, -1, 0])
    v = np.array([0, 1, 5])
    with pytest.raises(ValueError):
        _ = get_one_dim_solutions(n0, v)

    # Case 3: when bounds do not include an integer.
    v = np.array([1, 1, -2, 3])
    n0 = np.array([1, 2, 1, -1])
    xs = get_one_dim_solutions(n0, v)
    assert len(xs) == 0


def test_natural_solutions_specific():
    # Since random matrices are usually not bounded, we will not test random.
    # Test specific cases.
    a = np.array([[1, 3, 4, -3, -2], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]])
    b = np.array([0, 6, 6])
    n0, vs = solve_diophantines(a, b)
    xs = get_natural_solutions(n0, vs)
    ns = xs @ vs + n0
    ns = np.array(sorted(ns.tolist()), dtype=int)
    # Manually obtained solutions.
    ns_std = np.array(
        [
            [4, 0, 2, 0, 6],
            [3, 3, 0, 0, 6],
            [3, 2, 1, 1, 5],
            [3, 1, 2, 2, 4],
            [2, 4, 0, 2, 4],
            [3, 0, 3, 3, 3],
            [2, 3, 1, 3, 3],
            [2, 2, 2, 4, 2],
            [1, 5, 0, 4, 2],
            [2, 1, 3, 5, 1],
            [1, 4, 1, 5, 1],
            [2, 0, 4, 6, 0],
            [1, 3, 2, 6, 0],
            [0, 6, 0, 6, 0],
        ],
        dtype=int,
    )
    ns_std = np.array(sorted(ns_std.tolist()), dtype=int)
    npt.assert_array_equal(ns, ns_std)

    a = np.array([[1, 3, 4, -3, -2], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]])
    b = np.array([0, 6, 6]) * 2
    n0, vs = solve_diophantines(a, b)
    xs = get_natural_solutions(n0, vs, step=2)  # Test a step.
    ns = xs @ vs + n0
    ns = np.array(sorted(ns.tolist()), dtype=int)
    # Scaling test.
    ns_std = (
        np.array(
            [
                [4, 0, 2, 0, 6],
                [3, 3, 0, 0, 6],
                [3, 2, 1, 1, 5],
                [3, 1, 2, 2, 4],
                [2, 4, 0, 2, 4],
                [3, 0, 3, 3, 3],
                [2, 3, 1, 3, 3],
                [2, 2, 2, 4, 2],
                [1, 5, 0, 4, 2],
                [2, 1, 3, 5, 1],
                [1, 4, 1, 5, 1],
                [2, 0, 4, 6, 0],
                [1, 3, 2, 6, 0],
                [0, 6, 0, 6, 0],
            ],
            dtype=int,
        )
        * 2
    )
    ns_std = np.array(sorted(ns_std.tolist()), dtype=int)
    npt.assert_array_equal(ns, ns_std)

    a = np.array([[1, 3, 4, -2, -1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]])
    b = np.array([0, 6, 6])
    n0, vs = solve_diophantines(a, b)
    xs = get_natural_solutions(n0, vs)
    ns = xs @ vs + n0
    ns = np.array(sorted(ns.tolist()), dtype=int)
    # Manually obtained solutions.
    ns_std = np.array(
        [
            [4, 0, 2, 6, 0],
            [3, 3, 0, 6, 0],
            [4, 1, 1, 5, 1],
            [4, 2, 0, 4, 2],
            [5, 0, 1, 3, 3],
            [5, 1, 0, 2, 4],
            [6, 0, 0, 0, 6],
        ],
        dtype=int,
    )
    ns_std = np.array(sorted(ns_std.tolist()), dtype=int)
    npt.assert_array_equal(ns, ns_std)

    a = np.array(
        [
            [1, 2, 3, 4, -2, -1],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 1, -1, 0, 0, 0],
            [0, 0, 1, -1, 0, 0],
        ]
    )
    b = np.array([0, 12, 12, 0, 0])
    n0, vs = solve_diophantines(a, b)
    xs = get_natural_solutions(n0, vs)
    ns = xs @ vs + n0
    ns = np.array(sorted(ns.tolist()), dtype=int)
    # Manually obtained solutions.
    ns_std = np.array(
        [[12, 0, 0, 0, 0, 12], [9, 1, 1, 1, 6, 6], [6, 2, 2, 2, 12, 0]], dtype=int
    )
    ns_std = np.array(sorted(ns_std.tolist()), dtype=int)
    npt.assert_array_equal(ns, ns_std)

    a = np.array(
        [
            [1, 2, 3, 4, 5, 0, -2, -1, -1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    b = np.array([0, 3, 2, 1, 5, 1])
    n0, vs = solve_diophantines(a, b)
    xs = get_natural_solutions(n0, vs)
    ns = xs @ vs + n0
    ns = np.array(sorted(ns.tolist()), dtype=int)
    # Manually obtained solutions.
    ns_std = np.array(
        [
            [2, 2, 0, 0, 1, 1, 5, 0, 1],
            [1, 1, 1, 0, 1, 2, 5, 0, 1],
            [1, 2, 0, 0, 1, 2, 4, 1, 1],
            [0, 0, 2, 0, 1, 3, 5, 0, 1],
            [0, 1, 0, 1, 1, 3, 5, 0, 1],
            [0, 1, 1, 0, 1, 3, 4, 1, 1],
            [0, 2, 0, 0, 1, 3, 3, 2, 1],
        ],
        dtype=int,
    )
    ns_std = np.array(sorted(ns_std.tolist()), dtype=int)
    npt.assert_array_equal(ns, ns_std)

    a = np.array(
        [
            [1, 2, 3, 4, 5, 0, -2, -1, -1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    b = np.array([0, 3, 2, 1, 5, 1]) * 3
    n0, vs = solve_diophantines(a, b)
    xs = get_natural_solutions(n0, vs, step=3)
    ns = xs @ vs + n0
    ns = np.array(sorted(ns.tolist()), dtype=int)
    # Scalability test.
    ns_std = (
        np.array(
            [
                [2, 2, 0, 0, 1, 1, 5, 0, 1],
                [1, 1, 1, 0, 1, 2, 5, 0, 1],
                [1, 2, 0, 0, 1, 2, 4, 1, 1],
                [0, 0, 2, 0, 1, 3, 5, 0, 1],
                [0, 1, 0, 1, 1, 3, 5, 0, 1],
                [0, 1, 1, 0, 1, 3, 4, 1, 1],
                [0, 2, 0, 0, 1, 3, 3, 2, 1],
            ],
            dtype=int,
        )
        * 3
    )
    ns_std = np.array(sorted(ns_std.tolist()), dtype=int)
    npt.assert_array_equal(ns, ns_std)

    a = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    b = np.array([0, 1])
    n0, vs = solve_diophantines(a, b)
    xs = get_natural_solutions(n0, vs)
    ns = xs @ vs + n0
    ns = np.array(sorted(ns.tolist()), dtype=int)
    # Manually obtained solutions.
    ns_std = np.eye(5, dtype=int)
    ns_std = np.array(sorted(ns_std.tolist()), dtype=int)
    npt.assert_array_equal(ns, ns_std)


def test_flip_size():
    # Test bad flips.
    for _ in range(100):
        u = np.random.randint(low=-100, high=100, size=50)
        if u.sum() != 0:
            with pytest.raises(ValueError):
                _ = flip_size(u)
        else:
            assert flip_size(u) == u[u > 0].sum()


def test_row_matches():
    for _ in range(10):
        a = np.random.randint(low=-100, high=100, size=(100, 100))
        a_clean = np.array(list({tuple(r) for r in a}))
        n = len(a_clean)
        aa1 = a[
            np.random.choice(
                np.arange(2 * n // 3, dtype=int), size=2 * n // 3, replace=False
            )
        ]
        aa2 = a[
            np.random.choice(
                np.arange(n // 3, n, dtype=int), size=n - n // 3, replace=False
            )
        ]
        assert count_row_matches(aa1, aa2) == n // 3


def test_connectivity():
    # Pretty dumb pre-computed test.
    ns = np.array(
        [
            [4, 0, 2, 0, 6],
            [3, 3, 0, 0, 6],
            [3, 2, 1, 1, 5],
            [3, 1, 2, 2, 4],
            [2, 4, 0, 2, 4],
            [3, 0, 3, 3, 3],
            [2, 3, 1, 3, 3],
            [2, 2, 2, 4, 2],
            [1, 5, 0, 4, 2],
            [2, 1, 3, 5, 1],
            [1, 4, 1, 5, 1],
            [2, 0, 4, 6, 0],
            [1, 3, 2, 6, 0],
            [0, 6, 0, 6, 0],
        ],
        dtype=int,
    )
    u1 = np.array([-1, 2, -1, 1, -1], dtype=int)
    assert connectivity(u1, ns) == 8
    assert connectivity(-u1, ns) == 8

    u2 = np.array([0, -1, 1, 1, -1])
    assert connectivity(u2, ns) == 9
    assert connectivity(u2, ns) == 9

    assert connectivity(np.zeros(5, dtype=int), ns) == len(ns)
    assert connectivity([4, -6, 2, -6, 6], ns) == 1


def test_optimal_basis_specific():
    # Random matrices may not be bounded. Do specific tests only.
    # Do 3 pre-computed tests.
    # Test 1: LMTPO, loop also checked. Optimization went well.
    a = np.array([[1, 3, 4, -3, -2], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]])
    b = np.array([0, 6, 6])
    n0, vs = solve_diophantines(a, b)
    xs = get_natural_solutions(n0, vs)
    vs_opt = get_optimal_basis(n0, vs, xs)
    vs_std = np.array([[0, -1, 1, 1, -1], [-1, 1, 0, 2, -2]])
    assert_table_set_equal(vs_opt, vs_std)

    # Test 2: LNMTOF
    a = np.array(
        [
            [1, 2, 3, 4, -2, -1],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 1, -1, 0, 0, 0],
            [0, 0, 1, -1, 0, 0],
        ]
    )
    b = np.array([0, 1, 1, 0, 0])
    n0, vs = solve_diophantines(a, b * 6)
    xs = get_natural_solutions(n0, vs)
    vs_opt = get_optimal_basis(n0, vs, xs)
    vs_std = np.array([[-3, 1, 1, 1, 6, -6]])

    assert_table_set_equal(vs_opt, vs_std)

    # Test 3: Random Alloy
    a = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]])
    b = np.array([0, 1, 1])
    n0, vs = solve_diophantines(a, b * 1)
    xs = get_natural_solutions(n0, vs)
    vs_opt = get_optimal_basis(n0, vs, xs)
    vs_std = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]])

    assert_table_set_equal(vs_opt, vs_std)


def test_ergodic_vectors_specific():
    # Random matrices may not be bounded. Do specific tests only.
    # Do 1 pre-computed test.
    # Test 1: LMTOF
    # Pre-computed test
    a = np.array([[1, 3, 4, -2, -1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]])
    b = np.array([0, 6, 6])
    n0, vs = solve_diophantines(a, b)
    xs = get_natural_solutions(n0, vs)
    vs_opt = get_optimal_basis(n0, vs, xs)
    len(vs)
    xs_opt = get_natural_solutions(n0, vs_opt)
    vs_tab = get_ergodic_vectors(n0, vs_opt, xs_opt)
    vs_std = np.array([[0, -1, 1, 1, -1], [-1, 1, 0, 2, -2]])
    assert_table_set_equal(vs_tab, vs_std)

    # A non-ergodic table made ergodic.
    n02 = np.array([6, 0, 0, 0, 6], dtype=int)
    vs2 = np.array([[0, -1, 1, 1, -1], [-1, 2, -1, 1, -1]], dtype=int)
    xs2 = np.array([[0, 0], [1, 1], [2, 1], [2, 2], [3, 2], [3, 3], [4, 2]], dtype=int)
    assert np.all(xs2 @ vs2 + n02 >= 0)
    vs_tab2 = get_ergodic_vectors(n02, vs2, xs2)
    vs_std2 = np.array([[0, -1, 1, 1, -1], [-1, 2, -1, 1, -1], [-1, 1, 0, 2, -2]])
    assert_table_set_equal(vs_tab2, vs_std2)


def test_mask():
    for _ in range(10):
        vs = np.random.randint(low=-100, high=100, size=(30, 50))
        table = np.concatenate([(u, -u) for u in vs], axis=0)
        n = np.random.randint(low=0, high=100, size=50)
        aa = np.random.randint(low=-50, high=50, size=50)
        aa @ n
        max_n = np.random.randint(low=100, high=200, size=50)
        mask = flip_weights_mask(vs, n)
        assert len(mask) == len(table)
        assert np.all(np.any(table[~mask, :] + n < 0, axis=-1))
        assert np.all(np.all(table[mask, :] + n >= 0, axis=-1))

        mask = flip_weights_mask(vs, n, max_n=max_n)
        assert np.all(
            np.any(table[~mask, :] + n < 0, axis=-1)
            | np.any(table[~mask, :] + n > max_n, axis=-1)
        )
        assert np.all(
            np.all(table[mask, :] + n >= 0, axis=-1)
            & np.any(table[mask, :] + n <= max_n, axis=-1)
        )

        mask = flip_weights_mask(vs, n, max_n=300)
        assert np.all(np.any(table[~mask, :] + n < 0, axis=-1))
        assert np.all(np.all(table[mask, :] + n >= 0, axis=-1))


def test_choose_sections():
    counts = Counter()
    p = [0.1, 0, 0.3, 0.2, 0.1, 0.3]
    for _ in range(10000):
        counts[choose_section_from_partition(p)] += 1

    for i in range(len(p)):
        # assert abs(counts[i] / 10000 - p[i]) <= 0.05
        # May be this is not required as long as you trust
        # np.random
        if i == 1:
            assert counts[i] <= 1


@pytest.mark.parametrize("determinant", range(6, 13))
def test_yield_hermite_normal_forms(determinant):
    hnfs = [hnf for hnf in yield_hermite_normal_forms(determinant)]
    for hnf in hnfs:
        assert np.linalg.det(hnf) == pytest.approx(determinant)

    # make sure that all the HNFs are unique
    assert len(np.unique(hnfs, axis=0)) == len(hnfs)
