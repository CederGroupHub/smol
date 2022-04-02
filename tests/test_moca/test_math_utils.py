"""Test smol.moca.utils.math_utils."""
import pytest

from smol.moca.utils.math_utils import *
import numpy as np
import numpy.testing as npt

from collections import Counter


def test_gcd():
    for _ in range(10):
        a = np.random.randint(low=-10000, high=10000, size=100)
        gcd = gcd_list(a)
        a_p = a // gcd
        npt.assert_array_equal(np.round(a % gcd), 0)
        assert gcd_list(a_p) == 1
        a_0 = np.append(a, 0)
        gcd_0 = gcd_list(a_0)
        assert gcd_0 == gcd


def test_comb():
    # Just trivial running tests.
    for _ in range(10):
        a, b = np.random.randint(100, size=2).tolist()
        c = comb(a, b)
        if a < b:
            assert c == 0
        if a == b:
            assert c == 1


def test_rational():
    for _ in range(10):
        rand_num, rand_den = np.random.randint(low=-1000,
                                               high=1000,
                                               size=2).tolist()
        if rand_den < 0:
            rand_den = - rand_den
        perturbation = np.random.rand() * NUM_TOL * 0.5
        x = float(rand_num) / rand_den + perturbation
        num, den = rationalize_number(x, max_denominator=1000, dtol=NUM_TOL)
        if rand_num != 0:
            assert num == rand_num
            assert den == rand_den

        rand_num, rand_den = 0, np.random.randint(1000)
        perturbation = np.random.rand() * NUM_TOL * 0.5
        x = float(rand_num) / rand_den + perturbation
        num, den = rationalize_number(x, max_denominator=1000, dtol=NUM_TOL)
        assert num == 0
        assert den == 1

    # Test a bad case
    rand_num, rand_den = np.random.randint(low=-1000,
                                           high=1000,
                                           size=2).tolist()
    if rand_den < 0:
        rand_den = - rand_den
    perturbation = NUM_TOL * 5
    x = float(rand_num) / rand_den + perturbation
    with pytest.raises(ValueError):
       _, _ = rationalize_number(x, max_denominator=1000, dtol=NUM_TOL)


def test_integerize():
    for _ in range(10):
        a = np.random.randint(low=-10000,
                              high=10000,
                              size=100)
        aa = a.reshape((10, 10))
        den1, den2 = np.random.randint(1000, size=2).tolist()

        pert = np.random.rand(100) * NUM_TOL * 0.5
        a_t = a / den1 + pert
        aa_t = aa / den2 + pert.reshape((10, 10))

        a_r, den1_r = integerize_vector(a_t, max_denominator=1000,
                                        dtol=NUM_TOL)
        aa_r, den2_r = integerize_multiple(aa_t, max_denominator=1000,
                                           dtol=NUM_TOL)

        npt.assert_array_equal(a_r, a)
        npt.assert_array_equal(aa_r, aa)
        assert den1 == den1_r
        assert den2 == den2_r

    a = np.zeros(100, dtype=int)
    a_r, den_r = integerize_vector(a, max_denominator=1000,
                                   dtol=NUM_TOL)
    npt.assert_array_equal(a_r, a)
    assert den_r == 1


def test_snf():
    for _ in range(10):
        a = np.random.randint(low=-1000, high=1000,
                              size=(15, 20))
        rank = np.linalg.matrix_rank(a)
        s, m, t = compute_snf(a)
        assert m.shape == (15, 20)
        npt.assert_array_equal(m, s @ a @ t)
        m_diag = m.diagonal().copy()
        m_diag = m_diag[m_diag != 0]
        assert len(m_diag) == rank
        npt.assert_array_equal(m_diag, m.diagonal()[:rank])
        npt.assert_array_equal(m_diag, np.sort(m_diag))
        npt.assert_array_equal(*np.nonzero(m))

        x = np.random.randint(low=-100, high=100,
                              size=(100, 20 - rank))
        npt.assert_array_equal(a @ (t[:, rank:] @ x.T), 0)

    # Specific test from wikipedia
    a = np.array([[2, 4, 4], [-6, 6, 12], [10, 4, 16]])
    s, m, t = compute_snf(a)
    npt.assert_array_equal(m.diagonal(), [2, 2, 156])


def test_solve_diop():
    for _ in range(10):
        a = np.random.randint(low=-1000, high=1000,
                              size=(15, 20))
        nn = np.random.randint(low=-1000, high=1000,
                               size=20)
        rank = np.linalg.matrix_rank(a)
        b = a @ nn
        n0, vs = solve_diophantines(a, b)
        assert vs.shape == (20 - rank, 20)
        x = np.random.randint(low=-100, high=100,
                              size=(100, 20 - rank))
        ns = x @ vs + n0
        npt.assert_array_equal(a @ ns.T - b[:, None], 0)


def test_float_verts():
    for _ in range(10):
        a = np.random.randint(low=-1000, high=1000,
                              size=(10, 15))
        nn = np.random.randint(low=-1000, high=1000,
                               size=15)
        b = a @ nn
        verts = get_nonneg_float_vertices(a, b)
        assert verts.shape[1] == 15
        assert np.all(verts >= 0 - NUM_TOL)
        npt.assert_almost_equal(a @ verts.T - b[:, None], 0,
                                decimal=6)
        assert np.all(np.any(np.isclose(verts, 0, atol=NUM_TOL),
                             axis=-1))


def test_centroid():
    for _ in range(10):
        a = np.random.randint(low=-1000, high=1000,
                              size=(10, 15))
        nn = np.random.randint(low=-1000, high=1000,
                               size=15)
        b = a @ nn
        n0, vs = solve_diophantines(a, b)
        x_cent = get_natural_centroid(n0, vs)
        n_cent = vs @ x_cent + n0
        npt.assert_array_equal(a @ n_cent, b)
        # Can not test for optimality.


def test_one_dim_solution():
    for _ in range(10):
        n = np.random.randint(low=0, high=1000,
                              size=20)
        v = np.random.randint(low=-1000, high=1000,
                              size=20)
        x0 = np.random.randint(low=-10, high=10)
        n0 = n - x0 * v
        # Now this is going to have at least one solution.

        xs = get_one_dim_solutions(n0, v)
        assert len(xs) > 0
        assert np.all(np.outer(xs, v) + n0 >= 0)
        npt.assert_array_equal(xs,
                               np.arange(min(xs), max(xs) + 1,
                                         dtype=int))
        # Check that the bound is tightly filled.
        xs_oob = np.array([min(xs) - 1, max(xs) + 1], dtype=int)
        assert np.all(np.any(np.outer(xs_oob, v) + n0 < 0, axis=-1))

    # Test 3 cases with no solution.
    # Case 1: when bounds create an empty set.
    n0 = np.array([-2, 1, -1, 1, -1, 1])
    v = np.array([1, 1, 0, 2, 0, -1])
    xs = get_natural_solutions(n0, v)
    assert len(xs) == 0
    assert len(xs.shape) == 1
    # Case 2: actually a special case 1, when a v=0 while n0<0.
    n0 = np.array([1, -1, 0])
    v = np.array([0, 1, -5])
    xs = get_natural_solutions(n0, v)
    assert len(xs) == 0
    # Conter example of case 2.
    n0 = np.array([1, -1, 0])
    v = np.array([0, 1, 5])
    xs = get_natural_solutions(n0, v)
    npt.assert_array_equal(xs, [0, 1])
    # Case 3: when bounds do not include an integer.
    n0 = np.array([1, 1, -2, 3])
    v = np.array([1, 2, 1, -1])
    xs = get_natural_solutions(n0, v)
    assert len(xs) == 0


def test_natural_solutions():
    for _ in range(10):
        # So this will always have natural number solutions.
        # Control test size.
        a = np.random.randint(low=-8, high=8,
                              size=(3, 5))
        nn = np.random.randint(low=0, high=8,
                               size=5)
        b = a @ nn
        n0, vs = solve_diophantines(a, b)

        xs = get_natural_solutions(n0, vs)
        ns = xs @ vs + n0
        npt.assert_array_equal(a @ ns.T - b[:, None], 0)
        assert np.all(ns >= 0)

        # Test bounds are tight.
        for j in range(xs.shape[1]):
            i_max = np.argmax(xs[:, j])
            i_min = np.argmin(xs[:, j])
            x_max = xs[i_max].copy()
            x_min = xs[i_min].copy()
            x_max[j] += 1
            x_min[j] -= 1
            xs_oos = np.array((x_max, x_min))
            assert np.all(np.any(xs_oos @ vs + n0 < 0, axis=-1))

    # Test a specific case.
    a = np.array([[1, 3, 4, -3, -2],
                  [1, 1, 1, 0, 0],
                  [0, 0, 0, 1, 1]])
    b = np.array([0, 6, 6])
    n0, vs = solve_diophantines(a, b)
    xs = get_natural_solutions(n0, vs)
    ns = xs @ vs + n0
    ns = np.array(sorted(ns.tolist()), dtype=int)
    # Manually obtained solutions.
    ns_std = np.array([[4, 0, 2, 0, 6],
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
                       [0, 6, 0, 6, 0]], dtype=int)
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
        a_clean = np.array(list(set([tuple(r) for r in a])))
        n = len(a_clean)
        a1 = a[np.random.choice(np.arange(2 * n // 3, dtype=int),
                                size=2 * n // 3, replace=False)]
        a2 = a[np.random.choice(np.arange(n // 3, n, dtype=int),
                                size=n - n // 3, replace=False)]
        assert count_row_matches(a1, a2) == n // 3


def test_connectivity():
    # Pretty dumb pre-computed test.
    ns = np.array([[4, 0, 2, 0, 6],
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
                   [0, 6, 0, 6, 0]], dtype=int)
    u1 = np.array([-1, 2, -1, -1, 1], dtype=int)
    assert connectivity(u1, ns) == 8
    assert connectivity(-u1, ns) == 8

    u2 = np.array([0, -1, 1, 1, -1])
    assert connectivity(u2, ns) == 9
    assert connectivity(u2, ns) == 9

    assert connectivity(np.zeros(5, dtype=int), ns) == len(ns)
    assert connectivity([4, -6, 2, -6, 6], ns) == 1


def test_optimal_basis():
    for _ in range(10):
        a = np.random.randint(low=-8, high=8,
                              size=(3, 5))
        nn = np.random.randint(low=0, high=8,
                               size=5)
        b = a @ nn
        n0, vs = solve_diophantines(a, b)
        xs = get_natural_solutions(n0, vs)
        ns = xs @ vs + n0
        vs_opt = get_optimal_basis(n0, vs, xs)
        assert len(vs_opt) == len(vs)
        assert np.linalg.matrix_rank(vs_opt) == len(vs)
        npt.assert_array_equal(a @ (vs_opt + n0).T - b[:, None], 0)
        # Test if really been optimized.
        sizes_ori = sorted([flip_size(v) for v in vs])
        sizes_opt = sorted([flip_size(v) for v in vs_opt])
        assert np.all(sizes_ori >= sizes_opt)
        conn_ori = sorted([connectivity(v, ns) for v in vs])
        conn_opt = sorted([connectivity(v, ns) for v in vs_opt])
        assert np.all(conn_ori <= conn_opt)

    # Do a pre-computed test.
    a = np.array([[1, 3, 4, -3, -2],
                  [1, 1, 1, 0, 0],
                  [0, 0, 0, 1, 1]])
    b = np.array([0, 6, 6])
    n0, vs = solve_diophantines(a, b)
    xs = get_natural_solutions(n0, vs)
    vs_opt = get_optimal_basis(n0, vs, xs)
    vs_std = np.array([[0, -1, 1, 1, -1],
                       [-1, 1, 0, 2, -2]])
    table_opt = np.concatenate([vs_opt, -vs_opt], axis=0)
    table_opt = np.array(sorted(table_opt.tolist()), dtype=int)
    table_std = np.concatenate([vs_std, -vs_std], axis=0)
    table_std = np.array(sorted(table_std.tolist()), dtype=int)

    npt.assert_array_equal(table_opt, table_std)


def test_ergodic_vectors():
    for _ in range(10):
        a = np.random.randint(low=-8, high=8,
                              size=(3, 5))
        nn = np.random.randint(low=0, high=8,
                               size=5)
        b = a @ nn
        n0, vs = solve_diophantines(a, b)
        xs = get_natural_solutions(n0, vs)
        ns = xs @ vs + n0
        vs_tab = get_ergodic_vectors(n0, vs, xs)
        assert np.all(is_connected(n, vs_tab, ns) for n in ns)

    # Pre-computed test
    a = np.array([[1, 3, 4, -2, -1],
                  [1, 1, 1, 0, 0],
                  [0, 0, 0, 1, 1]])
    b = np.array([0, 6, 6])
    n0, vs = solve_diophantines(a, b)
    xs = get_natural_solutions(n0, vs)
    vs_opt = get_optimal_basis(n0, vs, xs)
    d = len(vs)
    xs_opt = (np.linalg.inv(vs_opt[:, :d]) @ vs[:, :d] @ xs.T).T
    vs_tab = get_ergodic_vectors(n0, vs_opt, xs_opt)
    vs_std = np.array([[0, 1, -1, 1, -1],
                       [-1, 1, 0, 2, -2]])
    table_tab = np.concatenate([vs_tab, -vs_tab], axis=0)
    table_tab = np.array(sorted(table_tab.tolist()), dtype=int)
    table_std = np.concatenate([vs_std, -vs_std], axis=0)
    table_std = np.array(sorted(table_std.tolist()), dtype=int)

    npt.assert_array_equal(table_tab, table_std)


def test_mask():
    for _ in range(10):
        vs = np.random.randint(low=-100, high=100, size=(30, 50))
        table = np.concatenate([(u, -u) for u in vs], axis=0)
        n = np.random.randint(low=0, high=100, size=50)
        max_n = np.random.randint(low=100, high=200, size=50)
        mask = flip_weights_mask(vs, n)
        assert len(mask) == len(table)
        assert np.all(np.any(table[~mask, :] + n < 0, axis=-1))
        assert np.all(np.all(table[mask, :] + n >= 0, axis=-1))

        mask = flip_weights_mask(vs, n, max_n=max_n)
        assert np.all(np.any(table[~mask, :] + n < 0, axis=-1)
                      | np.any(table[~mask, :] + n > max_n, axis=-1))
        assert np.all(np.all(table[mask, :] + n >= 0, axis=-1)
                      & np.any(table[~mask, :] + n <= max_n, axis=-1))

        mask = flip_weights_mask(vs, n, max_n=280)
        assert np.all(np.any(table[~mask, :] + n < 0, axis=-1)
                      | np.any(table[~mask, :] + n > max_n, axis=-1))
        assert np.all(np.all(table[mask, :] + n >= 0, axis=-1)
                      & np.any(table[~mask, :] + n <= max_n, axis=-1))


def test_choose_sections():
    counts = Counter()
    p = [0.1, 0, 0.3, 0.2, 0.1, 0.3]
    for _ in range(10000):
        counts[choose_section_from_partition(p)] += 1

    for i in range(len(p)):
        assert abs(counts[i] / 10000 - p[i]) <= 0.05
        if i == 1:
            assert counts[i] <= 1
