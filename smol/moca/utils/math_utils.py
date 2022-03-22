"""Mathematic utilities.

Including linear algebra, combinatorics and integer enumerations.
"""

__author___ = 'Fengyu Xie'

import math
import numpy as np
from functools import reduce
from itertools import combinations
from fractions import Fraction

import cvxpy as cp
import polytope as pc
from sympy import Matrix, eye, ZZ
from scipy.spatial import KDTree


# Global numerical tolerance in this module.
NUM_TOL = 1e-6


class DiopInfeasibleError(Exception):
    """Diophantine equations have no integer solution."""

    def __init__(self):
        message = "Diophantine equations have no integer solution!"
        super(DiopInfeasibleError, self).__init__(message)


class NaturalInfeasibleError(Exception):
    """Diophantine equations have no natural number solution."""

    def __init__(self):
        message = "Diophantine equations have no natural number solution!"
        super(NaturalInfeasibleError, self).__init__(message)


def GCD(a, b):
    """Euclidean Algorithm, giving positive GCD's."""
    return math.gcd(a, b)


def GCD_list(L):
    """Find GCD of a list of numbers."""
    if len(L) < 1:
        return None
    elif len(L) == 1:
        return L[0]
    else:
        return reduce(lambda a, b: GCD(a, b), L)


def LCM(a, b):
    """Find LCM of two numbers."""
    if a == 0 and b == 0:
        return 0
    elif a == 0 and b != 0:
        return b
    elif a != 0 and b == 0:
        return a
    else:
        return a * b // GCD(a, b)


def LCM_list(L):
    """Find LCM of a list of numbers."""
    if len(L) < 1:
        return None
    elif len(L) == 1:
        return L[0]
    else:
        return reduce(LCM, L)


# Combinatoric and intergerization
def comb(n, k):
    """Choose k from n.

    Written to be compatible with lower python versions.
    If python >= 3.8, will use math.comb.
    Otherwise, will be computed with math.factorial.
    """
    if hasattr(math, "comb"):
        return math.comb(n, k)
    else:
        return (math.factorial(n) //
                (math.factorial(n - k) * math.factorial(k)))


def rationalize_number(a, max_denominator=100, dtol=1E-5):
    """Find a rational number near real number within dtol.

    Args:
        a(float):
            A number to be rationalized
        max_denominator(int,default=100):
            Maximum allowed denominator.
        dtol(float,default=1E-5):
            Maximum allowed difference of variable a
            to its rational form
    Return:
        Numerator and denominator:
           int, int
    """
    f = Fraction.from_float(a).limit_denominator(max_denominator)
    numerator = f.numerator
    denominator = f.denominator

    if abs(float(numerator) / denominator - a) > dtol:
        raise ValueError("Can't find a rational number near " +
                         "{} within tolerance!".format(a))

    return numerator, denominator


def integerize_vector(v, max_denominator=100, dtol=1E-5):
    """Integerize all components of a vector v.

    Rationalize all components of a vector, then multiply
    the vector by the LCM of all the rationalized component's
    denominator, so that all vector components are converted
    to integers. We call this process 'intergerization' of a
    vector.
    Args:
        v(np.ndarray(float)):
            A vector to be rationalized
        max_denominator(int,default=100):
            Maximum allowed denominator.
        dtol(float,default=1E-5):
            Maximum allowed difference of variable a
            to its rational form
    Return:
        Integrized vector, LCM of denominators:
            np.ndarray, int
    """
    denos = []
    v = np.array(v)
    for c in v:
        _, deno = rationalize_number(
            c, max_denominator=max_denominator,
            dtol=dtol
        )
        denos.append(deno)
    lcm = LCM_list(denos)
    return np.array(np.round(v * lcm), dtype=np.int64), lcm


def integerize_multiple(vs, max_denominator=100, dtol=1E-5):
    """Integerize multiple vectors in a matrix.

    Args:
        vs(np.ndarray(float)):
            A matrix of vectors to be rationalized.
        max_denominator(int,default=100):
            Maximum allowed denominator.
        dtol(float,default=1E-5):
            Maximum allowed difference of variable a
            to its rational form.
    Return:
        Integerized vectors in the input shape, LCM of denominator:
            np.ndarray[int], int
    """
    vs = np.array(vs)
    shp = vs.shape
    vs_flatten = vs.flatten()
    vs_flat_int, mul = integerize_vector(vs_flatten,
                                         max_denominator=
                                         max_denominator,
                                         dtol=dtol)
    vs_int = np.reshape(vs_flat_int, shp)
    return vs_int, mul


# Integer linear algebra utilities.
def compute_snf(A, domain=ZZ):
    """Compute smith normal form of a matrix.

    Args:
        A(2D Arraylike of int):
            A matrix defined on some domain.
        domain(sympy.domain):
            The domain on which matrix m is defined. By default,
            is sympy.ZZ (the integer domain).

    Returns:
        s, m, t, m = s a t is the SNF:
            np.ndarray[int]
    """

    # Matrix pivoting operations.
    def leftmult(m, i0, i1, a, b, c, d):
        for j in range(m.cols):
            x, y = m[i0, j], m[i1, j]
            m[i0, j] = a * x + b * y
            m[i1, j] = c * x + d * y

    def rightmult(m, j0, j1, a, b, c, d):
        for i in range(m.rows):
            x, y = m[i, j0], m[i, j1]
            m[i, j0] = a * x + c * y
            m[i, j1] = b * x + d * y

    # Must convert to a sympy.Matrix.
    m = np.array(A).astype(int).copy()
    m = Matrix(m)
    s = eye(m.rows)
    t = eye(m.cols)
    last_j = -1
    for i in range(m.rows):
        for j in range(last_j + 1, m.cols):
            if not m.col(j).is_zero:
                break
        else:
            break
        if m[i, j] == 0:
            for ii in range(m.rows):
                if m[ii, j] != 0:
                    break
            leftmult(m, i, ii, 0, 1, 1, 0)
            leftmult(s, i, ii, 0, 1, 1, 0)
        rightmult(m, j, i, 0, 1, 1, 0)
        rightmult(t, j, i, 0, 1, 1, 0)
        j = i
        upd = True
        while upd:
            upd = False
            for ii in range(i + 1, m.rows):
                if m[ii, j] == 0:
                    continue
                upd = True
                if domain.rem(m[ii, j], m[i, j]) != 0:
                    coef1, coef2, g = domain.gcdex(int(m[i, j]),
                                                   int(m[ii, j]))
                    coef3 = domain.quo(m[ii, j], g)
                    coef4 = domain.quo(m[i, j], g)
                    leftmult(m, i, ii, coef1, coef2, -coef3, coef4)
                    leftmult(s, i, ii, coef1, coef2, -coef3, coef4)
                coef5 = domain.quo(m[ii, j], m[i, j])
                leftmult(m, i, ii, 1, 0, -coef5, 1)
                leftmult(s, i, ii, 1, 0, -coef5, 1)
            for jj in range(j + 1, m.cols):
                if m[i, jj] == 0:
                    continue
                upd = True
                if domain.rem(m[i, jj], m[i, j]) != 0:
                    coef1, coef2, g = domain.gcdex(int(m[i, j]),
                                                   int(m[i, jj]))
                    coef3 = domain.quo(m[i, jj], g)
                    coef4 = domain.quo(m[i, j], g)
                    rightmult(m, j, jj, coef1, -coef3, coef2, coef4)
                    rightmult(t, j, jj, coef1, -coef3, coef2, coef4)
                coef5 = domain.quo(m[i, jj], m[i, j])
                rightmult(m, j, jj, 1, -coef5, 0, 1)
                rightmult(t, j, jj, 1, -coef5, 0, 1)
        last_j = j

    for i1 in range(min(m.rows, m.cols)):
        for i0 in reversed(range(i1)):
            coef1, coef2, g = domain.gcdex(int(m[i0, i0]),
                                           int(m[i1, i1]))
            if g == 0:
                continue
            coef3 = domain.quo(m[i1, i1], g)
            coef4 = domain.quo(m[i0, i0], g)
            leftmult(m, i0, i1, 1, coef2, coef3, coef2 * coef3 - 1)
            leftmult(s, i0, i1, 1, coef2, coef3, coef2 * coef3 - 1)
            rightmult(m, i0, i1, coef1, 1 - coef1 * coef4, 1, -coef4)
            rightmult(t, i0, i1, coef1, 1 - coef1 * coef4, 1, -coef4)

    s = np.array(s).astype(int)
    m = np.array(m).astype(int)
    t = np.array(t).astype(int)
    return s, m, t


def solve_diophantines(A, b=None):
    """Solve diophantine equations An=b.

    We use Smith normal form to solve equations.
    If equation is not solvable, we will throw an
    error.

    Args:
        A(2D ArrayLike[int]):
            Matrix A in An=b.
        b(1D ArrayLike[int], default=None):
            Vector b in An=b.
            If not given, will set to zeros.

    Return:
        A base solution and base vectors (as rows):
        1D np.ndarray[int], 2D np.ndarray[int]
    """
    A = np.array(A, dtype=int)
    n, d = A.shape
    b = (np.array(b, dtype=int) if b is not None
         else np.zeros(d, dtype=int))
    # If you choose b=0, the equations may not have
    # any natural number solution!

    U, B, V = compute_snf(A)
    c = U @ b
    k = None
    for i in range(min(n, d)):
        if B[i, i] == 0:
            k = i
    k = min(n, d) if k is None else k

    # Check feasibility
    for i in range(k):
        if c[i] % B[i, i] != 0:
            raise DiopInfeasibleError

    # Get base solution
    n0 = V[:, :k] @ (c[: k] // B.diagonal()[: k])

    return n0, V[:, k:].transpose().copy()


def get_nonneg_float_vertices(A, b):
    """Get vertices of polytope An=b, n>=0.

    n is allowed to be non-negative floats.
    This result can be used to get the minimum
    supercell size by looking for the minimum
    integer than can integerize all these vertices,
    if b is written according to number of sites in
    a primitive cell.

    Args:
        A(2D np.ndarray[int]):
            A in An=b.
        b(2D np.ndarray[int]):
            b in An=b.
    Returns:
        Vertices of polytope An=b, n>=0 in float:
            np.ndarray[float]
    """
    A = np.array(A, dtype=int)
    b = np.array(b, dtype=int)

    n0, vs = solve_diophantines(A, b)
    poly = pc.Polytope(-1 * vs.transpose(), n0)

    verts = pc.extreme(poly)
    verts = verts @ vs + n0
    return verts


def get_natural_centroid(n0, vs):
    """Get the natural number solution closest to centroid.

    Done by linear programming, minimize:
    norm_2(x - centroid)**2
    s.t. n0 + sum_s x_s * v_s >= 0
    Where centroid is the center of the polytope
    bounded by those constraints.

    Note: Need cvxopt and cvxpy!

    Args:
        n0(1D ArrayLike[int]):
            An origin point of integer lattice.
        vs(2D ArrayLike[int]):
            Basis vectors of integer lattice.

    Returns:
        The natural number point on the grid closest to centroid ("x"):
            1D np.ndarray[int]
    """
    n0 = np.array(n0, dtype=int)
    vs = np.array(vs, dtype=int)
    n, d = vs.shape
    assert len(n0) == d
    poly = pc.Polytope(-vs.transpose(), n0)
    centroid = np.average(pc.extreme(poly), axis=0)
    x = cp.Variable(n, integer=True)
    constraints = [n0[i] + vs[:, i] @ x >= 0 for i in range(d)]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - centroid)),
                      constraints)
    _ = prob.solve()
    if x.value is None:
        raise NaturalInfeasibleError

    return np.array(np.round(x.value), dtype=int)


def get_one_dim_solutions(n0, v, integer_tol=NUM_TOL):
    """Solve one dimensional integer inequalities.

    This will solve n0 + v * x >= 0, give all
    integer solutions. x should be a single int.
    Args:
        n0(1D np.ndarray[int]):
            1 dimensional constraining factors
        v(1D np.ndarray[int]):
            1 dimensional constraining factors
        integer_tol(float): optional
            Tolerance of a number off integer
            value. If (number - round(number))
            <= integer_tol, it will be considered
            as an integer. Default is set by global
            NUM_TOL.
    Returns:
        All Integer solutions:
            1D np.ndarray[int]
    """
    x_min = -np.inf
    x_max = np.inf
    for i in range(len(v)):
        if v[i] > 0:
            l_margin = -n0[i] / v[i]
            if l_margin > x_min:
                x_min = l_margin
        elif v[i] < 0:
            r_margin = -n0[i] / v[i]
            if r_margin < x_max:
                x_max = r_margin
        else:
            if n0[i] < 0:  # Dim i not feasible.
                return np.array([], dtype=int)

    # If close to an integer, shift to it.
    # This is to prevent error in floor and ceil.
    x_min = (round(x_min)
             if abs(x_min - round(x_min)) <= integer_tol
             else x_min)
    x_max = (round(x_max)
             if abs(x_max - round(x_max)) <= integer_tol
             else x_max)

    n_min = np.ceil(x_min)
    n_max = np.floor(x_max)
    if n_min > n_max:
        return np.array([], dtype=int)
    else:
        return np.arange(n_min, n_max + 1, dtype=int)


def get_natural_solutions(n0, vs, integer_tol=NUM_TOL):
    """Enumerate all natural number solutions.

    Given the convex polytope n0 + sum_i x_i*v_i >= 0.
    Each time enumerate all possible integer x_0,
    fix x_0, then solve in each polytope with 1
    less dimension:
    n0 - x_0 * v_0 + sum_(i>0) x_i*v_i >= 0
    Recurse until all solutions are enumerated.

    Notice:
        This function is very costly! Do not
    use it with a large super-cell size!

    Args:
        n0(1D ArrayLike[int]):
            An origin point of integer lattice.
        vs(2D ArrayLike[int]):
            Basis vectors of integer lattice.
        integer_tol(float): optional
            Tolerance of a number off integer
            value. If (number - round(number))
            <= integer_tol, it will be considered
            as an integer. Default is set by global
            NUM_TOL.

    Returns:
        All natural number solutions ("x"):
            2D np.ndarray[int]
    """
    n0 = np.array(n0, dtype=int)
    vs = np.array(vs, dtype=int)

    n, d = vs.shape
    if n == 1:
        sols = get_one_dim_solutions(n0, vs[0, :],
                                     integer_tol=integer_tol)
        sols = sols.reshape(-1, 1)

        return sols

    p = pc.Polytope(-1 * vs.transpose(), n0)
    verts = pc.extreme(p)
    # Always branch the 1st dimension.
    x_min = np.min(verts[:, 0])
    x_max = np.max(verts[:, 0])

    x_min = (round(x_min)
             if abs(x_min - round(x_min)) <= integer_tol
             else x_min)
    x_max = (round(x_max)
             if abs(x_max - round(x_max)) <= integer_tol
             else x_max)

    n_min = np.ceil(x_min)
    n_max = np.floor(x_max)
    if n_min > n_max:
        return np.array([], dtype=int).reshape(-1, n)
    else:
        n_range = np.arange(n_min, n_max + 1, dtype=int)
        sols = []
        for m in n_range:
            n0_next = m * vs[0, :] + n0
            vs_next = vs[1:, :]
            sols_m = (get_natural_solutions(
                n0_next, vs_next,
                integer_tol=integer_tol)
            )
            if len(sols_m) > 0:
                sols_m = np.append(m * np.ones(len(sols_m),
                                               dtype=int),
                                   sols_m, axis=-1)
            else:
                sols_m = np.array([], dtype=int).reshape(-1, n)
            sols.append(sols_m)

        return np.concatenate(sols, axis=0)


# Flip table utilities
def flip_size(u, sublattice_dims=None):
    """Get metric of a direction.

    This function can compute flip size, absolute
    charge on one side of the reaction formula, etc.
    Args:
        u(1D ArrayLike[int]):
            A flip direction on the composition lattice.
            The components of u must be ordered, such that
            u is a simple concatenation of coordinates on
            each sub-lattice in sublattice_dims.
        sublattice_dims(1D ArrayLike[int]): optional
            Number of dimensions on each sub-lattice,
            namely the number of species types on each
            sub-lattice.
            Must have len(u) == sum(sublattice_dims).
            u must be sorted such that its components are ordered
            sub-lattice by sub-lattice.
            If not give, will consider all species in a same
            sub-lattice.

    Returns:
        Metric of direction u:
            int
    """
    if sublattice_dims is None:
        sublattice_dims = [len(u)]

    if len(u) != sum(sublattice_dims):
        raise ValueError("Sum of sub-lattice dimensions mismatch " +
                         "with total dimensions!")

    u = np.array(u, dtype=int)
    U = 0  # Flip size.
    begin = 0
    for d_sl in sublattice_dims:
        end = begin + d_sl
        u_sl = u[begin: end]
        U += np.sum(u_sl[u_sl > 0])
        begin = end

    return U


def count_row_matches(a1, a2):
    """Count number of matching rows in integer arrays.

    Args:
        a1, a2(2D Arraylike[int]):
            Arrays to compare
    Returns:
        Number of rows in common:
           int
    """
    a1 = np.array(a1, dtype=int)
    a2 = np.array(a2, dtype=int)
    s1 = set([tuple(r) for r in a1])
    s2 = set([tuple(r) for r in a2])
    return len(s1 & s2)


def connectivity(u, ns):
    """Compute graph connectivity contributed by vector.

    connectivity = number of (n',n) pair in natural
    number solutions ns, that n'-n = u or -u.
    i.e., suppose the natural number solutions as
    fully connected graph nodes, connectivity contributed
    by u is the number of edges parallel to u.

    Args:
        u(1D ArrayLike[int]):
            A flip direction on the composition lattice.
            The components of u must be ordered, such that
            u is a simple concatenation of coordinates on
            each sub-lattice in sublattice_dims.
        ns(2D ArrayLike[int]):
            A grid of natural number solutions.

    Returns:
        connectivity contributed by u: int
    """
    u = np.array(u, dtype=int)
    ns = np.array(ns, dtype=int)

    return (count_row_matches(ns, ns - u) +
            count_row_matches(ns, ns + u))


def get_optimal_basis(n0, vs, xs, sublattice_dims=None,
                      max_loops=100):
    """Get the optimal basis vectors to include in the flip table.

    Generate optimal basis vectors by:
        (1) Requiring basis vectors have minimal flip sizes;
        (2) Requiring maximum connectivity between compositions.
    To do this, we will iteratively execute the following steps
    to do a greedy optimization:
        1, For the current basis v1, .., vn, enumerate vectors:
        {v1, ..., vn} + {vi + vj | (i, j) in choose(n, 2)} +
        {vi - vj | (i, j) in choose(n, 2)}
        2, Sort these vectors, primarily by ascending flip size,
        secondarily by descending contribution to connectivity.
        3, Create an empty list. Iterate over sorted vectors.
        If the current we are looking at is not linearly dependent
        on the vectors already in the list, append it to the list.
        Stop the iteration until the list has n vectors.
    The above is called a cycle of optimization. We loop cycles
    to iteratively optimize basis set, until max number of loops
    has been reached, or the selected basis vectors no longer change.

    Notice:
        (1) This function is very costly! Do not use it with a large
    super-cell size!
        (2) The algorithm is supposed to give relatively good basis
    vectors, but its completeness has not been proven. Use it at your
    own risk.

    Args:
        n0(1D ArrayLike[int]):
            An origin point of integer lattice.
        vs(2D ArrayLike[int]):
            Basis vectors of integer lattice.
        xs(2D ArrayLike[int]):
            Solutions such that for each row x in xs,
            n0.T + x.T @ vs is a natural number point
            on the lattice.
            Shape = (n_solutions, n_basis_vectors)
        sublattice_dims(1D ArrayLike[int]): optional
            Number of dimensions on each sub-lattice,
            namely the number of species types on each
            sub-lattice.
            Must have len(u) == sum(sublattice_dims).
            u must be sorted such that it components are ordered
            sub-lattice by sub-lattice.
            If not give, will consider all species in a same
            sub-lattice.
        max_loops(int): optional
            Maximum number of optimization loops.

    Returns:
        Optimized basis vectors:
            2D np.ndarray[int]
    """
    n0 = np.array(n0, dtype=int)
    vs = np.array(vs, dtype=int)
    vs_opt = vs.copy()
    xs = np.array(xs, dtype=int)

    ns = xs @ vs + n0
    n, d = vs.shape

    def key_func(u):
        return (flip_size(u, sublattice_dims),
                -1 * connectivity(u, ns))

    # Make the first column always positive.
    def standardize_table(V):
        sign = (V[:, 0] >= 0).astype(int)
        mult = np.round((sign - 0.5) * 2).astype(int)
        return V * mult[:, None]

    def table_match(V1, V2):
        if V1.shape != V2.shape:
            return False
        V1 = standardize_table(V1)
        V2 = standardize_table(V2)
        return count_row_matches(V1, V2) == V1.shape[0]

    for _ in range(max_loops):
        V = vs_opt.copy()
        for i1, i2 in combinations(list(range(n))):
            np.concatenate(V, [vs[i1] + vs[i2], vs[i1] - vs[i2]],
                           axis=0)

        V = np.array(sorted(V, key=key_func), dtype=int)
        vs_new = np.array([], dtype=int).reshape(0, d)
        for i in range(len(V)):
            if len(vs_new) == n:
                break
            vs_current = np.concatenate(vs_new, [V[i]], axis=0)
            if np.linalg.matrix_rank(vs_current) \
                    == min(vs_current.shape):
                # Full rank.
                vs_new = vs_current.copy()

        if table_match(vs_new, vs):
            break
        vs_opt = vs_new.copy()

    return vs_opt


def get_ergodic_vectors(n0, vs, xs,
                        sublattice_dims=None, k=3):
    """Compute an ergodic flip table.

    Notice:
        1, The following algorithm only guarantees getting
        an ergodic flip table.
        It will also try to prioritize adding flip directions
        with  minimal flip sizes, but global optimality is not
        guaranteed.
        2, This algorithm does not guarantee finding the
        fewest number of directions to satisfy ergodicity at all.
        2, This process is NP-hard. Do not use it when the
        super-cell size is large.
    Args:
        n0(1D ArrayLike[int]):
            An origin point of integer lattice.
        vs(2D ArrayLike[int]):
            Basis vectors of integer lattice.
        xs(2D ArrayLike[int]):
            Solutions such that for each row x in xs,
            n0.T + x.T @ vs is a natural number point
            on the lattice.
            Shape = (n_solutions, n_basis_vectors)
        sublattice_dims(1D ArrayLike[int]): optional
            Number of dimensions on each sub-lattice,
            namely the number of species types on each
            sub-lattice.
            Must have len(u) == sum(sublattice_dims).
            u must be sorted such that it components are ordered
            sub-lattice by sub-lattice.
            If not give, will consider all species in a same
            sub-lattice.
        k(int): optional
            Find k-nearest neighbor of non-ergodic points,
            add those of them with minimal flip size to ensure
            ergodicity. Default is 3.

    Returns:
        Basis + Flip directions added to guarantee ergodicity:
            2D np.ndarray[int]
    """

    def is_connected(n, vs, ns):
        """Check whether a grid point is connected given table."""
        n = np.array(n, dtype=int)
        vs = np.array(vs, dtype=int)
        ns = np.array(ns, dtype=int)
        n_images = np.concatenate((vs, -vs), axis=0) + n
        return np.any(np.all(np.isclose(n_images[:, None, :],
                                        ns[None, :, :]),
                             axis=-1))

    def test_connected(vs, ns_disconnected, ns):
        return np.array([is_connected(n, vs, ns)
                         for n in ns_disconnected], dtype=bool)

    def candidate_key(u):
        return flip_size(u, sublattice_dims=sublattice_dims)

    n0 = np.array(n0, dtype=int)
    xs = np.array(xs, dtype=int)
    vs = np.array(vs, dtype=int)
    ns = xs @ vs + n0
    connected = test_connected(vs, ns, ns)
    ns_disconnected = ns[~ connected]

    tree = KDTree(ns)
    candidate_vectors = []
    connections_made = []
    # Get k nearest neighbors in Euclidean distance.
    for n in ns_disconnected:
        dists, points = tree.query(n, k=k)
        points = np.array(np.round(points), dtype=int)
        for point in points:
            u = point - n
            if (tuple(u.tolist()) in candidate_vectors or
                    tuple(-u.tolist()) in candidate_vectors):
                continue
            candidate_vectors.append(tuple(u.tolist()))

    candidate_vectors = sorted(candidate_vectors,
                               key=candidate_key)
    candidate_vectors = np.array(candidate_vectors, dtype=int)
    selected_vectors = vs.copy()
    ns_rem = ns_disconnected.copy()
    for u in candidate_vectors:
        vs_current = np.concatenate([u], selected_vectors, axis=0)
        connected = test_connected(vs_current, ns_rem, ns)
        selected_vectors = vs_current.copy()
        ns_rem = ns_rem[~ connected]
        if len(ns_rem) == 0:
            break

    return selected_vectors


# Probability selection tools
def choose_section_from_partition(p):
    """Choose one partition from multiple partitions.

    This function choose one section from a list with probability weights.
    Args:
        p(1D Arraylike[float]):
            Probabilities of each element. Will be normalized if not yet.
    Return:
        The index of randomly chosen element:
           int
    """
    p = np.array(p) / np.sum(p)
    return int(np.random.choice(len(p), p=p))