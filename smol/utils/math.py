"""Mathematic utilities.

Including linear algebra, combinatorics and integer enumerations.
"""

__author___ = "Fengyu Xie, Luis Barroso-Luque"

from fractions import Fraction
from itertools import combinations, product

import numpy as np
from monty.dev import requires
from scipy.linalg import null_space
from scipy.spatial import KDTree

try:
    import polytope as pc
except ImportError:
    pc = None

try:
    import cvxpy as cp
except ImportError:
    cp = None


# Global numerical tolerance in this module.
NUM_TOL = 1e-6


def yield_hermite_normal_forms(determinant):
    """Yield all hermite normal form matrices with given determinant.

    Args:
        determinant (int):
            determinant of hermite normal forms to be yielded

    Yields:
        ndarray: hermite normal form matrix with given determinant
    """
    for a in filter(lambda x: determinant % x == 0, range(1, determinant + 1)):
        quotient = determinant // a
        for c in filter(lambda x: quotient % x == 0, range(1, determinant // a + 1)):
            f = quotient // c
            for b, d, e in product(range(0, c), range(0, f), range(0, f)):
                yield np.array([[a, 0, 0], [b, c, 0], [d, e, f]], dtype=int)


def gcdex(a, b):
    """Extend Euclidean Algorithm."""
    if a == 0:
        return 0, 1, b

    x1, y1, g = gcdex(b % a, a)
    x = y1 - (b // a) * x1
    y = x1

    return x, y, g


def rationalize_number(a, max_denominator=1000, dtol=NUM_TOL):
    """Find a rational number near real number within dtol.

    Args:
        a(float):
            A number to be rationalized
        max_denominator(int): optional
            Maximum allowed denominator. Default 1000.
        dtol(float): optional
            Maximum allowed difference of variable a
            to its rational form. Default 1E-6.
            You must have 1/max_denominator > dtol!
    Return:
        Numerator and denominator (always positive):
           int, int
    """
    f = Fraction.from_float(a).limit_denominator(max_denominator)
    numerator = f.numerator
    denominator = f.denominator

    if abs(float(numerator) / denominator - a) > dtol:
        raise ValueError(
            "Can't find a rational number near " "{} within tolerance!".format(a)
        )

    return numerator, denominator


def integerize_vector(v, max_denominator=1000, dtol=NUM_TOL):
    """Integerize all components of a vector v.

    Rationalize all components of a vector, then multiply
    the vector by the LCM of all the rationalized component's
    denominator, so that all vector components are converted
    to integers. We call this process 'intergerization' of a
    vector.
    Args:
        v(np.ndarray(float)):
            A vector to be rationalized
        max_denominator(int,default=1000):
            Maximum allowed denominator.
        dtol(float,default=1E-6):
            Maximum allowed difference of variable a
            to its rational form.
        You must have 1/max_denominator > dtol!
    Return:
        Integrized vector, LCM of denominators:
            np.ndarray, int
    """
    denos = []
    v = np.array(v)
    for c in v:
        _, deno = rationalize_number(c, max_denominator=max_denominator, dtol=dtol)
        denos.append(deno)
    lcm = np.lcm.reduce(denos)
    return np.array(np.round(v * lcm), dtype=np.int64), lcm


def integerize_multiple(vs, max_denominator=1000, dtol=NUM_TOL):
    """Integerize multiple vectors in a matrix.

    Args:
        vs(np.ndarray(float)):
            A matrix of vectors to be rationalized.
        max_denominator(int,default=1000):
            Maximum allowed denominator.
        dtol(float,default=1E-6):
            Maximum allowed difference of variable to its rational form.
        You must have 1/max_denominator > dtol!
    Return:
        Integerized vectors in the input shape, LCM of denominator:
            np.ndarray[int], int
    """
    vs = np.array(vs)
    shp = vs.shape
    vs_flatten = vs.flatten()
    vs_flat_int, mul = integerize_vector(
        vs_flatten, max_denominator=max_denominator, dtol=dtol
    )
    vs_int = np.reshape(vs_flat_int, shp)
    return vs_int, mul


# Integer linear algebra utilities.
# Note: don't use sympy anymore. In sympy 1.9,
# a matrix m full of 0 will give is_zero=False,
# while in 1.5.1, it will give is_zero=True!
# Try to remove sympy dependency!
def compute_snf(a):
    """Compute smith normal form of a matrix.

    Args:
        a(2D arraylike of int):
            A matrix defined on some domain.

    Returns:
        s, m, t (np.ndarray of int):
            Smith decomposition of a, such that:
            m is the smith normal form and m = s a t.
    """

    def leftmult(m, i0, i1, a, b, c, d):  # Matrix pivoting operations.
        for j in range(m.shape[1]):
            x, y = m[i0, j], m[i1, j]
            m[i0, j] = a * x + b * y
            m[i1, j] = c * x + d * y

    def rightmult(m, j0, j1, a, b, c, d):  # Matrix pivoting operations.
        for i in range(m.shape[0]):
            x, y = m[i, j0], m[i, j1]
            m[i, j0] = a * x + c * y
            m[i, j1] = b * x + d * y

    # Must convert to a sympy.Matrix.
    m = np.round(a).astype(int).copy()
    s = np.eye(m.shape[0]).astype(int)
    t = np.eye(m.shape[1]).astype(int)
    last_j = -1
    for i in range(m.shape[0]):
        for j in range(last_j + 1, m.shape[1]):
            if not np.all(m[:, j] == 0):
                break
        else:
            break
        if m[i, j] == 0:
            for ii in range(m.shape[0]):
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
            for ii in range(i + 1, m.shape[0]):
                if m[ii, j] == 0:
                    continue
                upd = True
                if m[ii, j] % m[i, j] != 0:
                    coef1, coef2, g = gcdex(m[i, j], m[ii, j])
                    coef3 = m[ii, j] // g
                    coef4 = m[i, j] // g
                    leftmult(m, i, ii, coef1, coef2, -coef3, coef4)
                    leftmult(s, i, ii, coef1, coef2, -coef3, coef4)
                coef5 = m[ii, j] // m[i, j]
                leftmult(m, i, ii, 1, 0, -coef5, 1)
                leftmult(s, i, ii, 1, 0, -coef5, 1)
            for jj in range(j + 1, m.shape[1]):
                if m[i, jj] == 0:
                    continue
                upd = True
                if m[i, jj] % m[i, j] != 0:
                    coef1, coef2, g = gcdex(m[i, j], int(m[i, jj]))
                    coef3 = m[i, jj] // g
                    coef4 = m[i, j] // g
                    rightmult(m, j, jj, coef1, -coef3, coef2, coef4)
                    rightmult(t, j, jj, coef1, -coef3, coef2, coef4)
                coef5 = m[i, jj] // m[i, j]
                rightmult(m, j, jj, 1, -coef5, 0, 1)
                rightmult(t, j, jj, 1, -coef5, 0, 1)
        last_j = j

    for i1 in range(min(m.shape)):
        for i0 in reversed(range(i1)):
            coef1, coef2, g = gcdex(m[i0, i0], m[i1, i1])
            if g == 0:
                continue
            coef3 = m[i1, i1] // g
            coef4 = m[i0, i0] // g
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

    We use Smith normal form to solve equations. If equation is not solvable, we will
    throw an error.

    Note: If A decomposes to snf with large matrix elements, the numerical accuracy
    might have an issue! When this is the case, even if An=b has an integer solution,
    our function is not guaranteed to find it! But for most application uses, the
    numerical accuracy here should be enough.

    Args:
        A (2D ArrayLike[int]):
            Matrix A in An=b.
        b (1D ArrayLike[int], default=None):
            Vector b in An=b.
            If not given, will set to zeros.

    Return: (1D np.ndarray[int], 2D np.ndarray[int]
        A base solution and base vectors (as rows):
    """
    A = np.array(A, dtype=int)
    n, d = A.shape
    b = np.array(b, dtype=int) if b is not None else np.zeros(d, dtype=int)
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
            print("index:", i)
            print("c[i]:", c[i])
            print("b[ii]:", B[i, i])
            print("U:", U)
            print("B:", B)
            print("V:", V)
            assert np.allclose(B, U @ A @ V)
            raise ValueError("Diophantine equations A n = b are not feasible!")

    # Get base solution
    n0 = V[:, :k] @ (c[:k] // B.diagonal()[:k])

    return n0, V[:, k:].transpose().copy()


@requires(pc is not None, " 'polytope' package not found. Please install it.")
def get_nonneg_float_vertices(A, b):
    """Get vertices of polytope An=b, n>=0.

    n is allowed to be non-negative floats.
    This result can be used to get the minimum
    supercell size by looking for the minimum
    integer than can integerize all these vertices,
    if b is written according to number of sites in
    a primitive cell.

    Args:
        A (2D np.ndarray):
            A in An=b.
        b (2D np.ndarray):
            b in An=b.
    Returns:
        Vertices of polytope An=b, n>=0 in float:
            np.ndarray[float]
    """
    A = np.array(A)
    b = np.array(b)

    vs = null_space(A).transpose()  # each basis vector as a row.
    n0 = np.linalg.pinv(A) @ b
    poly = pc.Polytope(-1 * vs.transpose(), n0)

    verts = pc.extreme(poly)
    if len(verts) == 0:
        if pc.is_empty(poly):
            raise ValueError("Provided equation An=b is not feasible " "under n>=0.")
        else:
            raise ValueError(
                "Provided equation An=b is not fill " "dimensional under n>=0."
            )
    verts = verts @ vs + n0
    return verts


@requires(
    pc is not None and cp is not None,
    "'polytope' and 'cvxpy' packages are required. Please install them.",
)
def get_natural_centroid(
    n0, vs, sc_size, a_leq=None, b_leq=None, a_geq=None, b_geq=None
):
    """Get the natural number solution closest to centroid.

    Done by linear programming, minimize:
    norm_2(x - centroid)**2
    s.t. n0 + sum_s x_s * v_s >= 0
    Where centroid is the center of the polytope
    bounded by those constraints.

    Note: Need cvxopt and cvxpy!

    Args:
        n0 (1D ArrayLike[int]):
            An origin point of integer lattice.
        vs (2D ArrayLike[int]):
            Basis vectors of integer lattice.
        sc_size (int):
            Super-cell size with n0 as a base solution.
        a_leq (2D ArrayLike[int]), b_leq(1D ArrayLike[float]):
            Constraint A @ n <= b. Unit is per prim.
        a_geq (2D ArrayLike[int]), b_geq(1D ArrayLike[float]):
            Constraint A @ n >= b. Unit is per prim.

    Returns: 1D np.ndarray[int]
        The natural number point on the grid closest to centroid ("x"):
    """
    n0 = np.array(n0, dtype=int)
    vs = np.array(vs, dtype=int)
    n, d = vs.shape
    assert len(n0) == d
    poly = pc.Polytope(-vs.transpose(), n0)
    centroid = np.average(pc.extreme(poly), axis=0)
    x = cp.Variable(n, integer=True)
    constraints = [n0[i] + vs[:, i] @ x >= 0 for i in range(d)]
    if a_leq is not None and b_leq is not None:
        for a, bb in zip(a_leq, b_leq):
            constraints.append(a @ (n0 + x @ vs) <= bb * sc_size)
    if a_geq is not None and b_geq is not None:
        for a, bb in zip(a_geq, b_geq):
            constraints.append(a @ (n0 + x @ vs) >= bb * sc_size)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - centroid)), constraints)
    # Use gurobi if present.
    if "GUROBI" in cp.installed_solvers():
        _ = prob.solve(solver=cp.GUROBI)
    else:
        _ = prob.solve(
            solver=cp.ECOS_BB, max_iters=200, abstol=NUM_TOL, feastol=NUM_TOL
        )
    if x.value is None:
        raise ValueError("No feasible natural number composition found!")

    return np.array(np.round(x.value), dtype=int)


def get_one_dim_solutions(n0, v, integer_tol=NUM_TOL, step=1):
    """Solve one dimensional integer inequalities.

    This will solve n0 + v * x >= 0, give all
    integer solutions. x should be a single int.
    Args:
        n0 (1D ArrayLinke[int]:
            1 dimensional constraining factors
        v (1D ArrayLinke[int]:
            1 dimensional constraining factors
        integer_tol (float): optional
            Tolerance of a number off integer
            value. If (number - round(number))
            <= integer_tol, it will be considered
            as an integer. Default is set by global
            NUM_TOL.
        step (int): optional
            Step to skip when yielding solutions. For example,
            when step=2, will yield every 2 solutions.
            Default is 1, yield every solution.
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

    if x_min <= -np.inf or x_max >= np.inf:
        raise ValueError("Inequalities are not bounded!")
    # If close to an integer, shift to it.
    # This is to prevent error in floor and ceil.
    x_min = round(x_min) if abs(x_min - round(x_min)) <= integer_tol else x_min
    x_max = round(x_max) if abs(x_max - round(x_max)) <= integer_tol else x_max

    n_min = np.ceil(x_min)
    n_max = np.floor(x_max)
    if n_min > n_max:
        return np.array([], dtype=int)
    else:
        return np.arange(n_min, n_max + 1, step, dtype=int)


@requires(cp is not None, "'cvxpy' package is required. Please install it.")
def get_first_dim_extremes(a, b):
    """Solve extremes for x0 under ax<=b.

    Args:
        a, b (ArrayLike[int]):
            Constraints ax<=b. ax<=b must be feasible and
            bounded.
    Return:
        float, float:
            min x0 and max x0 when ax<=b is feasible.
    """
    a = np.array(a)
    b = np.array(b)
    n, d = a.shape
    if len(b) != n:
        raise ValueError(f"Constraint matrix {a} and " f"vector {b} does not match!")
    x1 = cp.Variable(d)
    prob1 = cp.Problem(cp.Minimize(x1[0]), [a @ x1 <= b])
    _ = prob1.solve()
    if x1.value is None:
        raise ValueError(f"Polytope a: {a}, b:{b} is empty or not bounded!")
    assert abs(prob1.value - x1.value[0]) <= NUM_TOL

    x2 = cp.Variable(d)
    prob2 = cp.Problem(cp.Maximize(x2[0]), [a @ x2 <= b])
    _ = prob2.solve()
    if x1.value is None:
        raise ValueError(f"Polytope a: {a}, b:{b} is empty or not bounded!")
    assert abs(prob2.value - x2.value[0]) <= NUM_TOL

    return prob1.value, prob2.value


def get_natural_solutions(n0, vs, integer_tol=NUM_TOL, step=1):
    """Enumerate all natural number solutions.

    Given the convex polytope n0 + sum_i x_i*v_i >= 0.
    Each time enumerate all possible integer x_0,
    fix x_0, then solve in each polytope with 1
    less dimension:
    n0 - x_0 * v_0 + sum_(i>0) x_i*v_i >= 0
    Recurse until all solutions are enumerated.

    Notice:
        1, This function is very costly! Do not
    use it with a large super-cell size!
        2, This function does not apply to any n0
    and vs. It only applies to bounded polytopes!

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
        step(int): optional
            Step to skip when yielding solutions in each
            dimension. For example,
            when step=2, will yield every 2 solutions in each
            dimension.
            Default is 1, yield every solution.

    Returns:
        All natural number solutions ("x"):
            2D np.ndarray[int]
    """
    n0 = np.array(n0, dtype=int)
    vs = np.array(vs, dtype=int)

    n, d = vs.shape
    if n == 1:
        sols = get_one_dim_solutions(n0, vs[0, :], integer_tol=integer_tol, step=step)
        sols = sols.reshape(-1, 1)

        return sols

    x_min, x_max = get_first_dim_extremes(-1 * vs.transpose(), n0)
    if x_min <= -np.inf or x_max >= np.inf:
        raise ValueError("Inequalities are not bounded!")
    # Do not use polytope module here. A polytope cannot be take extreme
    # if it is not feasible or has 0 volume.
    # Always branch the 1st dimension.

    x_min = round(x_min) if abs(x_min - round(x_min)) <= integer_tol else x_min
    x_max = round(x_max) if abs(x_max - round(x_max)) <= integer_tol else x_max

    n_min = np.ceil(x_min)
    n_max = np.floor(x_max)
    if n_min > n_max:
        return np.array([], dtype=int).reshape(-1, n)
    else:
        n_range = np.arange(n_min, n_max + 1, step, dtype=int)
        sols = []
        for m in n_range:
            n0_next = m * vs[0, :] + n0
            vs_next = vs[1:, :]
            sols_m = get_natural_solutions(
                n0_next, vs_next, integer_tol=integer_tol, step=step
            )
            if len(sols_m) > 0:
                sols_m = np.append(
                    m * np.ones(len(sols_m), dtype=int).reshape(-1, 1), sols_m, axis=-1
                )
            else:
                sols_m = np.array([], dtype=int).reshape(-1, n)
            sols.append(sols_m)

        return np.concatenate(sols, axis=0)


# Flip table utilities
def flip_size(u):
    """Get metric of a direction.

    This function can compute flip size, absolute
    charge on one side of the reaction formula, etc.
    Args:
        u(1D ArrayLike[int]):
            A flip direction on the composition lattice.
            The components of u must be ordered, such that
            u is a simple concatenation of components on
            each sub-lattice.
            We only check that sum(u) == 0. It is your
            responsibility to check that's also true
            per-sub-lattice.

    Returns:
        Metric of direction u:
            int
    """
    u = np.array(u, dtype=int)
    if np.sum(u) != 0:
        raise ValueError(f"Flip vector {u} does not" "conserve number of sites!")
    return np.sum(u[u > 0])


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
    s1 = {tuple(r) for r in a1}
    s2 = {tuple(r) for r in a2}
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
            u is a simple concatenation of components in
            each sub-lattice.
        ns(2D ArrayLike[int]):
            A grid of natural number solutions.

    Returns:
        connectivity contributed by u: int
    """
    u = np.array(u, dtype=int)
    ns = np.array(ns, dtype=int)

    # Connectivity of u must be the same as -u,
    # This is trivial conclusion from sets theory,
    # so do not double-count.
    return count_row_matches(ns, ns + u)


def is_connected(n, vs, ns):
    """Check whether a grid point is connected by flip vectors.

    Args:
        n(1D ArrayLike[int]):
            A grid point.
        vs(2D ArrayLike[int]):
            Table of flip vectors.
        ns(2D ArrayLike[int]):
            All grid points.
    Returns:
        bool.
    """
    n = np.array(n, dtype=int)
    vs = np.array(vs, dtype=int)
    ns = np.array(ns, dtype=int)
    n_images = np.concatenate((vs, -vs), axis=0) + n
    return np.any(np.all(np.isclose(n_images[:, None, :], ns[None, :, :]), axis=-1))


def get_optimal_basis(n0, vs, xs, max_loops=100):
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
        return flip_size(u), -1 * connectivity(u, ns)

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
        for i1, i2 in combinations(list(range(n)), 2):
            V = np.concatenate((V, [V[i1] + V[i2], V[i1] - V[i2]]), axis=0)

        V = np.array(sorted(V, key=key_func), dtype=int)
        vs_new = np.array([], dtype=int).reshape(0, d)
        for i in range(len(V)):
            if len(vs_new) == n:
                break
            vs_current = np.concatenate((vs_new, [V[i]]), axis=0)
            if np.linalg.matrix_rank(vs_current) == min(vs_current.shape):
                # Full rank.
                vs_new = vs_current.copy()

        if table_match(vs_new, vs_opt):
            break
        vs_opt = vs_new.copy()

    return vs_opt


def get_ergodic_vectors(n0, vs, xs, k=3):
    """Compute an ergodic flip table.

    Notice:
        1, The following algorithm only guarantees getting
        an ergodic flip table.
        It will also try to prioritize adding flip directions
        with  minimal flip sizes, but global optimality is not
        guaranteed. Currently also does not guarantee graph
        connectivity, if graphs can be divided as parts.
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
        k(int): optional
            Find k-nearest neighbor of non-ergodic points,
            add those of them with minimal flip size to ensure
            ergodicity. Default is 3. k>1 is required, otherwise
            will always return self.

    Returns:
        Basis + Flip directions added to guarantee ergodicity:
            2D np.ndarray[int]
    """

    def test_connected(vs, ns_disconnected, ns):
        return np.array([is_connected(n, vs, ns) for n in ns_disconnected], dtype=bool)

    def candidate_key(u):
        return flip_size(u)

    n0 = np.array(n0, dtype=int)
    xs = np.array(xs, dtype=int)
    vs = np.array(vs, dtype=int)
    ns = xs @ vs + n0
    connected = test_connected(vs, ns, ns)
    ns_disconnected = ns[~connected]
    if len(ns_disconnected) == 0:
        return vs

    tree = KDTree(ns)
    candidate_vectors = []
    # Get k nearest neighbors in Euclidean distance.
    for n in ns_disconnected:
        dists, point_ids = tree.query(n, k=k)
        if dists[0] == 0:
            point_ids = point_ids[1:]
        points = ns[point_ids, :]
        for point in points:
            u = point - n
            if (
                tuple(u.tolist()) in candidate_vectors
                or tuple((-u).tolist()) in candidate_vectors
            ):
                continue
            candidate_vectors.append(tuple(u.tolist()))

    candidate_vectors = sorted(candidate_vectors, key=candidate_key)
    candidate_vectors = np.array(candidate_vectors, dtype=int)
    selected_vectors = vs.copy()
    ns_rem = ns_disconnected.copy()
    for u in candidate_vectors:
        vs_current = np.concatenate((selected_vectors, [u]), axis=0)
        connected = test_connected(vs_current, ns_rem, ns)
        selected_vectors = vs_current.copy()
        ns_rem = ns_rem[~connected]
        if len(ns_rem) == 0:
            break

    return selected_vectors


def flip_weights_mask(flip_vectors, n, max_n=None):
    """Mark feasibility of flip vectors.

    If a flip direction leads to any n+v < 0, then it is marked
    infeasible. Generates a boolean mask, every two components
    marks whether a flip direction and its inverse is feasible
    given n at the current occupancy.
    Will be used by Tableflipper.

    Args:
        flip_vectors(1D ArrayLike[int]):
            Flip directions in the table (inverses not included).
        n(1D ArrayLike[int]):
            Amount of each specie on sublattices. Same as returned
            by occu_to_species_n.
        max_n(1D ArrayLike[int]): optional
            Maximum number of each species allowed. This is needed
            When the number of active sites != number of sites in
            some sub-lattice.

    Return:
        Direction and its inverse are feasible or not:
           1D np.ndarray[bool]
    """
    flip_vectors = np.array(flip_vectors, dtype=int)
    directions = np.concatenate([(u, -u) for u in flip_vectors], axis=0)
    if max_n is None:
        max_n = np.ones(len(n)) * np.inf
    elif isinstance(max_n, (int, np.int32, np.int64)):
        max_n = np.ones(len(n), dtype=int) * max_n
    else:
        max_n = np.array(max_n, dtype=int)
    return ~(
        np.any(directions + n < 0, axis=-1) | np.any(directions + n > max_n, axis=-1)
    )


# Probability selection tools
def choose_section_from_partition(probabilities, rng=None):
    """Choose one partition from multiple partitions.

    This function choose one section from a list with probability weights.
    Args:
        probabilities(1D Arraylike[float]):
            Probabilities of each section. Will be normalized if not yet.
        rng (np.Generator): optional
            The given PRNG must be the same instance as that used by the
            kernel and any bias terms, otherwise reproducibility will be
            compromised.
    Return:
        The index of randomly chosen element:
           int
    """
    if rng is None:
        rng = np.random.default_rng()
    p = np.array(probabilities)
    if np.allclose(p, 0):
        p = np.ones(len(p))
    if not np.all(p >= -NUM_TOL):
        raise ValueError("Probabilities contain negative number.")
    p = p / p.sum()
    return int(round(rng.choice(len(p), p=p)))
