"""Mathematic utilities.

Including linear algebra, combinatorics and integer enumerations.
"""

__author___ = 'Fengyu Xie'

import numpy as np
from functools import reduce
from itertools import combinations
import math

from sympy.solvers.diophantine import diop_linear  # Sympy 1.5.1

from sympy import symbols


def GCD(a, b):
    """Euclidean Algorithm, giving positive GCD's."""
    if round(a) != a or round(b) != b:
        raise ValueError("GCD input must be integers!")
    a = abs(a)
    b = abs(b)
    while a:
        a, b = b % a, a
    return b


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
def reverse_ordering(L, ordering):
    """Reverse mapping.

    Given a mapping order of list, reverse that order
    and return the original list
    """
    original_l = [0 for i in range(len(L))]
    for cur_id, ori_id in enumerate(ordering):
        original_l[ori_id] = L[cur_id]
    return original_l


def rationalize_number(a, dim_limiter=100, dtol=1E-5):
    """Find a rational number near real number within dtol.

    Args:
        a(float):
            A number to be rationalized
        dim_limiter(int,default=100):
            Maximum allowed denoninator.
        dtol(float,default=1E-5):
            Maximum allowed difference of variable a
            to its rational form
    Return:
        (int, int): (numerator, denominator)
    """
    if a == 0:
        return 0, 1
    for magnif in range(1, dim_limiter + 1):
        a_prime = int(round(magnif * a))
        if abs(a_prime / magnif - a) < dtol:
            return a_prime, magnif

    raise ValueError("Can't find a rational number near \
                      {} within tolerance!".format(a))


def integerize_vector(v, dim_limiter=100, dtol=1E-5):
    """Rationalize all components of a vector v.

    Multiply the vector by LCM of all the component's denominator, so
    that all vector components are converted to integers.

    We call this process 'intergerization' of a vector.
    Args:
        v(np.ndarray(float)):
            A vector to be rationalized
        dim_limiter(int,default=100):
            Maximum allowed denoninator.
        dtol(float,default=1E-5):
            Maximum allowed difference of variable a
            to its rational form
    Return:
        np.ndarray, int
    """
    denos = []
    v = np.array(v)
    for c in v:
        _, deno = rationalize_number(
                                    c, dim_limiter=dim_limiter,
                                    dtol=dtol
                                    )
        denos.append(deno)
    lcm = LCM_list(denos)
    return np.array(np.round(v * lcm), dtype=np.int64), lcm


def integerize_multiple(vs, dim_limiter=100, dtol=1E-5):
    """Flatten and integerize multiple vectors.

    Args:
        v(np.ndarray(float)):
            A vector to be rationalized.
        dim_limiter(int,default=100):
            Maximum allowed denoninator.
        dtol(float,default=1E-5):
            Maximum allowed difference of variable a
            to its rational form.
    Return:
        np.ndarray, int
    """
    vs = np.array(vs)
    shp = vs.shape
    vs_flatten = vs.flatten()
    vs_flat_int, mul = integerize_vector(vs_flatten,
                                         dim_limiter=dim_limiter,
                                         dtol=dtol)
    vs_int = np.reshape(vs_flat_int, shp)
    return vs_int, mul


def combinatorial_number(n, m):
    """Calculate the combinatorial number.

    Args:
        m(int):
            Chosen size.
        n(int):
            Choose from size.
    Return:
        int, the combinatorial number.
    """
    if m > n:
        return 0
    return (math.factorial(n) //
            (math.factorial(m) * math.factorial(n - m)))


def get_integer_grid(subspc_normv, right_side=0, limiters=None):
    """Enumerate all points on a diophantine lattice.

    Gives all integer grid points in a subspace (on a hyperplane
    defined by a normal vector). The normal vector's components
    should all be positive, non-zero, and sorted (from low to high).
    Also know as the standard integer enumeration problem.

    Note:
        Diophatine enumeration is NP-hard, so try not to enumerate
        when dimensions of diophantine equation is too high!

    Args:
        subspc_normv(1D ArrayLike[float]):
            Normal vector of the integer hyper plane.
        right_side(int,default=0):
            Right side intercept of the hyper plane equation.
            Normal vector @ coordinate = intercept.
        limiters(List[tuple(int,int)]|NoneType, default = None):
             Lowerbounds and upperbounds of enumeration in
             each dimension.

    Return:
        List[List[int]], enumerated integer coordinates.
    """
    d = len(subspc_normv)
    if limiters is None:
        limiters = [(-7, 8) for i in range(d)]

    grids = []
    if d < 1:
        raise ValueError('Dimensionality too low, can not enumerate!')

    elif d == 1:
        k = subspc_normv[-1]
        if k != 0:
            if (right_side % k == 0 and right_side // k >= limiters[-1][0]
               and right_side // k <= limiters[-1][1]):
                grids.append([right_side // k])
        else:
            if right_side == 0:
                for i in range(limiters[-1][0], limiters[-1][1] + 1):
                    grids.append([i])

    else:
        new_limiters = limiters[:-1]
        # Move the last variable to the right hand side of
        # the hyperplane expression
        grids = []
        for val in range(limiters[-1][0], limiters[-1][1]+1):
            partial_grids = get_integer_grid(
                                            subspc_normv[:-1],
                                            right_side -
                                            val * subspc_normv[-1],
                                            limiters=new_limiters
                                            )
            for p_grid in partial_grids:
                grids.append(p_grid + [val])

    return grids


def get_integer_base_solution(subspc_normv, right_side=0):
    """Get a base integer solution of a linear diophitine equation.

    Args:
        subspc_normv(1D ArrayLike[float]):
            Normal vector of the integer hyper plane.
        right_side(int,default=0):
            Right side intercept of the hyper plane equation.
            Normal vector @ coordinate = intercept.

    Return:
        List[int], base integer solution.
    """
    symtags = ','.join(["x_{}".format(i) for i in range(len(subspc_normv))])
    syms = symbols(symtags)

    if len(subspc_normv) == 0:
        raise ValueError("Given diophatine coefficients are empty!")
    if np.allclose(subspc_normv, np.zeros(len(subspc_normv))):
        raise ValueError("Diphatine equation has no free symbol!")

    expr = subspc_normv[0] * syms[0]
    for i in range(1, len(subspc_normv)):
        expr += subspc_normv[i] * syms[i]
    expr -= right_side

    return [round(float(e.subs({s: 0 for s in e.free_symbols})))
            for e in diop_linear(expr)]


def get_integer_basis(normal_vec, sl_flips_list=None):
    """Primitive cell vectors of a diophatine lattice.

    The integer points on a hyperplane n x = 0 can form a lattice.
    This function can find the primitive lattice vectors with
    smallest norm, and are closest to orthogonal.

    Norm is defined by:
        L = sum_over_sublats(max_in_sublat(x_i))

    Args:
        normal_vec(1D ArrayLike[int]):
            Normal vector of the integer hyperplane.
        sl_flips_list(List[List[int]]!NoneType, default = None):
            Match index of each coordinate dimension into sublattice.
             A list of indices. If none, each coordinate dimension
             will be mapped into independent sublattices.
    Return:
        List[np.ndarray(int)], basis of the integer lattice.
    """
    gcd = GCD_list(normal_vec)
    if gcd == 0:
        gcd = 1

    # Make vector co-primal
    normal_vec = np.array(normal_vec, dtype=np.int64) // gcd
    d = len(normal_vec)

    # Dimension of the Charge-neutral subspace
    if np.allclose(normal_vec, np.zeros(d)):
        D = d
    else:
        D = d-1

    if sl_flips_list is None:
        sl_flips_list = [[i] for i in range(d)]

    # Single out dimensions where normal vec components are 0.
    # On these directions, basis is just a unit vector.
    zero_ids = np.where(normal_vec == 0)[0]
    pos_ids = np.where(normal_vec < 0)[0]
    neg_ids = np.where(normal_vec > 0)[0]
    non_zero_ids = np.concatenate((pos_ids, neg_ids))

    unit_basis = []
    for idx in zero_ids:
        e = np.zeros(d, dtype=np.int64)
        e[idx] = 1
        unit_basis.append(e)

    d_remain = D - len(zero_ids)
    if d_remain == 0:
        return unit_basis
    else:
        # Convert the problem into standard form
        nv_partial = np.abs(normal_vec[non_zero_ids])
        table = list(enumerate(nv_partial))
        sorted_table = sorted(table, key=(lambda pair: pair[1]))
        nv_sorted = np.array([n for ori_id, n in sorted_table],
                             dtype=np.int64)
        order_sorted = np.array([ori_id for ori_id, n in sorted_table],
                                dtype=np.int64)

        # Estimate limiters
        d_nonzero = len(nv_partial)
        limiter_vectors = []
        for i, j in combinations(list(range(d_nonzero)), 2):
            gcd = GCD(nv_sorted[i], nv_sorted[j])
            lcm = (nv_sorted[i] * nv_sorted[j]) // gcd
            limiter_v = np.zeros(d_nonzero, dtype=np.int64)
            limiter_v[i] = lcm // nv_sorted[i]
            limiter_v[j] = -lcm // nv_sorted[j]
            limiter_vectors.append(limiter_v)

        limiter_ubs = np.max(np.vstack(limiter_vectors), axis=0)
        limiter_lbs = np.min(np.vstack(limiter_vectors), axis=0)
        limiters = list(zip(limiter_lbs, limiter_ubs))

        integer_grid = get_integer_grid(nv_sorted, limiters=limiters)

        basis_pool = []
        for point in integer_grid:
            reversely_ordered_point = reverse_ordering(point, order_sorted)
            basis = np.zeros(d, dtype=np.int64)
            for part_id, ori_id in enumerate(non_zero_ids):
                if ori_id in pos_ids:
                    basis[ori_id] = reversely_ordered_point[part_id]
                else:
                    basis[ori_id] = -1 * reversely_ordered_point[part_id]
            basis_pool.append(basis)

        basis_pool = sorted(basis_pool,
                            key=lambda v: (
                             formula_norm(v, sl_flips_list=sl_flips_list),
                             np.max(np.abs(v))))

        basis_pool = np.array(basis_pool)

        chosen_basis = []
        for v in basis_pool:
            # Full rank condition
            if (np.linalg.matrix_rank(np.vstack(chosen_basis + [v]))
               == len(chosen_basis) + 1):
                chosen_basis.append(v)
            if len(chosen_basis) == d_remain:
                # We have selected enough basis
                break

        return unit_basis + chosen_basis



def formula_norm(v, sl_flips_list):
    """Get formula norm.

    L = sum_over_sublats(max_in_sublat(|x_i|)).
      = total number of atoms flipped.
    """
    v = np.array(v)
    sl_form_sizes = []
    for sl in sl_flips_list:
        flip_nums = v[sl].tolist() + [-1 * sum(v[sl])]
        sl_form_size = 0
        for num in flip_nums:
            if num > 0:
                sl_form_size += num
        sl_form_sizes.append(sl_form_size)
    return sum(sl_form_sizes)


# Partition selection tools
def choose_section_from_partition(probs):
    """Choose one partition from multiple partitions.

    This function choose one section from a partition based on each section's
    normalized probability.
    Args:
        probs(1D Arraylike[float]):
            Probabilities of each sections. Will be normalized if not yet.
    Return:
        int, the index of randomly chosen section.
    """
    N_secs = len(probs)
    if N_secs < 1:
        raise ValueError("Segment can't be selected!")

    norm_probs = np.array(probs) / np.sum(probs)
    upper_bnds = np.array([sum(norm_probs[:i + 1]) for i in range(N_secs)])
    rand_seed = np.random.rand()

    for sec_id, sec_upper in enumerate(upper_bnds):
        if sec_id == 0:
            sec_lower = 0
        else:
            sec_lower = upper_bnds[sec_id - 1]
        if rand_seed >= sec_lower and rand_seed < sec_upper:
            return sec_id

    raise ValueError("Segment can't be selected.")
