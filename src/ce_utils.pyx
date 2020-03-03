# coding: utf-8
"""
Utilities for manipulating coordinates or list of coordinates, under periodic
boundary conditions or otherwise.
"""

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def delta_corr_single_flip(np.int_t[:] occu_f, np.int_t[:] occu_i,
                           int n_bit_orderings, orbits):
    """
    Computes the correlation difference between adjacent occupancies.
    (That differ by only a single flip)
    Args:
        occu_f (np.array):
            encoded occupancy vector with flip
        occu_i (np.array):
            encoded occupancy vector without flip
        n_bit_orderings (int):
            total number of bit orderings in expansion.
        orbits:
            Information of all orbits that include the flip site.

    Returns: array
        correlation vector difference
    """

    cdef int i, j, k, I, J, K, id, N, n
    cdef double p, pi, pf, r
    cdef const np.int_t[:, ::1] inds
    cdef const np.int_t[:, :, ::1] bitcbs
    cdef const np.float_t[:, :, ::1] bases
    out = np.zeros(n_bit_orderings)
    cdef np.float_t[:] o_view = out

    for bitcbs, id, inds, r, bases in orbits:
        I = inds.shape[0] # cluster index
        K = inds.shape[1] # index within cluster
        N = bitcbs.shape[0]
        for n in range(N):
            J = bitcbs.shape[1]
            p = 0
            for i in range(I):
                for j in range(J):
                    pf = 1
                    pi = 1
                    for k in range(K):
                        pf *= bases[k, bitcbs[n, j, k], occu_f[inds[i, k]]]
                        pi *= bases[k, bitcbs[n, j, k], occu_i[inds[i, k]]]
                    p += (pf - pi)
            o_view[id + n] = p / r / I*J

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def delta_ewald_corr_single_flip(np.int_t[:] final, np.int_t[:] init,
                                 int n_bit_orderings, clusters, int ind, int bit,
                                 np.float_t[:, :, :] ewald_matrices,
                                 np.int_t[:, :] ewald_inds, double size):

    cdef int i, j, k, I, J, K, l, add, sub
    cdef bint ok
    cdef np.int_t[:, :] inds
    out = np.zeros(len(ewald_matrices))
    cdef np.float_t[:] o_view = out
    cdef double r, o

    # values of -1 are vacancies and hence don't have ewald indices
    add = ewald_inds[ind, bit]
    sub = ewald_inds[ind, init[ind]]
    for l in range(ewald_matrices.shape[0]):
        o = 0
        for j in range(final.shape[0]):
            i = ewald_inds[j, final[j]]
            if i != -1 and add != -1:
                if i != add:
                    o += ewald_matrices[l, i, add] * 2
                else:
                    o += ewald_matrices[l, i, add]

        for j in range(init.shape[0]):
            i = ewald_inds[j, init[j]]
            if i != -1 and sub != -1:
                if i != sub:
                    o -= ewald_matrices[l, i, sub] * 2
                else:
                    o -= ewald_matrices[l, i, sub]

        o_view[l] = o / size
        l += 1

    return out
