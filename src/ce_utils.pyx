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
def corr_from_occupancy(np.int_t[::1] occu, int n_bit_orderings, orbit_list):
    """
    Computes the correlation vector for a given encoded occupancy vector
    Args:
        occu (np.array):
            encoded occupancy vector
        n_bit_orderings (int):
            total number of bit orderings in expansion.
        orbits:
            Information of all orbits that include the flip site.
            (bit_combos, orbit id, site indices, bases array)

    Returns: array
        correlation vector difference
    """

    cdef int i, j, k, I, J, K, id, n
    cdef double p, pi
    cdef const np.int_t[:, ::1] inds
    cdef const np.int_t[:, ::1] bits
    cdef const np.float_t[:, :, ::1] bases
    out = np.zeros(n_bit_orderings)
    cdef np.float_t[:] o_view = out
    o_view[0] = 1  # empty cluster

    for id, combos, bases, inds in orbit_list:
        I = inds.shape[0] # cluster index
        K = inds.shape[1] # index within cluster
        n = id
        for bits in combos:
            J = bits.shape[0]
            p = 0
            for i in range(I):
                for j in range(J):
                    pi = 1
                    for k in range(K):
                        pi *= bases[k, bits[j, k], occu[inds[i, k]]]
                    p += pi
            o_view[n] = p / (I*J)
            n += 1

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def delta_corr_single_flip(np.int_t[::1] occu_f, np.int_t[::1] occu_i,
                           int n_bit_orderings, site_orbit_list):
    """
    Computes the correlation difference between two occupancy
    vectors.
    Args:
        occu_f (np.array):
            encoded occupancy vector with flip
        occu_i (np.array):
            encoded occupancy vector without flip
        n_bit_orderings (int):
            total number of bit orderings in expansion.
        site_orbit_list:
            Information of all orbits that include the flip site.
            List of tuples each with
            (bit_combos, orbit id, site indices, ratio, bases array)


    Returns: array
        correlation vector difference
    """

    cdef int i, j, k, I, J, K, id, n
    cdef double p, pi, pf, r
    cdef const np.int_t[:, ::1] inds
    cdef const np.int_t[:, ::1] bits
    cdef const np.float_t[:, :, ::1] bases
    out = np.zeros(n_bit_orderings)
    cdef np.float_t[:] o_view = out

    for id, r, combos, bases, inds in site_orbit_list:
        I = inds.shape[0] # cluster index
        K = inds.shape[1] # index within cluster
        n = id
        for bits in combos:
            J = bits.shape[0]
            p = 0
            for i in range(I):
                for j in range(J):
                    pf = 1
                    pi = 1
                    for k in range(K):
                        pf *= bases[k, bits[j, k], occu_f[inds[i, k]]]
                        pi *= bases[k, bits[j, k], occu_i[inds[i, k]]]
                    p += (pf - pi)
            o_view[n] = p / r / (I*J)
            n += 1

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def delta_ewald_single_flip(np.int_t[:] final, np.int_t[:] init,
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
