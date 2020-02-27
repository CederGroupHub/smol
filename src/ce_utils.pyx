# coding: utf-8

"""
Utilities for manipulating coordinates or list of coordinates, under periodic
boundary conditions or otherwise.
"""
import numpy as np

cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def delta_corr_single_flip(np.int_t[:] final, np.int_t[:] init,
                           int n_bit_orderings, clusters):
    """
    Counts number of rows of a that are present in b
    Args:
        final, init: inital and final occupancies
        clusters: List of clusters that the flip can affect
    Returns:
        delta_corr vector from a single flip
    """

    cdef int i, j, k, I, J, K, l
    cdef bint ok
    cdef np.int_t[:, :] b, inds
    out = np.zeros(n_bit_orderings)
    cdef np.float_t[:] o_view = out
    cdef double r, o

    for sc_bits, sc_b_id, inds, r in clusters:
        l = sc_b_id
        I = inds.shape[0] # cluster index
        K = inds.shape[1] # index within cluster
        for b in sc_bits:
            J = b.shape[0] # index within bit array
            o = 0
            for i in range(I):
                for j in range(J):
                    ok = True
                    for k in range(K):
                        if final[inds[i, k]] != b[j, k]:
                            ok = False
                            break
                    if ok:
                        o += 1

                    ok = True
                    for k in range(K):
                        if init[inds[i, k]] != b[j, k]:
                            ok = False
                            break
                    if ok:
                        o -= 1

            o_view[l] = o / r / (I * J)
            l += 1

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

'''
corr = np.zeros(self.n_orbit_functions)
corr[0] = 1  # zero point cluster
for orb, inds in self.orbit_inds:
    c_occu = occu[inds]
    for i, bits in enumerate(orb.bit_combos):
        p = []
        for bit in bits:
            for oc in c_occu:
                p1 = 1
                for j, (b, sp) in enumerate(zip(bit, oc)):
                    p1 *= orb.bases_array[j, b, sp]
                p.append(p1)
        corr[i + orb.bit_id] = np.mean(p)
return corr
'''