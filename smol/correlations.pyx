#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True

"""
General cython utility functions to calculate correlation vectors and local
changes in correlation vectors a tad bit faster than pure python.
"""

__author__ = "Luis Barroso-Luque, William D. Richards"

import numpy as np
cimport numpy as np


cpdef corr_from_occupancy(const long[::1] occu,
                          const int num_corr_functions,
                          list orbit_list):
    """Computes the correlation vector for a given encoded occupancy string.

    Args:
        occu (ndarray):
            encoded occupancy vector
        num_corr_functions (int):
            total number of bit orderings in expansion.
        orbit_list:
            Information of all orbits.
            (orbit id, flat tensor index array, flat correlation tensor,
             site indices of clusters)

    Returns: array
        correlation vector for given occupancy
    """
    cdef int i, j, n, m, I, J, M, index
    cdef double p
    cdef const long[:, ::1] indices
    cdef const long[::1] tensor_indices
    cdef const double[:, ::1] corr_tensors
    out = np.zeros(num_corr_functions)
    cdef double[:] o_view = out
    o_view[0] = 1  # empty cluster

    for n, tensor_indices, corr_tensors, indices in orbit_list:
        M = corr_tensors.shape[0]  # index of bit combos
        I = indices.shape[0] # cluster index
        J = indices.shape[1] # index within cluster
        for m in range(M):
            p = 0
            for i in range(I):
                index = 0
                for j in range(J):
                    index += tensor_indices[j] * occu[indices[i, j]]
                p += corr_tensors[m, index]
            o_view[n] = p / I
            n += 1
    return out


cpdef delta_corr_single_flip(const long[::1] occu_f,
                             const long[::1] occu_i,
                             const int num_corr_functions,
                             list site_orbit_list):
    """Computes the correlation difference between two occupancy vectors.

    Args:
        occu_f (ndarray):
            encoded occupancy vector with flip
        occu_i (ndarray):
            encoded occupancy vector without flip
        num_corr_functions (int):
            total number of bit orderings in expansion.
        site_orbit_list:
            Information of all orbits that include the flip site.
            List of tuples each with
            (orbit id, cluster ratio, flat tensor index array,
             flat correlation tensor, site indices of clusters)


    Returns:
        ndarray: correlation vector difference
    """
    cdef int i, j, n, m, I, J, M, ind_i, ind_f
    cdef double p, ratio
    cdef const long[:, ::1] indices
    cdef const long[::1] tensor_indices
    cdef const double[:, ::1] corr_tensors
    out = np.zeros(num_corr_functions)
    cdef double[::1] o_view = out

    for n, ratio, tensor_indices, corr_tensors, indices in site_orbit_list:
        M = corr_tensors.shape[0] # index of bit combos
        I = indices.shape[0] # cluster index
        J = indices.shape[1] # index within cluster
        for m in range(M):
            p = 0
            for i in range(I):
                ind_i, ind_f = 0, 0
                for j in range(J):
                    ind_i += tensor_indices[j] * occu_i[indices[i, j]]
                    ind_f += tensor_indices[j] * occu_f[indices[i, j]]
                p += (corr_tensors[m, ind_f] - corr_tensors[m, ind_i])
            o_view[n] = p / ratio / I
            n += 1
    return out


cpdef indicator_delta_corr_single_flip(const long[::1] occu_f,
                                       const long[::1] occu_i,
                                       const int num_corr_functions,
                                       list site_orbit_list):
    """Local change in indicator basis correlation vector from single flip.

    Args:
        occu_f (ndarray):
            encoded occupancy vector with flip
        occu_i (ndarray):
            encoded occupancy vector without flip
        num_corr_functions (int):
            total number of bit orderings in expansion.
        site_orbit_list:
            Information of all orbits that include the flip site.
            List of tuples each with
            (orbit id, cluster ratio, bit_combos,
             bit_combo_indices site indices, bases array)

    Returns:
        ndarray: correlation vector difference
    """
    cdef int i, j, k, n, m, I, K, M
    cdef bint ok
    cdef const long[:, ::1] bit_combos, indices
    cdef const long[::1] bit_indices
    out = np.zeros(num_corr_functions)
    cdef double[::1] o_view = out
    cdef double r, o

    for n, r, bit_combos, bit_indices, _, indices in site_orbit_list:
        M = bit_indices.shape[0] # index of bit combos
        I = indices.shape[0] # cluster index
        K = indices.shape[1] # index within cluster
        for m in range(M - 1):
            o = 0
            for i in range(I):
                for j in range(bit_indices[m], bit_indices[m + 1]):
                    ok = True
                    for k in range(K):
                        if occu_f[indices[i, k]] != bit_combos[j, k]:
                            ok = False
                            break
                    if ok:
                        o += 1

                    ok = True
                    for k in range(K):
                        if occu_i[indices[i, k]] != bit_combos[j, k]:
                            ok = False
                            break
                    if ok:
                        o -= 1

            o_view[n] = o / r / (I * (bit_indices[m + 1] - bit_indices[m]))
            n += 1
    return out


cpdef delta_ewald_single_flip(const long[::1] occu_f,
                              const long[::1] occu_i,
                              const double[:, ::1] ewald_matrix,
                              const long[:, ::1] ewald_indices,
                              const int site_ind):
    """Compute the change in electrostatic interaction energy from a flip.

    Args:
        occu_f (ndarray):
            encoded occupancy vector with flip
        occu_i (ndarray):
            encoded occupancy vector without flip
        ewald_matrix (ndarray):
            Ewald matrix for electrostatic interactions
        ewald_indices (ndarray):
            2D array of indices corresponding to a specific site occupation
            in the ewald matrix
        site_ind (int):
            site index for site being flipped

    Returns:
        float: electrostatic interaction energy difference
    """
    cdef int i, j, k, add, sub
    cdef bint ok
    cdef double out = 0
    cdef double out_k

    # values of -1 are vacancies and hence don't have ewald indices
    add = ewald_indices[site_ind, occu_f[site_ind]]
    sub = ewald_indices[site_ind, occu_i[site_ind]]

    for k in range(occu_f.shape[0]):
        i = ewald_indices[k, occu_f[k]]
        out_k = 0
        if i != -1 and add != -1:
            if i != add:
                out_k = out_k + 2 * ewald_matrix[i, add]
            else:
                out_k = out_k + ewald_matrix[i, add]

        j = ewald_indices[k, occu_i[k]]
        if j != -1 and sub != -1:
            if j != sub:
                out_k = out_k - 2 * ewald_matrix[j, sub]
            else:
                out_k = out_k - ewald_matrix[j, sub]
        out += out_k
    return out


cpdef interactions_from_occupancy(const long[::1] occu,
                                  const int num_interactions,
                                  const double offset,
                                  list orbit_list):
    """Computes the cluster interaction vector for a given encoded occupancy string.
    Args:
        occu (ndarray):
            encoded occupancy vector
        num_interactions (int):
            total number of cluster interactions (orbits in cluster subspace).
        offset (float):
            eci value for the constant term.
        orbit_list:
            Information of all orbits.
            (flat tensor index array, flat cluster interaction tensor,
             site indices of clusters)
    Returns: array
        cluster interaction vector for given occupancy
    """
    cdef int n, i, j, I, J, index
    cdef double p
    cdef const long[:, ::1] indices
    cdef const long[::1] tensor_indices
    cdef const double[::1] interaction_tensors
    out = np.zeros(num_interactions)
    cdef double[:] o_view = out
    o_view[0] = offset  # empty cluster

    n = 1
    for tensor_indices, interaction_tensors, indices in orbit_list:
        I = indices.shape[0] # cluster index
        J = indices.shape[1] # index within cluster
        p = 0
        for i in range(I):
            index = 0
            for j in range(J):
                index += tensor_indices[j] * occu[indices[i, j]]
            p += interaction_tensors[index]
        o_view[n] = p / I
        n += 1

    return out


cpdef delta_interactions_single_flip(const long[::1] occu_f,
                                     const long[::1] occu_i,
                                     const int num_interactions,
                                     list site_orbit_list):
    """Computes the cluster interaction vector difference between two occupancy
    strings.
    Args:
        occu_f (ndarray):
            encoded occupancy vector with flip
        occu_i (ndarray):
            encoded occupancy vector without flip
        num_interactions (int):
            total number of cluster interactions (orbits in cluster subspace).
        site_orbit_list:
            Information of all orbits that include the flip site.
            List of tuples each with
            (cluster ratio, flat tensor index array, flat cluster interaction tensor,
             site indices of clusters)
    Returns:
        ndarray: cluster interaction vector difference
    """
    cdef int i, j, n, I, J, ind_i, ind_f
    cdef double p, ratio
    cdef const long[:, ::1] indices
    cdef const long[::1] tensor_indices
    cdef const double[::1] interaction_tensor
    out = np.zeros(num_interactions)
    cdef double[::1] o_view = out

    for n, ratio, tensor_indices, interaction_tensor, indices in site_orbit_list:
        I = indices.shape[0] # cluster index
        J = indices.shape[1] # index within cluster
        p = 0
        for i in range(I):
            ind_i, ind_f = 0, 0
            for j in range(J):
                ind_i += tensor_indices[j] * occu_i[indices[i, j]]
                ind_f += tensor_indices[j] * occu_f[indices[i, j]]
            p += (interaction_tensor[ind_f] - interaction_tensor[ind_i])
        o_view[n] = p / ratio / I
    return out
