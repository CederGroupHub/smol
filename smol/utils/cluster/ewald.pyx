"""Cython function to compute the change in electrostatic energy from a flips."""

__author__ = "William D. Richards"


cpdef double delta_ewald_single_flip(const long[::1] occu_f,
                                     const long[::1] occu_i,
                                     const double[:, ::1] ewald_matrix,
                                     const long[:, ::1] ewald_indices,
                                     const int site_ind) nogil:
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

    # single threaded seems good enough for the majority of systems
    # if multiple threads are ever needed this can be replaced with prange
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
