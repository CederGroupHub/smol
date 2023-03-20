"""Evaluator cython extension type to for fast computation of correlation vectors."""

__author__ = "Luis Barroso-Luque"


import numpy as np
from cython.parallel import prange

cimport numpy as np

from smol.utils.cluster_utils.container cimport (
    FloatArray1DContainer,
    IntArray2DContainer,
    OrbitContainer,
)
from smol.utils.cluster_utils.struct cimport FloatArray1D, IntArray2D, OrbitC


cdef class ClusterSpaceEvaluator(OrbitContainer):
    """ClusterSpaceEvaluator class is used to compute the correlation vectors.

    This extension type should rarely be used directly. Instead, use the
    ClusterSubspace class to create a cluster subspace instance and compute correlation
    vectors using the ClusterSubspace.corr_from_occupancy method.
    """

    cpdef np.ndarray[np.float64_t, ndim=1] corr_from_occupancy(
            self,
            const long[::1] occu,
            const int num_corr_functions,
            IntArray2DContainer cluster_indices,
    ):
        """Computes the correlation vector for a given encoded occupancy string.

        Args:
            occu (ndarray):
                encoded occupancy vector
            num_corr_functions (int):
                total number of bit orderings in expansion.
            cluster_indices (IntArray1DContainer):
                Container with pointers to arrays with indices of sites of all clusters
                in each orbit given as a container of arrays.

        Returns: array
            correlation vector for given occupancy
        """
        cdef int i, j, k, n, I, J, K, N, index, bit_id
        cdef double p
        cdef IntArray2D indices  # flattened tensor indices
        cdef OrbitC orbit

        out = np.zeros(num_corr_functions)
        cdef double[:] o_view = out
        o_view[0] = 1  # empty cluster

        for n in prange(self.size, nogil=True):  # loop thru orbits
            orbit = self.data[n]
            indices = cluster_indices.data[n]
            bit_id = orbit.bit_id
            I = orbit.correlation_tensors.size_r  # index of bit combos
            J = indices.size_r # cluster index
            K = indices.size_c # index within cluster
            N = orbit.correlation_tensors.size_c # size of single flattened tensor

            for i in range(I):  # loop thru bit combos
                p = 0
                for j in range(J):  # loop thru clusters
                    index = 0
                    for k in range(K):  # loop thru sites in cluster
                        index = index + orbit.tensor_indices.data[k] * occu[indices.data[j * K + k]]
                    # sum contribution of correlation of cluster k with occupancy at "index"
                    p = p + orbit.correlation_tensors.data[i * N + index]
                o_view[bit_id] = p / J
                bit_id = bit_id + 1

        return out

    cpdef np.ndarray[np.float64_t, ndim=1] interactions_from_occupancy(
            self, const long[::1] occu,
            const double offset,
            FloatArray1DContainer cluster_interaction_tensors,
            IntArray2DContainer cluster_indices,
    ):
        """Computes the cluster interaction vector for a given encoded occupancy string.
        Args:
            occu (ndarray):
                encoded occupancy vector
            offset (float):
                eci value for the constant term.
            cluster_interaction_tensors (IntArray1DContainer):
                Container with pointers to flattened cluster interaction tensors
            cluster_indices (IntArray1DContainer):
                Container with pointers to arrays with indices of sites of all clusters
                in each orbit given as a container of arrays.
        Returns: array
            cluster interaction vector for given occupancy
        """
        cdef int n, i, j, I, J, index
        cdef double p
        cdef IntArray2D indices
        cdef OrbitC orbit
        cdef FloatArray1D interaction_tensor

        out = np.zeros(self.size + 1)
        cdef double[:] o_view = out
        o_view[0] = offset  # empty cluster

        for n in prange(self.size, nogil=True):
            orbit = self.data[n]
            indices = cluster_indices.data[n]
            interaction_tensor = cluster_interaction_tensors.data[n]
            I = indices.size_r # cluster index
            J = indices.size_c  # index within cluster
            p = 0
            for i in range(I):
                index = 0
                for j in range(J):
                    index = index + orbit.tensor_indices.data[j] * occu[indices.data[i * J + j]]
                p = p + interaction_tensor.data[index]
            o_view[n + 1] = p / I

        return out
