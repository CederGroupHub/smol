"""Evaluator cython extension type to for fast computation of correlation vectors."""

__author__ = "Luis Barroso-Luque"


import cython
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
    """ClusterSpaceEvaluator is used to compute correlation and interaction vectors.

    This extension type should not be used directly. Instead, use the
    ClusterSubspace class to create a cluster subspace instance and compute correlation
    vectors using the ClusterSubspace.corr_from_occupancy method.
    """

    cpdef np.ndarray[np.float64_t, ndim=1] correlations_from_occupancy(
            self,
            const long[::1] occu,
            IntArray2DContainer cluster_indices,
    ):
        """Computes the correlation vector for a given encoded occupancy string.

        Args:
            occu (ndarray):
                encoded occupancy vector
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

        out = np.zeros(self.num_correlations + 1)
        cdef double[::1] o_view = out
        o_view[0] = 1  # empty cluster

        for n in prange(self.size, nogil=True):  # loop thru orbits
            orbit = self.data[n]
            indices = cluster_indices.data[n]
            bit_id = orbit.bit_id
            K = orbit.correlation_tensors.size_r  # index of bit combos
            J = indices.size_r # cluster index
            I = indices.size_c # index within cluster
            N = orbit.correlation_tensors.size_c # size of single flattened tensor

            for k in range(K):  # loop thru bit combos
                p = 0
                for j in range(J):  # loop thru clusters
                    index = 0
                    for i in range(I):  # loop thru sites in cluster
                        index = index + orbit.tensor_indices.data[i] * occu[indices.data[j * I + i]]
                    # sum contribution of correlation of cluster k with occupancy at "index"
                    p = p + orbit.correlation_tensors.data[k * N + index]
                o_view[bit_id] = p / J
                bit_id = bit_id + 1

        return out

    cpdef np.ndarray[np.float64_t, ndim=1] interactions_from_occupancy(
            self,
            const long[::1] occu,
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
        cdef double[::1] o_view = out
        o_view[0] = offset  # empty cluster

        for n in prange(self.size, nogil=True):
            orbit = self.data[n]
            indices = cluster_indices.data[n]
            interaction_tensor = cluster_interaction_tensors.data[n]
            J = indices.size_r # cluster index
            I = indices.size_c  # index within cluster
            p = 0
            for j in range(J):
                index = 0
                for i in range(I):
                    index = index + orbit.tensor_indices.data[i] * occu[indices.data[j * I + i]]
                p = p + interaction_tensor.data[index]
            o_view[n + 1] = p / J

        return out

@cython.final
cdef class LocalClusterSpaceEvaluator(ClusterSpaceEvaluator):
    """LocalClusterSpaceEvaluator used to compute correlation and interaction vectors.

    This extension type is meant to compute only the correlations or cluster
    interactions that include a specific site.

    Also allows to compute changes in cluster interactions and correlations from
    changes in a single site.

    This extension type should not be used directly. Instead, use corresponding
    Processor classes in smol.moca
    """

    cpdef np.ndarray[np.float64_t, ndim=1] delta_correlations_single_flip(
            self,
            const long[::1] occu_f,
            const long[::1] occu_i,
            IntArray2DContainer cluster_indices,
    ):
        """Computes the correlation difference between two occupancy vectors.

        Args:
            occu_f (ndarray):
                encoded occupancy vector with flip
            occu_i (ndarray):
                encoded occupancy vector without flip
            cluster_indices (IntArray2DContainer):
                Container with pointers to arrays with indices of sites of all clusters
                in each orbit given as a container of arrays.

        Returns:
            ndarray: correlation vector difference
        """
        cdef int i, j, k, n, I, J, K, N, ind_i, ind_f, bit_id
        cdef double p
        cdef IntArray2D indices  # flattened tensor indices
        cdef OrbitC orbit

        out = np.zeros(self.num_correlations + 1)
        cdef double[::1] o_view = out

        for n in prange(self.size, nogil=True):  # loop thru orbits
            orbit = self.data[n]
            indices = cluster_indices.data[n]
            bit_id = orbit.bit_id
            K = orbit.correlation_tensors.size_r  # index of bit combos
            J = indices.size_r # cluster index
            I = indices.size_c # index within cluster
            N = orbit.correlation_tensors.size_c # size of single flattened tensor

            for k in range(K):  # loop thru bit combos
                p = 0
                for j in range(J):  # loop thru clusters
                    ind_i, ind_f = 0, 0
                    for i in range(I):  # loop thru sites in cluster
                        ind_i = ind_i + orbit.tensor_indices.data[i] * occu_i[indices.data[j * I + i]]
                        ind_f = ind_f + orbit.tensor_indices.data[i] * occu_f[indices.data[j * I + i]]
                    # sum contribution of correlation of cluster k with occupancy at "index"
                    p = p + (orbit.correlation_tensors.data[k * N + ind_f] - orbit.correlation_tensors.data[k * N + ind_i])
                o_view[bit_id] = p / orbit.ratio / J
                bit_id = bit_id + 1

        return out

    cpdef np.ndarray[np.float64_t, ndim=1] delta_interactions_single_flip(
            self,
            const long[::1] occu_f,
            const long[::1] occu_i,
            FloatArray1DContainer cluster_interaction_tensors,
            IntArray2DContainer cluster_indices,

    ):
        """Computes the cluster interaction vector difference between two occupancy
        strings.
        Args:
            occu_f (ndarray):
                encoded occupancy vector with flip
            occu_i (ndarray):
                encoded occupancy vector without flip
            cluster_interaction_tensors (IntArray1DContainer):
                Container with pointers to flattened cluster interaction tensors
            cluster_indices (IntArray2DContainer):
                Container with pointers to arrays with indices of sites of all clusters
                in each orbit given as a container of arrays.

        Returns:
            ndarray: cluster interaction vector difference
        """
        cdef int i, j, n, I, J, ind_i, ind_f
        cdef double p
        cdef IntArray2D indices
        cdef OrbitC orbit
        cdef FloatArray1D interaction_tensor

        out = np.zeros(self.size + 1)
        cdef double[::1] o_view = out

        for n in prange(self.size, nogil=True):
            orbit = self.data[n]
            indices = cluster_indices.data[n]
            interaction_tensor = cluster_interaction_tensors.data[n]
            J = indices.size_r # cluster index
            I = indices.size_c # index within cluster
            p = 0
            for j in range(J):
                ind_i, ind_f = 0, 0
                for i in range(I):
                    ind_i = ind_i + orbit.tensor_indices.data[i] * occu_i[indices.data[j * I + i]]
                    ind_f = ind_f + orbit.tensor_indices.data[i] * occu_f[indices.data[j * I + i]]
                p = p + (interaction_tensor.data[ind_f] - interaction_tensor.data[ind_i])
            o_view[n + 1] = p / orbit.ratio / J

        return out
