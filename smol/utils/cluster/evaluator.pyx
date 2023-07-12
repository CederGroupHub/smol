"""Evaluator cython extension type to for fast computation of correlation vectors."""

__author__ = "Luis Barroso-Luque"


import cython
import numpy as np
from cython.parallel import prange

cimport numpy as np
from libc.math cimport fabs

from smol.utils.cluster.container cimport (
    FloatArray1DContainer,
    IntArray2DContainer,
    OrbitContainer,
)
from smol.utils.cluster.struct cimport FloatArray1D, IntArray2D, OrbitC


@cython.final
cdef class ClusterSpaceEvaluator(OrbitContainer):
    """ClusterSpaceEvaluator is used to compute correlation and interaction vectors.

    This extension type should not be used directly. Instead, use the
    ClusterSubspace class to create a cluster subspace instance and compute correlation
    vectors using the ClusterSubspace.corr_from_structure method.
    """

    def __cinit__(
            self,
            tuple orbit_data,
            int num_orbits,
            int num_corr_functions,
            int num_threads = 1,
            double offset = 0.0,
            tuple cluster_interaction_tensors = None
    ):
        """Initialize ClusterSpaceEvaluator extension type.

        Args:
            orbit_data (tuple):
                Tuple of tuples with orbit data. Each tuple contains the following data:
                    - orbit bit id
                    - orbit tensor indices
                    - orbit correlation tensors
            num_orbits (int):
                Number of orbits.
            num_corr_functions (int):
                Number of correlation functions.
            offset (float):
                interaction value for the constant term (i.e. the grand mean).
            cluster_interaction_tensors (tuple):
                Tuple of ndarrays cluster interaction tensors.
        """
        self.num_orbits = num_orbits
        self.num_corr = num_corr_functions
        self.num_threads = num_threads
        self.offset = offset

        if cluster_interaction_tensors is None:
            cluster_interaction_tensors = tuple(data[2].sum(axis=0) for data in orbit_data)

        self.cluster_interactions = FloatArray1DContainer(cluster_interaction_tensors)

    def __reduce__(self):
        """Return a tuple of the arguments needed to re-initialize the object."""
        return (
            ClusterSpaceEvaluator,
            (
                self._orbit_data,
                self.num_orbits,
                self.num_corr,
                self.num_threads,
                self.offset,
                self.cluster_interactions._arrays,
            ),
        )

    cpdef public void reset_data(
            self,
            tuple orbit_data,
            int num_orbits,
            int num_corr_functions,
    ):
        """Reset data of ClusterSpaceEvaluator extension type.

        Args:
            orbit_data (tuple):
                Tuple of tuples with orbit data. Each tuple contains the following data:
                    - orbit bit id
                    - orbit tensor indices
                    - orbit correlation tensors
            num_orbits (int):
                Number of orbits.
            num_corr_functions (int):
                Number of correlation functions.
        """
        self.num_orbits = num_orbits
        self.num_corr = num_corr_functions
        self.set_orbits(orbit_data)

    cpdef public void set_cluster_interactions(
            self, tuple cluster_interaction_tensors, double offset
    ):
        """Sets the cluster interaction tensors.

        Args:
            cluster_interaction_tensors (tuple):
                Tuple of ndarrays cluster interaction tensors.
            offset (float):
                interaction value for the constant term (i.e. the grand mean).
        """
        if len(cluster_interaction_tensors) != self.size:
            raise ValueError(
                "Number of cluster interaction tensors must be equal to the number of orbits."
            )
        self.cluster_interactions.set_arrays(cluster_interaction_tensors)
        self.offset = offset

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

        out = np.zeros(self.num_corr)
        cdef double[::1] o_view = out
        o_view[0] = 1  # empty cluster

        # loop thru orbits
        for n in prange(self.size, nogil=True, schedule="guided", num_threads=self.num_threads):
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
            IntArray2DContainer cluster_indices,
    ):
        """Computes the cluster interaction vector for a given encoded occupancy string.
        Args:
            occu (ndarray):
                encoded occupancy vector
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

        out = np.zeros(self.num_orbits)
        cdef double[::1] o_view = out
        o_view[0] = self.offset  # empty cluster

        for n in prange(self.size, nogil=True, schedule="guided", num_threads=self.num_threads):
            orbit = self.data[n]
            indices = cluster_indices.data[n]
            interaction_tensor = self.cluster_interactions.data[n]
            J = indices.size_r # cluster index
            I = indices.size_c  # index within cluster
            p = 0
            for j in range(J):
                index = 0
                for i in range(I):
                    index = index + orbit.tensor_indices.data[i] * occu[indices.data[j * I + i]]
                p = p + interaction_tensor.data[index]
            o_view[orbit.id] = p / J

        return out

    cpdef np.ndarray[np.float64_t, ndim=1] delta_correlations_from_occupancies(
            self,
            const long[::1] occu_f,
            const long[::1] occu_i,
            const double[::1] cluster_ratio,
            IntArray2DContainer cluster_indices,
    ):
        """Computes the correlation difference between two occupancy vectors.

        Args:
            occu_f (ndarray):
                encoded occupancy vector with flip
            occu_i (ndarray):
                encoded occupancy vector without flip
            cluster_ratio (ndarray):
                ratio of number of clusters in each entry of cluster_indices to the total
                number of the clusters in a structure for the corresponding cluster.
            cluster_indices (IntArray2DContainer):
                Container with pointers to arrays with indices of sites of all clusters
                in each orbit given as a container of arrays.

        Returns: ndarray
            correlation vector difference
        """
        cdef int i, j, k, n, I, J, K, N, ind_i, ind_f, bit_id
        cdef double p
        cdef IntArray2D indices  # flattened tensor indices
        cdef OrbitC orbit

        out = np.zeros(self.num_corr)
        cdef double[::1] o_view = out

        # loop thru orbits
        for n in prange(self.size, nogil=True, num_threads=self.num_threads):
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
                o_view[bit_id] = p / cluster_ratio[n] / J
                bit_id = bit_id + 1

        return out

    cpdef np.ndarray[np.float64_t, ndim=1] delta_interactions_from_occupancies(
            self,
            const long[::1] occu_f,
            const long[::1] occu_i,
            const double[::1] cluster_ratio,
            IntArray2DContainer cluster_indices,

    ):
        """Computes the cluster interaction vector difference between two occupancy
        strings.

        Args:
            occu_f (ndarray):
                encoded occupancy vector with flip
            occu_i (ndarray):
                encoded occupancy vector without flip
            cluster_ratio (ndarray):
                ratio of number of clusters in each entry of cluster_indices to the total
                number of the clusters in a structure for the corresponding cluster.
            cluster_indices (IntArray2DContainer):
                Container with pointers to arrays with indices of sites of all clusters
                in each orbit given as a container of arrays.

        Returns: ndarray
            cluster interaction vector difference
        """
        cdef int i, j, n, I, J, ind_i, ind_f
        cdef double p
        cdef IntArray2D indices
        cdef OrbitC orbit
        cdef FloatArray1D interaction_tensor

        out = np.zeros(self.num_orbits)
        cdef double[::1] o_view = out

        for n in prange(self.size, nogil=True, num_threads=self.num_threads):
            orbit = self.data[n]
            indices = cluster_indices.data[n]
            interaction_tensor = self.cluster_interactions.data[n]
            J = indices.size_r # cluster index
            I = indices.size_c # index within cluster
            p = 0
            for j in range(J):
                ind_i, ind_f = 0, 0
                for i in range(I):
                    ind_i = ind_i + orbit.tensor_indices.data[i] * occu_i[indices.data[j * I + i]]
                    ind_f = ind_f + orbit.tensor_indices.data[i] * occu_f[indices.data[j * I + i]]
                p = p + (interaction_tensor.data[ind_f] - interaction_tensor.data[ind_i])
            o_view[orbit.id] = p / cluster_ratio[n] / J

        return out

    cpdef np.ndarray[np.float64_t, ndim=1] corr_distances_from_occupancies(
            self,
            const long[::1] occu_f,
            const long[::1] occu_i,
            const double[::1] ref_corr_vector,
            IntArray2DContainer cluster_indices
    ):
        """Computes the absolute distance of two correlation vectors separated by a single
        flip and a given correlation vector.

        Unfortunately this scales just as bad as computing the full correlation vector.

        Args:
            occu_f (ndarray):
                encoded occupancy array with flip
            occu_i (ndarray):
                encoded occupancy array without flip
            ref_corr_vector (ndarray):
                reference correlation vector
            cluster_indices (IntArray2DContainer):
                Container with pointers to arrays with indices of sites of all clusters
                in each orbit given as a container of arrays.

        Returns:
            ndarray: 2D with correlation vector distances from reference for each of occu_i
            and occu_f
        """
        cdef int i, j, k, n, I, J, K, N, bit_id, ind_i, ind_f
        cdef double p_i, p_f
        cdef IntArray2D indices  # flattened tensor indices
        cdef OrbitC orbit

        out = np.zeros((2, self.num_corr))
        cdef double[:, ::1] o_view = out
        o_view[:, 0] = 0

        # loop thru orbits
        for n in prange(self.size, nogil=True, schedule="guided", num_threads=self.num_threads):
            orbit = self.data[n]
            indices = cluster_indices.data[n]
            bit_id = orbit.bit_id
            K = orbit.correlation_tensors.size_r  # index of bit combos
            J = indices.size_r # cluster index
            I = indices.size_c # index within cluster
            N = orbit.correlation_tensors.size_c # size of single flattened tensor

            for k in range(K):  # loop thru bit combos
                p_f, p_i = 0, 0
                for j in range(J):  # loop thru clusters
                    ind_f, ind_i = 0, 0
                    for i in range(I):  # loop thru sites in cluster
                        ind_f = ind_f + orbit.tensor_indices.data[i] * occu_f[indices.data[j * I + i]]
                        ind_i = ind_i + orbit.tensor_indices.data[i] * occu_i[indices.data[j * I + i]]
                    # sum contribution of correlation of cluster k with occupancy at "ind"
                    p_f = p_f + orbit.correlation_tensors.data[k * N + ind_f]
                    p_i = p_i + orbit.correlation_tensors.data[k * N + ind_i]
                o_view[1, bit_id] = fabs(p_f / J - ref_corr_vector[bit_id])
                o_view[0, bit_id] = fabs(p_i / J - ref_corr_vector[bit_id])
                bit_id = bit_id + 1

        return out

    cpdef np.ndarray[np.float64_t, ndim=1] interaction_distances_from_occupancies(
            self,
            const long[::1] occu_f,
            const long[::1] occu_i,
            const double[::1] ref_interaction_vector,
            IntArray2DContainer cluster_indices
    ):
        """Computes the absolute distance of two cluster interaction vectors separated by a
            single flip and a given correlation vector.

            Unfortunately this scales just as bad as computing the full interaction vector.

            Args:
                occu_f (ndarray):
                    encoded occupancy array with flip
                occu_i (ndarray):
                    encoded occupancy array without flip
                ref_interaction_vector (ndarray):
                    reference cluster interaction vector
                cluster_indices (IntArray2DContainer):
                    Container with pointers to arrays with indices of sites of all clusters
                    in each orbit given as a container of arrays.

            Returns:
                ndarray: 2D with cluster interaction vector distances from reference for each of
                occu_i and occu_f
            """
        cdef int n, i, j, I, J, ind_i, ind_f
        cdef double p_i, p_f
        cdef IntArray2D indices
        cdef OrbitC orbit
        cdef FloatArray1D interaction_tensor

        out = np.zeros((2, self.num_orbits))
        cdef double[:, ::1] o_view = out
        o_view[:, 0] = 0

        for n in prange(self.size, nogil=True, schedule="guided", num_threads=self.num_threads):
            orbit = self.data[n]
            indices = cluster_indices.data[n]
            interaction_tensor = self.cluster_interactions.data[n]
            J = indices.size_r # cluster index
            I = indices.size_c  # index within cluster
            p_f, p_i = 0, 0
            for j in range(J):
                ind_f, ind_i = 0, 0
                for i in range(I):
                    ind_f = ind_f + orbit.tensor_indices.data[i] * occu_f[indices.data[j * I + i]]
                    ind_i = ind_i + orbit.tensor_indices.data[i] * occu_i[indices.data[j * I + i]]
                p_f = p_f + interaction_tensor.data[ind_f]
                p_i = p_i + interaction_tensor.data[ind_i]
            o_view[1, orbit.id] = fabs(p_f / J - ref_interaction_vector[orbit.id])
            o_view[0, orbit.id] = fabs(p_i / J - ref_interaction_vector[orbit.id])

        return out
