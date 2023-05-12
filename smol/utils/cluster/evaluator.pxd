"""Evaluator cython extension type to for fast computation of correlation vectors."""

__author__ = "Luis Barroso-Luque"

cimport numpy as np

import cython

from smol.utils.cluster.container cimport (
    FloatArray1DContainer,
    IntArray2DContainer,
    OrbitContainer,
)


@cython.final
cdef class ClusterSpaceEvaluator(OrbitContainer):
    cdef int num_corr  # total number of correlation functions
    cdef int num_orbits  # total number of orbits
    cdef public double offset  # offset for the interaction tensor
    cdef FloatArray1DContainer cluster_interactions
    cdef public int num_threads

    cpdef public void reset_data(
            self,
            tuple orbit_data,
            int num_orbits,
            int num_corr_functions,
    )

    cpdef public void set_cluster_interactions(
            self, tuple cluster_interaction_tensors, double offset
    )

    cpdef np.ndarray[np.float64_t, ndim=1] correlations_from_occupancy(
            self,
            const long[::1] occu,
            IntArray2DContainer cluster_indices,
    )

    cpdef np.ndarray[np.float64_t, ndim=1] interactions_from_occupancy(
            self,
            const long[::1] occu,
            IntArray2DContainer cluster_indices,
    )

    cpdef np.ndarray[np.float64_t, ndim=1] delta_correlations_from_occupancies(
            self,
            const long[::1] occu_f,
            const long[::1] occu_i,
            const double[::1] cluster_ratio,
            IntArray2DContainer cluster_indices)

    cpdef np.ndarray[np.float64_t, ndim=1] delta_interactions_from_occupancies(
            self,
            const long[::1] occu_f,
            const long[::1] occu_i,
            const double[::1] cluster_ratio,
            IntArray2DContainer cluster_indices
    )

    cpdef np.ndarray[np.float64_t, ndim=1] corr_distances_from_occupancies(
            self,
            const long[::1] occu_f,
            const long[::1] occu_i,
            const double[::1] ref_corr_vector,
            IntArray2DContainer cluster_indices
    )

    cpdef np.ndarray[np.float64_t, ndim=1] interaction_distances_from_occupancies(
            self,
            const long[::1] occu_f,
            const long[::1] occu_i,
            const double[::1] ref_interaction_vector,
            IntArray2DContainer cluster_indices
    )
