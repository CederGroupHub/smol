"""Evaluator cython extension type to for fast computation of correlation vectors."""

__author__ = "Luis Barroso-Luque"

cimport numpy as np
import cython

from smol.utils.cluster_utils.container cimport (
    FloatArray1DContainer,
    IntArray2DContainer,
    OrbitContainer,
)


cdef class ClusterSpaceEvaluator(OrbitContainer):
    cpdef np.ndarray[np.float64_t, ndim=1] corr_from_occupancy(
            self,
            const long[::1] occu,
            const int num_corr_functions,  # TODO save this in the evaluator
            IntArray2DContainer cluster_indices,
    )

    cpdef np.ndarray[np.float64_t, ndim=1] interactions_from_occupancy(
            self,
            const long[::1] occu,
            const double offset,
            FloatArray1DContainer cluster_interaction_tensors,
            IntArray2DContainer cluster_indices,
    )

@cython.final
cdef class LocalClusterSpaceEvaluator(ClusterSpaceEvaluator):
    """LocalClusterSpaceEvaluator used to compute correlation and interaction vectors.

    This extension type is meant to compute only the correlations or cluster interactions
    that include a specific site.

    Also allows to compute changes in cluster interactions and

    This extension type should not be used directly. Instead, use corresponding
    Processor classes in smol.moca
    """

    cpdef np.ndarray[np.float64_t, ndim=1] delta_corr_single_flip(
            self,
            const long[::1] occu_f,
            const long[::1] occu_i,
            const int num_corr_functions,
            IntArray2DContainer cluster_indices)

    cpdef np.ndarray[np.float64_t, ndim=1] delta_interactions_single_flip(
            self,
            const long[::1] occu_f,
            const long[::1] occu_i,
            FloatArray1DContainer cluster_interaction_tensors,
            IntArray2DContainer cluster_indices
    )
