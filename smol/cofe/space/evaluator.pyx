"""Evaluator cython extension type to for fast computation of correlation vectors."""

__author__ = "Luis Barroso-Luque"


"""Evaluator cython extension type to for fast computation of correlation vectors."""

__author__ = "Luis Barroso-Luque"

from cython.parallel import prange
import numpy as np
cimport numpy as np
from smol.utils.cluster_utils.container cimport OrbitContainer, IntArray2DContainer
from smol.utils.cluster_utils.struct cimport OrbitC, IntArray2D


cdef class ClusterSpaceEvaluator(OrbitContainer):
    """ClusterSpaceEvaluator class is used to compute the correlation vectors.

    This extenstion type should rarely be used directly. Instead, use the
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
                indices of sites of all clusters in each orbit given as a container
                of arrays.
    
        Returns: array
            correlation vector for given occupancy
        """
        cdef int i, j, m, I, J, M, index, bit_id
        cdef double p
        cdef IntArray2D indices
        cdef OrbitC orbit

        out = np.zeros(num_corr_functions)

        cdef double[:] o_view = out
        o_view[0] = 1  # empty cluster

        for i in range(self.size):#, nogil=True):
            orbit = self.data[i]
            bit_id = orbit.bit_id
            indices = cluster_indices.data[i]
            M = orbit.correlation_tensors.size_r # index of bit combos
            I = indices.size_r # cluster index
            J = indices.size_c # index within cluster
            assert J == orbit.tensor_indices.size

            for m in range(M):
                p = 0
                for i in range(I):
                    index = 0
                    for j in range(J):
                        print(m, j)
                        print(indices.data[i * I + j], i * I + j, indices.size_r, indices.size_c)
                        index = index + orbit.tensor_indices.data[j] * occu[indices.data[i * I + j]]
                    print("setting p")
                    p = p + orbit.correlation_tensors.data[m * M + index]
                print(f"setting o_view with {p / I}")
                o_view[bit_id] = p / I
                bit_id = bit_id + 1

        return out
