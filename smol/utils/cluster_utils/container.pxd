"""Definitions of array and orbitC struct containers as cython extension types.

"""

__author__ = "Luis Barroso-Luque"

from smol.utils.cluster_utils.struct cimport (
    FloatArray1D,
    FloatArray2D,
    IntArray1D,
    IntArray2D,
    OrbitC,
)


cdef class OrbitContainer:
    cdef OrbitC* data
    cdef readonly int size
    cdef readonly int num_correlations

    cpdef public void set_orbits(self, list orbit_list)

    @staticmethod
    cdef OrbitC create_struct(
            int orbit_id,
            int bit_id,
            const double[:, ::1] correlation_tensors,
            const long[::1] tensor_indices,
    )


cdef class FloatArray2DContainer:
    cdef FloatArray2D* data
    cdef readonly int size

    cpdef public void set_arrays(self, tuple arrays)

    @staticmethod
    cdef FloatArray2D create_struct(const double[:, ::1] array)


cdef class FloatArray1DContainer:
    cdef FloatArray1D* data
    cdef readonly int size

    cpdef public void set_arrays(self, tuple arrays)

    @staticmethod
    cdef FloatArray1D create_struct(const double[::1] array)


cdef class IntArray1DContainer:
    cdef IntArray1D* data
    cdef readonly int size

    cpdef public void set_arrays(self, tuple arrays)

    @staticmethod
    cdef IntArray1D create_struct(const long[::1] array)


cdef class IntArray2DContainer:
    cdef IntArray2D* data
    cdef readonly int size

    cpdef public void set_arrays(self, tuple arrays)

    @staticmethod
    cdef IntArray2D create_struct(const long[:, ::1] array)
