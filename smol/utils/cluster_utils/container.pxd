__author__ = "Luis Barroso-Luque"

from smol.utils.cluster_utils.struct cimport _FloatArray1D, _FloatArray2D, OrbitC


cdef class OrbitContainer:
    cdef OrbitC* data
    cdef readonly int size

    cpdef public void set_orbits(self, list orbit_list)

    @staticmethod
    cdef OrbitC create_struct(
            int bit_id,
            float ratio,
            const double[:, ::1] correlation_tensors,
            const long[::1] tensor_indices
    )


cdef class FloatArray2DContainer:
    cdef _FloatArray2D* data
    cdef readonly int size

    cpdef public void set_arrays(self, list array_list)

    @staticmethod
    cdef _FloatArray2D create_struct(const double[:, ::1] array)


cdef class FloatArray1DContainer:
    cdef _FloatArray1D* data
    cdef readonly int size

    cpdef public void set_arrays(self, list array_list)

    @staticmethod
    cdef _FloatArray1D create_struct(const double[::1] array)
