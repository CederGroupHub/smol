from smol.utils.cluster_utils.struct cimport _FloatArray1D, _FloatArray2D

__author__ = "Luis Barroso-Luque"


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
