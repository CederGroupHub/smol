"""Definitions of array and orbitC struct containers as cython extension types.

"""

__author__ = "Luis Barroso-Luque"

import cython
from smol.utils.cluster.struct cimport (
    FloatArray1D,
    FloatArray2D,
    IntArray1D,
    IntArray2D,
    OrbitC,
)


cdef class OrbitContainer:
    cdef OrbitC* data
    cdef readonly int size
    cdef tuple _orbit_data

    cpdef public void set_orbits(self, tuple orbit_data) except *

    @staticmethod
    cdef OrbitC create_struct(
            int orbit_id,
            int bit_id,
            double[:, ::1] correlation_tensors,
            long[::1] tensor_indices,
    )


@cython.final
cdef class FloatArray2DContainer:
    cdef FloatArray2D* data
    cdef readonly int size
    cdef tuple _arrays

    cpdef public void set_arrays(self, tuple arrays) except *

    @staticmethod
    cdef FloatArray2D create_struct(double[:, ::1] array)


@cython.final
cdef class FloatArray1DContainer:
    cdef FloatArray1D* data
    cdef readonly int size
    cdef tuple _arrays

    cpdef public void set_arrays(self, tuple arrays)  except *

    @staticmethod
    cdef FloatArray1D create_struct(double[::1] array)


@cython.final
cdef class IntArray1DContainer:
    cdef IntArray1D* data
    cdef readonly int size
    cdef tuple _arrays

    cpdef public void set_arrays(self, tuple arrays)  except *

    @staticmethod
    cdef IntArray1D create_struct(long[::1] array)


@cython.final
cdef class IntArray2DContainer:
    cdef IntArray2D* data
    cdef readonly int size
    cdef tuple _arrays

    cpdef public void set_arrays(self, tuple arrays)  except *

    @staticmethod
    cdef IntArray2D create_struct(long[:, ::1] array)
