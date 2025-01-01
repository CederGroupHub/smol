"""Definitions of c structs

The structs are used to hold data needed to compute correlation and cluster
interaction vectors.
"""

__author__ = "Luis Barroso-Luque"

cimport numpy as np


cdef struct IntArray1D:
    np.int32_t* data
    np.int32_t size


cdef struct IntArray2D:
    np.int32_t* data
    np.int32_t size_r
    np.int32_t size_c


cdef struct FloatArray1D:
    np.float64_t* data
    np.int32_t size


cdef struct FloatArray2D:
    np.float64_t* data
    np.int32_t size_r
    np.int32_t size_c


cdef struct OrbitC:
    np.float64_t id
    np.int32_t bit_id
    IntArray1D tensor_indices
    FloatArray2D correlation_tensors
