"""Definitions of c structs

The structs are used to hold data needed to compute correlation and cluster
interaction vectors.
"""

__author__ = "Luis Barroso-Luque"

cimport numpy as np


cdef struct IntArray1D:
    np.int32_t* data
    int size


cdef struct IntArray2D:
    np.int32_t* data
    int size_r
    int size_c


cdef struct FloatArray1D:
    np.float64_t* data
    int size


cdef struct FloatArray2D:
    np.float64_t* data
    int size_r
    int size_c


cdef struct OrbitC:
    int id
    int bit_id
    IntArray1D tensor_indices
    FloatArray2D correlation_tensors
