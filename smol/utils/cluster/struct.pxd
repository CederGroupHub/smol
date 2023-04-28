"""Definitions of c structs

The structs are used to hold data needed to compute correlation and cluster
interaction vectors.
"""

__author__ = "Luis Barroso-Luque"


cdef struct IntArray1D:
    long* data
    int size


cdef struct IntArray2D:
    long* data
    int size_r
    int size_c


cdef struct FloatArray1D:
    double* data
    int size


cdef struct FloatArray2D:
    double* data
    int size_r
    int size_c


cdef struct OrbitC:
    int id
    int bit_id
    IntArray1D tensor_indices
    FloatArray2D correlation_tensors
