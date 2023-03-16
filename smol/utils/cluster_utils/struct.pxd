"""Definitions of c structs

The structs are used to hold data needed to compute correlation and cluster
interaction vectors.
"""

__author__ = "Luis Barroso-Luque"


cdef struct _IntArray1D:
    long* data
    int size


cdef struct _IntArray2D:
    long* data
    int size_r
    int size_c


cdef struct _FloatArray1D:
    double* data
    int size


cdef struct _FloatArray2D:
    double* data
    int size_r
    int size_c


cdef struct OrbitC:
    int bit_id
    float ratio
    _IntArray1D tensor_indices
    _FloatArray2D correlation_tensors
