from cpython.mem cimport PyMem_Free, PyMem_Malloc, PyMem_Realloc

from smol.utils.cluster_utils.struct cimport _FloatArray1D, _FloatArray2D

__author__ = "Luis Barroso-Luque"

# TODO make a "template" for all arraycontainers based on tempita

cdef class FloatArray2DContainer:
    def __cinit__(self, list array_list):
        self.size = len(array_list)
        self.data = <_FloatArray2D*> PyMem_Malloc(self.size * sizeof(_FloatArray2D))
        if not self.data:
            raise MemoryError()

        # populate orbits array
        self.set_arrays(array_list)

    cpdef public void set_arrays(self, list array_list):
        """Populated data using a list of 2D arrays."""
        cdef int i

        for array in array_list:
            if array.ndim != 2:
                raise ValueError("All arrays must be 2D.")

        # if different size then reallocate
        if len(array_list) != self.size:
            mem = <_FloatArray2D*> PyMem_Realloc(
                self.data, len(array_list) * sizeof(_FloatArray2D)
            )
            if not mem:
                raise MemoryError()
            self.size = len(array_list)
            self.data = mem

        for i in range(self.size):
            self.data[i] = FloatArray2DContainer.create_struct(array_list[i])

    @staticmethod
    cdef _FloatArray2D create_struct(const double[:, ::1] array):
        """Set the fields of a _FloatArray2D struct from memoryview."""
        cdef _FloatArray2D array_struct
        array_struct.size_r = array.shape[0]
        array_struct.size_c = array.shape[1]
        array_struct.data = &array[0, 0]
        return array_struct

    def __len__(self):
        return self.size

    def __dealloc__(self):
        PyMem_Free(self.data)


cdef class FloatArray1DContainer:
    def __cinit__(self, list array_list):
        self.size = len(array_list)
        self.data = <_FloatArray1D*> PyMem_Malloc(self.size * sizeof(_FloatArray1D))
        if not self.data:
            raise MemoryError()

        # populate orbits array
        self.set_arrays(array_list)

    cpdef public void set_arrays(self, list array_list):
        """Populated data using a list of 1D arrays."""
        cdef int i

        for array in array_list:
            if array.ndim != 1:
                raise ValueError("All arrays must be 1D.")

        # if different size then reallocate
        if len(array_list) != self.size:
            mem = <_FloatArray1D*> PyMem_Realloc(
                self.data, len(array_list) * sizeof(_FloatArray1D)
            )
            if not mem:
                raise MemoryError()
            self.size = len(array_list)
            self.data = mem

        for i in range(self.size):
            self.data[i] = FloatArray1DContainer.create_struct(array_list[i])

    @staticmethod
    cdef _FloatArray1D create_struct(const double[::1] array):
        """Set the fields of a _FloatArray1D struct from memoryview."""
        cdef _FloatArray1D array_struct
        array_struct.size = array.size
        array_struct.data = &array[0]
        return array_struct

    def __len__(self):
        return self.size

    def __dealloc__(self):
        PyMem_Free(self.data)
