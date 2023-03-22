"""Container extension classes that hold C arrays of structs to ndarray pointers."""

__author__ = "Luis Barroso-Luque"


cimport cython
cimport numpy as np
from cpython.mem cimport PyMem_Free, PyMem_Malloc, PyMem_Realloc

from smol.utils.cluster_utils.struct cimport (
    FloatArray1D,
    FloatArray2D,
    IntArray1D,
    IntArray2D,
    OrbitC,
)

# Is it possible to "template" for all arraycontainers based on tempita


cdef class OrbitContainer:
    def __cinit__(self, list orbit_list):
        self.size = len(orbit_list)
        self.data = <OrbitC*> PyMem_Malloc(self.size * sizeof(OrbitC))
        if not self.data:
            raise MemoryError()

        # populate orbits array
        self.set_orbits(orbit_list)

    cpdef public void set_orbits(self, list orbit_list):
        """Populated data using a list of orbit data.

        The orbit data should be given as a list of tuples where is tuple has the
        following information for the corresponding orbit:
        (orbit id, orbit bit_id, flattened correlation tensors, tensor indices)
        """
        cdef int i, num_corr

        # check that dataypes are correct
        for orbit_data in orbit_list:
            if not isinstance(orbit_data[0], int):
                raise TypeError("id must be an integer.")
            if not isinstance(orbit_data[1], int):
                raise TypeError("bit_id must be an integer.")
            if not isinstance(orbit_data[2], np.ndarray):
                raise TypeError("correlation_tensors must be a numpy array.")
            if not isinstance(orbit_data[3], np.ndarray):
                raise TypeError("tensor_indices must be a numpy array.")
            if orbit_data[2].ndim != 2:
                raise ValueError("correlation_tensors must be 2D.")
            if orbit_data[3].ndim != 1:
                raise ValueError("tensor_indices must be 1D.")

        # if different size then reallocate
        if len(orbit_list) != self.size:
            mem = <OrbitC*> PyMem_Realloc(
                self.data, len(orbit_list) * sizeof(OrbitC)
            )
            if not mem:
                raise MemoryError()
            self.size = len(orbit_list)
            self.data = mem

        num_corr = 0
        for i in range(self.size):
            num_corr += orbit_list[i][2].shape[0]
            self.data[i] = OrbitContainer.create_struct(
                orbit_list[i][0],
                orbit_list[i][1],
                orbit_list[i][2],
                orbit_list[i][3],
            )
        self.num_correlations = num_corr

    @staticmethod
    cdef OrbitC create_struct(
            int orbit_id,
            int bit_id,
            const double[:, ::1] correlation_tensors,
            const long[::1] tensor_indices,
    ):
        """Set the fields of a OrbitC struct from memoryviews."""
        cdef OrbitC orbit

        orbit.id = orbit_id
        orbit.bit_id = bit_id

        orbit.correlation_tensors.size_r = correlation_tensors.shape[0]
        orbit.correlation_tensors.size_c = correlation_tensors.shape[1]
        orbit.correlation_tensors.data = &correlation_tensors[0, 0]

        orbit.tensor_indices.size = tensor_indices.shape[0]
        orbit.tensor_indices.data = &tensor_indices[0]

        return orbit

    def __len__(self):
        return self.size

    def __dealloc__(self):
        PyMem_Free(self.data)


@cython.final
cdef class FloatArray2DContainer:
    def __cinit__(self, tuple arrays):
        self.size = len(arrays)
        self.data = <FloatArray2D*> PyMem_Malloc(self.size * sizeof(FloatArray2D))
        if not self.data:
            raise MemoryError()

        # populate orbits array
        self.set_arrays(arrays)

    cpdef public void set_arrays(self, tuple arrays):
        """Populated data using a list of 2D arrays."""
        cdef int i

        for array in arrays:
            if array.ndim != 2:
                raise ValueError("All arrays must be 2D.")

        # if different size then reallocate
        if len(arrays) != self.size:
            mem = <FloatArray2D*> PyMem_Realloc(
                self.data, len(arrays) * sizeof(FloatArray2D)
            )
            if not mem:
                raise MemoryError()
            self.size = len(arrays)
            self.data = mem

        for i in range(self.size):
            self.data[i] = FloatArray2DContainer.create_struct(arrays[i])


    @staticmethod
    cdef FloatArray2D create_struct(const double[:, ::1] array):
        """Set the fields of a _FloatArray2D struct from memoryview."""
        cdef FloatArray2D array_struct
        array_struct.size_r = array.shape[0]
        array_struct.size_c = array.shape[1]
        array_struct.data = &array[0, 0]
        return array_struct

    def __len__(self):
        return self.size

    def __dealloc__(self):
        PyMem_Free(self.data)


@cython.final
cdef class FloatArray1DContainer:
    def __cinit__(self, tuple arrays):
        self.size = len(arrays)
        self.data = <FloatArray1D*> PyMem_Malloc(self.size * sizeof(FloatArray1D))
        if not self.data:
            raise MemoryError()

        # populate orbits array
        self.set_arrays(arrays)

    cpdef public void set_arrays(self, tuple arrays):
        """Populated data using a list of 1D arrays."""
        cdef int i

        for array in arrays:
            if array.ndim != 1:
                raise ValueError("All arrays must be 1D.")

        # if different size then reallocate
        if len(arrays) != self.size:
            mem = <FloatArray1D*> PyMem_Realloc(
                self.data, len(arrays) * sizeof(FloatArray1D)
            )
            if not mem:
                raise MemoryError()
            self.size = len(arrays)
            self.data = mem

        for i in range(self.size):
            self.data[i] = FloatArray1DContainer.create_struct(arrays[i])

    @staticmethod
    cdef FloatArray1D create_struct(const double[::1] array):
        """Set the fields of a FloatArray1D struct from memoryview."""
        cdef FloatArray1D array_struct
        array_struct.size = array.size
        array_struct.data = &array[0]
        return array_struct

    def __len__(self):
        return self.size

    def __dealloc__(self):
        PyMem_Free(self.data)


@cython.final
cdef class IntArray1DContainer:
    def __cinit__(self, tuple arrays):
        self.size = len(arrays)
        self.data = <IntArray1D*> PyMem_Malloc(self.size * sizeof(IntArray1D))
        if not self.data:
            raise MemoryError()

        # populate orbits array
        self.set_arrays(arrays)

    cpdef public void set_arrays(self, tuple arrays):
        """Populated data using a list of 1D arrays."""
        cdef int i

        for array in arrays:
            if array.ndim != 1:
                raise ValueError("All arrays must be 1D.")

        # if different size then reallocate
        if len(arrays) != self.size:
            mem = <IntArray1D*> PyMem_Realloc(
                self.data, len(arrays) * sizeof(IntArray1D)
            )
            if not mem:
                raise MemoryError()
            self.size = len(arrays)
            self.data = mem

        for i in range(self.size):
            self.data[i] = IntArray1DContainer.create_struct(arrays[i])

    @staticmethod
    cdef IntArray1D create_struct(const long[::1] array):
        """Set the fields of a _FloatArray1D struct from memoryview."""
        cdef IntArray1D array_struct
        array_struct.size = array.size
        array_struct.data = &array[0]
        return array_struct

    def __len__(self):
        return self.size

    def __dealloc__(self):
        PyMem_Free(self.data)


@cython.final
cdef class IntArray2DContainer:
    def __cinit__(self, tuple arrays):
        self.size = len(arrays)
        self.data = <IntArray2D*> PyMem_Malloc(self.size * sizeof(IntArray2D))
        if not self.data:
            raise MemoryError()

        # populate orbits array
        self.set_arrays(arrays)

    def print_contents(self):
        #cdef long val
        for i in range(self.size):
            for j in range(self.data[i].size_r):
                for k in range(self.data[i].size_c):
                    print(self.data[i].data[j * self.data[i].size_c + k])

    cpdef public void set_arrays(self, tuple arrays):
        """Populated data using a list of 2D arrays."""
        cdef int i

        for array in arrays:
            if array.ndim != 2:
                raise ValueError("All arrays must be 2D.")

        # if different size then reallocate
        if len(arrays) != self.size:
            mem = <IntArray2D*> PyMem_Realloc(
                self.data, len(arrays) * sizeof(IntArray2D)
            )
            if not mem:
                raise MemoryError()
            self.size = len(arrays)
            self.data = mem

        for i in range(self.size):
            self.data[i] = IntArray2DContainer.create_struct(arrays[i])

    @staticmethod
    cdef IntArray2D create_struct(const long[:, ::1] array):
        """Set the fields of a _IntArray2D struct from memoryview."""
        cdef IntArray2D array_struct
        array_struct.size_r = array.shape[0]
        array_struct.size_c = array.shape[1]
        array_struct.data = &array[0, 0]
        return array_struct

    def __len__(self):
        return self.size

    def __dealloc__(self):
        PyMem_Free(self.data)
