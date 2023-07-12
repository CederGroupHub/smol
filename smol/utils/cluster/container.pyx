"""Container extension classes that hold C arrays of structs to ndarray pointers."""

__author__ = "Luis Barroso-Luque"


cimport cython
cimport numpy as np
from cpython.mem cimport PyMem_Free, PyMem_Malloc, PyMem_Realloc

from smol.utils.cluster.struct cimport (
    FloatArray1D,
    FloatArray2D,
    IntArray1D,
    IntArray2D,
    OrbitC,
)

# Is it possible to "template" for all array containers based on tempita


cdef class OrbitContainer:
    def __init__(self, tuple orbit_data, *args):
        """Python initialization

        Args:
            orbit_data (tuple):
                The orbit data should be given as a list of tuples where is tuple has
                the following information for the corresponding orbit:
                (orbit id, orbit bit_id, flattened correlation tensors, tensor indices)
        """
        # keep a python reference to the orbit_list so that it is not garbage collected
        self._orbit_data = orbit_data

    def __cinit__(self, tuple orbit_data, *args):
        self.size = len(orbit_data)
        self.data = <OrbitC*> PyMem_Malloc(self.size * sizeof(OrbitC))
        if not self.data:
            raise MemoryError()

        # populate orbits array
        self.set_orbits(orbit_data)

    def __reduce__(self):
        """Return a tuple of the arguments needed to re-initialize the object."""
        return OrbitContainer, (self._orbit_data,)

    cpdef public void set_orbits(self, tuple orbit_data) except *:
        """Populated data using a list of orbit data."""
        cdef int i

        # check that dataypes are correct
        for data in orbit_data:
            if not isinstance(data[0], int):
                raise TypeError("id must be an integer.")
            if not isinstance(data[1], int):
                raise TypeError("bit_id must be an integer.")
            if not isinstance(data[2], np.ndarray):
                raise TypeError("correlation_tensors must be a numpy array.")
            if not isinstance(data[3], np.ndarray):
                raise TypeError("tensor_indices must be a numpy array.")
            if data[2].ndim != 2:
                raise ValueError("correlation_tensors must be 2D.")
            if data[3].ndim != 1:
                raise ValueError("tensor_indices must be 1D.")

        # if different size then reallocate
        if len(orbit_data) != self.size:
            mem = <OrbitC*> PyMem_Realloc(
                self.data, len(orbit_data) * sizeof(OrbitC)
            )
            if not mem:
                raise MemoryError()

            self.size = len(orbit_data)
            self.data = mem

        for i, data in enumerate(orbit_data):
            self.data[i] = OrbitContainer.create_struct(
                data[0], data[1], data[2], data[3]
            )

        self._orbit_data = orbit_data

    @staticmethod
    cdef OrbitC create_struct(
            int orbit_id,
            int bit_id,
            double[:, ::1] correlation_tensors,
            long[::1] tensor_indices,
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
    def __init__(self, tuple arrays):
        # keep a python reference to the arrays so that it is not garbage collected
        self._arrays = arrays

    def __cinit__(self, tuple arrays):
        self.size = len(arrays)
        self.data = <FloatArray2D*> PyMem_Malloc(self.size * sizeof(FloatArray2D))
        if not self.data:
            raise MemoryError()

        # populate orbits array
        self.set_arrays(arrays)

    def __reduce__(self):
        """Return a tuple of the arguments needed to re-initialize the object."""
        return FloatArray2DContainer, (self._arrays,)

    cpdef public void set_arrays(self, tuple arrays) except *:
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

        self._arrays = arrays


    @staticmethod
    cdef FloatArray2D create_struct(double[:, ::1] array):
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
    def __init__(self, tuple arrays):
        # keep a python reference to the arrays so that it is not garbage collected
        self._arrays = arrays

    def __cinit__(self, tuple arrays):
        self.size = len(arrays)
        self.data = <FloatArray1D*> PyMem_Malloc(self.size * sizeof(FloatArray1D))
        if not self.data:
            raise MemoryError()

        # populate orbits array
        self.set_arrays(arrays)

    def __reduce__(self):
        """Return a tuple of the arguments needed to re-initialize the object."""
        return FloatArray1DContainer, (self._arrays,)

    cpdef public void set_arrays(self, tuple arrays) except *:
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

        self._arrays = arrays

    @staticmethod
    cdef FloatArray1D create_struct(double[::1] array):
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
    def __init__(self, tuple arrays):
        # keep a python reference to the arrays so that it is not garbage collected
        self._arrays = arrays

    def __cinit__(self, tuple arrays):
        self.size = len(arrays)
        self.data = <IntArray1D*> PyMem_Malloc(self.size * sizeof(IntArray1D))
        if not self.data:
            raise MemoryError()

        # populate orbits array
        self.set_arrays(arrays)

    def __reduce__(self):
        """Return a tuple of the arguments needed to re-initialize the object."""
        return IntArray1DContainer, (self._arrays,)

    cpdef public void set_arrays(self, tuple arrays) except *:
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

        self._arrays = arrays

    @staticmethod
    cdef IntArray1D create_struct(long[::1] array):
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
    def __init__(self, tuple arrays):
        # keep a python reference to the arrays so that it is not garbage collected
        self._arrays = arrays

    def __cinit__(self, tuple arrays):
        self.size = len(arrays)
        self.data = <IntArray2D*> PyMem_Malloc(self.size * sizeof(IntArray2D))
        if not self.data:
            raise MemoryError()

        # populate orbits array
        self.set_arrays(arrays)

    def __reduce__(self):
        """Return a tuple of the arguments needed to re-initialize the object."""
        return IntArray2DContainer, (self._arrays,)

    cpdef public void set_arrays(self, tuple arrays)  except *:
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

        self._arrays = arrays

    @staticmethod
    cdef IntArray2D create_struct(long[:, ::1] array):
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
