"""Simple mixin class that has an attributed/method that runs in parallel."""


from .._openmp_helpers import _openmp_effective_numthreads

DEFAULT_NUM_THREADS = _openmp_effective_numthreads(n_threads=2)


class SetNumThreads:
    """
    A descriptor used to set threads of an attributed object that has multi-threading.

    Right now it's only used for the Evaluator class.
    """

    def __init__(
        self, multithreaded_object_name: str, thread_attr_name: str = "num_threads"
    ):
        """Define the name of the multithread object and the threads attribute name."""
        # attribute that has multi-threading must have an int attribute
        # with name thread_attr_name
        self._obj_name = multithreaded_object_name
        self._attr_name = thread_attr_name

    def __get__(self, instance, objtype=None):
        """Get the number of threads used by the evaluator to compute correlations."""
        return getattr(getattr(instance, self._obj_name), self._attr_name)

    def __set__(self, instance, value):
        """Set the number of threads used by the evaluator to compute correlations."""
        if value is None:
            value = DEFAULT_NUM_THREADS

        if not isinstance(value, int):
            raise TypeError("num_threads must be an integer")

        max_threads = _openmp_effective_numthreads()
        if value > max_threads:
            raise ValueError(f"num_threads cannot be greater than {max_threads}.")

        obj = getattr(instance, self._obj_name)
        num_threads = _openmp_effective_numthreads(value)
        setattr(obj, self._attr_name, num_threads)
