"""Simple mixin class that has an attributed/method that runs in parallel."""

from os import cpu_count

DEFAULT_NUM_THREADS = 4 if cpu_count() >= 4 else cpu_count()


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

    def __get__(self, instance, owner):
        """Get the number of threads used by the evaluator to compute correlations."""
        return getattr(getattr(instance, self._obj_name), self._attr_name)

    def __set__(self, instance, value):
        """Set the number of threads used by the evaluator to compute correlations."""
        if value is None:
            value = DEFAULT_NUM_THREADS

        if not isinstance(value, int):
            raise TypeError("num_threads must be an integer")

        if value > cpu_count():
            raise ValueError("num_threads cannot be greater than the number of CPUs")

        obj = getattr(instance, self._obj_name)
        setattr(obj, self._attr_name, value)
