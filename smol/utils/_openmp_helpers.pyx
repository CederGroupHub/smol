# Helpers to safely access OpenMP routines
#
# no-op implementations are provided for the case where OpenMP is not available.
#
# All calls to OpenMP routines should be cimported from this module.
#
# Code based on sklearn.utils._openmp_helpers

import os

# Module level cache for cpu_count as we do not expect this to change during
# the lifecycle of a Python program.
_CPU_COUNT: int = 1


cdef extern from *:
    """
    #ifdef _OPENMP
        #include <omp.h>
        #define OPENMP_ENABLED 1
    #else
        #define OPENMP_ENABLED 0
        #define omp_lock_t int
        #define omp_init_lock(l) (void)0
        #define omp_destroy_lock(l) (void)0
        #define omp_set_lock(l) (void)0
        #define omp_unset_lock(l) (void)0
        #define omp_get_thread_num() 0
        #define omp_get_max_threads() 1
    #endif
    """
    bint OPENMP_ENABLED

    int omp_get_thread_num() noexcept nogil
    int omp_get_max_threads() noexcept nogil


cpdef _openmp_effective_numthreads(n_threads=None):
    """Determine the effective number of threads to be used for OpenMP calls

    - For ``n_threads = None``,
      - if the ``OMP_NUM_THREADS`` environment variable is set, return
        ``openmp.omp_get_max_threads()``
      - otherwise, default to 2 threads
      The result of ``omp_get_max_threads`` can be influenced by environment
      variable ``OMP_NUM_THREADS`` or at runtime by ``omp_set_num_threads``.

    - For ``n_threads > 0``, return this as the maximal number of threads for
      parallel OpenMP calls.

    - For ``n_threads < 0``, return the maximal number of threads minus
      ``|n_threads + 1|``. In particular ``n_threads = -1`` will use as many
      threads as there are available cores on the machine.

    - Raise a ValueError for ``n_threads = 0``.

    If smol is built without OpenMP support, always return 1.
    """
    if n_threads == 0:
        raise ValueError("n_threads = 0 is invalid")

    if not OPENMP_ENABLED:
        # OpenMP disabled at build-time => sequential mode
        return 1

    if os.getenv("OMP_NUM_THREADS"):
        # Fall back to user provided number of threads making it possible
        # to exceed the number of cpus.
        max_n_threads = omp_get_max_threads()
    else:
        max_n_threads = min(omp_get_max_threads(), os.cpu_count())

    if n_threads is None:
        return max_n_threads
    elif n_threads < 0:
        return max(1, max_n_threads + n_threads + 1)

    return n_threads
