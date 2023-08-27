import os
from importlib import reload

import pytest

from smol.utils import _openmp_helpers
from smol.utils.cluster import numthreads


class Dummy:
    num_threads = numthreads.SetNumThreads("_multithreaded_dummy")

    def __init__(self, num_threads=None):
        self._multithreaded_dummy = MultithreadedDummy()
        self.num_threads = num_threads


class MultithreadedDummy:
    def __init__(self):
        self.num_threads = 1


def test_openmp_numthreads_env_set(monkeypatch):
    # this is not completely working, it seems that 4 is the default number of threads
    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    reload(_openmp_helpers)
    reload(numthreads)
    dummy = Dummy()

    if _openmp_helpers._openmp_enabled():
        assert _openmp_helpers._openmp_effective_numthreads() == 4
        assert _openmp_helpers._openmp_effective_numthreads(2) == 2
        assert dummy.num_threads == 4
        assert Dummy(num_threads=1).num_threads == 1
        with pytest.warns(UserWarning):
            dummy.num_threads = 5
        assert dummy.num_threads == 4
    else:
        assert _openmp_helpers._openmp_effective_numthreads(2) == 1
        assert _openmp_helpers._openmp_effective_numthreads() == 1
        assert _openmp_helpers._openmp_effective_numthreads(-1) == 1
        assert dummy.num_threads == 1


def test_openmp_numthreads_no_set(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    reload(_openmp_helpers)
    reload(numthreads)
    dummy = Dummy()

    if _openmp_helpers._openmp_enabled():
        assert _openmp_helpers._openmp_effective_numthreads(2) == 2
        assert _openmp_helpers._openmp_effective_numthreads() == min(
            os.cpu_count(), _openmp_helpers._openmp_get_max_threads()
        )
        assert _openmp_helpers._openmp_effective_numthreads(-1) == min(
            os.cpu_count(), _openmp_helpers._openmp_get_max_threads()
        )
        assert (
            _openmp_helpers._openmp_effective_numthreads(-2)
            == min(os.cpu_count(), _openmp_helpers._openmp_get_max_threads()) - 1
        )
        assert dummy.num_threads == 2
        assert Dummy(num_threads=1).num_threads == 1
        with pytest.warns(UserWarning):
            dummy.num_threads = (
                min(os.cpu_count(), _openmp_helpers._openmp_get_max_threads()) + 1
            )
        assert dummy.num_threads == min(
            os.cpu_count(), _openmp_helpers._openmp_get_max_threads()
        )
    else:
        assert _openmp_helpers._openmp_effective_numthreads(2) == 1
        assert _openmp_helpers._openmp_effective_numthreads() == 1
        assert _openmp_helpers._openmp_effective_numthreads(-1) == 1
        assert dummy.num_threads == 1
