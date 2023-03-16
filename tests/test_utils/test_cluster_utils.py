import pytest

from smol.utils.cluster_utils.container import FloatArray2DContainer, FloatArray1DContainer


@pytest.mark.parametrize("FloatContainer, dim", [(FloatArray1DContainer, 1), (FloatArray2DContainer, 2)])
def test_float_container(FloatContainer, dim, rng):
    array_list = [rng.random(tuple(rng.integers(1, 5, size=dim))) for _ in range(10)]
    container = FloatContainer(array_list)
    assert len(container) == 10

    # test setting the same sized list (no memory reallocation)
    new_array_list = [rng.random(tuple(rng.integers(1, 5, size=dim))) for _ in range(10)]
    container.set_arrays(new_array_list)

    assert len(container) == 10

    # test setting a different sized list (needs memory reallocation)
    new_array_list = [rng.random(tuple(rng.integers(1, 5, size=dim))) for _ in range(12)]
    container.set_arrays(new_array_list)
    assert len(container) == 12

    # TODO figure out why exception raised in cython is not captured by pytest
    #with pytest.raises(Exception):
    #    array_list = [rng.random(tuple(rng.integers(1, 5, size=dim + 1))) for _ in range(10)]
    #    container.set_arrays(array_list)
