import pytest

from smol.utils.cluster_utils.container import FloatArray2DContainer, FloatArray1DContainer, OrbitContainer


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


def test_orbit_container(rng):
    orbit_list = []
    for i in range(10):
        bit_id = i
        ratio = rng.random()
        correlation_tensors = rng.random(tuple(rng.integers(1, 5, size=2)))
        tensor_indices = rng.integers(1, 5, size=correlation_tensors.shape[0])
        orbit_list.append((bit_id, ratio, correlation_tensors, tensor_indices))

    container = OrbitContainer(orbit_list)
    assert len(container) == 10

    # test setting the same sized list (no memory reallocation)
    new_orbit_list = []
    for i in range(10):
        bit_id = i
        ratio = rng.random()
        correlation_tensors = rng.random(tuple(rng.integers(1, 5, size=2)))
        tensor_indices = rng.integers(1, 5, size=correlation_tensors.shape[0])
        new_orbit_list.append((bit_id, ratio, correlation_tensors, tensor_indices))

    container.set_orbits(new_orbit_list)

    assert len(container) == 10

    # test setting a different sized list (needs memory reallocation)
    new_orbit_list = []
    for i in range(12):
        bit_id = i
        ratio = rng.random()
        correlation_tensors = rng.random(tuple(rng.integers(1, 5, size=2)))
        tensor_indices = rng.integers(1, 5, size=correlation_tensors.shape[0])
        new_orbit_list.append((bit_id, ratio, correlation_tensors, tensor_indices))

    container.set_orbits(new_orbit_list)
    assert len(container) == 12

    # TODO write test for invalid types and shapes
