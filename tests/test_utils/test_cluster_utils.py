import numpy as np
import pytest

from smol.utils.cluster.container import (
    FloatArray1DContainer,
    FloatArray2DContainer,
    IntArray1DContainer,
    IntArray2DContainer,
    OrbitContainer,
)


@pytest.mark.parametrize(
    "IntArrayContainer, dim", [(IntArray1DContainer, 1), (IntArray2DContainer, 2)]
)
def test_int_container(IntArrayContainer, dim, rng):
    # Use np.int_ to ensure cross-platform compatibility.
    arrays = tuple(
        rng.integers(1, 5, size=dim * (rng.integers(1, 5),)).astype(np.int_)
        for _ in range(10)
    )
    container = IntArrayContainer(arrays)
    assert len(container) == 10

    # test setting the same sized list (no memory reallocation)
    new_arrays = tuple(
        rng.integers(1, 5, size=dim * (rng.integers(1, 5),)).astype(np.int_)
        for _ in range(10)
    )
    container.set_arrays(new_arrays)

    assert len(container) == 10

    # test setting a different sized list (needs memory reallocation)
    new_arrays = tuple(
        rng.integers(1, 5, size=dim * (rng.integers(1, 5),)).astype(np.int_)
        for _ in range(12)
    )
    container.set_arrays(new_arrays)

    assert len(container) == 12

    with pytest.raises(ValueError):
        new_arrays = tuple(
            rng.integers(1, 5, size=(dim + 1) * (rng.integers(1, 5),)).astype(np.int_)
            for _ in range(12)
        )
        container.set_arrays(new_arrays)


@pytest.mark.parametrize(
    "FloatContainer, dim", [(FloatArray1DContainer, 1), (FloatArray2DContainer, 2)]
)
def test_float_container(FloatContainer, dim, rng):
    arrays = tuple(rng.random(tuple(rng.integers(1, 5, size=dim))) for _ in range(10))
    container = FloatContainer(arrays)
    assert len(container) == 10

    # test setting the same sized list (no memory reallocation)
    new_arrays = tuple(
        rng.random(tuple(rng.integers(1, 5, size=dim))) for _ in range(10)
    )
    container.set_arrays(new_arrays)

    assert len(container) == 10

    # test setting a different sized list (needs memory reallocation)
    new_arrays = tuple(
        rng.random(tuple(rng.integers(1, 5, size=dim))) for _ in range(12)
    )
    container.set_arrays(new_arrays)
    assert len(container) == 12

    with pytest.raises(ValueError):
        array_list = tuple(
            rng.random(tuple(rng.integers(1, 5, size=dim + 1))) for _ in range(10)
        )
        container.set_arrays(array_list)


def test_orbit_container(rng):
    orbit_data = []
    for i in range(10):
        orbit_id = i
        bit_id = rng.integers(1, 5, dtype=int)
        correlation_tensors = rng.random(tuple(rng.integers(1, 5, size=2)))
        tensor_indices = rng.integers(1, 5, size=correlation_tensors.shape[0]).astype(
            np.int_
        )
        orbit_data.append((orbit_id, bit_id, correlation_tensors, tensor_indices))

    container = OrbitContainer(tuple(orbit_data))
    assert len(container) == 10

    # test setting the same sized list (no memory reallocation)
    new_orbit_data = []
    for i in range(10):
        orbit_id = i
        bit_id = rng.integers(1, 5, dtype=int)
        correlation_tensors = rng.random(tuple(rng.integers(1, 5, size=2)))
        tensor_indices = rng.integers(1, 5, size=correlation_tensors.shape[0]).astype(
            np.int_
        )
        new_orbit_data.append((orbit_id, bit_id, correlation_tensors, tensor_indices))

    container.set_orbits(tuple(new_orbit_data))
    assert len(container) == 10

    # test setting a different sized list (needs memory reallocation)
    new_orbit_data = []
    for i in range(12):
        orbit_id = i
        bit_id = rng.integers(1, 5, dtype=int)
        correlation_tensors = rng.random(tuple(rng.integers(1, 5, size=2)))
        tensor_indices = rng.integers(1, 5, size=correlation_tensors.shape[0]).astype(
            np.int_
        )
        new_orbit_data.append((orbit_id, bit_id, correlation_tensors, tensor_indices))

    container.set_orbits(tuple(new_orbit_data))
    assert len(container) == 12

    with pytest.raises(TypeError):
        new_orbit_data[-1] = (
            "S",
            bit_id,
            correlation_tensors,
            tensor_indices,
        )  # orbit_id must be an int
        container.set_orbits(tuple(new_orbit_data))

    with pytest.raises(TypeError):
        new_orbit_data[-1] = (
            orbit_id,
            "X",
            correlation_tensors,
            tensor_indices,
        )  # bit_id must be an int
        container.set_orbits(tuple(new_orbit_data))

    with pytest.raises(TypeError):
        new_orbit_data[-1] = (orbit_id, bit_id, "X", tensor_indices)
        container.set_orbits(
            tuple(new_orbit_data)
        )  # correlation_tensors must be a numpy array

    with pytest.raises(TypeError):
        new_orbit_data[-1] = (orbit_id, bit_id, correlation_tensors, "X")
        container.set_orbits(
            tuple(new_orbit_data)
        )  # tensor_indices must be a numpy array

    with pytest.raises(ValueError):
        new_orbit_data[-1] = (
            orbit_id,
            bit_id,
            rng.integers(1, 5, size=10),
            tensor_indices,
        )
        container.set_orbits(tuple(new_orbit_data))  # correlation_tensors must be 2D

    with pytest.raises(ValueError):
        new_orbit_data[-1] = (
            orbit_id,
            bit_id,
            correlation_tensors,
            rng.integers(1, 5, size=(2, 2)),
        )
        container.set_orbits(tuple(new_orbit_data))  # tensor_indices must be 1D
