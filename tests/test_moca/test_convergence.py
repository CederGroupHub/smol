import numpy as np
import pytest

from smol.moca.analysis.convergence import (
    check_property_converged,
    determine_discard_number,
)


@pytest.fixture()
def prop_array_1():  # array with an outlier at the end
    array1 = [1, 2] * 100
    array1.append(10)
    return np.array(array1)


@pytest.fixture()
def prop_array_2(rng):
    # linearly decreasing array with some noise, then hold constant with noise
    array2 = []
    for i in np.arange(1, -0.001, -0.05):
        array2.append(i)
        random_samples = rng.normal(loc=i, scale=0.01, size=10)
        array2.extend(random_samples)

    more_random_samples = rng.normal(loc=0, scale=0.01, size=10000)
    array2.extend(more_random_samples)
    array2.append(0.005)
    return np.array(array2)


def test_check_property_converged(prop_array_1, prop_array_2):
    assert not check_property_converged(prop_array_1)
    assert not check_property_converged(prop_array_2[:200])
    assert check_property_converged(prop_array_2[1000:])


def test_determine_discard_number(prop_array_1, prop_array_2):
    assert not determine_discard_number(prop_array_1)
    assert not determine_discard_number(prop_array_2[:200])
    assert determine_discard_number(prop_array_2[1000:]) > 0
