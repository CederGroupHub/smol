import pytest

from smol._utils import class_name_from_str, derived_class_factory, get_subclasses
from smol.moca.kernel import (
    Metropolis,
    MulticellMetropolis,
    UniformlyRandom,
    WangLandau,
)
from smol.moca.kernel._base import MCKernelInterface, ThermalKernelMixin


@pytest.mark.parametrize(
    "class_str",
    [
        "TheBestClass",
        "The-Best-Class",
        "The-Best-class",
        "The-best-Class",
        "the-Best-Class",
        "The-best-class",
        "the-Best-class",
        "the-best-Class",
        "the-best-class",
    ],
)
def test_class_name_from_str(class_str):
    assert class_name_from_str(class_str) == "TheBestClass"


def test_get_subclasses():
    assert all(
        c in get_subclasses(MCKernelInterface).values()
        for c in [UniformlyRandom, Metropolis, WangLandau, MulticellMetropolis]
    )
    assert all(
        c in get_subclasses(ThermalKernelMixin).values()
        for c in [Metropolis, MulticellMetropolis]
    )


def test_derived_class_factory(single_canonical_ensemble):
    kernel = derived_class_factory(
        "Metropolis",
        MCKernelInterface,
        single_canonical_ensemble,
        "swap",
        temperature=500,
    )
    assert isinstance(kernel, Metropolis)
    assert isinstance(kernel, MCKernelInterface)

    with pytest.raises(NotImplementedError):
        derived_class_factory(
            "BeepBoop", MCKernelInterface, single_canonical_ensemble, "swap"
        )
