import pytest

from smol.moca.sampler.kernel import (
    MCKernel,
    Metropolis,
    ThermalKernel,
    UniformlyRandom,
)
from smol.utils import class_name_from_str, derived_class_factory, get_subclasses


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
        c in get_subclasses(MCKernel).values() for c in [UniformlyRandom, Metropolis]
    )
    assert all(c in get_subclasses(ThermalKernel).values() for c in [Metropolis])


def test_derived_class_factory(single_canonical_ensemble):
    kernel = derived_class_factory(
        "Metropolis", ThermalKernel, single_canonical_ensemble, "swap", temperature=500
    )
    assert isinstance(kernel, Metropolis)
    assert isinstance(kernel, MCKernel)

    with pytest.raises(NotImplementedError):
        derived_class_factory(
            "BeepBoop", ThermalKernel, single_canonical_ensemble, "swap"
        )
