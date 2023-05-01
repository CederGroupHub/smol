from abc import ABC, abstractmethod

import pytest

from smol.cofe.space.basis import (
    BasisIterator,
    ChebyshevIterator,
    IndicatorIterator,
    LegendreIterator,
    NumpyPolyIterator,
    PolynomialIterator,
    SinusoidIterator,
)
from smol.moca.kernel import Metropolis, UniformlyRandom, WangLandau
from smol.moca.kernel.base import MCKernelInterface, ThermalKernelMixin
from smol.moca.kernel.bias import (
    FugacityBias,
    MCBias,
    SquareChargeBias,
    SquareHyperplaneBias,
)
from smol.moca.kernel.mcusher import (
    Composite,
    Flip,
    MCUsher,
    MultiStep,
    Swap,
    TableFlip,
)
from smol.utils.class_utils import (
    class_name_from_str,
    derived_class_factory,
    get_subclasses,
)


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
    # test a few dummy classes
    class DummyABC(ABC):
        @abstractmethod
        def do_stuff(self):
            pass

    class DummyDummyABC(DummyABC):
        pass

    class DummyParent(DummyABC):
        def do_stuff(self):
            print("I'm doing stuff!")

    class DummyChild(DummyParent):
        def do_stuff(self):
            print("I'm doing more stuff!")

    assert all(
        c in get_subclasses(DummyABC).values() for c in [DummyParent, DummyChild]
    )
    assert DummyABC not in get_subclasses(DummyABC).values()
    assert DummyDummyABC not in get_subclasses(DummyABC).values()

    # now test classes in smol
    assert all(
        c in get_subclasses(MCKernelInterface).values()
        for c in [UniformlyRandom, Metropolis, WangLandau]
    )
    assert Metropolis in get_subclasses(ThermalKernelMixin).values()

    assert all(
        c in get_subclasses(MCUsher).values()
        for c in [Swap, Flip, MultiStep, Composite, TableFlip]
    )

    assert all(
        c in get_subclasses(MCBias).values()
        for c in [FugacityBias, SquareChargeBias, SquareHyperplaneBias]
    )

    assert all(
        c in get_subclasses(BasisIterator).values()
        for c in [
            SinusoidIterator,
            IndicatorIterator,
            LegendreIterator,
            ChebyshevIterator,
            PolynomialIterator,
        ]
    )

    # assert an abstract derived class is not included
    assert NumpyPolyIterator not in get_subclasses(BasisIterator).values()

    assert all(
        c in get_subclasses(BasisIterator).values()
        for c in [LegendreIterator, ChebyshevIterator, PolynomialIterator]
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
