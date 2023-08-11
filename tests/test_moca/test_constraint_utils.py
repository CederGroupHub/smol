"""Test utilities for managing composition constraints."""
import numpy.testing as npt
import pytest
from pymatgen.core import DummySpecies, Element, Species

# To test manager.
from smol.cofe.space.domain import Vacancy
from smol.moca.composition.constraints import (
    convert_constraint_string,
    handle_side_string,
)
from smol.moca.composition.space import CompositionSpace


def test_handle_side_string():
    # Good expressions.
    ag1 = Species("Ag", 1)
    mn = Element("Mn")
    mn3 = Species("Mn", 3)
    cu = Element("Cu")
    vac = Vacancy()
    x = DummySpecies("X")

    cases = [
        ("Ag+ + 2 Mn", ([(1, ag1, None), (2, mn, None)], 0)),
        (" Ag+ +  2 Mn ", ([(1, ag1, None), (2, mn, None)], 0)),
        ("- 1 Ag+ +3.5 Mn3+", ([(-1, ag1, None), (3.5, mn3, None)], 0)),
        ("- Ag+ + 3.5 Mn3+", ([(-1, ag1, None), (3.5, mn3, None)], 0)),
        ("Ag+(0) +2 Mn3+", ([(1, ag1, 0), (2, mn3, None)], 0)),
        ("Ag+(00) +2 Mn3+(10) + 1.5", ([(1, ag1, 0), (2, mn3, 10)], 1.5)),
        (
            "Vacancy(0) + Mn3+(10) - 4.1 Cu",
            ([(1, vac, 0), (1, mn3, 10), (-4.1, cu, None)], 0),
        ),
        ("X(100) + 2", ([(1, x, 100)], 2)),
    ]

    for s, standard in cases:
        assert handle_side_string(s) == standard

    # Bad test cases.
    # species string not separated from number in the front.
    with pytest.raises(Exception):
        _ = handle_side_string("Ag+ + 2Mn")
    # species string not separated from operator in the front.
    with pytest.raises(Exception):
        _ = handle_side_string("Ag+ +Mn")
    # Space exist between species and its sub-lattice index label.
    with pytest.raises(Exception):
        _ = handle_side_string("Ag+ (0) +Mn")
    # Space inserted within a species string
    with pytest.raises(Exception):
        _ = handle_side_string("Ag+, spin=1(0) +Mn")


@pytest.fixture
def bits():
    return [
        [Species("Li", 1), Species("Mn", 3), Species("Ti", 4), Vacancy()],
        [Species("Li", 1), Species("O", -2), Species("O", -1), Species("F", -1)],
    ]


def test_convert_constraint_string(bits):
    cases = [
        ("- Li+ +3 O2-  <= - 10.2", ([-1, 0, 0, 0, -1, 3, 0, 0], -10.2, "<=")),
        ("- Li+(0) + O2- = 2", ([-1, 0, 0, 0, 0, 1, 0, 0], 2, "=")),
        (
            "- Li+(0) + 5 O- - 1.5 >= 4 Li+(1) - Mn3+ + Vacancy(0)",
            ([-1, 1, 0, -1, -4, 0, 5, 0], 1.5, ">="),
        ),
        ("- Li+ + 3 F- == 2 Ti4+", ([-1, 0, -2, 0, -1, 0, 0, 3], 0, "==")),
    ]

    for s, standard in cases:
        assert standard == convert_constraint_string(s, bits)

    # Bad cases where symbols are not properly separated by spaces.
    with pytest.raises(ValueError):
        _ = convert_constraint_string("- Li+ +3 O2- <=- 10.2", bits)
    with pytest.raises(ValueError):
        _ = convert_constraint_string("- Li+ +3 O2->= - 10.2", bits)
    # Bad case when symbol is not allowed:
    with pytest.raises(ValueError):
        _ = convert_constraint_string("- Li+ +3 O2- > - 10.2", bits)
    with pytest.raises(ValueError):
        _ = convert_constraint_string("- Li+ +3 O2->= - 10.2", bits)
    # Bad case when species can not be found on any sub-lattice
    with pytest.raises(ValueError):
        _ = convert_constraint_string("- K+(0) +3 O2- <= - 10.2", bits)
    with pytest.raises(ValueError):
        _ = convert_constraint_string("- K+ +3 O2- <= - 10.2", bits)


def test_constraints_manager(bits):
    sublattice_sizes = [10, 6]

    # Charge goes first, then normalizations and others.
    a_eq_std = [
        [1, 3, 4, 0, 1, -2, -1, -1],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [-2, 0, 1, 0, -2, 0, 0, 0],
        [1, 1, 1, 2, 0, 4, -1, 3],
    ]
    b_eq_std = [0, 10, 6, 6, 4]
    a_leq_std = [[1, 0.5, 0, 2, 0, 0, -1, 3], [0, 0, 1, -1, 0.4, 0, 0, 0]]
    b_leq_std = [10, 0]
    # No more geq constraints stored in manager.

    # Test all allowed formats of input.
    other_constraints = [
        # converted to int.
        "-1 Li+ + 0.5 Ti4+ == 3",
        ([1, 1, 1, 2, 0, 4, -1, 3], 4, "="),
        # not converted to int.
        "Li+(0) + 0.5 Mn3+(0) + 2 Vacancy(0) - O-(1) + 3 F-(1) <= 10",
        "- Ti4+ + Vacancy >= 0.4 Li+(1)",
    ]

    comp_space = CompositionSpace(
        bits,
        sublattice_sizes,
        charge_neutral=True,
        other_constraints=other_constraints,
    )

    npt.assert_array_equal(a_eq_std, comp_space._A)
    npt.assert_array_equal(b_eq_std, comp_space._b)
    npt.assert_array_equal(a_leq_std, comp_space._A_leq)
    npt.assert_array_equal(b_leq_std, comp_space._b_leq)

    # Test None cases.
    other_constraints = [
        # converted to int.
        "-1 Li+ + 0.5 Ti4+ == 3",
        ([1, 1, 1, 2, 0, 4, -1, 3], 4, "="),
    ]
    comp_space = CompositionSpace(
        bits,
        sublattice_sizes,
        charge_neutral=True,
        other_constraints=other_constraints,
    )
    npt.assert_array_equal(a_eq_std, comp_space._A)
    npt.assert_array_equal(b_eq_std, comp_space._b)
    assert comp_space._A_leq is None
    assert comp_space._b_leq is None

    comp_space = CompositionSpace(
        bits,
        sublattice_sizes,
        charge_neutral=True,
    )
    npt.assert_array_equal(a_eq_std[:-2], comp_space._A)
    npt.assert_array_equal(b_eq_std[:-2], comp_space._b)
    assert comp_space._A_leq is None
    assert comp_space._b_leq is None
