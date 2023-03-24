"""Test constraints to the upper-bound problem."""
import pytest

from smol.solver.upper_bound.constraints import (
    get_upper_bound_composition_space_constraints,
    get_upper_bound_fixed_composition_constraints,
    get_upper_bound_normalization_constraints,
)
from smol.solver.upper_bound.variables import get_upper_bound_variables_from_sublattices

from .utils import get_random_neutral_variable_values, get_random_variable_values


def test_normalization(exotic_ensemble):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices, exotic_ensemble.num_sites
    )
    constraints = get_upper_bound_normalization_constraints(variables, variable_indices)
    assert len(constraints) == exotic_ensemble.num_sites - len(
        exotic_ensemble.restricted_sites
    )
    # Check with constraint compliant random occupancies.
    for _ in range(20):
        rand_val = get_random_variable_values(exotic_ensemble.sublattices)
        variables.value = rand_val
        for c in constraints:
            assert c.value()


def test_comp_space_constraints(exotic_ensemble, exotic_initial_occupancy):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices, exotic_ensemble.num_sites
    )
    constraints = get_upper_bound_composition_space_constraints(
        exotic_ensemble.sublattices,
        variables,
        variable_indices,
        initial_occupancy=exotic_initial_occupancy,
    )

    # Only 1 charge balance because normalization skipped.
    assert len(constraints) == 1
    for _ in range(20):
        rand_val = get_random_neutral_variable_values(
            exotic_ensemble.sublattices, exotic_initial_occupancy, variable_indices
        )
        variables.value = rand_val
        for c in constraints:
            assert c.value()

    # Add one more constraint to constrain some species amounts.
    constraints = get_upper_bound_composition_space_constraints(
        exotic_ensemble.sublattices,
        variables,
        variable_indices,
        other_constraints=[
            "Mn4+ == 1",  # Broken when force_flip, kept when canonical. 2nd.
            "Ti4+ = 2",  # Broken when force_flip, kept when canonical. 3rd.
            "Mn3+ + Mn2+ <= 3",  # Always true. 4th.
            "Mn4+ + Mn3+ + Mn2+ >= 7",  # Never true. GEQ comes after LEQ. 5th.
            " >= -1",  # Always True. Skipped.
            " <= 1.5",  # Always True. Skipped.
            " = 0.0",  # Always True. Skipped.
        ],
        initial_occupancy=exotic_initial_occupancy,
    )

    assert len(constraints) == 5
    # Check with force_flip.
    for _ in range(20):
        rand_val = get_random_neutral_variable_values(
            exotic_ensemble.sublattices,
            exotic_initial_occupancy,
            variable_indices,
            force_flip=True,
        )
        variables.value = rand_val
        results = [c.value() for c in constraints]
        assert results == [True, False, False, True, False]
    # Check with canonical.
    for _ in range(20):
        rand_val = get_random_neutral_variable_values(
            exotic_ensemble.sublattices,
            exotic_initial_occupancy,
            variable_indices,
            canonical=True,
        )
        variables.value = rand_val
        results = [c.value() for c in constraints]
        assert results == [True, True, True, True, False]

    # Bad test cases.
    with pytest.raises(ValueError):
        _ = get_upper_bound_composition_space_constraints(
            exotic_ensemble.sublattices,
            variables,
            variable_indices,
            other_constraints=[
                " == 10",  # Unsatisfiable.
            ],
            initial_occupancy=exotic_initial_occupancy,
        )
    with pytest.raises(ValueError):
        _ = get_upper_bound_composition_space_constraints(
            exotic_ensemble.sublattices,
            variables,
            variable_indices,
            other_constraints=[
                " >= 1",  # Unsatisfiable.
            ],
            initial_occupancy=exotic_initial_occupancy,
        )
    with pytest.raises(ValueError):
        _ = get_upper_bound_composition_space_constraints(
            exotic_ensemble.sublattices,
            variables,
            variable_indices,
            other_constraints=[
                " <= -1",  # Unsatisfiable.
            ],
            initial_occupancy=exotic_initial_occupancy,
        )


def test_fixed_composition_constraints(exotic_ensemble, exotic_initial_occupancy):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices, exotic_ensemble.num_sites
    )
    constraints = get_upper_bound_fixed_composition_constraints(
        exotic_ensemble.sublattices,
        variables,
        variable_indices,
        initial_occupancy=exotic_initial_occupancy,
    )

    # TODO: finish this.
