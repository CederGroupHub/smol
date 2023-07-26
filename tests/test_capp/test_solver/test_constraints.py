"""Test constraints to the upper-bound problem."""
import itertools

import pytest
from pymatgen.core import Species

from smol.capp.generate.groundstate.upper_bound.constraints import (
    get_composition_space_constraints,
    get_fixed_composition_constraints,
    get_normalization_constraints,
)
from smol.capp.generate.groundstate.upper_bound.variables import (
    get_variables_from_sublattices,
)
from smol.moca.occu_utils import get_dim_ids_table, occu_to_counts

from .utils import get_random_neutral_variable_values, get_random_variable_values


def test_normalization(solver_test_ensemble, solver_test_initial_occupancy):
    variables, variable_indices = get_variables_from_sublattices(
        solver_test_ensemble.sublattices,
        solver_test_ensemble.processor.structure,
        solver_test_initial_occupancy,
    )
    constraints = get_normalization_constraints(variables, variable_indices)
    assert len(constraints) == solver_test_ensemble.num_sites - len(
        solver_test_ensemble.restricted_sites
    )
    # Check with constraint compliant random occupancies.
    for _ in range(20):
        rand_val = get_random_variable_values(solver_test_ensemble.sublattices)
        variables.value = rand_val
        for c in constraints:
            assert c.value()


def test_comp_space_constraints(solver_test_ensemble, solver_test_initial_occupancy):
    variables, variable_indices = get_variables_from_sublattices(
        solver_test_ensemble.sublattices,
        solver_test_ensemble.processor.structure,
        solver_test_initial_occupancy,
    )
    constraints = get_composition_space_constraints(
        solver_test_ensemble.sublattices,
        variables,
        variable_indices,
        solver_test_ensemble.processor.structure,
    )

    # Only 1 charge balance because normalization skipped.
    assert len(constraints) == 1
    for _ in range(20):
        rand_val = get_random_neutral_variable_values(
            solver_test_ensemble.sublattices,
            solver_test_initial_occupancy,
            variable_indices,
        )
        variables.value = rand_val
        for c in constraints:
            assert c.value()

    # Add one more constraint to constrain some species amounts.
    constraints = get_composition_space_constraints(
        solver_test_ensemble.sublattices,
        variables,
        variable_indices,
        solver_test_ensemble.processor.structure,
        other_constraints=[
            "Mn4+ == 1",  # Broken when force_flip, kept when canonical. 2nd.
            "Ti4+ = 2",  # Broken when force_flip, kept when canonical. 3rd.
            "Mn4+ + Mn3+ + Mn2+ >= 7",  # Never true. 4th.
            "Mn3+ + Mn2+ <= 3",  # Always true. 5th.
            "0 >= -1",  # Always True. Skipped.
            "0 <= 1.5",  # Always True. Skipped.
            "0.0 = 0.0",  # Always True. Skipped.
        ],
    )

    assert len(constraints) == 5
    # Check with force_flip.
    for _ in range(20):
        rand_val = get_random_neutral_variable_values(
            solver_test_ensemble.sublattices,
            solver_test_initial_occupancy,
            variable_indices,
            force_flip=True,
        )
        variables.value = rand_val
        results = [c.value() for c in constraints]
        assert results == [True, False, False, False, True]
    # Check with canonical.
    for _ in range(20):
        rand_val = get_random_neutral_variable_values(
            solver_test_ensemble.sublattices,
            solver_test_initial_occupancy,
            variable_indices,
            canonical=True,
        )
        variables.value = rand_val
        results = [c.value() for c in constraints]
        assert results == [True, True, True, False, True]

    # Bad test cases.
    with pytest.raises(ValueError):
        _ = get_composition_space_constraints(
            solver_test_ensemble.sublattices,
            variables,
            variable_indices,
            solver_test_ensemble.processor.structure,
            other_constraints=[
                " == 10",  # Unsatisfiable.
            ],
        )
    with pytest.raises(ValueError):
        _ = get_composition_space_constraints(
            solver_test_ensemble.sublattices,
            variables,
            variable_indices,
            solver_test_ensemble.processor.structure,
            other_constraints=[
                " >= 1",  # Unsatisfiable.
            ],
        )
    with pytest.raises(ValueError):
        _ = get_composition_space_constraints(
            solver_test_ensemble.sublattices,
            variables,
            variable_indices,
            solver_test_ensemble.processor.structure,
            other_constraints=[
                " <= -1",  # Unsatisfiable.
            ],
        )
    with pytest.raises(ValueError):
        _ = get_composition_space_constraints(
            solver_test_ensemble.sublattices,
            variables,
            variable_indices,
            solver_test_ensemble.processor.structure,
            other_constraints=[
                "F- == 1000",  # Unsatisfiable.
            ],
        )
    with pytest.raises(ValueError):
        _ = get_composition_space_constraints(
            solver_test_ensemble.sublattices,
            variables,
            variable_indices,
            solver_test_ensemble.processor.structure,
            other_constraints=[
                "F- >= 1000",  # Unsatisfiable.
            ],
        )
    with pytest.raises(ValueError):
        _ = get_composition_space_constraints(
            solver_test_ensemble.sublattices,
            variables,
            variable_indices,
            solver_test_ensemble.processor.structure,
            other_constraints=[
                "F- <= 1",  # Unsatisfiable.
            ],
        )


def test_fixed_composition_constraints(
    solver_test_ensemble, solver_test_initial_occupancy
):
    variables, variable_indices = get_variables_from_sublattices(
        solver_test_ensemble.sublattices,
        solver_test_ensemble.processor.structure,
        solver_test_initial_occupancy,
    )
    bits = [s.species for s in solver_test_ensemble.sublattices]
    n_dims = sum([len(sl_species) for sl_species in bits])
    table = get_dim_ids_table(solver_test_ensemble.sublattices)
    fixed_counts = occu_to_counts(solver_test_initial_occupancy, n_dims, table)
    constraints = get_fixed_composition_constraints(
        solver_test_ensemble.sublattices,
        variables,
        variable_indices,
        solver_test_ensemble.processor.structure,
        fixed_composition=fixed_counts,
    )

    # F- is fixed and always satisfied, will not appear.
    assert len(constraints) == 8
    for _ in range(20):
        rand_val = get_random_neutral_variable_values(
            solver_test_ensemble.sublattices,
            solver_test_initial_occupancy,
            variable_indices,
            canonical=True,
        )  # Force canonical constraints, will always satisfy.
        variables.value = rand_val
        results = [c.value() for c in constraints]
        assert results == [True for _ in range(8)]
    flatten_bits = list(itertools.chain(*bits))
    flatten_bits.remove(Species("F", -1))
    assert len(flatten_bits) == 8
    ti_id = flatten_bits.index(Species("Ti", 4))
    mn4_id = flatten_bits.index(Species("Mn", 4))
    for _ in range(20):
        rand_val = get_random_neutral_variable_values(
            solver_test_ensemble.sublattices,
            solver_test_initial_occupancy,
            variable_indices,
            force_flip=True,
        )  # Force Ti-Mn4 flip, will not satisfy Ti, Mn constraints
        variables.value = rand_val
        results = [c.value() for c in constraints]
        assumed_results = [
            (False if i in [ti_id, mn4_id] else True) for i in range(len(flatten_bits))
        ]
        assert results == assumed_results
