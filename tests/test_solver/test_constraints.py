"""Test constraints to the upper-bound problem."""
import itertools

import pytest
from pymatgen.core import Species

from smol.moca.utils.occu import get_dim_ids_table, occu_to_counts
from smol.solver.upper_bound.constraints import (
    get_upper_bound_composition_space_constraints,
    get_upper_bound_fixed_composition_constraints,
    get_upper_bound_normalization_constraints,
)
from smol.solver.upper_bound.variables import get_upper_bound_variables_from_sublattices

from .utils import get_random_neutral_variable_values, get_random_variable_values


def test_normalization(exotic_ensemble, exotic_initial_occupancy):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices,
        exotic_ensemble.processor.structure,
        exotic_initial_occupancy,
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
        exotic_ensemble.sublattices,
        exotic_ensemble.processor.structure,
        exotic_initial_occupancy,
    )
    constraints = get_upper_bound_composition_space_constraints(
        exotic_ensemble.sublattices,
        variables,
        variable_indices,
        exotic_ensemble.processor.structure,
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
        exotic_ensemble.processor.structure,
        other_constraints=[
            "Mn4+ == 1",  # Broken when force_flip, kept when canonical. 2nd.
            "Ti4+ = 2",  # Broken when force_flip, kept when canonical. 3rd.
            "Mn4+ + Mn3+ + Mn2+ >= 7",  # Never true. GEQ comes after LEQ. 5th.
            "Mn3+ + Mn2+ <= 3",  # Always true. 4th.
            "0 >= -1",  # Always True. Skipped.
            "0 <= 1.5",  # Always True. Skipped.
            "0.0 = 0.0",  # Always True. Skipped.
        ],
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
            exotic_ensemble.processor.structure,
            other_constraints=[
                " == 10",  # Unsatisfiable.
            ],
        )
    with pytest.raises(ValueError):
        _ = get_upper_bound_composition_space_constraints(
            exotic_ensemble.sublattices,
            variables,
            variable_indices,
            exotic_ensemble.processor.structure,
            other_constraints=[
                " >= 1",  # Unsatisfiable.
            ],
        )
    with pytest.raises(ValueError):
        _ = get_upper_bound_composition_space_constraints(
            exotic_ensemble.sublattices,
            variables,
            variable_indices,
            exotic_ensemble.processor.structure,
            other_constraints=[
                " <= -1",  # Unsatisfiable.
            ],
        )
    with pytest.raises(ValueError):
        _ = get_upper_bound_composition_space_constraints(
            exotic_ensemble.sublattices,
            variables,
            variable_indices,
            exotic_ensemble.processor.structure,
            other_constraints=[
                "F- == 1000",  # Unsatisfiable.
            ],
        )
    with pytest.raises(ValueError):
        _ = get_upper_bound_composition_space_constraints(
            exotic_ensemble.sublattices,
            variables,
            variable_indices,
            exotic_ensemble.processor.structure,
            other_constraints=[
                "F- >= 1000",  # Unsatisfiable.
            ],
        )
    with pytest.raises(ValueError):
        _ = get_upper_bound_composition_space_constraints(
            exotic_ensemble.sublattices,
            variables,
            variable_indices,
            exotic_ensemble.processor.structure,
            other_constraints=[
                "F- <= 1",  # Unsatisfiable.
            ],
        )


def test_fixed_composition_constraints(exotic_ensemble, exotic_initial_occupancy):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices,
        exotic_ensemble.processor.structure,
        exotic_initial_occupancy,
    )
    bits = [s.species for s in exotic_ensemble.sublattices]
    n_dims = sum([len(sl_species) for sl_species in bits])
    table = get_dim_ids_table(exotic_ensemble.sublattices)
    fixed_counts = occu_to_counts(exotic_initial_occupancy, n_dims, table)
    constraints = get_upper_bound_fixed_composition_constraints(
        exotic_ensemble.sublattices,
        variables,
        variable_indices,
        exotic_ensemble.processor.structure,
        fixed_composition=fixed_counts,
    )

    # F- is fixed and always satisfied, will not appear.
    assert len(constraints) == 8
    for _ in range(20):
        rand_val = get_random_neutral_variable_values(
            exotic_ensemble.sublattices,
            exotic_initial_occupancy,
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
            exotic_ensemble.sublattices,
            exotic_initial_occupancy,
            variable_indices,
            force_flip=True,
        )  # Force Ti-Mn4 flip, will not satisfy Ti, Mn constraints
        variables.value = rand_val
        results = [c.value() for c in constraints]
        assumed_results = [
            (False if i in [ti_id, mn4_id] else True) for i in range(len(flatten_bits))
        ]
        assert results == assumed_results
