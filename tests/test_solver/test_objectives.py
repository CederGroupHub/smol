"""Test objective functions generation."""
import numpy as np

from smol.moca.processor import ClusterExpansionProcessor
from smol.solver.upper_bound.objectives import (
    get_upper_bound_objective_from_chemical_potentials,
    get_upper_bound_objective_from_decomposition_processor,
    get_upper_bound_objective_from_ewald_processor,
    get_upper_bound_objective_from_expansion_processor,
)
from smol.solver.upper_bound.variables import (
    get_occupancy_from_variables,
    get_upper_bound_variables_from_sublattices,
)

from .utils import get_random_variable_values


def test_expansion_upper(exotic_ensemble, exotic_initial_occupancy):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices, exotic_ensemble.num_sites
    )
    proc = exotic_ensemble.processor.processors[0]
    if isinstance(proc, ClusterExpansionProcessor):
        objective = get_upper_bound_objective_from_expansion_processor(
            exotic_ensemble.sublattices,
            variables,
            variable_indices,
            expansion_processor=proc,
            initial_occupancy=exotic_initial_occupancy,
        )
    else:
        objective = get_upper_bound_objective_from_decomposition_processor(
            exotic_ensemble.sublattices,
            variables,
            variable_indices,
            decomposition_processor=proc,
            initial_occupancy=exotic_initial_occupancy,
        )
    for _ in range(50):
        rand_val = get_random_variable_values(exotic_ensemble.sublattices)
        rand_occu = get_occupancy_from_variables(
            exotic_ensemble.sublattices,
            rand_val,
            variable_indices,
            exotic_initial_occupancy,
        )
        variables.value = rand_val
        energy_obj = objective.value
        energy_proc = np.dot(proc.compute_feature_vector(rand_occu), proc.coefs)
        assert np.isclose(energy_obj, energy_proc)


def test_ewald_upper(exotic_ensemble, exotic_initial_occupancy):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices, exotic_ensemble.num_sites
    )
    proc = exotic_ensemble.processor.processors[1]
    objective = get_upper_bound_objective_from_ewald_processor(
        exotic_ensemble.sublattices,
        variables,
        variable_indices,
        ewald_processor=proc,
        initial_occupancy=exotic_initial_occupancy,
    )
    for _ in range(50):
        # Should have the same ewald for either neutral or not neutral.
        rand_val = get_random_variable_values(exotic_ensemble.sublattices)
        rand_occu = get_occupancy_from_variables(
            exotic_ensemble.sublattices,
            rand_val,
            variable_indices,
            exotic_initial_occupancy,
        )
        variables.value = rand_val
        energy_obj = objective.value
        energy_proc = proc.compute_feature_vector(rand_occu) * proc.coefs
        assert np.isclose(energy_obj, energy_proc)


def test_chemical_potentials_upper(exotic_ensemble, exotic_initial_occupancy):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices, exotic_ensemble.num_sites
    )
    # Only test semi-grand ensemble.
    if exotic_ensemble.chemical_potentials is not None:
        objective = get_upper_bound_objective_from_chemical_potentials(
            exotic_ensemble.sublattices,
            variables,
            variable_indices,
            exotic_ensemble._chemical_potentials["table"],
            exotic_initial_occupancy,
        )
        for _ in range(50):
            rand_val = get_random_variable_values(exotic_ensemble.sublattices)
            rand_occu = get_occupancy_from_variables(
                exotic_ensemble.sublattices,
                rand_val,
                variable_indices,
                exotic_initial_occupancy,
            )
            variables.value = rand_val
            energy_obj = objective.value
            # The last number is chemical work.
            energy_true = exotic_ensemble.compute_feature_vector(rand_occu)[-1]
            assert np.isclose(energy_obj, energy_true)
