"""Test objective functions generation."""
import numpy as np
import numpy.testing as npt

from smol.capp.generate.groundstate.upper_bound.objectives import (
    get_terms_from_chemical_potentials,
    get_terms_from_decomposition_processor,
    get_terms_from_ewald_processor,
    get_terms_from_expansion_processor,
)
from smol.capp.generate.groundstate.upper_bound.terms import (
    get_auxiliary_variable_values,
    get_expression_and_auxiliary_from_terms,
)
from smol.capp.generate.groundstate.upper_bound.variables import (
    get_occupancy_from_variables,
    get_variables_from_sublattices,
)
from smol.moca.processor import ClusterExpansionProcessor

from .utils import (
    evaluate_correlations_from_variable_values,
    get_random_variable_values,
    validate_correlations_from_occupancy,
    validate_interactions_from_occupancy,
)


def test_expansion_upper(solver_test_ensemble, solver_test_initial_occupancy):
    variables, variable_indices = get_variables_from_sublattices(
        solver_test_ensemble.sublattices,
        solver_test_ensemble.processor.structure,
        solver_test_initial_occupancy,
    )
    proc = solver_test_ensemble.processor.processors[0]
    if isinstance(proc, ClusterExpansionProcessor):
        terms = get_terms_from_expansion_processor(
            variable_indices,
            expansion_processor=proc,
        )
        grouped_terms = get_terms_from_expansion_processor(
            variable_indices,
            expansion_processor=proc,
            group_output_by_function=True,
        )
    else:
        terms = get_terms_from_decomposition_processor(
            variable_indices,
            decomposition_processor=proc,
        )
        grouped_terms = get_terms_from_decomposition_processor(
            variable_indices,
            decomposition_processor=proc,
            group_output_by_orbit=True,
        )
    objective, aux, aux_indices, aux_cons = get_expression_and_auxiliary_from_terms(
        terms, variables
    )
    for _ in range(50):
        rand_val = get_random_variable_values(solver_test_ensemble.sublattices)
        aux_val = get_auxiliary_variable_values(rand_val, aux_indices)
        rand_occu = get_occupancy_from_variables(
            solver_test_ensemble.sublattices,
            rand_val,
            variable_indices,
        )
        if isinstance(proc, ClusterExpansionProcessor):
            validate_correlations_from_occupancy(proc, rand_occu)
        else:
            validate_interactions_from_occupancy(proc, rand_occu)

        corr = evaluate_correlations_from_variable_values(grouped_terms, rand_val)
        npt.assert_array_almost_equal(
            corr, proc.compute_feature_vector(rand_occu) / proc.size
        )
        variables.value = rand_val
        aux.value = aux_val
        energy_obj = objective.value
        energy_proc = np.dot(proc.compute_feature_vector(rand_occu), proc.coefs)
        assert np.isclose(energy_obj, energy_proc)
        for con in aux_cons:
            assert con.value()


def test_ewald_upper(solver_test_ensemble, solver_test_initial_occupancy):
    variables, variable_indices = get_variables_from_sublattices(
        solver_test_ensemble.sublattices,
        solver_test_ensemble.processor.structure,
        solver_test_initial_occupancy,
    )
    proc = solver_test_ensemble.processor.processors[1]
    terms = get_terms_from_ewald_processor(
        variable_indices,
        ewald_processor=proc,
    )
    objective, aux, aux_indices, aux_cons = get_expression_and_auxiliary_from_terms(
        terms, variables
    )
    for inds in aux_indices:
        assert len(inds) == 2  # No more than pair terms.

    for _ in range(50):
        # Should have the same ewald for either neutral or not neutral.
        rand_val = get_random_variable_values(solver_test_ensemble.sublattices)
        aux_val = get_auxiliary_variable_values(rand_val, aux_indices)
        rand_occu = get_occupancy_from_variables(
            solver_test_ensemble.sublattices,
            rand_val,
            variable_indices,
        )
        variables.value = rand_val
        aux.value = aux_val
        energy_obj = objective.value
        energy_proc = proc.compute_feature_vector(rand_occu) * proc.coefs
        assert np.isclose(energy_obj, energy_proc)
        for con in aux_cons:
            assert con.value()


def test_chemical_potentials_upper(solver_test_ensemble, solver_test_initial_occupancy):
    variables, variable_indices = get_variables_from_sublattices(
        solver_test_ensemble.sublattices,
        solver_test_ensemble.processor.structure,
        solver_test_initial_occupancy,
    )
    # Only test semi-grand ensemble.
    if solver_test_ensemble.chemical_potentials is not None:
        terms = get_terms_from_chemical_potentials(
            variable_indices,
            solver_test_ensemble._chemical_potentials["table"],
        )
        objective, aux, aux_indices, aux_cons = get_expression_and_auxiliary_from_terms(
            terms, variables
        )
        # No linearize to point terms.
        assert aux is None
        assert len(aux_indices) == 0
        assert len(aux_cons) == 0
        for _ in range(50):
            rand_val = get_random_variable_values(solver_test_ensemble.sublattices)
            rand_occu = get_occupancy_from_variables(
                solver_test_ensemble.sublattices,
                rand_val,
                variable_indices,
            )
            variables.value = rand_val
            energy_obj = objective.value
            # The last number is -1 * chemical work.
            energy_true = (
                -1 * solver_test_ensemble.compute_feature_vector(rand_occu)[-1]
            )
            assert np.isclose(energy_obj, energy_true)
