"""Test objective functions generation."""
from itertools import product

import cvxpy as cp
import numpy as np
import pytest

from smol.moca.processor import ClusterExpansionProcessor
from smol.solver.upper_bound.objectives import (
    get_auxiliary_variable_values,
    get_expression_and_auxiliary_from_terms,
    get_upper_bound_terms_from_chemical_potentials,
    get_upper_bound_terms_from_decomposition_processor,
    get_upper_bound_terms_from_ewald_processor,
    get_upper_bound_terms_from_expansion_processor,
)
from smol.solver.upper_bound.variables import (
    get_occupancy_from_variables,
    get_upper_bound_variables_from_sublattices,
)

from .utils import get_random_variable_values


def test_expression_from_terms():
    x = cp.Variable(2, boolean=True)
    # Empty expression.
    terms = []
    with pytest.raises(RuntimeError):
        _ = get_expression_and_auxiliary_from_terms(terms, x)
    # Expression with only constant.
    terms = [([], -1), ([], 100)]
    with pytest.raises(RuntimeError):
        _ = get_expression_and_auxiliary_from_terms(terms, x)
    # Correct expression, which is to be evaluated.
    # A very simple test case.
    # 0 + 2 x0 -3 x1 + x0 * x1.
    terms = [([], -1), ([], 1), ([0], 2), ([1], -3), ([0, 1], -1), ([1, 0], 2)]
    func, y, indices, aux_cons = get_expression_and_auxiliary_from_terms(terms, x)
    assert y.size == 1
    assert len(indices) == 1
    assert list(indices[0]) == [0, 1]
    assert len(aux_cons) == 3
    assert isinstance(func, cp.Expression)
    for val0, val1 in product(*[[0, 1], [0, 1]]):
        true_val = 2 * val0 - 3 * val1 + val0 * val1
        x.value = np.array([val0, val1], dtype=int)
        y.value = val0 * val1
        assert func.value == true_val
        # Auxiliary constraints should always be satisfied.
        for con in aux_cons:
            assert con.value()


def test_expansion_upper(exotic_ensemble, exotic_initial_occupancy):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices, exotic_ensemble.num_sites
    )
    proc = exotic_ensemble.processor.processors[0]
    if isinstance(proc, ClusterExpansionProcessor):
        terms = get_upper_bound_terms_from_expansion_processor(
            exotic_ensemble.sublattices,
            variable_indices,
            expansion_processor=proc,
            initial_occupancy=exotic_initial_occupancy,
        )
    else:
        terms = get_upper_bound_terms_from_decomposition_processor(
            exotic_ensemble.sublattices,
            variable_indices,
            decomposition_processor=proc,
            initial_occupancy=exotic_initial_occupancy,
        )
    objective, aux, aux_indices, aux_cons = get_expression_and_auxiliary_from_terms(
        terms, variables
    )
    for _ in range(50):
        rand_val = get_random_variable_values(exotic_ensemble.sublattices)
        aux_val = get_auxiliary_variable_values(rand_val, aux_indices)
        rand_occu = get_occupancy_from_variables(
            exotic_ensemble.sublattices,
            rand_val,
            variable_indices,
            exotic_initial_occupancy,
        )
        variables.value = rand_val
        aux.value = aux_val
        energy_obj = objective.value
        energy_proc = np.dot(proc.compute_feature_vector(rand_occu), proc.coefs)
        assert np.isclose(energy_obj, energy_proc)
        for con in aux_cons:
            assert con.value()


def test_ewald_upper(exotic_ensemble, exotic_initial_occupancy):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices, exotic_ensemble.num_sites
    )
    proc = exotic_ensemble.processor.processors[1]
    terms = get_upper_bound_terms_from_ewald_processor(
        exotic_ensemble.sublattices,
        variable_indices,
        ewald_processor=proc,
        initial_occupancy=exotic_initial_occupancy,
    )
    objective, aux, aux_indices, aux_cons = get_expression_and_auxiliary_from_terms(
        terms, variables
    )
    for inds in aux_indices:
        assert len(inds) == 2  # No more than pair terms.

    for _ in range(50):
        # Should have the same ewald for either neutral or not neutral.
        rand_val = get_random_variable_values(exotic_ensemble.sublattices)
        aux_val = get_auxiliary_variable_values(rand_val, aux_indices)
        rand_occu = get_occupancy_from_variables(
            exotic_ensemble.sublattices,
            rand_val,
            variable_indices,
            exotic_initial_occupancy,
        )
        variables.value = rand_val
        aux.value = aux_val
        energy_obj = objective.value
        energy_proc = proc.compute_feature_vector(rand_occu) * proc.coefs
        assert np.isclose(energy_obj, energy_proc)
        for con in aux_cons:
            assert con.value()


def test_chemical_potentials_upper(exotic_ensemble, exotic_initial_occupancy):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices, exotic_ensemble.num_sites
    )
    # Only test semi-grand ensemble.
    if exotic_ensemble.chemical_potentials is not None:
        terms = get_upper_bound_terms_from_chemical_potentials(
            exotic_ensemble.sublattices,
            variable_indices,
            exotic_ensemble._chemical_potentials["table"],
            exotic_initial_occupancy,
        )
        objective, aux, aux_indices, aux_cons = get_expression_and_auxiliary_from_terms(
            terms, variables
        )
        # No linearize to point terms.
        assert aux is None
        assert len(aux_indices) == 0
        assert len(aux_cons) == 0
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
