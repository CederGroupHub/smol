"""Handles boolean product terms in expression."""
from typing import List, Tuple

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint
from numpy.typing import ArrayLike


# TODO: in the future, allow writing ground-state input files from terms,
#  which may support non-linear objectives and constraints.
def get_auxiliary_variable_values(
    variable_values: ArrayLike, indices_in_auxiliary_products: List[List[int]]
) -> np.ndarray:
    """Get the value of auxiliary variables from site variables.

    Args:
        variable_values(ArrayLike of bool or int):
            Values of site variables.
        indices_in_auxiliary_products(list of lists of int):
            A list containing the indices of variables whose product equals to the
            corresponding auxiliary slack variable.
    Returns:
        np.ndarray:
            Values of auxiliary variables subjecting to auxiliary constraints,
            as 0 and 1.
    """
    variable_values = np.array(variable_values).astype(int)
    aux_values = np.ones(len(indices_in_auxiliary_products), dtype=int)
    for i, inds in enumerate(indices_in_auxiliary_products):
        aux_values[i] = np.product(variable_values[inds])

    return aux_values.astype(int)


def get_expression_and_auxiliary_from_terms(
    cluster_terms: List[Tuple[List[int], float, float]],
    variables: cp.Variable,
    coefficients_cutoff: float = 0.0,
) -> Tuple[cp.Expression, cp.Variable, List[List[int]], List[Constraint]]:
    """Convert the cluster terms into linear function and auxiliary variables.

    This function simplify duplicates and linearizes multi-site cluster terms.

    Args:
        cluster_terms(list of tuples of (list of int, float)):
            A list of tuples, each represents a cluster term in the energy
            representation, containing indices of variables to be taken product with,
            and the factor before the boolean product.
            Energy is taken per super-cell.
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        coefficients_cutoff(float): optional
            Minimum cutoff to the coefficient of terms. If the absolute value of a
            term coefficient is less than cutoff, it will not be included in the
            objective function. If no cutoff is given, will include every term.
    Returns:
        cp.Expression, cp.Variable, list of lists of int, list of Constraint:
            The linearized energy expression, auxiliary slack variables for each
            multi-body product term, a list containing the indices of variables whose
            product equals to the corresponding auxiliary slack variable, and
            linearize constraints for each multi-body product term.
    """
    # Simplify cluster terms first.
    sorted_terms = [
        (tuple(sorted(inds)), fac1, fac2) for inds, fac1, fac2 in cluster_terms
    ]
    simplified_terms = {}
    for inds, fac1, fac2 in sorted_terms:
        if inds not in simplified_terms:
            simplified_terms[inds] = fac1 * fac2
        else:
            simplified_terms[inds] += fac1 * fac2
    simplified_terms = {
        inds: coef
        for inds, coef in simplified_terms.items()
        if abs(coef) >= coefficients_cutoff
    }
    simplified_terms = sorted(simplified_terms.items(), key=lambda t: (len(t[0]), t[1]))

    n_const = len([inds for inds, _ in simplified_terms if len(inds) == 0])
    n_single = len([inds for inds, _ in simplified_terms if len(inds) == 1])
    n_slack = len([inds for inds, _ in simplified_terms if len(inds) > 1])
    if n_slack == 0:
        aux_variables = None
    else:
        aux_variables = cp.Variable(n_slack, boolean=True)
    aux_constraints = []
    # Vectorized objective functions to speed up compilation.
    # Only one constant term is possible at most.
    expression = simplified_terms[0][1] if n_const == 1 else 0
    # Summing point terms.
    point_inds = [
        inds[0] for inds, _ in sorted(simplified_terms[n_const : n_const + n_single])
    ]
    point_coefs = np.array(
        [c for _, c in sorted(simplified_terms[n_const : n_const + n_single])]
    )
    if n_single > 0:
        expression += variables[point_inds] @ point_coefs
    # Summing many-body terms.
    many_inds = [
        list(inds)
        for inds, _ in simplified_terms[
            n_const + n_single : n_const + n_single + n_slack
        ]
    ]
    many_coefs = np.array(
        [
            coef
            for _, coef in simplified_terms[
                n_const + n_single : n_const + n_single + n_slack
            ]
        ]
    )
    if n_slack > 0:
        expression += aux_variables @ many_coefs
    for aux_id, inds in enumerate(many_inds):
        for var_id in inds:
            aux_constraints.append(aux_variables[aux_id] <= variables[var_id])
        aux_constraints.append(
            aux_variables[aux_id] >= cp.sum(variables[inds]) + 1 - len(inds)
        )

    if not isinstance(expression, cp.Expression):
        raise RuntimeError(
            f"The energy function {expression} has no configuration"
            f" degree of freedom. Cannot be optimized!"
        )

    return expression, aux_variables, many_inds, aux_constraints
