"""Handles boolean product terms in expression."""
from typing import List, Tuple

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint
from numpy.typing import ArrayLike


# TODO: in the future, allow writing solver input files from terms,
#  which may support non-linear objectives and constraints.
def get_auxiliary_variable_values(
    variable_values: ArrayLike, indices_in_auxiliary_products: List[List[int]]
) -> np.ndarray:
    """Get the value of auxiliary variables from site variables.

    Args:
        variable_values(ArrayLike[bool, int]):
            Values of site variables.
        indices_in_auxiliary_products(list[list[int]]):
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
) -> Tuple[cp.Expression, cp.Variable, List[List[int]], List[Constraint]]:
    """Convert the cluster terms into linear function and auxiliary variables.

    This function simplify duplicates and linearizes multi-site cluster terms.

    Args:
        cluster_terms(list[tuple(list[int], float)]):
            A list of tuples, each represents a cluster term in the energy
            representation, containing indices of variables to be taken product with,
            and the factor before the boolean product.
            Energy is taken per super-cell.
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
    Returns:
        cp.Expression, cp.Variable, list[list[int]], list[Constraint]:
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

    expression = 0
    n_slack = len([inds for inds in simplified_terms.keys() if len(inds) > 1])
    if n_slack == 0:
        aux_variables = None
    else:
        aux_variables = cp.Variable(n_slack, boolean=True)
    indices_in_aux_products = []
    aux_constraints = []
    aux_id = 0
    for inds, fac in simplified_terms.items():
        # A constant addition term.
        if len(inds) == 0:
            expression += fac
        # A point term, no aux needed.
        elif len(inds) == 1:
            expression += variables[inds[0]] * fac
        # A product term, need an aux and constraints.
        else:
            expression += aux_variables[aux_id] * fac
            indices_in_aux_products.append(list(inds))
            for var_id in inds:
                aux_constraints.append(aux_variables[aux_id] <= variables[var_id])
            aux_constraints.append(
                aux_variables[aux_id] >= 1 - len(inds) + cp.sum(variables[list(inds)])
            )
            aux_id += 1

    if not isinstance(expression, cp.Expression):
        raise RuntimeError(
            f"The energy function {expression} has no configuration"
            f" degree of freedom. Cannot be optimized!"
        )

    return expression, aux_variables, indices_in_aux_products, aux_constraints
