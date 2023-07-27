"""Get constraints on variables from a processor."""
from numbers import Number
from typing import List, Tuple, Union

import cvxpy as cp
import numpy as np
from cvxpy import Expression
from cvxpy.constraints.constraint import Constraint
from pymatgen.core import Structure

from smol.capp.generate.groundstate.upper_bound.indices import (
    get_variable_indices_for_each_composition_component,
)
from smol.moca.composition.space import CompositionSpace
from smol.moca.ensemble import Sublattice

__author__ = "Fengyu Xie"


def get_normalization_constraints(
    variables: cp.Variable, variable_indices: List[List[int]]
) -> List[Constraint]:
    """Get normalization constraints of variables on each site.

    Args:
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list of lists of int):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
    Returns:
        list of Constraint.
    """
    constraints = []
    for site_indices in variable_indices:
        site_variable_indices = np.array(site_indices, dtype=int)
        site_variable_indices = site_variable_indices[site_variable_indices >= 0]
        if len(site_variable_indices) > 0:
            constraints.append(cp.sum(variables[site_variable_indices]) == 1)
    return constraints


def _extract_constraint_matrix(comp_space, constraint_type):
    """Get the constraint matrix of a specific type."""
    if constraint_type in ["==", "=", "eq"]:
        mat_a = comp_space._A
        vec_b = comp_space._b
        constraint_type = "=="
    elif constraint_type in ["<=", "leq"]:
        mat_a = comp_space._A_leq if comp_space._A_leq is not None else []
        vec_b = comp_space._b_leq if comp_space._b_leq is not None else []
        constraint_type = "<="
    else:
        raise NotImplementedError(f"Constraint type {constraint_type} not supported!")
    return mat_a, vec_b, constraint_type


def _get_formula_constraints(
    comp_space, constraint_type, variables, variables_per_component, skip_cids=None
):
    """Get a specific type of composition constraints."""
    mat_a, vec_b, constraint_type = _extract_constraint_matrix(
        comp_space, constraint_type
    )

    constraints = []
    skip_cids = skip_cids or []
    for cid, (a, b) in enumerate(zip(mat_a, vec_b)):
        # Null constraints.
        if cid in skip_cids:
            continue
        if np.allclose(a, 0):
            if constraint_type == "==" and not np.isclose(b, 0):
                raise ValueError(f"Unsatisfiable constraint an=b, a: {a}, b: {b}.")
            if constraint_type == "<=" and b < 0:
                raise ValueError(f"Unsatisfiable constraint an<=b, a: {a}, b: {b}.")
            continue

        expression = 0
        for dim_id, (indices, n_fixed) in enumerate(variables_per_component):
            # Active sub-lattice.
            if len(indices) > 0 and not np.isclose(a[dim_id], 0):
                expression += cp.sum(variables[indices]) * a[dim_id]
            expression += n_fixed * a[dim_id]
        if not isinstance(expression, Expression):
            if (constraint_type == "==" and expression != b) or (
                constraint_type == "<=" and expression > b
            ):
                raise ValueError(
                    f"Constraint {a} {constraint_type} {b} can never be"
                    f" satisfied because the number of restricted sites"
                    f" can not satisfy this requirement!"
                )
        else:
            if constraint_type == "==":
                constraints.append(expression == b)
            if constraint_type == "<=":
                constraints.append(expression <= b)

    return constraints


def get_composition_space_constraints(
    sublattices: List[Sublattice],
    variables: cp.Variable,
    variable_indices: List[List[int]],
    processor_structure: Structure,
    charge_balanced: bool = True,
    other_constraints: List[
        Union[str, Tuple[Union[dict, List[dict], List[Number]], Number, str]]
    ] = None,
) -> List[Constraint]:
    """Get constraints on species composition in CompositionSpace.

    Supports charge balance and other generic composition constraints.
    See moca.CompositionSpace.
    Args:
        sublattices(list of Sublattice):
            Sub-lattices to build the upper-bound problem on.
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list of lists of int):
            List of variable indices corresponding to each active site index and
            index of species in its site space. Inactive sites will be marked by
            either -1 or -2. See documentation in groundstate.upper_bound.variables.
        processor_structure(Structure):
            The supercell structure stored in a processor's structure attribute.
            The sub-lattices must match the processor structure, or they must be the result
            of splitting with the initial_occupancy. See smol.moca.sublattice for the
            explanation of splitting a sub-lattice.
        charge_balanced(bool): optional
            Whether to enforce charge balance. Default is True.
        other_constraints(list): optional
            Other constraints to set in composition space. See moca.composition
            for detailed description.
            Note that constraints are now to be satisfied with the number of sites
            in sub-lattices of the supercell instead of the primitive cell.
    Returns:
        list of Constraint: Constraints corresponding to the given composition space.
    """
    # Get the variable indices corresponding to each dimension in "counts" format.
    variables_per_component = get_variable_indices_for_each_composition_component(
        sublattices, variable_indices, processor_structure
    )

    # Create the composition space.
    bits = [s.species for s in sublattices]
    sizes = [len(s.sites) for s in sublattices]
    comp_space = CompositionSpace(
        bits,
        sizes,
        charge_neutral=charge_balanced,
        other_constraints=other_constraints,
        optimize_basis=False,
        table_ergodic=False,
    )

    # Charge balance is the first constraint in CompositionSpace.
    n_charge = 1 if comp_space.charge_neutral else 0
    n_sublattices = len(sublattices)
    # Normalization on each sublattice are not necessary because it is already satisfied
    # by normalization on each site.
    skip_cids = list(range(n_charge, n_charge + n_sublattices))
    if np.allclose(comp_space._A[0, :], 0) and np.isclose(comp_space._b[0], 0):
        # Neutral alloy, no need to constrain charge.
        skip_cids = [0] + skip_cids

    # EQ, LEQ and GEQ. No skip in LEQ and GEQ.
    constraints = []
    constraints.extend(
        _get_formula_constraints(
            comp_space, "eq", variables, variables_per_component, skip_cids
        )
    )
    constraints.extend(
        _get_formula_constraints(comp_space, "leq", variables, variables_per_component)
    )
    # All inequality constraints must be of leq type.

    return constraints


def get_fixed_composition_constraints(
    sublattices: List[Sublattice],
    variables: cp.Variable,
    variable_indices: List[List[int]],
    processor_structure: Structure,
    fixed_composition: List[int],
) -> List[Constraint]:
    """Fix the count of species in the super-cell.

    Used for searching ground-states in a canonical ensemble.
    Args:
        sublattices(list of Sublattice):
            Sub-lattices to build the upper-bound problem on.
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list of lists of int):
            List of variable indices corresponding to each active site index and
            index of species in its site space. Inactive sites will be marked by
            either -1 or -2. See documentation in groundstate.upper_bound.variables.
        processor_structure(Structure):
            The supercell structure stored in a processor's structure attribute.
            The sub-lattices must match the processor structure, or they must be the result
            of splitting with the initial_occupancy. See smol.moca.sublattice for the
            explanation of splitting a sub-lattice.
        fixed_composition(list of int):
            Amount of each species to be fixed in the SUPER-cell, in CompositionSpace
            "counts" format. You are fully responsible for setting the order of species
            in the list correctly as would be generated by a CompositionSpace.
    Return:
        list of Constraint: Constraints corresponding to a fixed composition on each
        sub-lattice.
    """
    # Get the variable indices corresponding to each dimension in "counts" format.
    variables_per_component = get_variable_indices_for_each_composition_component(
        sublattices, variable_indices, processor_structure
    )

    constraints = []
    for dim_id, (indices, n_fixed) in enumerate(variables_per_component):
        expression = 0
        if len(indices) > 0:
            expression += cp.sum(variables[indices])
        expression += n_fixed

        if n_fixed > fixed_composition[dim_id]:
            raise ValueError(
                f"Fixed composition {fixed_composition} can not"
                f" be satisfied because the {dim_id}'th component"
                f" is always occupied by {n_fixed} species, more than"
                f" the number allowed!"
            )
        if not isinstance(expression, Expression):
            if expression != fixed_composition[dim_id]:
                raise ValueError(
                    f"Fixed composition {fixed_composition} can not"
                    f" be satisfied because the {dim_id}'th component"
                    f" is restricted to be occupied by {expression} species,"
                    f" not equals to the number required!"
                )
        else:
            constraints.append(expression == fixed_composition[dim_id])

    return constraints
