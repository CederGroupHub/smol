"""Get constraints on variables from a processor."""
from numbers import Number
from typing import List, Tuple, Union

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint
from numpy.typing import ArrayLike

from smol.moca.composition import CompositionSpace
from smol.moca.ensemble import Sublattice
from smol.solver.upper_bound.utils.indices import (
    get_variable_indices_for_each_composition_component,
)

__author__ = "Fengyu Xie"


def get_upper_bound_normalization_constraints(
    variables: cp.Variable, variable_indices: List[List[int]]
) -> List[Constraint]:
    """Get normalization constraints of variables on each site.

    Args:
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
    Returns:
        List[Constraint].
    """
    return [
        cp.sum(variables[indices]) == 1
        for indices in variable_indices
        if len(indices) > 0
    ]


def get_upper_bound_composition_space_constraints(
    sublattices: List[Sublattice],
    variables: cp.Variable,
    variable_indices: List[List[int]],
    other_constraints: List[
        Union[str, Tuple[Union[dict, List[dict], List[Number]], Number, str]]
    ] = None,
    initial_occupancy: ArrayLike = None,
) -> List[Constraint]:
    """Get constraints on species composition in CompositionSpace.

    Supports charge balance and other generic composition constraints.
    See moca.CompositionSpace.
    Args:
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
        other_constraints(list): optional
            Other constraints to set in composition space. See moca.composition
            for detailed description.
            Note that constraints are now to be satisfied with the number of sites
            in sub-lattices of the supercell instead of the primitive cell.
        initial_occupancy(ArrayLike): optional
            An initial occupancy used to set the occupancy of manually restricted
            sites that may have more than one allowed species.
            Must be provided if any site has been manually restricted.
    Returns:
        list[Constraint]: Constraints corresponding to the given composition space.
    """
    # Get the variable indices corresponding to each dimension in "counts" format.
    variables_per_component = get_variable_indices_for_each_composition_component(
        sublattices, variable_indices, initial_occupancy
    )

    # Create the composition space.
    bits = [s.species for s in sublattices]
    sizes = [len(s.sites) for s in sublattices]
    comp_space = CompositionSpace(
        bits,
        sizes,
        charge_balanced=True,
        other_constraints=other_constraints,
        optimize_basis=False,
        table_ergodic=False,
    )

    # Charge balance is the first constraint in CompositionSpace.
    n_charge = 1 if comp_space.charge_balanced else 0
    n_sublattices = len(sublattices)
    # Normalization on each sublattice are not necessary because it is already satisfied
    # by normalization on each site.
    skip_cids = list(range(n_charge, n_charge + n_sublattices))
    if np.allclose(comp_space._A[0, :], 0) and np.isclose(comp_space._b[0], 0):
        # Neutral alloy, no need to constrain charge.
        skip_cids = [0] + skip_cids
    constraints = []
    for cid, (a, b) in enumerate(zip(comp_space._A, comp_space._b)):
        # Null constraints.
        if cid in skip_cids or (np.allclose(a, 0) and np.isclose(b, 0)):
            continue
        if np.allclose(a, 0) and not np.isclose(b, 0):
            raise ValueError(f"Unsatisfiable constraint an=b, a: {a}, b: {b}.")

        expression = 0
        for dim_id, (indices, n_fixed) in enumerate(variables_per_component):
            # Active sub-lattice.
            if len(indices) > 0:
                expression += cp.sum(variables[indices]) * a[dim_id]
            expression += n_fixed * a[dim_id]
        constraints.append(expression == b)

    # LEQ.
    for cid, (a, b) in enumerate(zip(comp_space._A_leq, comp_space._b_leq)):
        # Null constraints.
        if np.allclose(a, 0) and b >= 0:
            continue
        if np.allclose(a, 0) and b < 0:
            raise ValueError(f"Unsatisfiable constraint an<=b, a: {a}, b: {b}.")

        expression = 0
        for dim_id, (indices, n_fixed) in enumerate(variables_per_component):
            # Active sub-lattice.
            if len(indices) > 0:
                expression += cp.sum(variables[indices]) * a[dim_id]
            expression += n_fixed * a[dim_id]
        constraints.append(expression <= b)

    # GEQ.
    for cid, (a, b) in enumerate(zip(comp_space._A_geq, comp_space._b_geq)):
        # Null constraints.
        if np.allclose(a, 0) and b <= 0:
            continue
        if np.allclose(a, 0) and b > 0:
            raise ValueError(f"Unsatisfiable constraint an>=b, a: {a}, b: {b}.")

        expression = 0
        for dim_id, (indices, n_fixed) in enumerate(variables_per_component):
            # Active sub-lattice.
            if len(indices) > 0:
                expression += cp.sum(variables[indices]) * a[dim_id]
            expression += n_fixed * a[dim_id]
        constraints.append(expression >= b)

    return constraints


def get_upper_bound_fixed_composition_constraints(
    sublattices: List[Sublattice],
    variables: cp.Variable,
    variable_indices: List[List[int]],
    fixed_composition: List[int],
    initial_occupancy: ArrayLike = None,
) -> List[Constraint]:
    """Fix the count of species in the super-cell.

    Used for searching ground-states in a canonical ensemble.
    Args:
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
        fixed_composition(list[int]):
            Amount of each species to be fixed in the SUPER-cell, in CompositionSpace
            "counts" format. You are fully responsible for setting the order of species
            in the list correctly as would be generated by a CompositionSpace.
        initial_occupancy(ArrayLike): optional
            An initial occupancy used to set the occupancy of manually restricted
            sites that may have more than one allowed species.
            Must be provided if any site has been manually restricted.
    Return:
        list[Constraint]: Constraints corresponding to a fixed composition on each
        sub-lattice.
    """
    # Get the variable indices corresponding to each dimension in "counts" format.
    variables_per_component = get_variable_indices_for_each_composition_component(
        sublattices, variable_indices, initial_occupancy
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
        constraints.append(expression == fixed_composition[dim_id])

    return constraints
