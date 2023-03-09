"""Get constraints on variables from a processor."""
from typing import List

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint

from smol.moca import CompositionSpace
from smol.solver.upper_bound.indices import get_dim_id_to_var_ids_mapping

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


# Not directly tested.
def get_upper_bound_composition_space_constraints(
    variables: cp.Variable,
    variable_indices: List[List[int]],
    composition_space: CompositionSpace,
    supercell_size: int,
    sublattice_sites: List[List[int]],
) -> List[Constraint]:
    """Get constraints on species composition with CompositionSpace.

    Supports charge balance and other generic composition constraints.
    See moca.CompositionSpace.
    Args:
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
        composition_space(CompositionSpace):
            Composition space with all constraints encoded in the "counts" format.
        supercell_size(int):
            Size of the super-cell in the number of primitive cells.
        sublattice_sites(list[list[int]]):
            Index of sites in each sub-lattice of a super-cell.
            variable_indices, composition_space and sublattice_sites must be generated
            from the same processor!
    Returns:
        list[Constraint]: Constraints corresponding to the given composition space.
    """
    # Get the variable indices corresponding to each dimension in "counts" format.
    dim_id_to_var_ids = get_dim_id_to_var_ids_mapping(
        composition_space.dim_ids, variable_indices, sublattice_sites
    )

    # Charge balance is the first constraint in CompositionSpace.
    n_charge = 1 if composition_space.charge_balanced else 0
    n_sublattices = len(composition_space.dim_ids)
    # Normalization on each sublattice are not necessary because it is already satisfied
    # by normalization on each site.
    skip_cids = list(range(n_charge, n_charge + n_sublattices))
    constraints = []
    for cid, (a, b) in enumerate(zip(composition_space._A, composition_space._b)):
        # Null constraints.
        if cid in skip_cids or (np.allclose(a, 0) and np.isclose(b, 0)):
            continue
        if np.allclose(a, 0) and not np.isclose(b, 0):
            raise ValueError(f"Unsatisfiable constraint an=b, a: {a}, b: {b}.")

        aa = np.round(a).astype(int)
        bb = int(np.round(b) * supercell_size)

        expression = 0
        for dim_id, indices in enumerate(dim_id_to_var_ids):
            # Active sub-lattice.
            if isinstance(indices, list):
                expression += cp.sum(variables[indices]) * aa[dim_id]
            else:
                expression += indices * aa[dim_id]
        constraints.append(expression == bb)

    # LEQ.
    for cid, (a, b) in enumerate(
        zip(composition_space._A_leq, composition_space._b_leq)
    ):
        # Null constraints.
        if np.allclose(a, 0) and b >= 0:
            continue
        if np.allclose(a, 0) and b < 0:
            raise ValueError(f"Unsatisfiable constraint an<=b, a: {a}, b: {b}.")

        aa = a
        bb = b * supercell_size

        expression = 0
        for dim_id, indices in enumerate(dim_id_to_var_ids):
            # Active sub-lattice.
            if isinstance(indices, list):
                expression += cp.sum(variables[indices]) * aa[dim_id]
            else:
                expression += indices * aa[dim_id]
        constraints.append(expression <= bb)

    # GEQ.
    for cid, (a, b) in enumerate(
        zip(composition_space._A_geq, composition_space._b_geq)
    ):
        # Null constraints.
        if np.allclose(a, 0) and b <= 0:
            continue
        if np.allclose(a, 0) and b > 0:
            raise ValueError(f"Unsatisfiable constraint an>=b, a: {a}, b: {b}.")

        aa = a
        bb = b * supercell_size

        expression = 0
        for dim_id, indices in enumerate(dim_id_to_var_ids):
            # Active sub-lattice.
            if isinstance(indices, list):
                expression += cp.sum(variables[indices]) * aa[dim_id]
            else:
                expression += indices * aa[dim_id]
        constraints.append(expression >= bb)

    return constraints


def get_upper_bound_fixed_composition_constraints(
    variables: cp.Variable,
    variable_indices: List[List[int]],
    fixed_composition: List[int],
    dim_ids_in_sublattices: List[List[int]],
    sublattice_sites: List[List[int]],
) -> List[Constraint]:
    """Fix the count of species in the super-cell.

    Used for searching ground-states in a canonical ensemble.
    Args:
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
        fixed_composition(list[int]):
            Amount of each species to be fixed in the super-cell, in CompositionSpace
            "counts" format. You are fully responsible for setting the order of species
            in the list correctly as would be generated by a CompositionSpace.
        dim_ids_in_sublattices(List[List[int]]):
            Indices of "counts" format composition vector corresponding to each species
            on each sub-lattice.
        sublattice_sites(list[list[int]]):
            Index of sites in each sub-lattice of a super-cell.
            variable_indices, composition_space and sublattice_sites must be generated
            from the same processor!
    Return:
        list[Constraint]: Constraints corresponding to a fixed composition on each
        sub-lattice.
    """
    dim_id_to_var_ids = get_dim_id_to_var_ids_mapping(
        dim_ids_in_sublattices, variable_indices, sublattice_sites
    )

    constraints = []
    for dim_id, indices in enumerate(dim_id_to_var_ids):
        if isinstance(indices, list):
            constraints.append(cp.sum(variables[indices]) == fixed_composition[dim_id])
        elif fixed_composition[dim_id] != indices:
            raise ValueError(
                f"Composition component {dim_id} is an inactive"
                f" sub-lattice with {indices} sites, but"
                f" fixed_composition only has {fixed_composition[dim_id]}"
                f" sites!"
            )

    return constraints
