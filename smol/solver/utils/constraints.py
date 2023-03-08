"""Get constraints on variables from a processor."""
from typing import List, Union

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint

from smol.moca import CompositionSpace


def get_normalization_constraints(
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
    return [cp.sum(variables[indices]) == 1 for indices in variable_indices]


def get_dim_id_to_var_ids_mapping(
    sublattice_dim_ids: List[List[int]],
    variable_indices: List[List[int]],
    sublattice_sites: List[List[int]],
) -> List[Union[List[int], int]]:
    """Get mapping from composition vector component index to variable indices.

    Args:
        sublattice_dim_ids(list[list[int]]):
            Index of composition vector component for each species in each sub-lattice.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
        sublattice_sites(list[list[int]]):
            Index of sites in each sub-lattice of a super-cell.
            variable_indices, composition_space and sublattice_sites must be generated
            from the same processor!
    Returns:
        list[list[int]|int]:
            Variable indices corresponding to each component index if the sub-lattice has
            more than one species, or the number of sites in sub-lattice if the sub-lattice
            has only one species.
    """
    n_dims = sum([len(dims) for dims in sublattice_dim_ids])
    dim_id_to_var_ids = [[] for _ in range(n_dims)]
    for sublattice_id, dim_ids in enumerate(sublattice_dim_ids):
        sites = sublattice_sites[sublattice_id]
        if len(dim_ids) == 1:  # Inactive sub-lattice.
            dim_id = dim_ids[0]
            # Save the number of sites as all occupied by a single species.
            dim_id_to_var_ids[dim_id] = len(sites)
        else:
            for species_id, dim_id in enumerate(dim_ids):
                dim_id_to_var_ids[dim_id] = [
                    variable_indices[site_id][species_id] for site_id in sites
                ]
    return dim_id_to_var_ids


def get_composition_space_constraints(
    variables: cp.Variable,
    variable_indices: List[List[int]],
    composition_space: CompositionSpace,
    supercell_size: int,
    sublattice_sites: List[List[int]],
) -> List[Constraint]:
    """Get constraints on species composition with CompositionSpace.

    Supports charge balance and other generic composition constraints. See moca.CompositionSpace.
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
        List[Constraint].
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
                expression += variables[indices] * aa[dim_id]
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
                expression += variables[indices] * aa[dim_id]
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
                expression += variables[indices] * aa[dim_id]
            else:
                expression += indices * aa[dim_id]
        constraints.append(expression >= bb)

    return constraints
