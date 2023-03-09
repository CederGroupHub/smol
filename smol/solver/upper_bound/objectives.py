"""Generate cvxpy objectives from processor."""
from itertools import product
from numbers import Number
from typing import List, Union

import cvxpy as cp
from numpy.typing import ArrayLike

from smol.cofe.space.domain import get_allowed_species
from smol.moca.processor import (
    ClusterDecompositionProcessor,
    ClusterExpansionProcessor,
    EwaldProcessor,
)
from smol.solver.upper_bound.indices import (
    get_dim_id_to_var_ids_mapping,
    get_ewald_id_to_var_id_mapping,
)

__author__ = "Fengyu Xie"


def get_upper_bound_objective_from_expansion_processor(
    variables: cp.Variable,
    variable_indices: List[List[int]],
    expansion_processor: ClusterExpansionProcessor,
) -> Union[cp.Expression, Number]:
    """Get the objective function from cluster expansion processor.

    Args:
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
        expansion_processor(ClusterExpansionProcessor):
            A cluster expansion processor to generate objective function with.
    Returns:
        cp.Expression:
            Expression for the cluster expansion energy in a super-cell.
    """
    coefs = expansion_processor.coefs
    function_terms = [0 for _ in range(expansion_processor.num_corr_functions)]
    function_terms[0] = 1  # empty cluster

    orbit_list = expansion_processor._orbit_list
    for n, tensor_indices, corr_tensors, indices in orbit_list:
        n_bit_combos = corr_tensors.shape[0]  # index of bit combos
        n_clusters = indices.shape[0]  # cluster index
        n_sites = indices.shape[1]  # index within cluster
        for bid in range(n_bit_combos):
            p = 0
            for cid in range(n_clusters):
                # Get possible occupancy states for each site.
                site_states = []
                for sid_in_cluster in range(n_sites):
                    sid_in_supercell = indices[cid, sid_in_cluster]
                    site_states.append(
                        list(range(len(variable_indices[sid_in_supercell])))
                    )

                # Only one of these cluster states can be active when a solution is valid.
                for cluster_state in product(*site_states):
                    state_index_in_tensor = 0
                    product_term = 1
                    # Get index to query.
                    for sid_in_cluster in range(n_sites):
                        sid_in_supercell = indices[cid, sid_in_cluster]
                        vid_in_site = cluster_state[sid_in_cluster]
                        state_index_in_tensor += (
                            tensor_indices[sid_in_cluster] * vid_in_site
                        )
                        var_id = variable_indices[sid_in_supercell][vid_in_site]
                        product_term *= variables[var_id]
                    p += corr_tensors[bid, state_index_in_tensor] * product_term
            function_terms[n] = p / n_clusters
            n += 1

    expression = 0
    for function, coef in zip(function_terms, coefs):
        expression += function * coef

    return expression * expansion_processor.size


def get_upper_bound_objective_from_decomposition_processor(
    variables: cp.Variable,
    variable_indices: List[List[int]],
    decomposition_processor: ClusterDecompositionProcessor,
) -> Union[cp.Expression, Number]:
    """Get the objective function from cluster expansion processor.

    Args:
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
        decomposition_processor(ClusterDecompositionProcessor):
            A cluster decomposition processor to generate objective function with.
    Returns:
        cp.Expression:
            Expression for the cluster expansion energy in a super-cell.
    """
    coefs = decomposition_processor.coefs
    orbit_terms = [0 for _ in range(decomposition_processor.n_orbits)]
    orbit_terms[0] = decomposition_processor._fac_tensors[0]  # empty cluster

    n = 1
    orbit_list = decomposition_processor._orbit_list
    for tensor_indices, interaction_tensors, indices in orbit_list:
        n_clusters = indices.shape[0]  # cluster index.
        n_sites = indices.shape[1]  # index within cluster.
        p = 0
        for cid in range(n_clusters):
            site_states = []
            for sid_in_cluster in range(n_sites):
                sid_in_supercell = indices[cid, sid_in_cluster]
                site_states.append(list(range(len(variable_indices[sid_in_supercell]))))

                # Only one of these cluster states can be active when a solution is valid.
                for cluster_state in product(*site_states):
                    state_index_in_tensor = 0
                    product_term = 1
                    # Get index to query.
                    for sid_in_cluster in range(n_sites):
                        sid_in_supercell = indices[cid, sid_in_cluster]
                        vid_in_site = cluster_state[sid_in_cluster]
                        state_index_in_tensor += (
                            tensor_indices[sid_in_cluster] * vid_in_site
                        )
                        var_id = variable_indices[sid_in_supercell][vid_in_site]
                        product_term *= variables[var_id]
                    p += interaction_tensors[state_index_in_tensor] * product_term

        orbit_terms[n] = p / n_clusters
        n += 1

    expression = 0
    for orbit, coef in zip(orbit_terms, coefs):
        expression += orbit * coef

    return expression * decomposition_processor.size


def get_upper_bound_objective_from_ewald_processor(
    variables: cp.Variable,
    variable_indices: List[List[int]],
    ewald_processor: EwaldProcessor,
) -> Union[cp.Expression, Number]:
    """Get the objective function from cluster expansion processor.

    Args:
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
        ewald_processor(ClusterDecompositionProcessor):
            A cluster decomposition processor to generate objective function with.
            You are fully responsible for checking variables have the correct site
            orderings as will be in ewald_processor.
    Returns:
        cp.Expression:
            Expression for the ewald energy in a super-cell.
    """
    ewald_matrix = ewald_processor.ewald_matrix

    # Should be square matrix.
    n_ewald_rows = ewald_matrix.shape[0]

    ewald_to_var_ids = get_ewald_id_to_var_id_mapping(
        n_ewald_rows, variable_indices, get_allowed_species(ewald_processor.structure)
    )
    expression = 0
    # Do not diagonalize because ewald matrix already contains 1/2.
    for i in n_ewald_rows:
        for j in n_ewald_rows:
            row_var_id = ewald_to_var_ids[i]
            col_var_id = ewald_to_var_ids[j]
            row_var = variables[row_var_id] if row_var_id != -1 else 1
            col_var = variables[col_var_id] if col_var_id != -1 else 1
            expression += ewald_matrix[i, j] * row_var * col_var

    # No need to divide.
    return expression


def get_upper_bound_objective_from_chemical_potentials(
    variables: cp.Variable,
    variable_indices: List[List[int]],
    dim_ids_in_sublattices: List[List[int]],
    sublattice_sites: List[List[int]],
    chemical_potentials: ArrayLike,
) -> Union[cp.Expression, Number]:
    """Get the objective function from chemical potentials.

    Args:
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
        dim_ids_in_sublattices(List[List[int]]):
            Indices of "counts" format composition vector corresponding to each species
            on each sub-lattice.
        sublattice_sites(list[list[int]]):
            Index of sites in each sub-lattice of a super-cell.
            variable_indices, composition_space and sublattice_sites must be generated
            from the same processor!
        chemical_potentials(ArrayLike):
            Chemical potentials corresponding to each species on each sub-lattie, and
            should have the same length and ordering as components in CompSpace's
            "counts" format. You are fully responsible for setting it correctly, with
            the same type of species on different sub-lattices having the same chemical
            potential.
            Inactive species are always considered to have 0 chemical potential, regardless
            of the value set.
    Returns:
        cp.Expression:
            Expression for the chemical work energy in a super-cell.
    """
    dim_id_to_var_ids = get_dim_id_to_var_ids_mapping(
        dim_ids_in_sublattices, variable_indices, sublattice_sites
    )

    expression = 0
    for dim_id, var_ids in enumerate(dim_id_to_var_ids):
        if isinstance(var_ids, list):
            expression += cp.sum(variables[var_ids]) * chemical_potentials[dim_id]
        else:
            expression += var_ids * 0

    return expression
