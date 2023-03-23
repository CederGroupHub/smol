"""Generate cvxpy objectives from processor."""
from itertools import product
from numbers import Number
from typing import List, Union

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike

from smol.moca.ensemble import Sublattice
from smol.moca.processor import (
    ClusterDecompositionProcessor,
    ClusterExpansionProcessor,
    EwaldProcessor,
)
from smol.solver.upper_bound.utils.indices import map_ewald_indices_to_variable_indices

__author__ = "Fengyu Xie"


def get_upper_bound_objective_from_expansion_processor(
    sublattices: List[Sublattice],
    variables: cp.Variable,
    variable_indices: List[List[int]],
    expansion_processor: ClusterExpansionProcessor,
    initial_occupancy: ArrayLike = None,
) -> Union[cp.Expression, Number]:
    """Get the objective function from cluster expansion processor.

    Args:
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
        expansion_processor(ClusterExpansionProcessor):
            A cluster expansion processor to generate objective function with.
        initial_occupancy(ArrayLike): optional
            An initial occupancy used to set the occupancy of manually restricted
            sites that may have more than one allowed species.
            Must be provided if any site has been manually restricted.
    Returns:
        cp.Expression:
            Expression for the cluster expansion energy in a super-cell.
    """
    coefs = expansion_processor.coefs
    function_terms = [0 for _ in range(expansion_processor.num_corr_functions)]
    function_terms[0] = 1  # empty cluster

    num_sites = len(variable_indices)
    site_sublattice_ids = np.zeros(num_sites, dtype=int) - 1
    for sublattice_id, sublattice in enumerate(sublattices):
        site_sublattice_ids[sublattice.sites] = sublattice_id

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
                    # Active site.
                    if len(variable_indices[sid_in_supercell]) > 0:
                        site_states.append(
                            list(range(len(variable_indices[sid_in_supercell])))
                        )
                    else:
                        sublattice_id = site_sublattice_ids[sid_in_supercell]
                        sublattice = sublattices[sublattice_id]
                        # Inactive sublattice.
                        if len(sublattice.species) == 1:
                            state_id = 0
                        elif len(sublattice.species) == 0:
                            raise ValueError(
                                f"Encountered empty sub-lattice on site"
                                f" {sid_in_supercell}."
                                f" Sub-lattice: {sublattice}."
                            )
                        # Active sublattice, but site manually restricted.
                        else:
                            if initial_occupancy is None:
                                raise ValueError(
                                    f"Site {sid_in_supercell} was manually restricted"
                                    f" in sub-lattice: {sublattice}, but no initial"
                                    f" occupancy was specified!"
                                )
                            # Manually fixed to the only initial species.
                            state_id = np.where(
                                sublattice.encoding
                                == initial_occupancy[sid_in_supercell]
                            )[0][0]
                        site_states.append([state_id])

                # Only one of these cluster states can be active when a solution is valid.
                for cluster_state in product(*site_states):
                    state_index_in_tensor = 0
                    product_term = 1
                    # Get index to query.
                    for sid_in_cluster in range(n_sites):
                        sid_in_supercell = indices[cid, sid_in_cluster]
                        sublattice_id = site_sublattice_ids[sid_in_supercell]
                        sublattice = sublattices[sublattice_id]
                        species_id_in_site = cluster_state[sid_in_cluster]
                        code_in_site = sublattice.encoding[species_id_in_site]
                        state_index_in_tensor += (
                            tensor_indices[sid_in_cluster] * code_in_site
                        )
                        if len(variable_indices[sid_in_supercell]) > 0:
                            var_id = variable_indices[sid_in_supercell][
                                species_id_in_site
                            ]
                            product_term *= variables[var_id]
                    p += corr_tensors[bid, state_index_in_tensor] * product_term
            function_terms[n] = p / n_clusters
            n += 1

    expression = 0
    for function, coef in zip(function_terms, coefs):
        expression += function * coef

    return expression * expansion_processor.size


def get_upper_bound_objective_from_decomposition_processor(
    sublattices: List[Sublattice],
    variables: cp.Variable,
    variable_indices: List[List[int]],
    decomposition_processor: ClusterDecompositionProcessor,
    initial_occupancy: ArrayLike = None,
) -> Union[cp.Expression, Number]:
    """Get the objective function from cluster expansion processor.

    Args:
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
        decomposition_processor(ClusterDecompositionProcessor):
            A cluster decomposition processor to generate objective function with.
        initial_occupancy(ArrayLike): optional
            An initial occupancy used to set the occupancy of manually restricted
            sites that may have more than one allowed species.
            Must be provided if any site has been manually restricted.
    Returns:
        cp.Expression:
            Expression for the cluster expansion energy in a super-cell.
    """
    coefs = decomposition_processor.coefs
    orbit_terms = [0 for _ in range(decomposition_processor.n_orbits)]
    orbit_terms[0] = decomposition_processor._fac_tensors[0]  # empty cluster

    num_sites = len(variable_indices)
    site_sublattice_ids = np.zeros(num_sites, dtype=int) - 1
    for sublattice_id, sublattice in enumerate(sublattices):
        site_sublattice_ids[sublattice.sites] = sublattice_id

    n = 1
    orbit_list = decomposition_processor._orbit_list
    for tensor_indices, interaction_tensors, indices in orbit_list:
        n_clusters = indices.shape[0]  # cluster index.
        n_sites = indices.shape[1]  # index within cluster.
        p = 0
        for cid in range(n_clusters):
            # Get possible occupancy states for each site.
            site_states = []
            for sid_in_cluster in range(n_sites):
                sid_in_supercell = indices[cid, sid_in_cluster]
                # Active site.
                if len(variable_indices[sid_in_supercell]) > 0:
                    site_states.append(
                        list(range(len(variable_indices[sid_in_supercell])))
                    )
                else:
                    sublattice_id = site_sublattice_ids[sid_in_supercell]
                    sublattice = sublattices[sublattice_id]
                    # Inactive sublattice.
                    if len(sublattice.species) == 1:
                        state_id = 0
                    elif len(sublattice.species) == 0:
                        raise ValueError(
                            f"Encountered empty sub-lattice on site"
                            f" {sid_in_supercell}."
                            f" Sub-lattice: {sublattice}."
                        )
                    # Active sublattice, but site manually restricted.
                    else:
                        if initial_occupancy is None:
                            raise ValueError(
                                f"Site {sid_in_supercell} was manually restricted"
                                f" in sub-lattice: {sublattice}, but no initial"
                                f" occupancy was specified!"
                            )
                        # Manually fixed to the only initial species.
                        state_id = np.where(
                            sublattice.encoding == initial_occupancy[sid_in_supercell]
                        )[0][0]
                    site_states.append([state_id])

                # Only one of these cluster states can be active when a solution is valid.
                for cluster_state in product(*site_states):
                    state_index_in_tensor = 0
                    product_term = 1
                    # Get index to query.
                    for sid_in_cluster in range(n_sites):
                        sid_in_supercell = indices[cid, sid_in_cluster]
                        sublattice_id = site_sublattice_ids[sid_in_supercell]
                        sublattice = sublattices[sublattice_id]
                        species_id_in_site = cluster_state[sid_in_cluster]
                        code_in_site = sublattice.encoding[species_id_in_site]

                        state_index_in_tensor += (
                            tensor_indices[sid_in_cluster] * code_in_site
                        )
                        if len(variable_indices[sid_in_supercell]) > 0:
                            var_id = variable_indices[sid_in_supercell][
                                species_id_in_site
                            ]
                            product_term *= variables[var_id]
                    p += interaction_tensors[state_index_in_tensor] * product_term

        orbit_terms[n] = p / n_clusters
        n += 1

    expression = 0
    for orbit, coef in zip(orbit_terms, coefs):
        expression += orbit * coef

    return expression * decomposition_processor.size


def get_upper_bound_objective_from_ewald_processor(
    sublattices: List[Sublattice],
    variables: cp.Variable,
    variable_indices: List[List[int]],
    ewald_processor: EwaldProcessor,
    initial_occupancy: ArrayLike = None,
) -> Union[cp.Expression, Number]:
    """Get the objective function from cluster expansion processor.

    Args:
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
        ewald_processor(ClusterDecompositionProcessor):
            A cluster decomposition processor to generate objective function with.
            You are fully responsible for checking variables have the correct site
            orderings as will be in ewald_processor.
        initial_occupancy(ArrayLike): optional
            An initial occupancy used to set the occupancy of manually restricted
            sites that may have more than one allowed species.
            Must be provided if any site has been manually restricted.
    Returns:
        cp.Expression:
            Expression for the ewald energy in a super-cell.
    """
    ewald_matrix = ewald_processor.ewald_matrix

    # Should be square matrix.
    n_ewald_rows = ewald_matrix.shape[0]

    ewald_to_var_ids = map_ewald_indices_to_variable_indices(
        sublattices, variable_indices, initial_occupancy
    )
    expression = 0
    # Do not diagonalize because ewald matrix already contains 1/2.
    for i in n_ewald_rows:
        for j in n_ewald_rows:
            row_var_id = ewald_to_var_ids[i]
            col_var_id = ewald_to_var_ids[j]
            # Some species are never possible on some sites because of
            # manual restriction. Should skip any associated element.
            if row_var_id == -2 or col_var_id == -2:
                continue
            row_var = variables[row_var_id] if row_var_id != -1 else 1
            col_var = variables[col_var_id] if col_var_id != -1 else 1
            expression += ewald_matrix[i, j] * row_var * col_var

    # No need to multiply because system size already included.
    return expression * ewald_processor.coefs


def get_upper_bound_objective_from_chemical_potentials(
    sublattices: List[Sublattice],
    variables: cp.Variable,
    variable_indices: List[List[int]],
    chemical_table: ArrayLike,
    initial_occupancy: ArrayLike = None,
) -> Union[cp.Expression, Number]:
    """Get the objective function from chemical potentials.

    Notice: returns the -mu N term. Negation already included.
    Args:
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
        chemical_table(ArrayLike):
            Chemical potentials corresponding to each species (cols) on each site
            (rows). Simply use ensemble._chemical_potential["table"].
        initial_occupancy(ArrayLike): optional
            An initial occupancy used to set the occupancy of manually restricted
            sites that may have more than one allowed species.
            Must be provided if any site has been manually restricted.
    Returns:
        cp.Expression:
            Expression for the chemical work energy in a super-cell.
    """
    num_sites = len(variable_indices)
    site_sublattice_ids = np.zeros(num_sites, dtype=int) - 1
    for sublattice_id, sublattice in enumerate(sublattices):
        site_sublattice_ids[sublattice.sites] = sublattice_id

    expression = 0
    for site_id in range(num_sites):
        sublattice_id = site_sublattice_ids[site_id]
        sublattice = sublattices[sublattice_id]
        # Active site.
        if site_id in sublattice.active_sites:
            codes = sublattice.encoding
            expression += (
                chemical_table[site_id, codes] @ variables[variable_indices[site_id]]
            )
        # Inactive sub-lattice.
        elif len(sublattice.species) == 1:
            expression += chemical_table[site_id, sublattice.encoding[0]]
        elif len(sublattice.species) == 0:
            raise ValueError(
                f"Encountered empty sub-lattice on site"
                f" {site_id}. Sub-lattice: {sublattice}."
            )
        # Manually restricted site.
        else:
            if initial_occupancy is None:
                raise ValueError(
                    f"Site {site_id} was manually restricted"
                    f" in sub-lattice: {sublattice}, but no initial"
                    f" occupancy was specified!"
                )
            expression += chemical_table[site_id, initial_occupancy[site_id]]

    # No need to multiply because system size already included.
    return -1 * expression
