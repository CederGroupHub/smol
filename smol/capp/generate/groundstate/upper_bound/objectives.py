"""Generate energy terms from processor to be converted into cvxpy."""
from copy import deepcopy
from itertools import product
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from smol.capp.generate.groundstate.upper_bound.indices import (
    map_ewald_indices_to_variable_indices,
)
from smol.moca.processor import (
    ClusterDecompositionProcessor,
    ClusterExpansionProcessor,
    EwaldProcessor,
)

__author__ = "Fengyu Xie"


def get_terms_from_expansion_processor(
    variable_indices: List[List[int]],
    expansion_processor: ClusterExpansionProcessor,
    group_output_by_function: bool = False,
) -> Union[
    List[Tuple[List[int], float, float]], List[List[Tuple[List[int], float, float]]]
]:
    """Get the cluster terms from cluster expansion processor.

    Args:
        variable_indices(list of lists of int):
            List of variable indices corresponding to each active site index and
            index of species in its site space. Inactive sites will be marked by
            either -1 or -2. See documentation in groundstate.upper_bound.variables.
        expansion_processor(ClusterExpansionProcessor):
            A cluster expansion processor to generate objective function with.
        group_output_by_function(bool): optional
            Whether to group the resulting terms by correlation functions, such
            that the output will be a list of lists, with each sub-list corresponding
            to a correlation function. Default is false.
            This is used specifically for checking the value of each correlation
            function in the objective function.
    Returns:
        list of tuples of (list of int, float, float):
            A list of tuples, each represents a cluster term in the energy
            representation, containing indices of variables to be taken product with,
            the correlation factor, and the corresponding coefficient.
            Energy is taken per super-cell.
            If group_output_by_function is enabled, the output will be a list of lists,
            with each sub-list corresponding to a correlation function.
    """
    coefs = expansion_processor.coefs
    # Store variable indices and the constant product in the cluster term.
    cluster_terms = [
        [] for _ in range(expansion_processor.cluster_subspace.num_corr_functions)
    ]
    cluster_terms[0] = [([], 1.0)]

    space = expansion_processor.cluster_subspace
    sc_matrix = expansion_processor.supercell_matrix
    mappings = space.supercell_orbit_mappings(sc_matrix)
    for orbit, mapping in zip(space.orbits, mappings):
        # Use un-flatten version now for easier access.
        n = deepcopy(orbit.bit_id)
        corr_tensors = orbit.correlation_tensors
        n_bit_combos = corr_tensors.shape[0]  # index of bit combos
        mapping = np.array(mapping, dtype=int)
        n_clusters = mapping.shape[0]  # cluster image index
        n_sites = mapping.shape[1]  # index within cluster
        for bid in range(n_bit_combos):
            for cid in range(n_clusters):
                # Get possible occupancy states for each site.
                site_states = []
                for sid_in_cluster in range(n_sites):
                    sid_in_supercell = mapping[cid, sid_in_cluster]
                    # Indices corresponding to un-split site space.
                    site_variable_indices = np.array(
                        variable_indices[sid_in_supercell], dtype=int
                    )
                    # -1 should never co-appear with non-negative variable index
                    # on a site.
                    # Some active variables exist.
                    if np.any(site_variable_indices >= 0):
                        site_states.append(
                            np.where(site_variable_indices >= 0)[0].tolist()
                        )
                    # Nothing active, then one must be -1. Can not have multiple -1.
                    else:
                        site_states.append(
                            np.where(site_variable_indices == -1)[0].tolist()
                        )

                # Only one of these cluster states can be active when a solution is valid.
                for cluster_state in product(*site_states):
                    # Tuple can be used to query a single element.
                    cluster_factor = corr_tensors[tuple((bid, *cluster_state))]
                    cluster_variable_indices = []
                    # Get index to query.
                    for sid_in_cluster in range(n_sites):
                        sid_in_supercell = mapping[cid, sid_in_cluster]
                        species_id_in_site = cluster_state[sid_in_cluster]
                        var_id = variable_indices[sid_in_supercell][species_id_in_site]
                        if var_id >= 0:
                            cluster_variable_indices.append(var_id)
                        else:
                            assert var_id == -1
                    # Divide by n_clusters to normalize.
                    cluster_terms[n].append(
                        [cluster_variable_indices, cluster_factor / n_clusters]
                    )
            n += 1

    # Put in system size and coefficients info.
    if not group_output_by_function:
        return [
            (inds, factor, coef * expansion_processor.size)
            for terms, coef in zip(cluster_terms, coefs)
            for inds, factor in terms
        ]
    else:
        return [
            [(inds, factor, coef * expansion_processor.size) for inds, factor in terms]
            for terms, coef in zip(cluster_terms, coefs)
        ]


def get_terms_from_decomposition_processor(
    variable_indices: List[List[int]],
    decomposition_processor: ClusterDecompositionProcessor,
    group_output_by_orbit: bool = False,
) -> Union[
    List[Tuple[List[int], float, float]], List[List[Tuple[List[int], float, float]]]
]:
    """Get the cluster terms from cluster decomposition processor.

    Args:
        variable_indices(list of lists of int):
            List of variable indices corresponding to each active site index and
            index of species in its site space. Inactive sites will be marked by
            either -1 or -2. See documentation in groundstate.upper_bound.variables.
        decomposition_processor(ClusterDecompositionProcessor):
            A cluster decomposition processor to generate objective function with.
        group_output_by_orbit(bool): optional
            Whether to group the resulting terms by orbits, such
            that the output will be a list of lists, with each sub-list corresponding
            to an orbit. Default is false.
            This is used specifically for checking the value of each interaction
            term in the objective function.
    Returns:
        list of tuples of (list of int, float, float):
            A list of tuples, each represents a cluster term in the energy
            representation, containing indices of variables to be taken product with,
            the interaction factor and the multiplicity coefficient.
            Energy is taken per super-cell.
            If group_output_by_orbit is enabled, the output will be a list of lists,
            with each sub-list corresponding to an orbit.
    """
    coefs = decomposition_processor.coefs  # Actually multiplicities.
    orbit_terms = [
        [] for _ in range(decomposition_processor.cluster_subspace.num_orbits)
    ]
    orbit_tensors = decomposition_processor._interaction_tensors
    # Use list in inner-layer as tuple does not support value assignment.
    orbit_terms[0] = [([], orbit_tensors[0])]

    n = 1
    space = decomposition_processor.cluster_subspace
    sc_matrix = decomposition_processor.supercell_matrix
    mappings = space.supercell_orbit_mappings(sc_matrix)
    for mapping in mappings:
        mapping = np.array(mapping, dtype=int)
        n_clusters = mapping.shape[0]  # cluster index.
        n_sites = mapping.shape[1]  # index within cluster.
        for cid in range(n_clusters):
            # Get possible occupancy states for each site.
            site_states = []
            for sid_in_cluster in range(n_sites):
                sid_in_supercell = mapping[cid, sid_in_cluster]
                # Indices corresponding to un-split site space.
                site_variable_indices = np.array(
                    variable_indices[sid_in_supercell], dtype=int
                )
                # -1 should never co-appear with non-negative variable index
                # on a site.
                # Some active variables exist.
                if np.any(site_variable_indices >= 0):
                    site_states.append(np.where(site_variable_indices >= 0)[0].tolist())
                # Nothing active, then one must be -1. Can not have multiple -1.
                else:
                    site_states.append(
                        np.where(site_variable_indices == -1)[0].tolist()
                    )

            # Only one of these cluster states can be active when a solution is valid.
            for cluster_state in product(*site_states):
                cluster_factor = orbit_tensors[n][tuple(cluster_state)]
                cluster_variable_indices = []
                # Get index to query.
                for sid_in_cluster in range(n_sites):
                    sid_in_supercell = mapping[cid, sid_in_cluster]
                    species_id_in_site = cluster_state[sid_in_cluster]

                    var_id = variable_indices[sid_in_supercell][species_id_in_site]
                    if var_id >= 0:
                        cluster_variable_indices.append(var_id)
                    else:
                        assert var_id == -1
                # Divide by n_clusters to normalize.
                orbit_terms[n].append(
                    [cluster_variable_indices, cluster_factor / n_clusters]
                )

        n += 1

    # Put in system size and coefficients info.
    if not group_output_by_orbit:
        return [
            (inds, factor, coef * decomposition_processor.size)
            for terms, coef in zip(orbit_terms, coefs)
            for inds, factor in terms
        ]
    else:
        return [
            [
                (inds, factor, coef * decomposition_processor.size)
                for inds, factor in terms
            ]
            for terms, coef in zip(orbit_terms, coefs)
        ]


def get_terms_from_ewald_processor(
    variable_indices: List[List[int]],
    ewald_processor: EwaldProcessor,
) -> List[Tuple[List[int], float, float]]:
    """Get the objective function from cluster expansion processor.

    Args:
        variable_indices(list of lists of int):
            List of variable indices corresponding to each active site index and
            index of species in its site space. Inactive sites will be marked by
            either -1 or -2. See documentation in groundstate.upper_bound.variables.
        ewald_processor(ClusterDecompositionProcessor):
            A cluster decomposition processor to generate objective function with.
            You are fully responsible for checking variables have the correct site
            orderings as will be in ewald_processor.
    Returns:
        list of tuples of (list of int, float, float):
            A list of tuples, each represents a cluster term in the energy
            representation, containing indices of variables to be taken product with,
            and the factors before the boolean product.
            Energy is taken per super-cell.
    """
    ewald_matrix = ewald_processor.ewald_matrix
    ewald_cluster_terms = []

    # Should be square matrix.
    n_ewald_rows = ewald_matrix.shape[0]

    ewald_to_var_ids = map_ewald_indices_to_variable_indices(
        ewald_processor.structure,
        variable_indices,
    )
    # Do not diagonalize because ewald matrix already contains 1/2.
    for i in range(n_ewald_rows):
        for j in range(n_ewald_rows):
            row_var_id = ewald_to_var_ids[i]
            col_var_id = ewald_to_var_ids[j]
            # Some species are never possible on some sites because of
            # manual restriction. Should skip any associated element.
            if row_var_id == -2 or col_var_id == -2:
                continue

            ewald_factor = ewald_matrix[i, j]
            ewald_variable_indices = []
            if row_var_id != -1:
                ewald_variable_indices.append(row_var_id)
            # Diagonal should also be a point term.
            if col_var_id != -1 and col_var_id != row_var_id:
                ewald_variable_indices.append(col_var_id)
            # Ewald processor coefficient is forced to be a 1D array.
            # Should convert it back.
            ewald_cluster_terms.append(
                (ewald_variable_indices, ewald_factor, float(ewald_processor.coefs))
            )

    # No need to multiply because system size is already included.
    return ewald_cluster_terms


def get_terms_from_chemical_potentials(
    variable_indices: List[List[int]],
    chemical_table: ArrayLike,
) -> List[Tuple[List[int], float, float]]:
    """Get the objective function from chemical potentials.

    Notice: returns the -mu N term. Negation already included.
    Args:
        variable_indices(list of lists of int):
            List of variable indices corresponding to each active site index and
            index of species in its site space. Inactive sites will be marked by
            either -1 or -2. See documentation in groundstate.upper_bound.variables.
        chemical_table(ArrayLike):
            Chemical potentials corresponding to each species (cols) on each site
            (rows). Simply use ensemble._chemical_potential["table"].
    Returns:
        list of tuples of (list of int, float):
            A list of tuples, each represents a cluster term in the energy
            representation, containing indices of variables to be taken product with,
            and the factor before the boolean product.
            Energy is taken per super-cell.
    """
    num_sites = len(variable_indices)

    point_terms = []
    for site_id in range(num_sites):
        # Assume table column indices match exactly with species codes.
        # Only an ensemble created from cluster expansion can satisfy that.
        for code, var_id in enumerate(variable_indices[site_id]):
            mu = chemical_table[site_id, code]
            # Multiply -1 to give E - mu * N.
            if var_id >= 0:
                point_terms.append(([var_id], -1 * mu, 1.0))
            elif var_id == -1:
                point_terms.append(([], -1 * mu, 1.0))

    # No need to multiply because system size already included.
    return point_terms
