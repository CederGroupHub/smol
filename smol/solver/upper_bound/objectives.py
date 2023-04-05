"""Generate cvxpy objectives from processor."""
from copy import deepcopy
from itertools import product
from numbers import Number
from typing import List, Tuple

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint
from numpy.typing import ArrayLike

from smol.moca.ensemble import Sublattice
from smol.moca.processor import (
    ClusterDecompositionProcessor,
    ClusterExpansionProcessor,
    EwaldProcessor,
)
from smol.solver.upper_bound.utils.indices import map_ewald_indices_to_variable_indices

__author__ = "Fengyu Xie"


def get_expression_and_auxiliary_from_terms(
    cluster_terms: List[Tuple[List[int], Number]], variables: cp.Variable
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
    sorted_terms = [(tuple(sorted(inds)), fac) for inds, fac in cluster_terms]
    simplified_terms = {}
    for inds, fac in sorted_terms:
        if inds not in simplified_terms:
            simplified_terms[inds] = fac
        else:
            simplified_terms[inds] += fac

    expression = 0
    n_slack = len([inds for inds in simplified_terms.keys() if len(inds) > 1])
    if n_slack == 0:
        aux_variables = None
    else:
        aux_variables = cp.Variable(n_slack, boolean=True)
    indices_in_aux_products = [[] for _ in range(n_slack)]
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


def get_auxiliary_variable_values(
    variable_values: ArrayLike[int], indices_in_auxiliary_products: List[List[int]]
) -> ArrayLike[Number]:
    """Get the value of auxiliary variables.

    Args:
        variable_values(np.ndarray):
            Values of site variables.
        indices_in_auxiliary_products(list[list[int]]):
            A list containing the indices of variables whose product equals to the
            corresponding auxiliary slack variable.
    Returns:
        np.ndarray:
            Values of auxiliary variables subjecting to auxiliary constraints.
    """
    variable_values = np.array(variable_values).astype(int)
    aux_values = np.ones(len(indices_in_auxiliary_products), dtype=int)
    for i, inds in enumerate(indices_in_auxiliary_products):
        aux_values[i] = np.product(variable_values[inds])

    return aux_values.astype(int)


def get_upper_bound_terms_from_expansion_processor(
    sublattices: List[Sublattice],
    variable_indices: List[List[int]],
    expansion_processor: ClusterExpansionProcessor,
    initial_occupancy: ArrayLike = None,
) -> List[Tuple[List[int], Number]]:
    """Get the cluster terms from cluster expansion processor.

    Args:
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
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
        list[tuple(list[int], float)]:
            A list of tuples, each represents a cluster term in the energy
            representation, containing indices of variables to be taken product with,
            and the factor before the boolean product.
            Energy is taken per super-cell.
    """
    coefs = expansion_processor.coefs
    # Store variable indices and the constant product in the cluster term.
    cluster_terms = [[] for _ in range(expansion_processor.num_corr_functions)]
    cluster_terms[0] = [([], 1)]

    num_sites = len(variable_indices)
    site_sublattice_ids = np.zeros(num_sites, dtype=int) - 1
    for sublattice_id, sublattice in enumerate(sublattices):
        site_sublattice_ids[sublattice.sites] = sublattice_id

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
                    pass
                    # Tuple can be used to query a single element.
                    cluster_factor = corr_tensors[tuple((bid, *cluster_state))]
                    cluster_variable_indices = []
                    # Get index to query.
                    for sid_in_cluster in range(n_sites):
                        sid_in_supercell = mapping[cid, sid_in_cluster]
                        species_id_in_site = cluster_state[sid_in_cluster]
                        if len(variable_indices[sid_in_supercell]) > 0:
                            var_id = variable_indices[sid_in_supercell][
                                species_id_in_site
                            ]
                            cluster_variable_indices.append(var_id)
                    # Divide by n_clusters to normalize.
                    cluster_terms[n].append(
                        (cluster_variable_indices, cluster_factor / n_clusters)
                    )
            n += 1

    # Put in system size and coefficients info.
    return [
        (inds, factor * coef * expansion_processor.size)
        for terms, coef in zip(cluster_terms, coefs)
        for inds, factor in terms
    ]


def get_upper_bound_terms_from_decomposition_processor(
    sublattices: List[Sublattice],
    variable_indices: List[List[int]],
    decomposition_processor: ClusterDecompositionProcessor,
    initial_occupancy: ArrayLike = None,
) -> List[Tuple[List[int], Number]]:
    """Get the cluster terms from cluster decomposition processor.

    Args:
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
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
        list[tuple(list[int], float)]:
            A list of tuples, each represents a cluster term in the energy
            representation, containing indices of variables to be taken product with,
            and the factor before the boolean product.
            Energy is taken per super-cell.
    """
    coefs = decomposition_processor.coefs  # Actually multiplicities.
    orbit_terms = [[] for _ in range(decomposition_processor.n_orbits)]
    # TODO: Change this to _interaction_tensors after merging luis/optimize4 branch.
    orbit_tensors = decomposition_processor._fac_tensors
    orbit_terms[0] = [([], orbit_tensors[0])]

    num_sites = len(variable_indices)
    site_sublattice_ids = np.zeros(num_sites, dtype=int) - 1
    for sublattice_id, sublattice in enumerate(sublattices):
        site_sublattice_ids[sublattice.sites] = sublattice_id

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
                cluster_factor = orbit_tensors[tuple((n, *cluster_state))]
                cluster_variable_indices = []
                # Get index to query.
                for sid_in_cluster in range(n_sites):
                    sid_in_supercell = mapping[cid, sid_in_cluster]
                    species_id_in_site = cluster_state[sid_in_cluster]

                    if len(variable_indices[sid_in_supercell]) > 0:
                        var_id = variable_indices[sid_in_supercell][species_id_in_site]
                        cluster_variable_indices.append(var_id)
                # Divide by n_clusters to normalize.
                orbit_terms[n].append(
                    (cluster_variable_indices, cluster_factor / n_clusters)
                )

        n += 1

    # Put in system size and coefficients info.
    return [
        (inds, factor * coef * decomposition_processor.size)
        for terms, coef in zip(orbit_terms, coefs)
        for inds, factor in terms
    ]


def get_upper_bound_terms_from_ewald_processor(
    sublattices: List[Sublattice],
    variable_indices: List[List[int]],
    ewald_processor: EwaldProcessor,
    initial_occupancy: ArrayLike = None,
) -> List[Tuple[List[int], Number]]:
    """Get the objective function from cluster expansion processor.

    Args:
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
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
        list[tuple(list[int], float)]:
            A list of tuples, each represents a cluster term in the energy
            representation, containing indices of variables to be taken product with,
            and the factor before the boolean product.
            Energy is taken per super-cell.
    """
    ewald_matrix = ewald_processor.ewald_matrix
    ewald_cluster_terms = []

    # Should be square matrix.
    n_ewald_rows = ewald_matrix.shape[0]

    ewald_to_var_ids = map_ewald_indices_to_variable_indices(
        sublattices, ewald_processor.structure, variable_indices, initial_occupancy
    )
    # Do not diagonalize because ewald matrix already contains 1/2.
    for i in n_ewald_rows:
        for j in n_ewald_rows:
            row_var_id = ewald_to_var_ids[i]
            col_var_id = ewald_to_var_ids[j]
            # Some species are never possible on some sites because of
            # manual restriction. Should skip any associated element.
            if row_var_id == -2 or col_var_id == -2:
                continue

            ewald_factor = ewald_matrix[i, j] * ewald_processor.coefs
            ewald_variable_indices = []
            if row_var_id != -1:
                ewald_variable_indices.append(row_var_id)
            # Diagonal should also be a point term.
            if col_var_id != -1 and col_var_id != row_var_id:
                ewald_variable_indices.append(col_var_id)
            ewald_cluster_terms.append((ewald_variable_indices, ewald_factor))

    # No need to multiply because system size is already included.
    return ewald_cluster_terms


def get_upper_bound_terms_from_chemical_potentials(
    sublattices: List[Sublattice],
    variable_indices: List[List[int]],
    chemical_table: ArrayLike,
    initial_occupancy: ArrayLike = None,
) -> List[Tuple[List[int], Number]]:
    """Get the objective function from chemical potentials.

    Notice: returns the -mu N term. Negation already included.
    Args:
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
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
        list[tuple(list[int], float)]:
            A list of tuples, each represents a cluster term in the energy
            representation, containing indices of variables to be taken product with,
            and the factor before the boolean product.
            Energy is taken per super-cell.
    """
    num_sites = len(variable_indices)
    site_sublattice_ids = np.zeros(num_sites, dtype=int) - 1
    for sublattice_id, sublattice in enumerate(sublattices):
        site_sublattice_ids[sublattice.sites] = sublattice_id

    point_terms = []
    for site_id in range(num_sites):
        sublattice_id = site_sublattice_ids[site_id]
        sublattice = sublattices[sublattice_id]
        # Active site.
        if site_id in sublattice.active_sites:
            codes = sublattice.encoding
            for var_id, mu in zip(
                variable_indices[site_id], chemical_table[site_id, codes]
            ):
                # Multiply by -1 because E - mu N.
                point_terms.append(([var_id], -1 * mu))
        # Inactive sub-lattice.
        elif len(sublattice.species) == 1:
            mu = chemical_table[site_id, sublattice.encoding[0]]
            point_terms.append(([], -1 * mu))
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
            mu = chemical_table[site_id, initial_occupancy[site_id]]
            point_terms.append(([], -1 * mu))

    # No need to multiply because system size already included.
    return point_terms
