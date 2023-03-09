"""Handle conversion of variable indices."""

from typing import Any, List, Tuple, Union

from smol.cofe.space.domain import Vacancy

__author__ = "Fengyu Xie"


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
            Index of variables corresponding to each index component if the sub-lattice is active,
            or the number of sites in sub-lattice if the sub-lattice is inactive.
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


def get_ewald_id_to_var_id_mapping(
    n_ewald_rows: int,
    variable_indices: List[List[int]],
    allowed_species: Union[List[Any], Tuple[Any]],
) -> List[int]:
    """Get mapping from ewald matrix row indices to variable indices.

    Args:
        n_ewald_rows(int):
            Number of rows in EwaldProcessor's ewald matrix.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            species indices in its site space.
        allowed_species(list[list[Species]]):
            Allowed species in the super-cell.
            You are fully responsible to check allowed species are generated from
            the same super-cell structure as cvxpy variables such that the mappings
            between them will be correct.
    Returns:
        list[int]:
            Index of cvxpy variable corresponding to each ewald matrix row. If a row
            does not have a corresponding variable (e.g., inactive site), it will be
            marked with -1.
    """
    ewald_var_ids = [-1 for _ in range(n_ewald_rows)]
    current_ewald_id = 0
    for site_vids, site_species in zip(variable_indices, allowed_species):
        # Inactive site, query species on it.
        if len(site_vids) == 0:
            # Has index in ewald, but not in cvxpy variables.
            if not isinstance(site_species[0], Vacancy):
                current_ewald_id += 1
        else:
            for spec_id, spec in enumerate(site_species):
                if not isinstance(site_species[0], Vacancy):
                    ewald_var_ids[current_ewald_id] = site_vids[spec_id]
                    current_ewald_id += 1
    return ewald_var_ids
