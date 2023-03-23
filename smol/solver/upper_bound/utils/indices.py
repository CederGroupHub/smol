"""Handle conversion of variable indices."""

from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike

from smol.cofe.space.domain import Vacancy
from smol.moca.ensemble import Sublattice
from smol.moca.utils.occu import get_dim_ids_by_sublattice

__author__ = "Fengyu Xie"


def get_variable_indices_for_each_composition_component(
    sublattices: List[Sublattice],
    variable_indices: List[List[int]],
    initial_occupancy: ArrayLike = None,
) -> List[Tuple[List[int], int]]:
    """Get variables and the number of restricted sites for each composition component.

    Composition components are the components in a vector representation of composition
    defined as the "counts" format in moca.composition.
    Args:
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            index of species in its site space. Inactive sites will be empty.
        initial_occupancy(ArrayLike): optional
            An initial occupancy used to set the occupancy of manually restricted
            sites that may have more than one allowed species.
            Must be provided if any site has been manually restricted.
    Returns:
        list[tuple[list[int], int]]:
            A list of tuples containing indices of variables falling under each
            composition component, and the number of sites manually restricted
            or naturally inactive to always be occupied by the species corresponding
            to this component. Used to quickly generate composition constraints.
    """
    bits = [sublattice.species for sublattice in sublattices]
    sublattice_dim_ids = get_dim_ids_by_sublattice(bits)
    n_dims = sum([len(dims) for dims in sublattice_dim_ids])

    num_sites = len(variable_indices)
    site_sublattice_ids = np.zeros(num_sites, dtype=int) - 1
    for sublattice_id, sublattice in enumerate(sublattices):
        site_sublattice_ids[sublattice.sites] = sublattice_id

    var_ids_for_dims = [([], 0) for _ in range(n_dims)]
    for site_id in range(num_sites):
        sublattice_id = site_sublattice_ids[site_id]
        sublattice = sublattices[sublattice_id]
        dim_ids = sublattice_dim_ids[sublattice_id]
        if site_id in sublattice.active_sites:
            for species_id, dim_id in enumerate(dim_ids):
                var_ids_for_dims[dim_id][0].append(variable_indices[species_id])
        # Inactive sites always contribute to one certain component.
        elif len(sublattice.species) == 1:
            var_ids_for_dims[dim_ids[0]][1] += 1
        elif len(sublattice.species) == 0:
            raise ValueError(
                f"Encountered empty sub-lattice on site {site_id}."
                f" Sub-lattice: {sublattice}."
            )
        else:
            if initial_occupancy is None:
                raise ValueError(
                    f"Site {site_id} was manually restricted in"
                    f" sub-lattice: {sublattice}, but no initial"
                    f" occupancy was specified!"
                )
            else:
                # Manually fixed to initial occupancy.
                species_id = np.where(
                    sublattice.encoding == initial_occupancy[site_id]
                )[0][0]
                dim_id = dim_ids[species_id]
                var_ids_for_dims[dim_id][1] += 1

    return var_ids_for_dims


def map_ewald_indices_to_variable_indices(
    sublattices: List[Sublattice],
    variable_indices: List[List[int]],
    initial_occupancy: ArrayLike = None,
) -> List[int]:
    """Map row indices in ewald matrix to indices of boolean variables.

    Args:
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            index of species in its site space. Inactive sites will be empty.
        initial_occupancy(ArrayLike): optional
            An initial occupancy used to set the occupancy of manually restricted
            sites that may have more than one allowed species.
            Must be provided if any site has been manually restricted.
    Returns:
        list[int]:
            Index of cvxpy variable corresponding to each ewald matrix row. If a
            row corresponds to a site with only one allowed species or the species
            on a manually restricted site in a pre-defined initial occupancy,
            it will be marked as -1 (always occupied).
            If a row corresponds to a site manually restricted to
            be occupied by other species rather than the species corresponding to
            this row, it will be marked as -2 (never occupied).
            This is to quickly parse the ewald term into the ewald objective
            function.
    """
    num_sites = len(variable_indices)
    site_sublattice_ids = np.zeros(num_sites, dtype=int) - 1
    for sublattice_id, sublattice in enumerate(sublattices):
        site_sublattice_ids[sublattice.sites] = sublattice_id

    ewald_ids_to_var_ids = []
    for site_id in range(num_sites):
        sublattice = sublattices[site_sublattice_ids[site_id]]
        site_space = sublattice.species
        for species_id, species in enumerate(site_space):
            # Vacancies are always skipped.
            if isinstance(species, Vacancy):
                continue
            if site_id in sublattice.active_sites:
                var_id = variable_indices[site_id][species_id]
            # Inactive sub-lattice. Fixed to always effective.
            elif len(sublattice.species) == 1:
                var_id = -1
            elif len(sublattice.species) == 0:
                raise ValueError(
                    f"Encountered empty sub-lattice on site {site_id}."
                    f" Sub-lattice: {sublattice}."
                )
            # Manually restricted site.
            else:
                if initial_occupancy is None:
                    raise ValueError(
                        f"Site {site_id} was manually restricted in"
                        f" sub-lattice: {sublattice}, but no initial"
                        f" occupancy was specified!"
                    )
                else:
                    # Site manually restricted to initial occupancy.
                    if sublattice.encoding[species_id] == initial_occupancy[site_id]:
                        # Fixed to always effective.
                        var_id = -1
                    else:
                        # Fixed to never effective.
                        var_id = -2
            ewald_ids_to_var_ids.append(var_id)

    return ewald_ids_to_var_ids
