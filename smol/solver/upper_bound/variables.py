"""Get variables from processor."""
from typing import List, Tuple

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike

from smol.moca.ensemble import Sublattice

__author__ = "Fengyu Xie"


def get_upper_bound_variables_from_sublattices(
    sublattices: List[Sublattice],
    num_sites: int,
) -> Tuple[cp.Variable, List[List[int]]]:
    """Get cvxpy boolean variables for the upper-bound problem from processor.

    Inactive sites (sites will only 1 allowed species) will not be added into variables.
    Args:
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
        num_sites(int):
            Total number of sites. Must match the number of sites in all sub-lattices.
    Returns:
        cp.Variable, list[list[int]]:
          Flatten variables for each active site and species; list of variable
          index corresponding to each active site index and species indices in
          its site space. Inactive or restricted sites will not have any variable.
    """
    n_variables = 0
    variable_indices = []
    site_sublattice_ids = np.zeros(num_sites, dtype=int) - 1
    for sublattice_id, sublattice in enumerate(sublattices):
        site_sublattice_ids[sublattice.sites] = sublattice_id
    if np.any(site_sublattice_ids < 0):
        raise ValueError(
            f"Total number of sites {num_sites} does not match"
            " total number of sites in sublattices!"
        )
    for site_id in range(num_sites):
        sublattice = sublattices[site_sublattice_ids[site_id]]
        # Only active sites are assigned with variables.
        if site_id in sublattice.active_sites:
            variable_indices.append(
                list(range(n_variables, n_variables + len(sublattice.species)))
            )
            n_variables += len(sublattice.species)
        else:
            variable_indices.append([])
    return cp.Variable(n_variables, boolean=True), variable_indices


def get_occupancy_from_variables(
    sublattices: List[Sublattice],
    variable_values: ArrayLike,
    variable_indices: List[List[int]],
    initial_occupancy: ArrayLike = None,
) -> np.ndarray:
    """Get encoded occupancy array from value of variables.

    Args:
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
        variable_values(ArrayLike):
            Value of cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            index of species in its site space. Inactive sites will be empty.
        initial_occupancy(ArrayLike): optional
            An initial occupancy used to set the occupancy of manually restricted
            sites that may have more than one allowed species.
            Must be provided if any site has been manually restricted.
    Returns:
        np.ndarray: Encoded occupancy string.
    """
    values = np.round(variable_values).astype(int)

    num_sites = len(variable_indices)
    occu = np.zeros(num_sites, dtype=int) - 1
    site_sublattice_ids = np.zeros(num_sites, dtype=int) - 1
    for sublattice_id, sublattice in enumerate(sublattices):
        site_sublattice_ids[sublattice.sites] = sublattice_id

    # Not considering species encoding order yet.
    for site_id, indices in enumerate(variable_indices):
        sublattice = sublattices[site_sublattice_ids[site_id]]
        # Active.
        if len(indices) > 0:
            species_ids_on_site = np.where(values[indices] == 1)[0]
            if len(species_ids_on_site) > 1:
                raise ValueError(f"More than one species occupied site {site_id}!")
            occu[site_id] = sublattice.encoding[species_ids_on_site[0]]
        # Naturally inactive.
        elif len(sublattice.species) == 1:
            occu[site_id] = sublattice.encoding[0]
        # Empty sublattice.
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
                # Simply copy initial occupancy over.
                occu[site_id] = initial_occupancy[site_id]
    if np.any(occu < 0):
        raise ValueError(f"Variables does not match given indices: {variable_indices}!")

    return occu


def get_variable_values_from_occupancy(
    sublattices: List[Sublattice],
    occupancy: ArrayLike,
    variable_indices: List[List[int]],
) -> np.ndarray:
    """Get value of variables from encoded occupancy array.

    Args:
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
        occupancy(ArrayLike): optional
            An encoded occupancy array. Does not check whether it satisfies
            constraints.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            index of species in its site space. Inactive sites will be empty.

    Returns:
        np.ndarray: Value of boolean variables in 0 and 1.
    """
    # Variable indices are continuous.
    num_variables = (
        max((max(sub) if len(sub) > 0 else -1) for sub in variable_indices) + 1
    )
    values = np.zeros(num_variables, dtype=int)

    num_sites = len(variable_indices)
    site_sublattice_ids = np.zeros(num_sites, dtype=int) - 1
    for sublattice_id, sublattice in enumerate(sublattices):
        site_sublattice_ids[sublattice.sites] = sublattice_id

    for site_id, var_ids in enumerate(variable_indices):
        sublattice_id = site_sublattice_ids[site_id]
        sublattice = sublattices[sublattice_id]

        if len(var_ids) > 0:
            active_var_ids_on_site = np.where(
                occupancy[site_id] == sublattice.encoding
            )[0]
            active_var_ids = np.array(var_ids, dtype=int)[active_var_ids_on_site]
            values[active_var_ids] = 1

    # No check, just return.
    return values
