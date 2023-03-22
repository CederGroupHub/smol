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
    variables: cp.Variable,
    variable_indices: List[List[int]],
    sublattices: List[Sublattice],
    initial_occupancy: ArrayLike = None,
) -> np.ndarray:
    """Get encoded occupancy array from variables.

    Args:
        variables(cp.Variable):
            cvxpy variables storing the ground-state result.
        variable_indices(list[list[int]]):
            List of variable indices corresponding to each active site index and
            index of species in its site space. Inactive sites will be empty.
        sublattices(list[Sublattice]):
            Sub-lattices to build the upper-bound problem on.
        initial_occupancy(ArrayLike): optional
            An initial occupancy used to set the occupancy of manually restricted
            sites that may have more than one allowed species.
            Must be provided if any site has been manually restricted.
    Returns:
        np.ndarray: Encoded occupancy string.
    """
    if variables.value is None:
        raise ValueError("CVX variables are not solved yet!")
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
            species_ids_on_site = np.where(variables.value[indices])[0]
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
