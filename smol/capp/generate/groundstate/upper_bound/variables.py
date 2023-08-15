"""Get variables from processor."""
from typing import List, Tuple

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike
from pymatgen.core import Structure

from smol.capp.generate.groundstate.upper_bound.indices import (
    get_sublattice_indices_by_site,
)
from smol.cofe.space.domain import get_allowed_species
from smol.moca.ensemble import Sublattice

__author__ = "Fengyu Xie"


def get_variables_from_sublattices(
    sublattices: List[Sublattice],
    structure: Structure,
    initial_occupancy: ArrayLike = None,
) -> Tuple[cp.Variable, List[List[int]]]:
    """Get cvxpy boolean variables for the upper-bound problem from a list sublattices.

    Inactive sites (sites will only 1 allowed species) will not be added into variables.
    Args:
        sublattices (list of Sublattice):
            Sub-lattices to build the upper-bound problem on.
            Note:1, Must contain all sub-lattices in the ensemble! 2, Must either be
            original sub-lattices computed from the structure, or sub-lattices
            split from those original sub-lattices with the initial_occupancy, such that
            the encoding of species can correctly refer to the species index in the
            structure's site_space!
        structure (Structure):
            The supercell structure stored in a processor's structure attribute that
            was used to create the given sub-lattices.
            The sub-lattices must match the processor structure, or they must be the
            result of splitting with the initial_occupancy. See smol.moca.sublattice for
            the explanation of splitting a sub-lattice.
        initial_occupancy (ArrayLike):
            An encoded occupancy string used to determine which species should occupy a
            manually restricted site in an active sub-lattice. If any site has been
            manually restricted, this argument will be mandatory.

    Returns:
        cp.Variable, list of lists of int:
          Flatten variables for each species on active sites; list of variable
          index corresponding to each site and species indices in
          its site space. Each sub-list will have the same shape as the corresponding
          original site space in structure.
          There are two cases when marking an inactive site:
          If a species is not allowed in the current sub-lattice setting (possibly due
          to a split sub-lattice does not contain this species), it will be marked as
          -2. If a species is enforced to occupy the site by manual site restriction,
          or due to the sub-lattice is inactive (i.e, sub-lattice only allows one
          species), then it will be marked by -1. In neither case a variable will be
          created on this site.
    """
    n_variables = 0
    num_sites = len(structure)
    # Original site spaces before any potential sub-lattice split and restriction.
    orig_site_spaces = get_allowed_species(structure)

    variable_indices = []
    site_sublattice_ids = get_sublattice_indices_by_site(sublattices)

    for site_id in range(num_sites):
        sublattice = sublattices[site_sublattice_ids[site_id]]
        current_site_space = sublattice.species
        orig_site_space = orig_site_spaces[site_id]
        # Only active sites are assigned with variables.
        site_variable_indices = []
        for species in orig_site_space:
            # Species in the site space after splitting.
            if species in current_site_space:
                # The current_site_space is active.
                if len(current_site_space) > 1:
                    # The current site is not manually restricted. Assign a variable.
                    if site_id in sublattice.active_sites:
                        site_variable_indices.append(n_variables)
                        n_variables += 1
                    # The current site is manually restricted. Need initial occupancy.
                    else:
                        if initial_occupancy is None:
                            raise ValueError(
                                f"Site {site_id} was manually restricted"
                                f" in sub-lattice: {sublattice}, but no initial"
                                f" occupancy was specified!"
                            )

                        # The species is exactly the enforced species as in
                        # initial_occupancy. Mark as -1 (always true).
                        # Encoding should match that of range(len(orig_site_space)).
                        if (
                            initial_occupancy[site_id]
                            == sublattice.encoding[current_site_space.index(species)]
                        ):
                            site_variable_indices.append(-1)
                        # The species is exactly the enforced species as in
                        # initial_occupancy. Mark as -2 (always false).
                        else:
                            site_variable_indices.append(-2)
                # The sub-lattice is inactive because it only has one species.
                # The site will always be occupied by this species. Mark as -1
                # (always true).
                else:
                    if len(current_site_space) == 0:
                        raise ValueError(
                            f"Got empty site space in sub-lattice" f" {sublattice}!"
                        )
                    site_variable_indices.append(-1)
            # Species no longer in site space after splitting. Mark as -2
            # (always false).
            else:
                site_variable_indices.append(-2)
        variable_indices.append(site_variable_indices)

    # Add a name to variable for better identification.
    return cp.Variable(n_variables, name="s", boolean=True), variable_indices


def get_occupancy_from_variables(
    sublattices: List[Sublattice],
    variable_values: ArrayLike,
    variable_indices: List[List[int]],
) -> np.ndarray:
    """Get encoded occupancy array from value of variables.

    Args:
        sublattices(list of Sublattice):
            Sub-lattices to build the upper-bound problem on.
        variable_values(ArrayLike):
            Value of cvxpy variables storing the ground-state result.
        variable_indices(list of lists of int):
            List of variable indices corresponding to each site index and
            the species in its site space.
    Returns:
        np.ndarray: Encoded occupancy string.
    """
    values = np.round(variable_values).astype(int)

    num_sites = len(variable_indices)
    occu = np.zeros(num_sites, dtype=int) - 1
    site_sublattice_ids = get_sublattice_indices_by_site(sublattices)

    # Not considering species encoding order yet.
    for site_id, indices in enumerate(variable_indices):
        sublattice = sublattices[site_sublattice_ids[site_id]]
        site_indices = np.array(indices, dtype=int)
        site_var_inds = site_indices[site_indices >= 0]
        # Active.
        if len(site_var_inds) > 0:
            species_ids_in_current_space = np.where(values[site_var_inds] == 1)[0]
            if len(species_ids_in_current_space) > 1:
                raise ValueError(f"More than one species occupied site {site_id}!")
            occu[site_id] = sublattice.encoding[species_ids_in_current_space[0]]
        # Inactive (manually restricted or site space is inactive.)
        # Simply set to the enforced code.
        else:
            occu[site_id] = np.where(site_indices == -1)[0][0]
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
        sublattices(list of Sublattice):
            Sub-lattices to build the upper-bound problem on.
        occupancy(ArrayLike): optional
            An encoded occupancy array. Does not check whether it satisfies
            constraints.
        variable_indices(list of lists of int):
            List of variable indices corresponding to each active site index and
            index of species in its site space. Inactive sites will be empty.

    Returns:
        np.ndarray: Value of boolean variables in 0 and 1.
    """
    # Variable indices are continuous.
    num_variables = max(max(sub) for sub in variable_indices) + 1
    values = np.zeros(num_variables, dtype=int)

    site_sublattice_ids = get_sublattice_indices_by_site(sublattices)

    for site_id, indices in enumerate(variable_indices):
        sublattice = sublattices[site_sublattice_ids[site_id]]
        site_indices = np.array(indices, dtype=int)
        site_var_inds = site_indices[site_indices >= 0]

        # Active site with variables assigned.
        if len(site_var_inds) > 0:
            active_var_id_in_current_space = np.where(
                occupancy[site_id] == sublattice.encoding
            )[0][0]
            active_var_id = site_var_inds[active_var_id_in_current_space]
            values[active_var_id] = 1

    # No check here, just return.
    return values
