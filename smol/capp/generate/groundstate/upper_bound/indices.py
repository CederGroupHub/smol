"""Handle conversion of variable indices."""

from typing import List, Tuple

import numpy as np
from pymatgen.core import Structure

from smol.cofe.space.domain import Vacancy, get_allowed_species
from smol.moca.ensemble import Sublattice

__author__ = "Fengyu Xie"


def get_variable_indices_for_each_composition_component(
    sublattices: List[Sublattice],
    variable_indices: List[List[int]],
    structure: Structure,
) -> List[Tuple[List[int], int]]:
    """Get variables and the number of restricted sites for each composition component.

    Composition components are the components in a vector representation of composition
    defined as the "counts" format in moca.composition.
    Args:
        sublattices(list of Sublattice):
            Sub-lattices to build the upper-bound problem on.
        variable_indices(list of lists of int):
            List of variable indices corresponding to each active site index and
            index of species in its site space. Inactive sites will be marked by
            either -1 or -2. See documentation in groundstate.upper_bound.variables.
        structure(Structure):
            The supercell structure stored in a processor's structure attribute.
            The sub-lattices must match the processor structure, or they must be the result
            of splitting with the initial_occupancy. See smol.moca.sublattice for the
            explanation of splitting a sub-lattice.
    Returns:
        list of tuples of (list of int, int):
            A list of tuples containing indices of variables falling under each
            composition component, and the number of sites manually restricted
            or naturally inactive to always be occupied by the species corresponding
            to this component. Used to quickly generate composition constraints.
    """
    orig_site_spaces = get_allowed_species(structure)

    var_ids_for_dims = []
    for sublattice in sublattices:
        for species in sublattice.species:
            active_var_inds = []
            num_fixed = 0
            for site_id in sublattice.sites:
                orig_site_space = orig_site_spaces[site_id]
                orig_sp_id = orig_site_space.index(species)
                var_ind = variable_indices[site_id][orig_sp_id]
                # Active variable.
                if var_ind >= 0:
                    active_var_inds.append(var_ind)
                # Always true.
                elif var_ind == -1:
                    num_fixed += 1
            var_ids_for_dims.append((active_var_inds, num_fixed))

    return var_ids_for_dims


def map_ewald_indices_to_variable_indices(
    structure: Structure,
    variable_indices: List[List[int]],
) -> List[int]:
    """Map row indices in ewald matrix to indices of boolean variables.

    Args:
        structure(Structure):
            The structure attribute of ensemble Processor. This is required
            to correctly parse the rows in the ewald_structure of an
            EwaldTerm, as the given sub-lattices might come form a split
            ensemble, whose site_spaces can not match the ewald_structure.
        variable_indices(list of lists of int):
            List of variable indices corresponding to each active site index and
            index of species in its site space. Inactive sites will be marked by
            either -1 or -2. See documentation in groundstate.upper_bound.variables.
    Returns:
        list of int:
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

    orig_site_spaces = get_allowed_species(structure)

    ewald_ids_to_var_ids = []
    for site_id in range(num_sites):
        for orig_sp_id, orig_species in enumerate(orig_site_spaces[site_id]):
            # Vacancies are always skipped.
            if isinstance(orig_species, Vacancy):
                continue
            ewald_ids_to_var_ids.append(variable_indices[site_id][orig_sp_id])

    return ewald_ids_to_var_ids


def get_sublattice_indices_by_site(sublattices: List[Sublattice]) -> np.array:
    """Get the indices of sub-lattice to which each site belong.

    Args:
        sublattices(list of Sublattice):
            Sub-lattices to build the upper-bound problem on.
            Must contain all sites in the ensemble, and the site indices must start from
            0 and be continuous.
    Returns:
        1D np.ndarray of int:
            An array containing the sub-lattice index where each site belongs to.
            Used to quickly access the sub-lattice information.
    """
    num_sites = sum([len(sublattice.sites) for sublattice in sublattices])
    site_sublattice_ids = np.zeros(num_sites, dtype=int) - 1
    for sublattice_id, sublattice in enumerate(sublattices):
        site_sublattice_ids[sublattice.sites] = sublattice_id
    if np.any(site_sublattice_ids < 0):
        raise ValueError(
            f"Provided sub-lattices: {sublattices} do not contain all"
            f" sites in ensemble, or the site-indices are not continuous!"
        )
    return site_sublattice_ids
