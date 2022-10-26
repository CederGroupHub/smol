"""Utility functions to handle encoded occupation arrays."""

__author__ = "Fengyu Xie"

import numpy as np


def get_dim_ids_by_sublattice(bits):
    """Get the component index of each species in vector n.

    Args:
        bits(List[List[Specie|Vacancy|Element]]):
           Species on each sub-lattice.
    Returns:
        Component index of each species on each sublattice in vector n:
           List[List[int]]
    """
    dim_ids = []
    dim_id = 0
    for species in bits:
        dim_ids.append(list(range(dim_id, dim_id + len(species))))
        dim_id += len(species)
    return dim_ids


# Parsing will be faster based on table.
def get_dim_ids_table(sublattices, active_only=False):
    """Get the dimension indices of all (site, code) in n-representation.

    This will be used to efficiently map occupancy to n-representation.
    Args:
        sublattices(smol.moca.Sublattice):
            All sub-lattices, active or not. The union
            of all sub-lattices' sites must be
            range(number of sites).
        active_only(bool): optional
            If true, will count un-restricted sites on all
            sub-lattices only. Default to false, will count
            all sites and sub-lattices.
    """
    n_row = sum(len(sublatt.sites) for sublatt in sublattices)
    n_col = max(max(sublatt.encoding) for sublatt in sublattices) + 1

    table = np.zeros((n_row, n_col), dtype=int) - 1
    dim_id = 0
    for sl_id, sublatt in enumerate(sublattices):
        for code in sublatt.encoding:
            if active_only:
                sites = sublatt.active_sites
            else:
                sites = sublatt.sites
            sites = sites.astype(int)  # in case sites are void.
            table[sites, code] = dim_id
            dim_id += 1
    return table


# Utilities for parsing occupation to composition.
def occu_to_species_list(occupancy, n_dims, dim_ids_table):
    """Get occupancy status of each sub-lattice.

    Get table of the indices of sites that are occupied by each specie on
    sub-lattices, from an encoded occupancy array.

    Args:
        occupancy(1d Arraylike[int]):
            An array representing encoded occupancy, can be list.
        n_dims (int):
            Number of dimensions in the unconstrained composition space,
            namely, the number of components in "counts" format vector
            representation.
        dim_ids_table(2D arrayLike[int]):
            Dimension indices of each site and code in n-representation.
            Rows correspond to site index, while columns correspond to
            species code. if a site is not active, all codes (columns)
            will give -1.

    Return:
        Index of sites occupied by each species, sublattices concatenated:
            List[List[int]]
    """
    occu = np.array(occupancy, dtype=int)
    if len(occu) != len(dim_ids_table):
        raise ValueError(
            f"Occupancy size {len(occu)} does not match "
            f"table size {len(dim_ids_table)}!"
        )
    dim_ids = dim_ids_table[np.arange(len(occu), dtype=int), occu]
    all_sites = np.arange(len(occu), dtype=int)
    return [all_sites[dim_ids == i].tolist() for i in range(n_dims)]


def occu_to_counts(occupancy, n_dims, dim_ids_table):
    """Count number of species from occupation array.

    Get the count of each species on sub-lattices from an encoded
    occupancy array. ("counts" format)
    Args:
        occupancy(1d Arraylike[int]):
            An array representing encoded occupancy, can be list.
        n_dims (int):
            Number of dimensions in the unconstrained composition space,
            namely, the number of components in "counts" format vector
            representation.
        dim_ids_table(2D arrayLike[int]):
            Dimension indices of each site and code in n-representation.
            Rows correspond to site index, while columns correspond to
            species code. if a site is not active, all codes (columns)
            will give -1.

    Return:
        Amount of each species, concatenated by sub-lattice:
            1D np.ndarray[int]
    """
    occu = np.array(occupancy, dtype=int)
    if len(occu) != len(dim_ids_table):
        raise ValueError(
            f"Occupancy size {len(occu)} does not match "
            f"table size {len(dim_ids_table)}!"
        )
    dim_ids = dim_ids_table[np.arange(len(occu), dtype=int), occu]
    n = np.zeros(n_dims, dtype=int)
    dims, counts = np.unique(dim_ids, return_counts=True)
    n[dims[dims >= 0]] = counts[dims >= 0]
    return n


def delta_counts_from_step(occu, step, n_dims, dim_ids_table):
    """Get the change of species amounts from MC step.

    Args:
        occu(1D arrayLike[int]):
            Encoded occupation array.
        step(List[tuple(int,int)]):
            List of tuples recording (site_id, code_of_species_to
            _replace_with).
        n_dims (int):
            Number of dimensions in the unconstrained composition space,
            namely, the number of components in "counts" format vector
            representation.
        dim_ids_table(2D arrayLike[int]):
            Dimension indices of each site and code in n-representation.
            Rows correspond to site index, while columns correspond to
            species code.

    Return:
        Change of species amounts (in "counts" format):
            1D np.ndarray[int]
    """
    occu_now = np.array(occu, dtype=int)
    dim_ids_table = np.array(dim_ids_table, dtype=int)
    delta_n = np.zeros(n_dims, dtype=int)
    # A step may involve a site multiple times.
    for site, code in step:
        code_ori = occu_now[site]
        dim_ori = dim_ids_table[site, code_ori]
        dim_nex = dim_ids_table[site, code]
        if dim_ori < 0 or dim_nex < 0:
            raise ValueError(
                f"Inactive sites or impossible codes " f"involved in step {step}!"
            )
        delta_n[dim_ori] -= 1
        delta_n[dim_nex] += 1
        occu_now[site] = code

    return delta_n
