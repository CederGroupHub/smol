"""Utility functions to handle encoded occupation arrays."""

__author__ = "Fengyu Xie"

import numpy as np
import itertools


# Utilities for parsing occupation, used in charge-neutral semigrand flip table
def occu_to_species_stat(occupancy, all_sublattices, active_only=False):
    """Make compstat format from occupation array.

    Get a statistics table of each specie on sublattices from an encoded
    occupancy array.
    Args:
        occupancy(np.ndarray like):
            An array representing encoded occupancy, can be list.
        all_sublattices(smol.moca.Sublattice):
            All sublattices in the super cell, regardless of activeness.
        active_only(Boolean):
            If true, will count un-restricted sites only. Default to false.

    Return:
        species_stat(2D list of ints/floats)
            Is a statistics of number of species on each sublattice.
            1st dimension: sublattices
            2nd dimension: number of each specie on that specific sublattice.
            Dimensions same as moca.sampler.mcushers.CorrelatedUsher.bits.
    """
    occu = np.array(occupancy)

    return [[int(round((occu[s.active_sites if active_only else s.sites]
                        == sp_id).sum()))
            for sp_id in range(len(s.species))]
            for s in all_sublattices]


def occu_to_species_list(occupancy, all_sublattices, active_only=False):
    """Get occupation status of each sublattice.

    Get table of the indices of sites that are occupied by each specie on
    sublattices, from an encoded occupancy array.

    Args:
        occupancy(np.ndarray like):
            An array representing encoded occupancy, can be list.
        all_sublattices(smol.moca.Sublattice):
            All sublattices in the super cell, regardless of activeness.
        active_only(Boolean):
            If true, will count un-restricted sites only. Default to false.

    Return:
        species_list(3d list of ints):
            Is a statistics of indices of sites occupied by each specie.
            1st dimension: sublattices
            2nd dimension: species on a sublattice
            3rd dimension: site ids occupied by that specie
    """
    occu = np.array(occupancy)

    if active_only:
        species_list = [[s.active_sites[occu[s.active_sites] == sp_id].tolist()
                         for sp_id in range(len(s.species))]
                        for s in all_sublattices]
    else:
        species_list = [[s.sites[occu[s.sites] == sp_id].tolist()
                         for sp_id in range(len(s.species))]
                        for s in all_sublattices]
    return species_list


def delta_ucoords_from_step(occu, step, comp_space, all_sublattices):
    """Get the change of unconstrained coordinates from mcmcusher step.

    Args:
        occu(1D arrayLike):
            Encoded occupation array.
        step(List[type(int.int)]):
            List of tuples corresponding to single flips
            in a step.
        bits(List[List[Specie]]):
            A list specifying the species that can occupy
            each sublattice.
        comp_space(smol.CompSpace):
            composition space object.
        all_sublattices(smol.moca.Sublattice):
            All sublattices in the super cell, regardless of activeness.

        Note: comp_space and all_sublattices must match.
    Return:
        np.ndarray, change of constrained coordinates.
    """
    if len(step) == 0:
        return np.zeros(comp_space.unconstr_dim)

    occu_0 = occu.copy()
    occu_1 = occu.copy()
    step = np.array(step, dtype=int)
    occu_1[step[:, 0]] = step[:, 1]

    sc_size = len(all_sublattices[0].sites) // comp_space.sl_sizes[0]
    compstat_0 = occu_to_species_stat(occu_0, all_sublattices)
    ucoords_0 = comp_space.translate_format(compstat_0,
                                            from_format='compstat',
                                            to_format='unconstr',
                                            sc_size=sc_size)
    compstat_1 = occu_to_species_stat(occu_1, all_sublattices)
    ucoords_1 = comp_space.translate_format(compstat_1,
                                            from_format='compstat',
                                            to_format='unconstr',
                                            sc_size=sc_size)
    return np.array(ucoords_1) - np.array(ucoords_0)


def delta_ccoords_from_step(occu, step, comp_space, all_sublattices):
    """Get the change of constrained coordinates from mcmcusher step.

    Args:
        occu(1D arrayLike):
            Encoded occupation array.
        step(List[type(int.int)]):
            List of tuples corresponding to single flips
            in a step.
        bits(List[List[Specie]]):
            A list specifying the species that can occupy
            each sublattice.
        comp_space(smol.CompSpace):
            composition space object.
        all_sublattices(smol.moca.Sublattice):
            All sublattices in the super cell, regardless of activeness.

        Note: comp_space and all_sublattices must match.
    Return:
        np.ndarray, change of constrained coordinates.
    """
    if len(step) == 0:
        return np.zeros(comp_space.dim)

    occu_0 = occu.copy()
    occu_1 = occu.copy()
    step = np.array(step, dtype=int)
    occu_1[step[:, 0]] = step[:, 1]

    sc_size = len(all_sublattices[0].sites) // comp_space.sl_sizes[0]
    compstat_0 = occu_to_species_stat(occu_0, all_sublattices)
    ccoords_0 = comp_space.translate_format(compstat_0,
                                            from_format='compstat',
                                            to_format='constr',
                                            sc_size=sc_size)
    compstat_1 = occu_to_species_stat(occu_1, all_sublattices)
    ccoords_1 = comp_space.translate_format(compstat_1,
                                            from_format='compstat',
                                            to_format='constr',
                                            sc_size=sc_size)
    return np.array(ccoords_1) - np.array(ccoords_0)


def flip_weights_mask(flip_table, comp_stat):
    """Mask pre-assiged flip weights.

    On the edge or surface of constrained composition space, some flip
    directions may not be possible. We shall re-adjust their weights to
    0 to improve efficiency of step proposal.

    Will be used by Tableflipper.

    Args:
        flip_table(list[dict]):
            The flip table.
        comp_stat(list[list[int]]):
            Number of each specie on each sublattice. Generated
            by occu_to_species_stat.

    Return:
        np.array: Weights mask of each flip direction, length =
                  2 * len(flip_table). Impossible directions
                  will be masked out with 0.
    """
    mask = []

    for idx in range(2*len(flip_table)):
        fid = idx // 2
        direction = idx % 2
        flip = flip_table[fid]

        # Forward direction.
        allowed = 1
        if direction == 0:
            for sl_id in flip['from']:
                for sp_id in flip['from'][sl_id]:
                    dn = flip['from'][sl_id][sp_id]
                    n0 = comp_stat[sl_id][sp_id]
                    if n0 < dn:
                        allowed = 0
                        break
                if allowed == 0:
                    break

        # Backward direction.
        if direction == 1:
            for sl_id in flip['to']:
                for sp_id in flip['to'][sl_id]:
                    dn = flip['to'][sl_id][sp_id]
                    n0 = comp_stat[sl_id][sp_id]
                    if n0 < dn:
                        allowed = 0
                        break
                if allowed == 0:
                    break

        mask.append(allowed)

    return np.array(mask)
