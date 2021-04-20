"""Utility functions to handle encoded occupation arrays."""

__author__ = "Fengyu Xie"

import numpy as np
import itertools


# Utilities for parsing occupation, used in charge-neutral semigrand flip table
def occu_to_species_stat(occupancy, bits, sublat_list_sc):
    """Make compstat format from occupation array.

    Get a statistics table of each specie on sublattices from an encoded
    occupancy array.
    Args:
        occupancy(np.ndarray like):
            An array representing encoded occupancy, can be list.
        bits(list of lists of Specie):
            A list of species on each sublattice, must correspond to occupancy
            encoding table
        sublat_list_sc(List of lists of ints):
            A list storing sublattice sites in a SUPER cell

    Return:
        species_stat(2D list of ints/floats)
            Is a statistics of number of species on each sublattice.
            1st dimension: sublattices
            2nd dimension: number of each specie on that specific sublattice.
            Dimensions same as moca.sampler.mcushers.CorrelatedUsher.bits.
    """
    occu = np.array(occupancy)

    species_stat = [[0 for i in range(len(sl_bits))] for sl_bits in bits]

    for s_id, sp_code in enumerate(occu):
        sl_id = None
        for i, sl in enumerate(sublat_list_sc):
            if s_id in sl:
                sl_id = i
                break
        if sl_id is None:
            continue  # Site is inactive.

        species_stat[sl_id][sp_code] += 1

    return species_stat


def occu_to_species_list(occupancy, bits, sublat_list_sc):
    """Get occupation status of each sublattice.

    Get table of the indices of sites that are occupied by each specie on
    sublattices, from an encoded occupancy array.
    Args:
        occupancy(np.ndarray like):
            An array representing encoded occupancy, can be list.
        bits(list of lists of Specie):
            A list of species on each sublattice, must correspond to
            occupancy encoding table
        sublat_list_sc(List of lists of ints):
            A list storing sublattice sites in a SUPER cell
    Return:
        species_list(3d list of ints):
            Is a statistics of indices of sites occupied by each specie.
            1st dimension: sublattices
            2nd dimension: species on a sublattice
            3rd dimension: site ids occupied by that specie
    """
    occu = np.array(occupancy)

    species_list = [[[] for i in range(len(sl_bits))] for sl_bits in bits]

    for site_id, sp_id in enumerate(occu):
        sl_id = None
        for i, sl in enumerate(sublat_list_sc):
            if site_id in sl:
                sl_id = i
                break
        if sl_id is None:
            continue  # Site is inactive.

        species_list[sl_id][sp_id].append(site_id)

    return species_list


def delta_ccoords_from_step(occu, step, bits, sublat_list_sc, base_vecs,
                            base_norm=None):
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
        sublat_list_sc(List[List[int]]):
            List of sublattice site indices.
        base_vecs(2D ArrayLike):
            Base vectors in the charge neutral compositional
            space.
        base_norm(1D arrayLike, Optional):
            Normal vector to the charge neutral subspace.
            Usually, the parameters in the linear charge
            neutral diophantine equation are given.

    Return:
        np.ndarray, change of constrained coordinates.
    """
    del_compstat = [[0 for sp_id in range(len(sl))] for sl in bits]

    for s_id, sp_id_to in step:
        sl_id = None
        for i, sl in enumerate(sublat_list_sc):
            if s_id in sl:
                sl_id = i
                break
        if sl_id is None:
            raise ValueError("Step constains a site {} not found in any \
                             sublattice!".format(s_id))
        sp_id_from = occu[s_id]
        del_compstat[sl_id][sp_id_from] -= 1
        del_compstat[sl_id][sp_id_to] += 1

    del_ucoords = list(itertools.chain(*[sl[:-1] for sl in del_compstat]))
    del_ucoords = np.array(del_ucoords)

    if len(base_vecs) == len(del_ucoords):
        # Non-ionic systems.
        del_ccoords = np.linalg.inv(np.array(base_vecs).T) @ del_ucoords
    elif len(base_vecs) == len(del_ucoords) - 1:
        if base_norm is None:
            raise ValueError("No normal vector given for ionic system case.")
        R = np.vstack((base_vecs, base_norm))
        del_ccoords = np.linalg.inv(R.T) @ del_ucoords
        slack = del_ccoords[-1]
        del_ccoords = del_ccoords[:-1]

        if abs(slack) > 1E-5:
            raise ValueError("Given flip can not be represented in charge \
                             neutral space.")
    else:
        raise ValueError("Number of basis vectors supplied does not match \
                          the dimensionality of the compositional space.")

    return del_ccoords
