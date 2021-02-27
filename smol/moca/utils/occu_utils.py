__author__ = "Fengyu Xie"

"""
Utility functions to handle encoded occupation arrays.
"""
import numpy as np

# Utilities for parsing occupation, used in charge-neutral semigrand flip table
def occu_to_species_stat(occupancy,bits,sublat_list_sc):
    """
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

    Returns:
        species_stat(2D list of ints/floats)
            Is a statistics of number of species on each sublattice.
            1st dimension: sublattices
            2nd dimension: number of each specie on that specific sublattice.
            Dimensions same as moca.sampler.mcushers.CorrelatedUsher.bits          
    """
    occu = np.array(occupancy)

    species_stat = [[0 for i in range(len(sl_bits))] for sl_bits in bits]

    for s_id,sp_code in enumerate(occu):
        sl_id = None
        for i,sl in enumerate(sublat_list_sc):
            if s_id in sl:
                sl_id = i
                break
        if sl_id is None:
            raise ValueError("Occupancy site {} can not be matched to a sublattice!".format(s_id))   
        species_stat[sl_id][sp_code]+=1

    return species_stat

def occu_to_species_list(occupancy,bits,sublat_list_sc):
    """
    Get table of the indices of sites that are occupied by each specie on sublattices,
    from an encoded occupancy array.
    Inputs:
        occupancy(np.ndarray like):
            An array representing encoded occupancy, can be list.
        bits(list of lists of Specie):
            A list of species on each sublattice, must correspond to occupancy
            encoding table
        sublat_list_sc(List of lists of ints):
            A list storing sublattice sites in a SUPER cell
    Returns:
        species_list(3d list of ints):
            Is a statistics of indices of sites occupied by each specie.
            1st dimension: sublattices
            2nd dimension: species on a sublattice
            3rd dimension: site ids occupied by that specie
    """
    occu = np.array(occupancy)
    
    species_list = [[[] for i in range(len(sl_bits))] for sl_bits in bits]

    for site_id,sp_id in enumerate(occu):
        sl_id = None
        for i,sl in enumerate(sublat_list_sc):
            if site_id in sl:
                sl_id = i
                break
        if sl_id is None:
            raise ValueError("Occupancy site {} can not be matched to a sublattice!".format(s_id))   

        species_list[sl_id][sp_id].append(site_id)

    return species_list

def direction_from_step(occu,step,sublat_list_sc):
    """
    Get the direction of flip from encoded occupancy array,
    mcmcstep and sublattice list of the supercell.

    Does not check validity of the step. 
    Any non-flip directions will be considered as swap.

    Args:
        occu(1D ArrayLike of int):
            encoded occupancy array
        step(List[type(int.int)]):
            List of tuples corresponding to single flips 
            in a step
        sublat_list_sc(List[List[int]]):
            List of sublattice site indices.
    """
    flip = {'from':{},'to':{}}
    for s_id,sp_id in step:
        sl_id = None
        for i,sl in enumerate(sublat_list_sc):
            if s_id in sl:
                sl_id = s_id
                break
        if sl_id is None:
            raise ValueError("Step contains a site {} not found in any sublattice!".format(s_id))

        if sl_id not in flip['from']:
            flip['from'][sl_id]={}
        if sl_id not in flip['to']:
            flip['to'][sl_id]={}          

        sp_from = occu[s_id]
        sp_to = sp_id
        if sp_from not in flip['from'][sl_id]:
            flip['from'][sl_id][sp_from]=1
        else:
            flip['from'][sl_id][sp_from]+=1
        if sp_to not in flip['to'][sl_id]:
            flip['to'][sl_id][sp_from]=1
        else:
            flip['to'][sl_id][sp_from]+=1

    direction = 0
    for f_id,tflip in enumerate(flip_combs):
        tflip_reversed = {'from':tflip['to'],'to':tflip['from']}
        if flip == tflip:
            direction = f_id+1
        elif flip == tflip_reversed:
            direction = -(f_id+1)

    return direction

