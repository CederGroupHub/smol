__author__ = 'Fengyu Xie'

"""
Compositional space utilities function. Also includes functions that deals with table filps in charge neutral compspace.
"""

from .math_utils import combinatorial_number
from smol.cofe.space.domain import get_species

# Composition linkage number for Charge neutral semi-grand flip rules
def get_n_links(comp_stat,flip_table):
    """
    Get the total number of configurations reachable by a single flip in flip_table
    set.
    comp_stat:
        a list of lists, same as the return value of comp_utils.occu_to_compstat, 
        is a statistics of occupying species on each sublattice.
    flip_table:
        a list of dictionaries, each representing a charge-conserving, minimal flip
        in the compositional space.
    Output:
        n_links:
            A list of integers, length = 2*len(flip_table), giving number of possible 
            flips along each flip_table.
            Even index 2*i : mult_flip i forward direction
            Odd index 2*i+1: mult_flip i reverse direction
    """
    n_links = [0 for i in range(2*len(flip_table))]

    for op_id,mult_flip in enumerate(flip_table):
        #Forward direction
        n_forward = 1
        n_to_flip_on_sl = [0 for i in range(len(comp_stat))]
        for sl_id in mult_flip['from']:
            for sp_id in mult_flip['from'][sl_id]:
                n = comp_stat[sl_id][sp_id]
                m = mult_flip['from'][sl_id][sp_id]
                n_forward = n_forward*combinatorial_number(n,m)
                n_to_flip_on_sl[sl_id] += m

        for sl_id in mult_flip['to']:
            for sp_id in mult_flip['to'][sl_id]:
                n = n_to_flip_on_sl[sl_id]
                m = mult_flip['to'][sl_id][sp_id]
                n_forward = n_forward*combinatorial_number(n,m)
                n_to_flip_on_sl[sl_id] -= m

        for n in n_to_flip_on_sl:
            if n!=0:
                raise ValueError("Number of species on both sides of mult_flip can not match!")

        #Reverse direction    
        n_reverse = 1
        n_to_flip_on_sl = [0 for i in range(len(comp_stat))]
        for sl_id in mult_flip['to']:
            for sp_id in mult_flip['to'][sl_id]:
                n = comp_stat[sl_id][sp_id]
                m = mult_flip['to'][sl_id][sp_id]
                n_reverse = n_reverse*combinatorial_number(n,m)
                n_to_flip_on_sl[sl_id] += m

        for sl_id in mult_flip['from']:
            for sp_id in mult_flip['from'][sl_id]:
                n = n_to_flip_on_sl[sl_id]
                m = mult_flip['from'][sl_id][sp_id]
                n_reverse = n_reverse*combinatorial_number(n,m)
                n_to_flip_on_sl[sl_id] -= m

        for n in n_to_flip_on_sl:
            if n!=0:
                raise ValueError("Number of species on both sides of mult_flip can not match!")

        n_links[2*op_id] = n_forward
        n_links[2*op_id+1] = n_reverse

    return n_links
