"""
A few testing utilities that may be useful to just import and run.
Some of these are borrowed from pymatgen test scripts.
"""

import json
import random
import numpy as np
from copy import deepcopy
from monty.json import MontyDecoder, MSONable

from smol.moca.utils.math_utils import GCD_list
from smol.moca.comp_space import CompSpace

def assert_msonable(obj, test_if_subclass=True):
    """
    Tests if obj is MSONable and tries to verify whether the contract is
    fulfilled.
    By default, the method tests whether obj is an instance of MSONable.
    This check can be deactivated by setting test_if_subclass to False.
    """
    if test_if_subclass:
        assert isinstance(obj, MSONable)
    assert obj.as_dict() == obj.__class__.from_dict(obj.as_dict()).as_dict()
    _ = json.loads(obj.to_json(), cls=MontyDecoder)


def gen_random_occupancy(sublattices, num_sites):
    """Generate a random encoded occupancy according to a list of sublattices.

    Args:
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        num_sites (int):
            Total number of sites

    Returns:
        ndarray: encoded occupancy
    """
    rand_occu = np.zeros(num_sites, dtype=int)
    for sublatt in sublattices:
        codes = range(len(sublatt.site_space))
        rand_occu[sublatt.sites] = np.random.choice(codes,
                                                    size=len(sublatt.sites),
                                                    replace=True)
    return rand_occu

def gen_random_neutral_occupancy(sublattices, num_sites):
    """Generate charge neutral occupancies according to a list of sublattices.
       Occupancies are encoded.

    Args:
        sublattices (Sequence of Sublattice):
           A sequence of sublattices, must be all sublattices, no matter
           active or not.
        num_sites (int):
           Total number of sites

    Returns:
        ndarray: encoded occupancy, charge neutral guaranteed.
    """
    rand_occu = np.zeros(num_sites, dtype=int)
    bits = [sl.species for sl in sublattices]
    sl_sizes = [len(sl.sites) for sl in sublattices]
    sc_size = GCD_list(sl_sizes)

    sl_sizes_prim = np.array(sl_sizes)//sc_size
    comp_space = CompSpace(bits,sl_sizes_prim)

    random_comp = random.choice(comp_space.int_grids(sc_size=sc_size,form='compstat'))

    sites = []
    assignments = []
    for sl,sl_comp in zip(sublattices,random_comp):
        sl_sites = list(sl.sites)
        random.shuffle(sl_sites)
        sites.extend(sl_sites)
        for sp_id,sp_n in enumerate(sl_comp):
            assignments.extend([sp_id for i in range(sp_n)])

    rand_occu[sites] = assignments
    return rand_occu
