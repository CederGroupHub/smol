"""
A few testing utilities that may be useful to just import and run.
Some of these are borrowed from pymatgen test scripts.
"""

import json
import random
from  itertools import chain
import numpy as np
from monty.json import MontyDecoder, MSONable
from pymatgen.core import Composition

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


def gen_random_occupancy(sublattices, inactive_sublattices):
    """Generate a random encoded occupancy according to a list of sublattices.

    Args:
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        num_sites (int):
            Total number of sites

    Returns:
        ndarray: encoded occupancy
    """
    num_sites = sum(
        len(sl.sites) for sl in chain(sublattices, inactive_sublattices))
    rand_occu = np.zeros(num_sites, dtype=int)
    for sublatt in sublattices:
        codes = range(len(sublatt.site_space))
        rand_occu[sublatt.sites] = np.random.choice(codes,
                                                    size=len(sublatt.sites),
                                                    replace=True)
    return rand_occu


def gen_random_structure(prim, size=3):
    """Generate an random ordered structure from a disordered prim

    Args:
        prim (pymatgen.Structure):
            disordered primitive structure:
        size (optional):
            size argument to structure.make_supercell

    Returns:
        ordered structure
    """
    structure = prim.copy()
    structure.make_supercell(size)
    for site in structure:
        site.species = Composition(
            {random.choice(list(site.species.keys())): 1})
    return structure