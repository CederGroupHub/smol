"""
A few testing utilities that may be useful to just import and run.
Some of these are borrowed from pymatgen test scripts.
"""

import json
import random
from itertools import chain

import numpy as np
from monty.json import MontyDecoder, MSONable
from pymatgen.core import Composition, Element

from smol.cofe.space.domain import Vacancy


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
    num_sites = sum(len(sl.sites) for sl in chain(sublattices, inactive_sublattices))
    rand_occu = np.zeros(num_sites, dtype=int)
    for sublatt in sublattices:
        codes = range(len(sublatt.site_space))
        rand_occu[sublatt.sites] = np.random.choice(
            codes, size=len(sublatt.sites), replace=True
        )
    return rand_occu


def gen_random_neutral_occupancy(sublattices, inactive_sublattices, lam=10):
    """Generate a random encoded occupancy according to a list of sublattices.

    Args:
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        num_sites (int):
            Total number of sites

    Returns:
        ndarray: encoded occupancy
    """

    def get_charge(sp):
        if isinstance(sp, (Element, Vacancy)):
            return 0
        else:
            return sp.oxi_state

    def charge(occu, sublattices, inactives):
        charge = 0
        for sl in sublattices:
            for site in sl.sites:
                charge += get_charge(sl.species[occu[site]])
        for sl in inactives:
            for site in sl.sites:
                assert occu[site] == 0
                assert len(sl.site_space) == 1
                charge += get_charge(list(sl.site_space.keys())[0])
        return charge

    def flip(occu, sublattices, inactives, lam=10):
        sl = random.choice(sublattices)
        site = random.choice(sl.sites)
        sp = random.choice(list({i for i in range(len(sl.site_space))} - {occu[site]}))
        occu_next = occu.copy()
        occu_next[site] = sp
        C = charge(occu, sublattices, inactives)
        C_next = charge(occu_next, sublattices, inactives)
        accept = np.log(np.random.random()) < -lam * (C_next**2 - C**2)
        if accept and C != 0:
            return occu_next.copy(), C_next
        else:
            return occu.copy(), C

    occu = gen_random_occupancy(sublattices, inactive_sublattices)
    for _ in range(10000):
        occu, C = flip(occu, sublattices, inactive_sublattices, lam=lam)
        if C == 0:
            return occu.copy()

    raise TimeoutError("Can not generate a neutral occupancy in 10000 flips!")


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
        site.species = Composition({random.choice(list(site.species.keys())): 1})
    return structure


def gen_fake_training_data(prim_structure, n=10):
    """Generate a fake structure, energy training set."""

    training_data = []
    for energy in np.random.random(n):
        struct = gen_random_structure(prim_structure, size=np.random.randint(2, 6))
        energy *= -len(struct)
        training_data.append((struct, energy))
    return training_data
