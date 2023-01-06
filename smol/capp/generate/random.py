"""Tools for generation of fully random occupancies and structures."""

__author__ = "Luis Barroso-Luque, Fengyu Xie"


import numpy as np
from pymatgen.core import Element

from smol.cofe.space.domain import Vacancy


def gen_random_ordered_occupancy(
    sublattices, composition=None, charge_neutral=False, rng=None
):
    """Generate a random encoded occupancy according to a list of sublattices.

    Args:
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        composition (Composition): optional
            A pymatgen Compositions that the generated occupancy should be.
        charge_neutral (bool): optional
            If True, the generated occupancy will be charge neutral. Oxidation states
            must be present in sublattices, if a composition is given this option is
            ignored.
        rng (optional): {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}
            A RNG, seed or otherwise to initialize defauly_rng

    Returns:
        ndarray: encoded occupancy
    """
    if composition is None:
        if charge_neutral:
            return _gen_neutral_occu(sublattices, rng=rng)
        else:
            return _gen_unconstrained_ordered_occu(sublattices, rng=rng)
    else:
        return _gen_composition_ordered_occu(sublattices, composition, rng=rng)


def _gen_unconstrained_ordered_occu(sublattices, rng=None):
    """Generate a random encoded occupancy according to a list of sublattices.

    Args:
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        rng (optional): {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}
            A RNG, seed or otherwise to initialize defauly_rng

    Returns:
        ndarray: encoded occupancy
    """
    num_sites = sum(len(sl.sites) for sl in sublattices)
    rand_occu = np.zeros(num_sites, dtype=int)
    rng = np.random.default_rng(rng)
    for sublatt in sublattices:
        rand_occu[sublatt.sites] = rng.choice(
            sublatt.encoding, size=len(sublatt.sites), replace=True
        )
    return rand_occu


def _gen_neutral_occu(sublattices, lam=10, rng=None):
    """Generate a random encoded occupancy according to a list of sublattices.

    Args:
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        rng (optional): {None, int, array_like[ints], SeedSequence, BitGenerator, Generator},
            A RNG, seed or otherwise to initialize defauly_rng

    Returns:
        ndarray: encoded occupancy
    """
    rng = np.random.default_rng(rng)

    def get_charge(sp):
        if isinstance(sp, (Element, Vacancy)):
            return 0
        else:
            return sp.oxi_state

    def charge(occu, sublattices):
        charge = 0
        for sl in sublattices:
            for site in sl.sites:
                sp_id = sl.encoding.tolist().index(occu[site])
                charge += get_charge(sl.species[sp_id])
        return charge

    def flip(occu, sublattices, lam=10):
        actives = [s for s in sublattices if s.is_active]
        sl = rng.choice(actives)
        site = rng.choice(sl.sites)
        code = rng.choice(list(set(sl.encoding) - {occu[site]}))
        occu_next = occu.copy()
        occu_next[site] = code
        C = charge(occu, sublattices)
        C_next = charge(occu_next, sublattices)
        accept = np.log(rng.random()) < -lam * (C_next**2 - C**2)
        if accept and C != 0:
            return occu_next.copy(), C_next
        else:
            return occu.copy(), C

    occu = _gen_unconstrained_ordered_occu(sublattices)
    for _ in range(10000):
        occu, C = flip(occu, sublattices, lam=lam)
        if C == 0:
            return occu.copy()

    raise TimeoutError("Can not generate a neutral occupancy in 10000 flips!")


def _gen_composition_ordered_occu(sublattices, composition, rng=None):
    """Generate a random occupancy satisfying a given composition.

    Args:
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        composition (Composition): optional
            A pymatgen Compositions that the generated occupancy should be.
        rng (optional): {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}
            A RNG, seed or otherwise to initialize defauly_rng

    Returns:
        ndarray: encoded occupancy
    """
    return


def _composition_compatiblity(sublattices, composition):
    """Check if a composition is compatible with a list of sublattices.

    Args:
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        composition (Composition): optional
            A pymatgen Compositions that the generated occupancy should be.

    Returns:
        bool: True if compatible, False otherwise.
    """
    if composition.num_atoms > 1.0:  # turn into a fractional composition
        composition = composition.frac_composition

    return
