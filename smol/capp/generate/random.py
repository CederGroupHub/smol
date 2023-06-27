"""Tools for generation of fully random occupancies."""

__author__ = "Luis Barroso-Luque, Fengyu Xie"

import warnings

import numpy as np
from pymatgen.core import Composition, Element

from smol.cofe.space.domain import SiteSpace, Vacancy


# TODO alloy encoding/decoding and also allow it for initial occus in smapler, etc
def generate_random_ordered_occupancy(
    processor,
    composition=None,
    charge_neutral=False,
    tol=1e-6,
    encoded=True,
    rng=None,
    **kwargs,
):
    """Generate a random encoded occupancy according to a list of sublattices.

    Args:
        processor (Processor):
            A processor object that represents the supercell space.
        composition (Sequence of Composition): optional
            A sequence of pymatgen Compositions for each sublattice specifying the
            composition for the generated occupancy. Must be in the same order as the
            sublattices.
        charge_neutral (bool): optional
            If True, the generated occupancy will be charge neutral. Oxidation states
            must be present in sublattices, if a composition is given this option is
            ignored.
        tol (float): optional
            Tolerance for the composition check, only used if a composition is given.
        encoded (bool): optional
            If True then occupancy is given as integer array otherwise as a list of
            Species.
        rng (optional): {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}
            A RNG, seed or otherwise to initialize defauly_rng

    Returns:
        ndarray: encoded occupancy
    """
    sublattices = processor.get_sublattices()

    if composition is None:
        if charge_neutral:
            occu = _gen_neutral_occu(sublattices, rng=rng, **kwargs)
        else:
            occu = _gen_unconstrained_ordered_occu(sublattices, rng=rng, **kwargs)
    else:
        occu = _gen_composition_ordered_occu(
            sublattices, composition, tol, rng=rng, **kwargs
        )

    if not encoded:
        occu = processor.decode_occupancy(occu)

    return occu


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
    return np.ascontiguousarray(rand_occu, dtype=int)


def _gen_neutral_occu(sublattices, lam=10, num_attempts=10000, rng=None):
    """Generate a random encoded occupancy according to a list of sublattices.

    Args:
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        lam (float): optional
            TODO explanation...
        num_attempts (int): optional
            number of flip attempts to generate occupancy.
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
    for _ in range(num_attempts):
        occu, C = flip(occu, sublattices, lam=lam)
        if C == 0:
            return np.ascontiguousarray(occu, dtype=int)

    raise TimeoutError(
        f"Can not generate a neutral occupancy in {num_attempts} attempts!"
    )


def _gen_composition_ordered_occu(sublattices, composition, tol, rng=None):
    """Generate a random occupancy satisfying a given composition.

    Args:
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        composition (Sequence of Composition): optional
            A sequence of pymatgen Compositions for each sublattice specifying the
            composition for the generated occupancy. Must be in the same order as the
            sublattices.
        tol (float):
            Tolerance for the composition check.
        rng (optional): {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}
            A RNG, seed or otherwise to initialize defauly_rng

    Returns:
        ndarray: encoded occupancy
    """
    rng = np.random.default_rng(rng)
    compositions = _composition_compatiblity(sublattices, composition, tol, rng=rng)
    occu = np.zeros(sum(len(sl.sites) for sl in sublattices), dtype=int)

    for composition, sublattice in zip(compositions, sublattices):
        # create a dummy site space to account for vacancies
        composition = SiteSpace(composition)
        all_sites = list(sublattice.sites.copy())
        for sp, code in zip(sublattice.species, sublattice.encoding):
            num_sp = round(composition[sp] * len(sublattice.sites))
            sites = rng.choice(all_sites, size=num_sp, replace=False)
            occu[sites] = code
            all_sites = [i for i in all_sites if i not in sites]

    return np.ascontiguousarray(occu, dtype=int)


def _composition_compatiblity(sublattices, composition, tol, rng=None):
    """Check if a composition is compatible with a list of sublattices.

    if only a single composition is given for more than one sublattice, a list of
    split compositions is returned.

    Args:
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        composition (Composition or Sequence of composition): optional
            A pymatgen Compositions that the generated occupancy should be. If a
            sequence of compositions is given the order must correspond to the order
            of sublattices.
        tol (float): optional
            Tolerance for the composition check.
        rng (optional): {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}
            A RNG, seed or otherwise to initialize defauly_rng

    Returns:
        list of Composition: A list of compositions that are compatible
    """
    # check that all species in composition appear in sublattices
    if isinstance(composition, Composition):
        compositions = [composition]
    else:
        compositions = composition

    for i, (comp, sl) in enumerate(zip(compositions, sublattices)):
        if any(sp not in sl.site_space for sp in comp):
            raise ValueError(
                "species are present in composition that are not in sublattices."
            )
        if comp.num_atoms > 1:
            warnings.warn(
                "A given sublattice composition is not normalized. \n"
                "Will be turned to a fractional composition."
            )
            compositions[i] = comp.fractional_composition

    # check if the compositions are compatible with the size of the sublattices
    for composition, sublattice in zip(compositions, sublattices):
        total = 0
        for concentration in composition.values():
            num_sites = len(sublattice.sites) * concentration
            if abs(round(num_sites) - num_sites) > tol:
                raise ValueError("composition is not compatible with supercell size.")
            total += round(num_sites)

        if total > len(sublattice.sites) + tol:
            raise ValueError("composition is not compatible with supercell size.")

    return compositions
