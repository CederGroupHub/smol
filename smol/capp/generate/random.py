"""Tools for generation of fully random occupancies."""

__author__ = "Luis Barroso-Luque, Fengyu Xie"

from collections import defaultdict

import numpy as np
from pymatgen.core import Composition, Element

from smol.cofe.space.domain import Vacancy


# TODO alloy encoding/decoding and also allow it for initial occus in smapler, etc
def gen_random_ordered_occupancy(
    processor,
    composition=None,
    charge_neutral=False,
    tol=1e-6,
    encoded=True,
    rng=None,
):
    """Generate a random encoded occupancy according to a list of sublattices.

    Args:
        processor (Processor):
            A processor object that represents the supercell space.
        composition (Composition): optional
            A pymatgen Compositions that the generated occupancy should be.
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
            occu = _gen_neutral_occu(sublattices, rng=rng)
        else:
            occu = _gen_unconstrained_ordered_occu(sublattices, rng=rng)
    else:
        occu = _gen_composition_ordered_occu(sublattices, composition, tol, rng=rng)

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
    return rand_occu


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
            return occu.copy()

    raise TimeoutError(
        f"Can not generate a neutral occupancy in {num_attempts} attempts!"
    )


def _gen_composition_ordered_occu(sublattices, composition, tol, rng=None):
    """Generate a random occupancy satisfying a given composition.

    Args:
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        composition (Composition): optional
            A pymatgen Compositions that the generated occupancy should be.
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
        all_sites = list(sublattice.sites.copy())
        for sp, code in zip(sublattice.species, sublattice.encoding):
            num_sp = round(composition[sp] * len(sublattice.sites))
            sites = rng.choice(all_sites, size=num_sp, replace=False)
            occu[sites] = code
            all_sites = [i for i in all_sites if i not in sites]

    return occu


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
    if isinstance(composition, Composition):
        composition = [composition]

    compositions = []
    for comp in composition:
        if comp.num_atoms > 1.0:  # turn into a fractional composition
            comp = comp.fractional_composition
        compositions.append(comp)

    # check that all species in composition appear in sublattices
    if len(compositions) == 1:
        # this validation could be moved into the _split_composition function
        species = {sp for sl in sublattices for sp in sl.species}
        if any(sp not in species for sp in compositions[0]):
            raise ValueError(
                "species are present in composition that are not in sublattices."
            )
        compositions = _split_composition_into_sublattices(
            compositions[0], sublattices, rng=rng
        )
    else:
        for comp, sl in zip(compositions, sublattices):
            if any(sp not in sl.site_space for sp in comp):
                raise ValueError(
                    "species are present in composition that are not in sublattices."
                )

    # check if the compositions are compatible with the size of the sublattices
    for composition, sublattice in zip(compositions, sublattices):
        total = 0
        for concentration in composition.values():
            num_sites = len(sublattice.sites) * concentration
            if abs(round(num_sites) - num_sites) > tol:
                return ValueError("composition is not compatible with supercell size.")
            total += round(num_sites)

        if abs(total - len(sublattice.sites)) > tol:
            return ValueError("composition is not compatible with supercell size.")

    return compositions


def _split_composition_into_sublattices(composition, sublattices, rng=None):
    """Split a given composition into several compositions according to a list of sublattices.

    The split is done randomly for all overlapping species.

    No check is made on the compatibility of the composition with the sublattices.

    Args:
        composition (Composition):
            A pymatgen Compositions to be split into Compositions commensurate with
            given sublattices.
        sublattices (Sequence of Sublattice):
            A sequence of sublattices
        rng (optional): {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}
            A RNG, seed or otherwise to initialize defauly_rng

    Returns:
        list: list of split compositions
    """
    rng = np.random.default_rng(rng)

    # dictionary of species and the sublattices in which they appear
    # TODO this can be a util function because I'm pretty sure its used in TableSwap
    species_in_sublattices = defaultdict(list)
    for sl in sublattices:
        for sp in sl.site_space.composition:
            species_in_sublattices[sp].append(sl)

    # split the composition into sublattices and record in a dictionary with
    # species as keys and then a dictionary of the sublattices and the composition
    # of that species in that sublattice
    total_size = sum(len(sl.sites) for sl in sublattices)
    compositions_in_sublattices = {}
    for sp, sublatts in species_in_sublattices.items():
        sl_sizes = np.array([len(sl.sites) for sl in sublatts])
        sp_compositions = np.zeros(len(sublatts))

        max_allowed = composition[sp] * total_size
        for i in range(len(sublatts) - 1):
            sp_compositions[i] = rng.random() * max_allowed / sl_sizes[i]
            max_allowed -= sp_compositions[i] * sl_sizes[i]

        sp_compositions[-1] = (
            total_size * composition[sp] - sum(sl_sizes * sp_compositions)
        ) / sl_sizes[-1]

        compositions_in_sublattices[sp] = {
            sl.site_space: comp for sl, comp in zip(sublatts, sp_compositions)
        }

    # now unwrap the into Composition objects for each sublattice
    compositions = [
        Composition(
            {
                sp: compositions_in_sublattices[sp][sl.site_space]
                for sp in sl.site_space.composition
            }
        )
        for sl in sublattices
    ]

    return compositions
