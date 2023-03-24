"""Utilities to perform solver tests. Only valid for exotic ensemble."""
import random

import numpy as np
from pymatgen.core import Species

from smol.cofe.space.domain import Vacancy
from smol.moca.sampler.mcusher import Swap
from smol.solver.upper_bound.variables import get_variable_values_from_occupancy


def get_random_exotic_occu(sublattices):
    # Only use this on orig_sublattices.
    # Li+ 9 Mn2+ 2 Mn3+ 1 Mn4+ 1 Ti4+ 2 Vac 5 O2- 8 O- 2 F- 10
    cation_sublattice = None
    anion_sublattice = None
    for sublattice in sublattices:
        if Species("Li", 1) in sublattice.species:
            cation_sublattice = sublattice
        if Species("O", -2) in sublattice.species:
            anion_sublattice = sublattice

    li_va_sites = np.random.choice(cation_sublattice.sites, size=14, replace=False)
    li_sites = np.random.choice(li_va_sites, size=9, replace=False)
    va_sites = np.setdiff1d(li_va_sites, li_sites)
    mn_ti_sites = np.setdiff1d(cation_sublattice.sites, li_va_sites)
    ti_sites = np.random.choice(mn_ti_sites, size=2, replace=False)
    mn_sites = np.setdiff1d(mn_ti_sites, ti_sites)
    mn2_sites = np.random.choice(mn_sites, size=2, replace=False)
    mn34_sites = np.setdiff1d(mn_sites, mn2_sites)
    mn3_sites = np.random.choice(mn34_sites, size=1, replace=False)
    mn4_sites = np.setdiff1d(mn34_sites, mn3_sites)

    o_sites = np.random.choice(anion_sublattice.sites, size=10, replace=False)
    o2_sites = np.random.choice(o_sites, size=8, replace=False)
    o1_sites = np.setdiff1d(o_sites, o2_sites)
    f_sites = np.setdiff1d(anion_sublattice.sites, o_sites)

    li_code = cation_sublattice.encoding[
        cation_sublattice.species.index(Species("Li", 1))
    ]
    mn2_code = cation_sublattice.encoding[
        cation_sublattice.species.index(Species("Mn", 2))
    ]
    mn3_code = cation_sublattice.encoding[
        cation_sublattice.species.index(Species("Mn", 3))
    ]
    mn4_code = cation_sublattice.encoding[
        cation_sublattice.species.index(Species("Mn", 4))
    ]
    ti_code = cation_sublattice.encoding[
        cation_sublattice.species.index(Species("Ti", 4))
    ]
    va_code = cation_sublattice.encoding[cation_sublattice.species.index(Vacancy())]
    o2_code = anion_sublattice.encoding[
        anion_sublattice.species.index(Species("O", -2))
    ]
    o1_code = anion_sublattice.encoding[
        anion_sublattice.species.index(Species("O", -1))
    ]
    f_code = anion_sublattice.encoding[anion_sublattice.species.index(Species("F", -1))]

    occu = np.zeros(40, dtype=int) - 1
    occu[li_sites] = li_code
    occu[mn2_sites] = mn2_code
    occu[mn3_sites] = mn3_code
    occu[mn4_sites] = mn4_code
    occu[ti_sites] = ti_code
    occu[va_sites] = va_code
    occu[o2_sites] = o2_code
    occu[o1_sites] = o1_code
    occu[f_sites] = f_code

    assert np.all(occu >= 0)

    return occu


def get_random_variable_values(sublattices):
    # Not always charge balanced.
    num_sites = sum(len(s.sites) for s in sublattices)
    site_sublattice_ids = np.zeros(num_sites, dtype=int) - 1
    for sublattice_id, sublattice in enumerate(sublattices):
        site_sublattice_ids[sublattice.sites] = sublattice_id

    values = []
    for site_id in range(num_sites):
        sl_id = site_sublattice_ids[site_id]
        sublattice = sublattices[sl_id]
        if len(sublattice.species) > 1 and site_id in sublattice.active_sites:
            site_vals = [0 for _ in range(len(sublattice.species))]
            site_vals[random.choice(range(len(sublattice.species)))] = 1
            values.extend(site_vals)

    return np.array(values, dtype=int)


def get_random_neutral_occupancy(
    sublattices, initial_occupancy, canonical=False, force_flip=False
):
    # Guarantee charge balanced.
    ti_code = None
    mn4_code = None
    ti_sites = None
    mn4_sites = None
    for sublattice in sublattices:
        if Species("Ti", 4) in sublattice.species:
            ti_code = sublattice.encoding[sublattice.species.index(Species("Ti", 4))]
            mn4_code = sublattice.encoding[sublattice.species.index(Species("Mn", 4))]
            ti_sites = sublattice.sites[initial_occupancy[sublattice.sites] == ti_code]
            mn4_sites = sublattice.sites[
                initial_occupancy[sublattice.sites] == mn4_code
            ]
    if canonical:
        threshold = 1.1
    elif not force_flip:
        threshold = 0.5
    else:
        threshold = -0.1

    if random.random() > threshold:
        # Ti-Mn flip that would not change charge.
        if random.random() > 0.5:
            flip = [(random.choice(ti_sites), mn4_code)]
        else:
            flip = [(random.choice(mn4_sites), ti_code)]
    else:
        swapper = Swap(sublattices)
        flip = swapper.propose_step(initial_occupancy)

    # Apply that flip.
    rand_occu = initial_occupancy.copy()
    for site_id, code in flip:
        rand_occu[site_id] = code
    return rand_occu


def get_random_neutral_variable_values(
    sublattices, initial_occupancy, variable_indices, canonical=False, force_flip=False
):
    # Guarantee charge balanced.
    rand_occu = get_random_neutral_occupancy(
        sublattices, initial_occupancy, canonical=canonical, force_flip=force_flip
    )
    return get_variable_values_from_occupancy(sublattices, rand_occu, variable_indices)
