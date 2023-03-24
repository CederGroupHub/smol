"""Exotic cases where sub-lattices contains split and manually fixed sites."""
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
from pymatgen.core import Lattice, Species, Structure

from smol.cofe import ClusterExpansion, ClusterSubspace
from smol.cofe.extern import EwaldTerm
from smol.cofe.space.domain import Vacancy
from smol.moca.ensemble import Ensemble


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


@pytest.fixture(scope="module")
def exotic_prim():
    lat = Lattice.from_parameters(2, 2, 2, 60, 60, 60)
    return Structure(
        lat,
        [
            {
                "Li+": 1 / 6,
                "Mn2+": 1 / 6,
                "Mn3+": 1 / 6,
                "Mn4+": 1 / 6,
                "Ti4+": 1 / 6,
            },
            {
                "O2-": 1 / 3,
                "O-": 1 / 3,
                "F-": 1 / 3,
            },
        ],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


@pytest.fixture(scope="module", params=["sinusoid", "indicator"])
def exotic_subspace(exotic_prim, request):
    # Use sinusoid basis to test if useful.
    space = ClusterSubspace.from_cutoffs(
        exotic_prim, {2: 6, 3: 3, 4: 3}, basis=request.param
    )
    space.add_external_term(EwaldTerm)
    return space


@pytest.fixture(scope="module")
def exotic_coefs(exotic_subspace):
    exotic_coefs = np.empty(exotic_subspace.num_corr_functions + 1)
    exotic_coefs[0] = -10
    n_pair = len(exotic_subspace.function_inds_by_size[2])
    n_tri = len(exotic_subspace.function_inds_by_size[3])
    n_quad = len(exotic_subspace.function_inds_by_size[4])
    i = 1
    exotic_coefs[i : i + n_pair] = np.random.random(size=n_pair)
    i += n_pair
    exotic_coefs[i : i + n_tri] = np.random.random(size=n_tri) * 0.4
    i += n_tri
    exotic_coefs[i : i + n_quad] = np.random.random(size=n_quad) * 0.1

    exotic_coefs[-1] = 0.2
    return exotic_coefs


@pytest.fixture(scope="module")
def exotic_expansion(exotic_subspace, exotic_coefs):
    return ClusterExpansion(exotic_subspace, exotic_coefs)


@pytest.fixture(
    scope="module",
    params=list(product(["canonical", "semigrand"], ["expansion, decomposition"])),
)
def orig_ensemble(exotic_expansion, request):
    if request.param[0] == "semigrand":
        chemical_potentials = {
            "Li+": 0,
            "Mn2+": 0,
            "Mn3+": 0,
            "Mn4+": 0,
            "Ti4+": 0,
            "Vacancy": 0,
            "O2-": 0,
            "O2-": 0,
            "F-": 0,
        }
    else:
        chemical_potentials = None
    return Ensemble.from_cluster_expansion(
        exotic_expansion,
        np.diag([5, 2, 2]),
        request.param[1],
        chemical_potentials=chemical_potentials,
    )


@pytest.fixture(scope="module")
# Cation sub-lattice sorted in front of anion sub-lattice.
def orig_sublattices(orig_ensemble):
    return orig_ensemble.sublattices


@pytest.fixture(scope="module")
def exotic_initial_occupancy(orig_sublattices):
    return get_random_exotic_occu(orig_sublattices)


@pytest.fixture(scope="module")
def exotic_ensemble(orig_ensemble, exotic_initial_occupancy):
    cation_id = None
    anion_id = None
    for sl_id in range(2):
        if Species("Li", 1) in orig_ensemble.sublattices[sl_id].species:
            cation_id = sl_id
        if Species("O", -2) in orig_ensemble.sublattices[sl_id].species:
            anion_id = sl_id

    # Split the cation sublattice.
    new_ensemble = deepcopy(orig_ensemble)

    # Manually restrict 3 random li sites.
    li_code = new_ensemble.sublattices[cation_id].encoding[
        new_ensemble.sublattices[cation_id].species.index(Species("Li", 1))
    ]
    li_sites = new_ensemble.sublattices[cation_id].sites[
        np.where(
            exotic_initial_occupancy[new_ensemble.sublattices[cation_id]] == li_code
        )[0][0]
    ]
    li_restricts = np.random.choice(li_sites, size=3, replace=False)
    new_ensemble.restrict_sites(li_restricts)

    # Manually restrict 2 random O2- sites.
    o2_code = new_ensemble.sublattices[anion_id].encoding[
        new_ensemble.sublattices[cation_id].species.index(Species("O", -2))
    ]
    o2_sites = new_ensemble.sublattices[anion_id].sites[
        np.where(
            exotic_initial_occupancy[new_ensemble.sublattices[anion_id]] == o2_code
        )[0][0]
    ]
    o2_restricts = np.random.choice(o2_sites, size=2, replace=False)
    new_ensemble.restrict_sites(o2_restricts)

    ca_partitions = [
        [Species("Li", 1), Vacancy()],
        [Species("Mn", 2), Species("Mn", 3), Species("Mn", 4), Species("Ti", 4)],
    ]
    an_partitions = [[Species("O", -2), Species("O", -1)], [Species("F"), -1]]
    new_ensemble.split_sublattice_by_species(
        cation_id, exotic_initial_occupancy, ca_partitions
    )
    # Sub-lattices all updated after partition.
    anion_id = anion_id if cation_id > anion_id else anion_id + 1
    new_ensemble.split_sublattice_by_species(
        anion_id, exotic_initial_occupancy, an_partitions
    )

    # Check if sites a correctly restricted.
    for site in li_restricts:
        assert site in new_ensemble.restricted_sites
    for site in o2_restricts:
        assert site in new_ensemble.restricted_sites
