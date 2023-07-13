"""Exotic cases where sub-lattices contains split and manually fixed sites."""
from itertools import product

import numpy as np
import pytest
from pymatgen.core import Lattice, Species, Structure

from smol.cofe import ClusterExpansion, ClusterSubspace
from smol.cofe.extern import EwaldTerm
from smol.cofe.space.domain import Vacancy
from smol.moca.ensemble import Ensemble

from .utils import get_random_solver_test_occu


# The following fixtures are designed specifically for testing the ground-state solver.
@pytest.fixture(scope="module")
def solver_test_prim():
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


@pytest.fixture(scope="module", params=["indicator", "sinusoid"])
def solver_test_subspace(solver_test_prim, request):
    # Use sinusoid basis to test if useful.
    space = ClusterSubspace.from_cutoffs(
        solver_test_prim, {2: 3, 3: 2.1}, basis=request.param
    )
    space.add_external_term(EwaldTerm())
    return space


@pytest.fixture(scope="module")
def solver_test_coefs(solver_test_subspace):
    solver_test_coefs = np.zeros(solver_test_subspace.num_corr_functions + 1)
    solver_test_coefs[0] = -10
    n_pair = len(solver_test_subspace.function_inds_by_size[2])
    n_tri = len(solver_test_subspace.function_inds_by_size[3])
    n_quad = 0
    i = 1
    solver_test_coefs[i : i + n_pair] = np.random.normal(size=n_pair)
    i += n_pair
    solver_test_coefs[i : i + n_tri] = np.random.normal(size=n_tri) * 0.4
    i += n_tri
    solver_test_coefs[i : i + n_quad] = np.random.normal(size=n_quad) * 0.1

    solver_test_coefs[-1] = 0.2
    return solver_test_coefs


@pytest.fixture(scope="module")
def solver_test_expansion(solver_test_subspace, solver_test_coefs):
    return ClusterExpansion(solver_test_subspace, solver_test_coefs)


@pytest.fixture(
    scope="module",
    params=list(product(["canonical", "semigrand"], ["expansion", "decomposition"])),
)
def orig_ensemble(solver_test_expansion, request):
    if request.param[0] == "semigrand":
        chemical_potentials = {
            "Li+": np.random.normal(),
            "Mn2+": np.random.normal(),
            "Mn3+": np.random.normal(),
            "Mn4+": np.random.normal(),
            "Ti4+": np.random.normal(),
            "Vacancy": np.random.normal(),
            "O2-": np.random.normal(),
            "O-": np.random.normal(),
            "F-": np.random.normal(),
        }
    else:
        chemical_potentials = None
    return Ensemble.from_cluster_expansion(
        solver_test_expansion,
        np.diag([5, 2, 2]),
        request.param[1],
        chemical_potentials=chemical_potentials,
    )


@pytest.fixture(scope="module")
# Cation sub-lattice sorted in front of anion sub-lattice.
def orig_sublattices(orig_ensemble):
    return orig_ensemble.sublattices


@pytest.fixture(scope="module")
def solver_test_initial_occupancy(orig_sublattices):
    return get_random_solver_test_occu(orig_sublattices)


@pytest.fixture(scope="module")
def solver_test_ensemble(orig_ensemble, solver_test_initial_occupancy):
    cation_id = None
    anion_id = None
    for sl_id in range(2):
        if Species("Li", 1) in orig_ensemble.sublattices[sl_id].species:
            cation_id = sl_id
        if Species("O", -2) in orig_ensemble.sublattices[sl_id].species:
            anion_id = sl_id

    # Split the cation sublattice.
    new_ensemble = Ensemble.from_dict(orig_ensemble.as_dict())

    # Manually restrict 3 random li sites, 1 Vacancy site.
    cation_sites = new_ensemble.sublattices[cation_id].sites
    li_code = new_ensemble.sublattices[cation_id].encoding[
        new_ensemble.sublattices[cation_id].species.index(Species("Li", 1))
    ]
    li_sites = new_ensemble.sublattices[cation_id].sites[
        np.where(solver_test_initial_occupancy[cation_sites] == li_code)[0]
    ]
    li_restricts = np.random.choice(li_sites, size=3, replace=False)
    new_ensemble.restrict_sites(li_restricts)
    va_code = new_ensemble.sublattices[cation_id].encoding[
        new_ensemble.sublattices[cation_id].species.index(Vacancy())
    ]
    va_sites = new_ensemble.sublattices[cation_id].sites[
        np.where(solver_test_initial_occupancy[cation_sites] == va_code)[0]
    ]
    va_restricts = np.random.choice(va_sites, size=1, replace=False)
    new_ensemble.restrict_sites(va_restricts)

    # Manually restrict 2 random O2- sites.
    anion_sites = new_ensemble.sublattices[anion_id].sites
    o2_code = new_ensemble.sublattices[anion_id].encoding[
        new_ensemble.sublattices[anion_id].species.index(Species("O", -2))
    ]
    o2_sites = new_ensemble.sublattices[anion_id].sites[
        np.where(solver_test_initial_occupancy[anion_sites] == o2_code)[0]
    ]
    o2_restricts = np.random.choice(o2_sites, size=2, replace=False)
    new_ensemble.restrict_sites(o2_restricts)

    ca_partitions = [
        [Species("Li", 1), Vacancy()],
        [Species("Mn", 2), Species("Mn", 3), Species("Mn", 4), Species("Ti", 4)],
    ]
    an_partitions = [[Species("O", -2), Species("O", -1)], [Species("F", -1)]]
    new_ensemble.split_sublattice_by_species(
        cation_id, solver_test_initial_occupancy, ca_partitions
    )
    # Sub-lattices all updated after partition.
    anion_id = anion_id if cation_id > anion_id else anion_id + 1
    new_ensemble.split_sublattice_by_species(
        anion_id, solver_test_initial_occupancy, an_partitions
    )

    # Check if sites a correctly restricted.
    assert len(new_ensemble.sublattices) == 4
    for site in li_restricts:
        assert site in new_ensemble.restricted_sites
    for site in va_restricts:
        assert site in new_ensemble.restricted_sites
    for site in o2_restricts:
        assert site in new_ensemble.restricted_sites

    return new_ensemble
