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

from .utils import get_random_exotic_occu


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


@pytest.fixture(scope="module", params=["sinusoid"])
def exotic_subspace(exotic_prim, request):
    # Use sinusoid basis to test if useful.
    space = ClusterSubspace.from_cutoffs(
        exotic_prim, {2: 4, 3: 3, 4: 2}, basis=request.param
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
    assert len(new_ensemble.sublattices) == 4
    for site in li_restricts:
        assert site in new_ensemble.restricted_sites
    for site in o2_restricts:
        assert site in new_ensemble.restricted_sites

    return new_ensemble
