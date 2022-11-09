from collections import defaultdict
from copy import deepcopy

import numpy as np
import numpy.testing as npt
from pymatgen.entries.computed_entries import ComputedStructureEntry

from smol.cofe.extern import EwaldTerm
from smol.cofe.wrangling import (
    max_ewald_energy_indices,
    unique_corr_vector_indices,
    weights_energy_above_composition,
    weights_energy_above_hull,
)
from smol.cofe.wrangling.tools import _energies_above_composition


def test_unique_corr_indices(structure_wrangler):
    indices, duplicates = unique_corr_vector_indices(
        structure_wrangler, "energy", return_compliment=True
    )
    assert set(indices) | set(duplicates) == {
        i for i in range(structure_wrangler.num_structures)
    }

    feature_matrix = structure_wrangler.feature_matrix
    energies = structure_wrangler.get_property_vector("energy")
    feature_nodupe = feature_matrix[indices, :]
    # No duplicacy
    assert feature_nodupe.shape == np.unique(feature_nodupe, axis=0).shape
    feature_nodupe_sorted = sorted(feature_nodupe.tolist())
    feature_unique_sorted = sorted(np.unique(feature_nodupe, axis=0).tolist())
    npt.assert_allclose(feature_nodupe_sorted, feature_unique_sorted)

    for rid, row in zip(indices, feature_nodupe):
        dupe = np.all(np.isclose(feature_matrix, row), axis=1)
        npt.assert_equal(np.min(energies[dupe]), energies[rid])


def test_ewald_energy_indices(structure_wrangler):
    e_tol = 0.01

    def get_ewald_energies(wrangler):
        ewald_energy = None
        for term in wrangler.cluster_subspace.external_terms:
            if isinstance(term, EwaldTerm):
                ewald_energy = [i["features"][-1] for i in wrangler.entries]

        if ewald_energy is None:
            ewald_energy = []
            for entry in wrangler.entries:
                struct = entry.structure
                matrix = entry.data["supercell_matrix"]
                map = entry.data["site_mapping"]
                occu = wrangler.cluster_subspace.occupancy_from_structure(
                    struct, encode=True, scmatrix=matrix, site_mapping=map
                )

                supercell = wrangler.cluster_subspace.structure.copy()
                supercell.make_supercell(matrix)
                size = wrangler.cluster_subspace.num_prims_from_matrix(matrix)
                term = EwaldTerm().value_from_occupancy(occu, supercell) / size
                ewald_energy.append(term)
        return ewald_energy

    indices, compliments = max_ewald_energy_indices(
        structure_wrangler, e_tol, return_compliment=True
    )
    assert set(indices) | set(compliments) == {
        i for i in range(structure_wrangler.num_structures)
    }

    comps = [
        entry.structure.composition.reduced_composition
        for entry in structure_wrangler.entries
    ]
    energies = np.array(get_ewald_energies(structure_wrangler))
    comp_min_energies = defaultdict(lambda: np.inf)
    for i in indices:
        if energies[i] < comp_min_energies[comps[i]]:
            comp_min_energies[comps[i]] = energies[i]

    for i in indices:
        assert energies[i] - comp_min_energies[comps[i]] <= e_tol


def test_weights_above_composition(structure_wrangler):
    weights = weights_energy_above_composition(
        structure_wrangler.structures,
        structure_wrangler.get_property_vector("energy", False),
        temperature=1000,
    )
    is_comp_min = np.isclose(weights, 1, atol=1e-8)
    comps = [s.composition.reduced_composition for s in structure_wrangler.structures]
    comps_min = [c for is_min, c in zip(is_comp_min, comps) if is_min]

    # For each composition, at least one structure in minimum set.
    assert set(comps) == set(comps_min)
    # These weights were generated from fixed lno data, but can not be used with random
    # wrangler.
    # We can not check correctness of energies and weights with random wrangler.
    # expected = np.array([0.85637358, 0.98816678, 1., 0.59209449, 1.,
    #             0.92882071, 0.87907454, 0.94729315, 0.40490513, 0.82484222,
    #             0.81578984, 1., 0.89615121, 0.92893004, 0.81650693,
    #             0.6080223 , 0.94848913, 0.92135297, 0.92326977, 0.83995635,
    #             1., 0.94663979, 1., 0.9414506, 1.])
    # assert np.allclose(expected, structure_wrangler.get_weights('comp'))
    # sizes = np.array(list(map(len, structure_wrangler.structures)))
    sizes = structure_wrangler.sizes * len(
        structure_wrangler.cluster_subspace.structure
    )
    energies = structure_wrangler.get_property_vector("energy", False)

    e_above = _energies_above_composition(structure_wrangler.structures, energies)
    comp_min_energies = defaultdict(lambda: np.inf)
    for i in range(len(energies)):
        if energies[i] / sizes[i] < comp_min_energies[comps[i]]:
            comp_min_energies[comps[i]] = energies[i] / sizes[i]
    e_comps = np.array([comp_min_energies[comps[i]] for i in range(len(energies))])
    npt.assert_almost_equal(energies / sizes, e_comps + e_above, decimal=8)


def test_weights_above_hull(structure_wrangler):
    weights = weights_energy_above_hull(
        structure_wrangler.structures,
        structure_wrangler.get_property_vector("energy", False),
        structure_wrangler.cluster_subspace.structure,
        temperature=1000,
    )
    # expected = np.array([0.85637358, 0.98816678, 1., 0.56916328, 0.96127103,
    #    0.89284844, 0.84502889, 0.91060546, 0.40490513, 0.82484222,
    #    0.81578984, 1., 0.89615121, 0.92893004, 0.81650693,
    #    0.58819251, 0.91755548, 0.89130433, 0.89315862, 0.81256235,
    #    0.9673864 , 0.91576647, 1., 0.9414506 , 1])
    # self.assertTrue(np.allclose(expected, structure_wrangler.get_weights('hull')))
    weights_c = weights_energy_above_composition(
        structure_wrangler.structures,
        structure_wrangler.get_property_vector("energy", False),
        temperature=1000,
    )
    structure_wrangler.get_property_vector("energy", False)
    np.array(list(map(len, structure_wrangler.structures)))

    # Weak tests on the hull generation:
    # 1, Hull must always be no higher than min comp energies
    # 2, There must be at least one comp with min energy in hull.
    # Add a small tolerance to avoid occasional fails.
    assert np.all(weights_c + 1e-7 >= weights)
    is_comp_min = np.isclose(weights_c, 1, atol=1e-8)
    on_hull = np.isclose(weights, 1, atol=1e-8)
    assert np.sum(on_hull & is_comp_min) > 0


# TODO improve test by precalculating expected energies and filtered structs
def test_filter_by_ewald(structure_wrangler):
    n_structs = structure_wrangler.num_structures
    indices = max_ewald_energy_indices(structure_wrangler, max_relative_energy=0.0)
    assert len(indices) < n_structs
    indices, compliment = max_ewald_energy_indices(
        structure_wrangler, max_relative_energy=0.0, return_compliment=True
    )
    assert len(indices) + len(compliment) == n_structs

    # with ewaldterm now
    structure_wrangler.cluster_subspace.add_external_term(EwaldTerm())
    structure_wrangler.update_features()
    indices2 = max_ewald_energy_indices(structure_wrangler, max_relative_energy=0.0)
    npt.assert_array_equal(indices, indices2)


def test_filter_duplicate_corr_vectors(structure_wrangler, rng):
    # add some repeat structures with infinite energy
    dup_entries = []
    for i in range(5):
        ind = rng.integers(structure_wrangler.num_structures)
        dup_entry = ComputedStructureEntry(
            structure_wrangler.structures[ind].copy(),
            np.inf,
            data=deepcopy(structure_wrangler.entries[ind].data),
        )
        dup_entries.append(dup_entry)

    final = structure_wrangler.num_structures
    structure_wrangler._entries += dup_entries
    n_structs = structure_wrangler.num_structures

    assert structure_wrangler.num_structures == final + len(dup_entries)
    assert np.inf in structure_wrangler.get_property_vector("energy")
    indices = unique_corr_vector_indices(structure_wrangler, property_key="energy")
    assert len(indices) < n_structs
    assert len(indices) == n_structs - len(dup_entries)
    assert np.inf not in structure_wrangler.get_property_vector("energy")[indices]
    indices, compliment = unique_corr_vector_indices(
        structure_wrangler, property_key="energy", return_compliment=True
    )
    assert len(indices) + len(compliment) == n_structs
    indices = unique_corr_vector_indices(
        structure_wrangler, property_key="energy", filter_by="max"
    )
    assert np.inf in structure_wrangler.get_property_vector("energy")[indices]
