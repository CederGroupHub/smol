import numpy as np
import numpy.testing as npt
import pytest

from smol.cofe import ClusterExpansion
from smol.utils.cluster.correlations import (
    corr_from_occupancy,
    delta_corr_single_flip,
    delta_interactions_single_flip,
    interactions_from_occupancy,
)
from smol.utils.exceptions import StructureMatchError
from tests.utils import compute_cluster_interactions, gen_random_ordered_structure

ATOL = 2e4 * np.finfo(float).eps


# test the new evaluator vs the legacy functions
@pytest.mark.xfail(raises=StructureMatchError)
def test_corr_from_occu(cluster_subspace, supercell_matrix, rng):
    supercell_matrix = 2 * supercell_matrix

    # legacy orbit lists
    mappings = cluster_subspace.supercell_orbit_mappings(supercell_matrix)
    orbit_list = []
    for orbit, cluster_inds in zip(cluster_subspace.orbits, mappings):
        orbit_list.append(
            (
                orbit.bit_id,
                orbit.flat_tensor_indices,
                orbit.flat_correlation_tensors,
                cluster_inds,
            )
        )

    orbit_list_ratio = []
    for orbit, cluster_inds in zip(cluster_subspace.orbits, mappings):
        orbit_list_ratio.append(
            (
                orbit.bit_id,
                1.0,
                orbit.flat_tensor_indices,
                orbit.flat_correlation_tensors,
                cluster_inds,
            )
        )

    ratio = np.ones(len(cluster_subspace.orbits))
    indices = cluster_subspace.get_orbit_indices(supercell_matrix)

    for _ in range(10):
        structure = gen_random_ordered_structure(
            cluster_subspace.structure, supercell_matrix
        )

        # test correlations
        occu = cluster_subspace.occupancy_from_structure(
            structure, encode=True, scmatrix=supercell_matrix
        )

        # legacy cython function
        corr = corr_from_occupancy(occu, len(cluster_subspace), orbit_list)

        npt.assert_allclose(
            corr,
            cluster_subspace.evaluator.correlations_from_occupancy(
                occu, indices.container
            ),
        )
        npt.assert_allclose(
            corr,
            cluster_subspace.corr_from_structure(structure, scmatrix=supercell_matrix),
        )

        # test correlation changes
        occu_f = occu.copy()
        occu_f[0] = 0 if occu[0] != 0 else 1
        occu_f[1] = 0 if occu[0] != 0 else 1
        corr_f = corr_from_occupancy(occu_f, len(cluster_subspace), orbit_list)

        npt.assert_allclose(
            delta_corr_single_flip(
                occu_f, occu, len(cluster_subspace), orbit_list_ratio
            ),
            cluster_subspace.evaluator.delta_correlations_from_occupancies(
                occu_f, occu, ratio, indices.container
            ),
        )

        # check that it also matches the full difference
        npt.assert_allclose(
            corr_f - corr,
            cluster_subspace.evaluator.delta_correlations_from_occupancies(
                occu_f, occu, ratio, indices.container
            ),
            atol=ATOL,
        )


def test_interactions_from_occu(cluster_subspace, supercell_matrix, rng):
    supercell_matrix = 2 * supercell_matrix
    eci = rng.random(len(cluster_subspace))
    expansion = ClusterExpansion(cluster_subspace, eci)
    flat_interaction_tensors = tuple(
        np.ravel(t, order="C") for t in expansion.cluster_interaction_tensors[1:]
    )
    offset = expansion.cluster_interaction_tensors[0]

    cluster_subspace.evaluator.set_cluster_interactions(
        flat_interaction_tensors,
        expansion.cluster_interaction_tensors[0],
    )
    # legacy orbit lists
    mappings = cluster_subspace.supercell_orbit_mappings(supercell_matrix)
    orbit_list = []
    for orbit, cluster_inds, flat_tensor in zip(
        cluster_subspace.orbits, mappings, flat_interaction_tensors
    ):
        orbit_list.append(
            (
                orbit.flat_tensor_indices,
                flat_tensor,
                cluster_inds,
            )
        )

    orbit_list_ratio = []
    for orbit, cluster_inds, flat_tensor in zip(
        cluster_subspace.orbits, mappings, flat_interaction_tensors
    ):
        orbit_list_ratio.append(
            (
                orbit.id,
                1.0,
                orbit.flat_tensor_indices,
                flat_tensor,
                cluster_inds,
            )
        )

    ratio = np.ones(len(cluster_subspace.orbits))
    indices = cluster_subspace.get_orbit_indices(supercell_matrix)

    for _ in range(10):
        structure = gen_random_ordered_structure(
            cluster_subspace.structure, supercell_matrix
        )

        # test correlations
        occu = cluster_subspace.occupancy_from_structure(
            structure, encode=True, scmatrix=supercell_matrix
        )

        # compare with explicit calculation of interactions
        # (see utils.compute_cluster_interactions)
        interactions = compute_cluster_interactions(
            expansion, structure, scmatrix=supercell_matrix
        )

        # legacy cython function
        npt.assert_allclose(
            interactions,
            interactions_from_occupancy(
                occu, cluster_subspace.num_orbits, offset, orbit_list
            ),
        )

        # calculations from cluster expansion
        npt.assert_allclose(
            interactions,
            expansion.cluster_interactions_from_structure(
                structure, scmatrix=supercell_matrix
            ),
        )

        # explicit calls to evaluator
        npt.assert_allclose(
            interactions,
            cluster_subspace.evaluator.interactions_from_occupancy(
                occu, indices.container
            ),
        )

        # test correlation changes
        occu_f = occu.copy()
        occu_f[0] = 0 if occu[0] != 0 else 1

        # legacy cython function
        npt.assert_allclose(
            delta_interactions_single_flip(
                occu_f, occu, cluster_subspace.num_orbits, orbit_list_ratio
            ),
            cluster_subspace.evaluator.delta_interactions_from_occupancies(
                occu_f, occu, ratio, indices.container
            ),
        )

        # check that it also matches a full difference with > 1 flips
        occu_f[1] = 0 if occu[0] != 0 else 1
        interactions_f = interactions_from_occupancy(
            occu_f, cluster_subspace.num_orbits, offset, orbit_list
        )

        npt.assert_allclose(
            interactions_f - interactions,
            cluster_subspace.evaluator.delta_interactions_from_occupancies(
                occu_f, occu, ratio, indices.container
            ),
            atol=ATOL,
        )
