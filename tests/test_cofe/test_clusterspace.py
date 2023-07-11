from itertools import combinations

import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.analysis.structure_matcher import (
    OrderDisorderElementComparator,
    StructureMatcher,
)
from pymatgen.core import Species, Structure
from pymatgen.util.coord import is_coord_subset_pbc

from smol.cofe import ClusterSubspace, PottsSubspace
from smol.cofe.space.clusterspace import get_complete_mapping, invert_mapping
from smol.cofe.space.constants import SITE_TOL
from smol.cofe.space.domain import Vacancy, get_allowed_species
from smol.utils.cluster import get_orbit_data
from smol.utils.cluster.evaluator import ClusterSpaceEvaluator
from smol.utils.exceptions import StructureMatchError
from tests.utils import assert_msonable, assert_pickles, gen_random_ordered_structure

pytestmark = pytest.mark.filterwarnings("ignore:All bit combos have been removed")


def test_from_cutoffs(structure):
    cutoffs = {2: 5, 3: 4, 4: 4}
    for increment in np.arange(0, 3, 1):
        cutoffs.update(
            {k: v + increment / (n + 1) for n, (k, v) in enumerate(cutoffs.items())}
        )
        subspace = ClusterSubspace.from_cutoffs(structure, cutoffs)
        tight_subspace = ClusterSubspace.from_cutoffs(structure, subspace.cutoffs)
        assert len(subspace) == len(tight_subspace)
        npt.assert_allclose(
            np.array(list(subspace.cutoffs.values())),
            np.array(list(tight_subspace.cutoffs.values())),
        )


def test_orbits(cluster_subspace):
    # test that all orbits generated are unique
    assert len(cluster_subspace) == cluster_subspace.num_corr_functions
    assert len(cluster_subspace.orbits) + 1 == cluster_subspace.num_orbits

    for o1, o2 in combinations(cluster_subspace.orbits, 2):
        assert o1 != o2


def test_cutoffs(cluster_subspace, cluster_cutoffs):
    for s, c in cluster_subspace.cutoffs.items():
        assert cluster_cutoffs[s] >= c


def test_orbits_by_diameter(cluster_subspace):
    previous_diameter = -1
    for diameter, orbits in cluster_subspace.orbits_by_diameter.items():
        assert all(np.isclose(o.base_cluster.diameter, diameter) for o in orbits)
        assert diameter > previous_diameter
        previous_diameter = diameter


def test_orbits_from_cutoffs(cluster_subspace, cluster_cutoffs):
    # Get all of them
    max_cutoff = max(cluster_cutoffs.values())
    assert all(
        o1 == o2
        for o1, o2 in zip(
            cluster_subspace.orbits, cluster_subspace.orbits_from_cutoffs(max_cutoff)
        )
    )
    for upper, lower in ((5, 0), (6, 3), (5, 2)):
        orbs = cluster_subspace.orbits_from_cutoffs(upper, lower)
        assert len(orbs) < len(cluster_subspace.orbits)
        assert all(lower <= o.base_cluster.diameter <= upper for o in orbs)

    # Test with dict
    upper = {2: 4.5, 3: 3.5}
    orbs = cluster_subspace.orbits_from_cutoffs(upper)
    assert len(orbs) < len(cluster_subspace.orbits)
    assert all(
        o.base_cluster.diameter <= upper[2] for o in orbs if len(o.base_cluster) == 2
    )
    assert all(
        o.base_cluster.diameter <= upper[3] for o in orbs if len(o.base_cluster) == 3
    )

    # Test for only pairs
    upper = {2: 4.5}
    orbs = cluster_subspace.orbits_from_cutoffs(upper)
    assert len(orbs) < len(cluster_subspace.orbits)
    assert all(
        o.base_cluster.diameter <= upper[2] for o in orbs if len(o.base_cluster) == 2
    )
    assert all(len(o.base_cluster) == 2 for o in orbs)

    # bad cuttoffs
    assert len(cluster_subspace.orbits_from_cutoffs(2, 4)) == 0


def test_functions_inds_by_size(cluster_subspace):
    indices = cluster_subspace.function_inds_by_size
    # check that all orbit functions are in there...
    assert sum(len(i) for i in indices.values()) == len(cluster_subspace) - 1
    fun_orb_ids = cluster_subspace.function_orbit_ids
    # Now check sizes are correct.
    for s, inds in indices.items():
        assert all(
            s == len(cluster_subspace.orbits[fun_orb_ids[i] - 1].base_cluster)
            for i in inds
        )


def test_functions_inds_by_cutoffs(cluster_subspace):
    indices = cluster_subspace.function_inds_from_cutoffs(6)
    # check that all of them are in there.
    assert len(indices) == len(cluster_subspace) - 1
    fun_orb_ids = cluster_subspace.function_orbit_ids
    for upper, lower in ((4, 0), (5, 3), (3, 1)):
        indices = cluster_subspace.function_inds_from_cutoffs(upper, lower)
        assert len(indices) < len(cluster_subspace)
        assert all(
            lower
            <= cluster_subspace.orbits[fun_orb_ids[i] - 1].base_cluster.diameter
            <= upper
            for i in indices
        )


@pytest.mark.parametrize("orthonormal", [(True, False)])
def test_site_bases(cluster_subspace, basis_name, orthonormal, rng):
    subspace = cluster_subspace.copy()  # copy it to keep original state
    subspace.change_site_bases(basis_name, orthonormal=orthonormal)
    if orthonormal:
        assert subspace.basis_orthogonal
        assert subspace.basis_orthonormal

    structure = gen_random_ordered_structure(subspace.structure, rng=rng)
    if cluster_subspace.basis_type == subspace.basis_type and not orthonormal:
        npt.assert_array_almost_equal(
            subspace.corr_from_structure(structure),
            cluster_subspace.corr_from_structure(structure),
        )
    else:
        assert not np.allclose(
            subspace.corr_from_structure(structure),
            cluster_subspace.corr_from_structure(structure),
        )


# TODO These can probably be improved to check odd and specific cases we want
#  to watch out for
def test_supercell_matrix_from_structure(cluster_subspace, rng):
    # Simple scaling
    supercell = cluster_subspace.structure.copy()
    supercell.make_supercell(2)
    sc_matrix = cluster_subspace.scmatrix_from_structure(supercell)
    assert np.linalg.det(sc_matrix) == pytest.approx(8)

    # A more complex supercell_structure
    m = np.array([[0, 5, 3], [-2, 0, 2], [-2, 4, 3]])
    supercell = cluster_subspace.structure.copy()
    supercell.make_supercell(m)

    sc_matrix = cluster_subspace.scmatrix_from_structure(supercell)
    assert np.linalg.det(sc_matrix) == pytest.approx(abs(np.linalg.det(m)))

    # Test a slightly distorted structure
    supercell = cluster_subspace.structure.copy()
    # up to 2% strain
    supercell.apply_strain(rng.uniform(-0.02, 0.02, size=3))
    supercell.make_supercell(2)
    sc_matrix = cluster_subspace.scmatrix_from_structure(supercell)
    assert np.linalg.det(sc_matrix) == pytest.approx(8)


@pytest.mark.xfail(raises=StructureMatchError)
def test_refine_structure(cluster_subspace, rng):
    supercell = cluster_subspace.structure.copy()
    supercell.make_supercell(3)
    structure = gen_random_ordered_structure(
        cluster_subspace.structure, size=3, rng=rng
    )
    structure.apply_strain(rng.uniform(-0.01, 0.01, size=3))
    refined_structure = cluster_subspace.refine_structure(structure)

    assert not np.allclose(  # check that distorted structure is not equivalent
        supercell.lattice.parameters, structure.lattice.parameters
    )
    npt.assert_allclose(
        supercell.lattice.parameters, refined_structure.lattice.parameters
    )
    npt.assert_array_almost_equal(
        cluster_subspace.corr_from_structure(structure),
        cluster_subspace.corr_from_structure(refined_structure),
    )


def test_remove_orbits(cluster_subspace, rng):
    subspace = cluster_subspace.copy()  # make copy
    remove_num = rng.integers(2, subspace.num_orbits - 1)
    ids_to_remove = rng.choice(
        range(1, subspace.num_orbits), size=remove_num, replace=False
    )
    subspace.remove_orbits(ids_to_remove)

    assert len(subspace.orbits) == len(cluster_subspace.orbits) - remove_num
    assert subspace.num_orbits == cluster_subspace.num_orbits - remove_num
    # check that cached_property is reset
    assert (
        len([o for _, os in subspace.orbits_by_diameter.items() for o in os])
        == len(cluster_subspace.orbits) - remove_num
    )

    for i, orbit in enumerate(cluster_subspace.orbits):
        if i + 1 in ids_to_remove:
            assert orbit not in subspace.orbits
        else:
            assert orbit in subspace.orbits

    corr_inds = [
        i
        for i in range(len(cluster_subspace))
        if cluster_subspace.function_orbit_ids[i] not in ids_to_remove
    ]
    structure = gen_random_ordered_structure(subspace.structure, size=2, rng=rng)
    npt.assert_allclose(
        cluster_subspace.corr_from_structure(structure)[corr_inds],
        subspace.corr_from_structure(structure),
    )

    # remove all orbits of a certain size and make sure key is removed
    size = rng.choice(list(subspace.orbits_by_size.keys()))
    ids_to_remove = [o.id for o in subspace.orbits_by_size[size]]
    subspace.remove_orbits(ids_to_remove)
    assert size not in subspace.orbits_by_size.keys()

    with pytest.raises(ValueError):
        subspace.remove_orbits([-1])
    with pytest.raises(ValueError):
        subspace.remove_orbits([subspace.num_orbits + 1])
    with pytest.raises(ValueError):
        subspace.remove_orbits([0])


def test_remove_corr_functions(cluster_subspace, rng):
    subspace = cluster_subspace.copy()  # make copy
    remove_num = rng.integers(2, len(subspace) - 1)
    ids_to_remove = rng.choice(range(1, len(subspace)), size=remove_num, replace=False)
    subspace.remove_corr_functions(ids_to_remove)

    assert len(subspace) == len(cluster_subspace) - remove_num
    assert (
        subspace.num_corr_functions == cluster_subspace.num_corr_functions - remove_num
    )

    corr_inds = [i for i in range(len(cluster_subspace)) if i not in ids_to_remove]
    structure = gen_random_ordered_structure(subspace.structure, size=2, rng=rng)
    npt.assert_allclose(
        cluster_subspace.corr_from_structure(structure)[corr_inds],
        subspace.corr_from_structure(structure),
    )
    with pytest.warns(UserWarning):
        bid = subspace.orbits[-1].bit_id
        ids = list(range(bid, bid + len(subspace.orbits[-1])))
        subspace.remove_corr_functions(ids)


@pytest.mark.xfail(raises=StructureMatchError)
def test_orbit_mappings(cluster_subspace, supercell_matrix, rng):
    # check that all supercell_structure index groups map to the correct
    # primitive cell sites, and check that max distance under supercell
    # structure pbc is less than the max distance without pbc

    supercell_struct = cluster_subspace.structure.copy()
    supercell_struct.make_supercell(supercell_matrix)
    fcoords = np.array(supercell_struct.frac_coords)

    for orb, inds in zip(
        cluster_subspace.orbits,
        cluster_subspace.supercell_orbit_mappings(supercell_matrix),
    ):
        for x in inds:
            pbc_radius = np.max(
                supercell_struct.lattice.get_all_distances(fcoords[x], fcoords[x])
            )
            # primitive cell fractional coordinates
            new_fc = np.dot(fcoords[x], supercell_matrix)
            assert orb.base_cluster.diameter + 1e-7 > pbc_radius
            found = False
            for equiv in orb.clusters:
                if is_coord_subset_pbc(equiv.frac_coords, new_fc, atol=SITE_TOL):
                    found = True
                    break
            assert found

    # check that the matrix was cached
    m_hash = tuple(sorted(tuple(s.tolist()) for s in supercell_matrix))
    orbit_indices = cluster_subspace._supercell_orbit_inds[m_hash]
    assert orbit_indices.arrays is cluster_subspace.supercell_orbit_mappings(
        supercell_matrix
    )
    assert orbit_indices.container.size == len(
        cluster_subspace.supercell_orbit_mappings(supercell_matrix)
    )

    evaluator = ClusterSpaceEvaluator(
        get_orbit_data(cluster_subspace.orbits),
        cluster_subspace.num_orbits,
        cluster_subspace.num_corr_functions,
    )
    # Test that symmetrically equivalent matrices really produce the
    # same correlation vector for the same occupancy.
    structures = [
        gen_random_ordered_structure(
            cluster_subspace.structure, size=supercell_matrix, rng=rng
        )
        for _ in range(10)
    ]
    matrix2 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) @ np.array(
        supercell_matrix, dtype=int
    )
    matrix3 = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]]) @ np.array(
        supercell_matrix, dtype=int
    )

    # Because of different site_mappings, when you change super-cell,
    # you must re-generate occupancy strings altogether even if your
    # super-cell matrices are symmetrically equivalent.
    occus = [
        cluster_subspace.occupancy_from_structure(
            structure, scmatrix=supercell_matrix, encode=True
        )
        for structure in structures
    ]
    occus2 = [
        cluster_subspace.occupancy_from_structure(
            structure, scmatrix=matrix2, encode=True
        )
        for structure in structures
    ]
    occus3 = [
        cluster_subspace.occupancy_from_structure(
            structure, scmatrix=matrix3, encode=True
        )
        for structure in structures
    ]

    corrs = np.array(
        [
            evaluator.correlations_from_occupancy(
                occu, cluster_subspace.get_orbit_indices(supercell_matrix).container
            )
            for occu in occus
        ]
    )

    corrs2 = np.array(
        [
            evaluator.correlations_from_occupancy(
                occu, cluster_subspace.get_orbit_indices(matrix2).container
            )
            for occu in occus2
        ]
    )

    corrs3 = np.array(
        [
            evaluator.correlations_from_occupancy(
                occu, cluster_subspace.get_orbit_indices(matrix3).container
            )
            for occu in occus3
        ]
    )

    # Symmetrically equivalent matrices should give the same correlation
    # function for the same structure, when the orbit indices mappings
    # are re-generated.
    npt.assert_array_almost_equal(corrs, corrs2)
    npt.assert_array_almost_equal(corrs, corrs3)

    # Symmetrically equivalent matrices should give the same correlation
    # vectors on the same structure, when using the default orbit mapping.
    cluster_subspace._supercell_orbit_inds = {}
    for structure, expected in zip(structures, corrs):
        predicted = cluster_subspace.corr_from_structure(
            structure, scmatrix=supercell_matrix
        )
        npt.assert_array_almost_equal(predicted, expected)

        predicted = cluster_subspace.corr_from_structure(structure, scmatrix=matrix2)
        npt.assert_array_almost_equal(predicted, expected)

        predicted = cluster_subspace.corr_from_structure(structure, scmatrix=matrix3)
        npt.assert_array_almost_equal(predicted, expected)

        predicted = cluster_subspace.corr_from_structure(structure)
        npt.assert_array_almost_equal(predicted, expected)


def test_get_aliased_orbits(cluster_subspace, supercell_matrix):
    # Verify that
    # 1) site mappings for aliased orbits are indeed an identical set,
    # 2) each orbit is counted only once,
    # 3) New implementation is the same with the old one.
    aliased_orbs = cluster_subspace.get_aliased_orbits(supercell_matrix)
    sc_orbit_maps = cluster_subspace.supercell_orbit_mappings(supercell_matrix)
    accounted_orbs = []
    for orb_tup in aliased_orbs:
        assert sum(o in accounted_orbs for o in orb_tup) == 0
        accounted_orbs.extend(orb_tup)
        for i, orb_id_i in enumerate(orb_tup[:-1]):
            orbit_map_i = tuple(
                tuple(sorted(inds_i)) for inds_i in sc_orbit_maps[orb_id_i - 1]
            )
            for orb_id_j in orb_tup[i + 1 :]:
                orbit_map_j = tuple(
                    tuple(sorted(inds_j)) for inds_j in sc_orbit_maps[orb_id_j - 1]
                )
                assert set(orbit_map_i) == set(orbit_map_j)

    aliased_orbits_std = []
    for orb_i, orb_map_i in enumerate(sc_orbit_maps):
        orb_i_id = orb_i + 1
        aliased = False
        orbit_i_aliased = [orb_i_id]
        sorted_orb_map_i = {tuple(sorted(c_map)) for c_map in orb_map_i}

        for orb_j, orb_map_j in enumerate(sc_orbit_maps):
            if orb_i == orb_j:
                continue
            orb_j_id = orb_j + 1
            sorted_orb_map_j = {tuple(sorted(c_map)) for c_map in orb_map_j}

            if sorted_orb_map_i == sorted_orb_map_j:
                aliased = True
                orbit_i_aliased.append(orb_j_id)

        orbit_i_aliased = tuple(sorted(orbit_i_aliased))
        if aliased:
            aliased_orbits_std.append(orbit_i_aliased)

    aliased_orbits_std = sorted(list(set(aliased_orbits_std)), key=lambda x: x[0])
    assert aliased_orbits_std == aliased_orbs


def test_periodicity_and_symmetry(cluster_subspace, supercell_matrix, rng):
    structure = gen_random_ordered_structure(
        cluster_subspace.structure, size=2, rng=rng
    )
    larger_structure = structure.copy()
    larger_structure.make_supercell(supercell_matrix)

    corr = cluster_subspace.corr_from_structure(structure)

    larger_scmatrix = supercell_matrix @ np.eye(3) * 2
    npt.assert_allclose(
        corr,
        cluster_subspace.corr_from_structure(
            larger_structure, scmatrix=larger_scmatrix
        ),
    )
    # Sometimes when supercell_matrix is very skewed, you should provide it.
    # Structure matcher is not good at finding very off-diagonal sc matrices.

    cm = OrderDisorderElementComparator()
    sm = StructureMatcher(allow_subset=True, comparator=cm, scale=True)

    def apply_operation(op, s):
        """Apply on fractional coordinates only."""
        return Structure(
            s.lattice, [st.species for st in s], op.operate_multi(s.frac_coords)
        )

    for symop in cluster_subspace.symops:
        # Be very careful with symop formats, whether they are fractional space
        # ops or not. By default, SpaceGroupAnalyzer gives fractional format,
        # while PointGroupAnalyzer gives cartesian format!
        prim = cluster_subspace.structure
        prim_op = apply_operation(symop, prim)
        # print("op:", symop)
        # print("prim:\n", prim)
        # print("prim_op:\n", prim_op)
        assert sm.fit(prim, prim_op)

        structure_op = apply_operation(symop, structure)
        npt.assert_allclose(corr, cluster_subspace.corr_from_structure(structure_op))


def test_equality(single_subspace, rng):
    subspace = single_subspace.copy()
    assert subspace == single_subspace
    subspace.change_site_bases("legendre")
    assert subspace == single_subspace
    subspace.remove_orbits(rng.choice(range(1, subspace.num_orbits), 2))
    assert subspace != single_subspace


def test_contains(single_subspace, rng):
    for orbit in single_subspace.orbits:
        assert orbit in single_subspace

    subspace = single_subspace.copy()
    orb_ids = rng.choice(range(1, subspace.num_orbits), 2)
    subspace.remove_orbits(orb_ids)
    for i, orbit in enumerate(single_subspace.orbits):
        if i + 1 in orb_ids:
            assert orbit not in subspace
        else:
            assert orbit in subspace


def test_msonable(cluster_subspace_ewald, rng):
    # force caching some orb indices for a few random structures
    _ = repr(cluster_subspace_ewald)  # can probably do better testing than this...
    _ = str(cluster_subspace_ewald)

    for _ in range(2):
        size = rng.integers(1, 4)
        s = gen_random_ordered_structure(
            cluster_subspace_ewald.structure, size=size, rng=rng
        )
        _ = cluster_subspace_ewald.corr_from_structure(s)

    assert_msonable(cluster_subspace_ewald)

    subspace = ClusterSubspace.from_dict(cluster_subspace_ewald.as_dict())
    for key in cluster_subspace_ewald._supercell_orbit_inds.keys():
        for arr1, arr2 in zip(
            subspace._supercell_orbit_inds[key].arrays,
            cluster_subspace_ewald._supercell_orbit_inds[key].arrays,
        ):
            npt.assert_array_equal(arr1, arr2)

    assert len(cluster_subspace_ewald.external_terms) == len(subspace.external_terms)
    npt.assert_allclose(
        subspace.corr_from_structure(s), cluster_subspace_ewald.corr_from_structure(s)
    )


def test_potts_subspace(cluster_subspace, rng):
    potts_subspace = PottsSubspace.from_cutoffs(
        cluster_subspace.structure, cluster_subspace.cutoffs
    )
    assert len(potts_subspace.orbits) == len(cluster_subspace.orbits)

    # check sizes and bits included in each orbit
    for porbit, corbit in zip(potts_subspace.orbits, cluster_subspace.orbits):
        assert len(porbit.site_spaces) == len(corbit.site_spaces)
        assert len(porbit.site_bases) == len(corbit.site_bases)
        assert len(porbit.bit_combos) > len(corbit.bit_combos)

        for i, site_space in enumerate(porbit.site_spaces):
            bits_i = np.concatenate([b[:, i] for b in porbit.bit_combos])
            assert all(j in bits_i for j in site_space.codes)

    # check decorations
    for _ in range(10):
        i = rng.choice(range(1, potts_subspace.num_corr_functions))
        o_id = potts_subspace.function_orbit_ids[i]
        orbit = potts_subspace.orbits[o_id - 1]
        fdeco = potts_subspace.get_function_decoration(i)
        odeco = potts_subspace.get_orbit_decorations(o_id)
        assert fdeco == odeco[i - orbit.bit_id]
        assert all(  # all decorations include valid species
            deco[i] in species
            for deco in fdeco
            for i, species in enumerate(orbit.site_spaces)
        )

    # test removing last bit combo
    potts_subspace1 = PottsSubspace.from_cutoffs(
        cluster_subspace.structure, cluster_subspace.cutoffs, remove_last_cluster=True
    )
    for o1, o2 in zip(potts_subspace.orbits, potts_subspace1.orbits):
        assert len(o1) - 1 == len(o2)

    assert_msonable(potts_subspace)


def test_invert_mapping_table():
    forward = [[], [], [1], [1], [1], [2, 4], [3, 4], [2, 3], [5, 6, 7]]
    backward = [[], [2, 3, 4], [5, 7], [6, 7], [5, 6], [8], [8], [8], []]

    forward_invert = [sorted(sub) for sub in invert_mapping(forward)]
    backward_invert = [sorted(sub) for sub in invert_mapping(backward)]

    assert forward_invert == backward
    assert backward_invert == forward


def test_get_complete_mapping():
    forward = [[], [], [1], [1], [1], [2, 4], [3, 4], [2, 3], [5, 6, 7]]
    backward = [[], [2, 3, 4], [5, 7], [6, 7], [5, 6], [8], [8], [8], []]

    forward_full = [
        [],
        [],
        [1],
        [1],
        [1],
        [1, 2, 4],
        [1, 3, 4],
        [1, 2, 3],
        [1, 2, 3, 4, 5, 6, 7],
    ]
    backward_full = [
        [],
        [2, 3, 4, 5, 6, 7, 8],
        [5, 7, 8],
        [6, 7, 8],
        [5, 6, 8],
        [8],
        [8],
        [8],
        [],
    ]

    forward_comp = [sorted(sub) for sub in get_complete_mapping(forward)]
    backward_comp = [sorted(sub) for sub in get_complete_mapping(backward)]

    assert forward_comp == forward_full
    assert backward_comp == backward_full


# fixed tests for LiCaBr structure
def test_numbers_fixed(single_subspace):
    # Test the total generated orbits, orderings and clusters are
    # as expected.
    assert single_subspace.num_orbits == 27
    assert single_subspace.num_corr_functions == 124
    assert single_subspace.num_clusters == 377


def test_func_orbit_ids_fixed(single_subspace):
    assert len(single_subspace.function_orbit_ids) == 124
    assert len(set(single_subspace.function_orbit_ids)) == 27


def test_function_hierarchy_fixed(single_subspace):
    hierarchy = single_subspace.function_hierarchy()
    assert sorted(hierarchy[0]) == []
    assert sorted(hierarchy[-1]) == [17, 21]
    assert sorted(hierarchy[15]) == []
    assert sorted(hierarchy[35]) == [5, 7, 10]
    assert sorted(hierarchy[56]) == [7, 8, 16]
    assert sorted(hierarchy[75]) == [7, 14, 20]
    assert sorted(hierarchy[95]) == [13, 19, 21]
    assert sorted(hierarchy[115]) == [13, 19, 21]


def test_orbit_hierarchy_fixed(single_subspace):
    hierarchy = single_subspace.orbit_hierarchy()
    assert sorted(hierarchy[0]) == []  # empty
    assert sorted(hierarchy[1]) == []  # point
    assert sorted(hierarchy[3]) == [1, 2]  # distinct site pair
    assert sorted(hierarchy[4]) == [1]  # same site pair
    assert sorted(hierarchy[15]) == [3, 6]  # triplet
    assert sorted(hierarchy[-1]) == [6, 7]


def test_corr_from_structure(single_subspace, rng):
    structure = Structure(
        single_subspace.structure.lattice,
        [
            "Li+",
        ]
        * 2
        + ["Ca+"]
        + ["Br-"],
        single_subspace.structure.frac_coords,
    )
    corr = single_subspace.corr_from_structure(structure)
    assert len(corr) == single_subspace.num_corr_functions + len(
        single_subspace.external_terms
    )
    assert corr[0] == 1

    cs = ClusterSubspace.from_cutoffs(
        single_subspace.structure, {2: 5}, basis="indicator"
    )

    # make an ordered supercell_structure
    s = single_subspace.structure.copy()
    s.make_supercell([2, 1, 1])
    species = ("Li+", "Ca+", "Li+", "Ca+", "Br-", "Br-")
    coords = (
        (0.125, 0.25, 0.25),
        (0.625, 0.25, 0.25),
        (0.375, 0.75, 0.75),
        (0.25, 0.5, 0.5),
        (0, 0, 0),
        (0.5, 0, 0),
    )
    s = Structure(s.lattice, species, coords)
    assert len(cs.corr_from_structure(s)) == 22

    expected = [
        1,
        0.5,
        0.25,
        0,
        0.5,
        0,
        0.375,
        0,
        0.0625,
        0.25,
        0.125,
        0,
        0.25,
        0.125,
        0.125,
        0,
        0,
        0.25,
        0,
        0.125,
        0,
        0.1875,
    ]
    npt.assert_allclose(cs.corr_from_structure(s), expected)

    # Test occu_from_structure
    occu = [
        Vacancy(),
        Species("Li", 1),
        Species("Ca", 1),
        Species("Li", 1),
        Vacancy(),
        Species("Ca", 1),
        Species("Br", -1),
        Species("Br", -1),
    ]
    assert all(s1 == s2 for s1, s2 in zip(occu, cs.occupancy_from_structure(s)))

    # shuffle sites and check correlation still works
    for _ in range(10):
        rng.shuffle(s)
        npt.assert_allclose(cs.corr_from_structure(s), expected)


def test_periodicity(single_subspace):
    # Check to see if a supercell of a smaller structure gives the same corr
    m = np.array([[2, 0, 0], [0, 2, 0], [0, 1, 1]])
    supercell = single_subspace.structure.copy()
    supercell.make_supercell(m)
    s = Structure(
        supercell.lattice,
        ["Ca+", "Li+", "Li+", "Br-", "Br-", "Br-", "Br-"],
        [
            [0.125, 1, 0.25],
            [0.125, 0.5, 0.25],
            [0.375, 0.5, 0.75],
            [0, 0, 0],
            [0, 0.5, 1],
            [0.5, 1, 0],
            [0.5, 0.5, 0],
        ],
    )
    a = single_subspace.corr_from_structure(s)
    s.make_supercell([2, 1, 1])
    b = single_subspace.corr_from_structure(s)
    npt.assert_allclose(a, b)


def test_site_basis_rotation(cluster_subspace, rng):
    cs = cluster_subspace.copy()
    if not cs.basis_orthogonal:
        with pytest.raises(RuntimeError):
            cs.rotate_site_basis(1, np.pi / 4)
        cs.change_site_bases("sinusoid", orthonormal=True)

    with pytest.raises(ValueError):
        cs.rotate_site_basis(len(cs.orbits_by_size[1]) + 2, np.pi)

    cs1 = cs.copy()
    # print(cs.site_rotation_matrix)
    cs1.rotate_site_basis(1, np.pi / 4)
    # print(cs1.site_rotation_matrix)
    for i in range(5):
        structure = gen_random_ordered_structure(cs.structure, rng=rng)
        for j in range(5):
            coefs1 = 10 * np.random.random(len(cs))
            coefs = coefs1.copy()
            coefs = cs1.site_rotation_matrix.T @ coefs
            eci = coefs / cs.function_ordering_multiplicities
            eci1 = coefs1 / cs1.function_ordering_multiplicities
            norms = np.array(
                [
                    np.sum(
                        cs.function_ordering_multiplicities[
                            np.array(cs.function_orbit_ids) == i
                        ]
                        * eci[np.array(cs.function_orbit_ids) == i] ** 2
                    )
                    for i in range(len(cs.orbits) + 1)
                ]
            )
            norms1 = np.array(
                [
                    np.sum(
                        cs1.function_ordering_multiplicities[
                            np.array(cs.function_orbit_ids) == i
                        ]
                        * eci1[np.array(cs1.function_orbit_ids) == i] ** 2
                    )
                    for i in range(len(cs.orbits) + 1)
                ]
            )
            # test ECI invariance
            assert np.allclose(norms, norms1)
            # test correlation vector invariance
            assert np.allclose(
                cs1.site_rotation_matrix @ cs.corr_from_structure(structure),
                cs1.corr_from_structure(structure),
            )
            # test values are the same
            assert np.isclose(
                np.dot(cs.corr_from_structure(structure), coefs),
                np.dot(cs1.corr_from_structure(structure), coefs1),
            )
            # test rotation matrix is invertible and abs(det) = 1
            # binary site rotations may lead to det = -1 ?
            assert np.isclose(abs(np.linalg.det(cs1.site_rotation_matrix)), 1)


def _encode_occu(occu, bits):
    return np.array([bit.index(sp) for sp, bit in zip(occu, bits)])


def test_vs_CASM_pairs(single_structure):
    species = [{"Li+": 0.1}] * 3 + ["Br-"]
    coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75), (0.5, 0.5, 0.5), (0, 0, 0))
    structure = Structure(single_structure.lattice, species, coords)
    cs = ClusterSubspace.from_cutoffs(structure, {2: 6}, basis="indicator")
    evaluator = ClusterSpaceEvaluator(
        get_orbit_data(cs.orbits), cs.num_orbits, cs.num_corr_functions
    )
    spaces = get_allowed_species(structure)
    m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    indices = cs.get_orbit_indices(m)

    # last two clusters are switched from CASM output (occupancy basis)
    # all_li (ignore casm point term)
    occu = _encode_occu([Species("Li", 1), Species("Li", 1), Species("Li", 1)], spaces)
    corr = evaluator.correlations_from_occupancy(occu, indices.container)
    npt.assert_allclose(corr, np.array([1] * 12))

    # all_vacancy
    occu = _encode_occu([Vacancy(), Vacancy(), Vacancy()], spaces)
    corr = evaluator.correlations_from_occupancy(occu, indices.container)
    npt.assert_allclose(corr, np.array([1] + [0] * 11))
    # octahedral
    occu = _encode_occu([Vacancy(), Vacancy(), Species("Li", 1)], spaces)
    corr = evaluator.correlations_from_occupancy(occu, indices.container)
    npt.assert_allclose(corr, [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    # tetrahedral
    occu = _encode_occu([Species("Li", 1), Species("Li", 1), Vacancy()], spaces)
    corr = evaluator.correlations_from_occupancy(occu, indices.container)
    npt.assert_allclose(corr, [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    # mixed
    occu = _encode_occu([Species("Li", 1), Vacancy(), Species("Li", 1)], spaces)
    corr = evaluator.correlations_from_occupancy(occu, indices.container)
    npt.assert_allclose(corr, [1, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0, 0.5, 1])
    # single_tet
    occu = _encode_occu([Species("Li", 1), Vacancy(), Vacancy()], spaces)
    corr = evaluator.correlations_from_occupancy(occu, indices.container)
    npt.assert_allclose(corr, [1, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0])


def test_vs_CASM_triplets(single_structure):
    """
    Test vs casm generated correlation with occupancy basis.
    """
    species = [{"Li+": 0.1}] * 3 + ["Br-"]
    coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75), (0.5, 0.5, 0.5), (0, 0, 0))
    structure = Structure(single_structure.lattice, species, coords)
    cs = ClusterSubspace.from_cutoffs(structure, {2: 6, 3: 4.5}, basis="indicator")
    evaluator = ClusterSpaceEvaluator(
        get_orbit_data(cs.orbits), cs.num_orbits, cs.num_corr_functions
    )
    spaces = get_allowed_species(structure)
    m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    indices = cs.get_orbit_indices(m)

    # last two pair terms are switched from CASM output (occupancy basis)
    # all_vacancy (ignore casm point term)
    occu = _encode_occu([Vacancy(), Vacancy(), Vacancy()], spaces)
    corr = evaluator.correlations_from_occupancy(occu, indices.container)
    npt.assert_allclose(corr, np.array([1] + [0] * 18))
    # all Li
    occu = _encode_occu([Species("Li", 1), Species("Li", 1), Species("Li", 1)], spaces)
    corr = evaluator.correlations_from_occupancy(occu, indices.container)
    npt.assert_allclose(corr, np.array([1] * 19))
    # octahedral
    occu = _encode_occu([Vacancy(), Vacancy(), Species("Li", 1)], spaces)
    corr = evaluator.correlations_from_occupancy(occu, indices.container)
    npt.assert_allclose(corr, [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])

    # tetrahedral
    occu = _encode_occu([Species("Li", 1), Species("Li", 1), Vacancy()], spaces)
    corr = evaluator.correlations_from_occupancy(occu, indices.container)
    npt.assert_allclose(corr, [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
    # mixed
    occu = _encode_occu([Species("Li", 1), Vacancy(), Species("Li", 1)], spaces)
    corr = evaluator.correlations_from_occupancy(occu, indices.container)
    npt.assert_allclose(
        corr,
        [1, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0, 0.5, 1, 0, 0, 0.5, 0.5, 0.5, 0.5, 1],
    )
    # single_tet
    occu = _encode_occu([Species("Li", 1), Vacancy(), Vacancy()], spaces)
    corr = evaluator.correlations_from_occupancy(occu, indices.container)
    npt.assert_allclose(
        corr, [1, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.5, 0.5, 0]
    )


def test_vs_CASM_multicomp(single_structure):
    cs = ClusterSubspace.from_cutoffs(single_structure, {2: 5}, basis="indicator")
    evaluator = ClusterSpaceEvaluator(
        get_orbit_data(cs.orbits), cs.num_orbits, cs.num_corr_functions
    )
    m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    spaces = get_allowed_species(single_structure)
    indices = cs.get_orbit_indices(m)

    # mixed
    occu = _encode_occu([Vacancy(), Species("Li", 1), Species("Li", 1)], spaces)
    corr = evaluator.correlations_from_occupancy(occu, indices.container)
    npt.assert_allclose(
        corr, [1, 0.5, 0, 1, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 1, 0, 0, 0.5, 0, 0, 0]
    )

    # Li-tet Ca-oct
    occu = _encode_occu([Vacancy(), Species("Li", 1), Species("Ca", 1)], spaces)
    corr = evaluator.correlations_from_occupancy(occu, indices.container)
    npt.assert_allclose(
        corr, [1, 0.5, 0, 0, 1, 0, 0.5, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1, 0, 0.5, 0, 0]
    )


def test_pickles(single_subspace):
    assert_pickles(single_subspace)
