import random
from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pytest

from smol.cofe import StructureWrangler
from smol.cofe.extern import EwaldTerm
from tests.utils import assert_msonable, gen_fake_training_data, gen_random_structure


def test_add_data(structure_wrangler):
    for struct, energy in gen_fake_training_data(
        structure_wrangler.cluster_subspace.structure
    ):
        structure_wrangler.add_data(struct, {"energy": energy}, weights={"random": 2.0})
    struct = gen_random_structure(structure_wrangler.cluster_subspace.structure)
    energy = -len(struct) * np.random.random()
    structure_wrangler.add_data(struct, {"energy": energy}, weights={"random": 3.0})

    assert all(w == 2.0 for w in structure_wrangler.get_weights("random")[:-1])
    assert (
        len(structure_wrangler.get_weights("random"))
        == structure_wrangler.num_structures
    )
    assert structure_wrangler.get_weights("random")[-1] == 3.0
    assert structure_wrangler.available_properties == ["energy"]
    assert structure_wrangler.available_weights == ["random"]

    # heavily distorted structure
    struct = struct.copy()
    struct.apply_strain(0.2)
    with pytest.raises(Exception):
        structure_wrangler.add_data(struct, {"energy": energy}, raise_failed=True)

    with pytest.warns(UserWarning):
        structure_wrangler.add_data(struct, {"energy": energy}, raise_failed=False)

    with pytest.raises(AttributeError):
        structure_wrangler.add_properties("test", structure_wrangler.sizes[:-2])

    structure_wrangler.add_properties(
        "normalized_energy",
        structure_wrangler.get_property_vector("energy", normalize=True),
    )

    items = structure_wrangler._items
    structure_wrangler.remove_all_data()
    assert len(structure_wrangler.structures) == 0

    # Test passing supercell matrices
    for item in items:
        structure_wrangler.add_data(
            item["structure"], item["properties"], supercell_matrix=item["scmatrix"]
        )
    assert len(structure_wrangler.structures) == len(items)
    structure_wrangler.remove_all_data()
    assert len(structure_wrangler.structures) == 0

    # Test passing site mappings
    for item in items:
        structure_wrangler.add_data(
            item["structure"], item["properties"], site_mapping=item["mapping"]
        )

    assert len(structure_wrangler._items) == len(items)
    structure_wrangler.remove_all_data()
    # test passing both
    for item in items:
        structure_wrangler.add_data(
            item["structure"],
            item["properties"],
            supercell_matrix=item["scmatrix"],
            site_mapping=item["mapping"],
        )
    assert len(structure_wrangler.structures) == len(items)

    # Add more properties to test removal
    structure_wrangler.add_properties(
        "normalized", structure_wrangler.get_property_vector("energy", normalize=True)
    )
    structure_wrangler.add_properties(
        "normalized1", structure_wrangler.get_property_vector("energy", normalize=True)
    )
    assert all(
        prop in ["energy", "normalized_energy", "normalized", "normalized1"]
        for prop in structure_wrangler.available_properties
    )
    structure_wrangler.remove_properties(
        "normalized_energy", "normalized", "normalized1"
    )
    assert structure_wrangler.available_properties == ["energy"]

    with pytest.warns(RuntimeWarning):
        structure_wrangler.remove_properties("blab")


def test_append_data_items(structure_wrangler):
    items = structure_wrangler._items
    structure_wrangler.remove_all_data()
    assert len(structure_wrangler.structures) == 0
    with pytest.raises(ValueError):
        structure_wrangler.append_data_items([{"b": 1}])

    data_items = []
    for item in items:
        data_items.append(
            structure_wrangler.process_structure(
                item["structure"],
                item["properties"],
                supercell_matrix=item["scmatrix"],
                site_mapping=item["mapping"],
            )
        )

    structure_wrangler.append_data_items(data_items)
    assert len(structure_wrangler.data_items) == len(items)


def test_data_indices(structure_wrangler):
    test = np.random.choice(range(structure_wrangler.num_structures), 5)
    train = np.setdiff1d(range(structure_wrangler.num_structures), test)
    structure_wrangler.add_data_indices("test", test)
    structure_wrangler.add_data_indices("train", train)
    assert all(key in structure_wrangler.available_indices for key in ["test", "train"])
    with pytest.raises(ValueError):
        structure_wrangler.add_data_indices(
            "bla",
            [
                structure_wrangler.num_structures,
            ],
        )

    with pytest.raises(TypeError):
        structure_wrangler.add_data_indices("foo", 77)


def test_properties(structure_wrangler):
    assert structure_wrangler.feature_matrix.shape == (
        structure_wrangler.num_structures,
        structure_wrangler.cluster_subspace.num_corr_functions,
    )
    assert (
        len(structure_wrangler.occupancy_strings) == structure_wrangler.num_structures
    )
    num_prim_sites = len(structure_wrangler.cluster_subspace.structure)
    for struct, occu, size in zip(
        structure_wrangler.structures,
        structure_wrangler.occupancy_strings,
        structure_wrangler.sizes,
    ):
        assert len(struct) <= len(occu)  # < with vacancies
        assert size * num_prim_sites == len(occu)


def test_remove_structures(structure_wrangler):
    total = len(structure_wrangler.structures)
    s = structure_wrangler.structures[np.random.randint(0, total)]
    structure_wrangler.remove_structure(s)
    assert len(structure_wrangler.structures) == total - 1
    with pytest.raises(ValueError):
        structure_wrangler.remove_structure(s)


def test_update_features(structure_wrangler):
    shape = structure_wrangler.feature_matrix.shape
    structure_wrangler.cluster_subspace.add_external_term(EwaldTerm())
    structure_wrangler.update_features()
    assert shape[1] + 1 == structure_wrangler.feature_matrix.shape[1]
    structure_wrangler.cluster_subspace._external_terms = []
    structure_wrangler.update_features()


def test_get_gram_matrix(structure_wrangler):
    G = structure_wrangler.get_gram_matrix()
    assert G.shape == 2 * (structure_wrangler.num_features,)
    npt.assert_array_equal(G, G.T)
    npt.assert_array_almost_equal(np.ones(G.shape[0]), G.diagonal())

    rows = np.random.choice(
        range(structure_wrangler.num_structures), structure_wrangler.num_structures - 2
    )
    cols = np.random.choice(
        range(structure_wrangler.num_features), structure_wrangler.num_features - 4
    )
    G = structure_wrangler.get_gram_matrix(rows=rows, cols=cols, normalize=False)
    assert G.shape == 2 * (structure_wrangler.num_features - 4,)
    npt.assert_array_equal(G, G.T)
    assert not np.allclose(np.ones(G.shape[0]), G.diagonal())


def test_get_similarity_matrix(structure_wrangler):
    S = structure_wrangler.get_similarity_matrix()
    assert S.shape == 2 * (structure_wrangler.num_features,)
    npt.assert_array_equal(S, S.T)
    npt.assert_array_equal(S.diagonal(), np.ones(S.shape[0]))

    rows = np.random.choice(
        range(structure_wrangler.num_structures), structure_wrangler.num_structures - 2
    )
    cols = np.random.choice(
        range(structure_wrangler.num_features), structure_wrangler.num_features - 4
    )

    S = structure_wrangler.get_similarity_matrix(rows=rows, cols=cols)
    assert S.shape == 2 * (structure_wrangler.num_features - 4,)
    npt.assert_array_equal(S, S.T)
    npt.assert_array_equal(np.ones(S.shape[0]), S.diagonal())


def test_matrix_properties(structure_wrangler):
    assert structure_wrangler.get_condition_number() >= 1
    rows = np.random.choice(range(structure_wrangler.num_structures), 16)
    cols = np.random.choice(range(structure_wrangler.num_features), 10)
    assert structure_wrangler.get_condition_number() >= 1
    assert structure_wrangler.get_condition_number(rows, cols) >= 1
    assert structure_wrangler.get_feature_matrix_rank(
        rows, cols
    ) >= structure_wrangler.get_feature_matrix_rank(cols=cols[:-3])


def test_get_orbit_rank(structure_wrangler):
    for _ in range(10):
        oid = random.choice(
            range(1, len(structure_wrangler.cluster_subspace.orbits) + 1)
        )
        orb_size = structure_wrangler.cluster_subspace.orbits[oid - 1]
        assert structure_wrangler.get_feature_matrix_orbit_rank(oid) <= len(orb_size)


def test_get_duplicate_corr_inds(structure_wrangler):
    ind = np.random.randint(structure_wrangler.num_structures)
    dup_item = deepcopy(structure_wrangler.data_items[ind])
    with pytest.warns(UserWarning):
        structure_wrangler.add_data(dup_item["structure"], dup_item["properties"])

    assert [
        ind,
        structure_wrangler.num_structures - 1,
    ] in structure_wrangler.get_duplicate_corr_indices()


def test_get_matching_corr_duplicate_inds(structure_wrangler):
    ind = np.random.randint(structure_wrangler.num_structures)
    dup_item = deepcopy(structure_wrangler.data_items[ind])
    ind2 = np.random.randint(structure_wrangler.num_structures)
    dup_item2 = deepcopy(structure_wrangler.data_items[ind2])
    # change the structure for this one:
    dup_item2["structure"] = np.random.choice(
        [s for s in structure_wrangler.structures if s != dup_item2["structure"]]
    )
    structure_wrangler.append_data_items([dup_item, dup_item2])
    expected_matches = [[ind, structure_wrangler.num_structures - 2]]
    assert all(
        i in matches
        for matches, expected in zip(
            structure_wrangler.get_matching_corr_duplicate_indices(), expected_matches
        )
        for i in expected
    )


def test_get_constant_features(structure_wrangler):
    ind = np.random.randint(1, structure_wrangler.num_features)
    for item in structure_wrangler.data_items:
        item["features"][ind] = 3.0  # make constant
    assert ind in structure_wrangler.get_constant_features()
    assert 0 not in structure_wrangler.get_constant_features()


def test_msonable(structure_wrangler):
    structure_wrangler.metadata["key"] = 4
    d = structure_wrangler.as_dict()
    sw = StructureWrangler.from_dict(d)
    npt.assert_array_equal(
        sw.get_property_vector("energy"),
        structure_wrangler.get_property_vector("energy"),
    )
    npt.assert_array_equal(
        sw.get_weights("random"), structure_wrangler.get_weights("random")
    )
    assert all(
        [s1 == s2 for s1, s2 in zip(sw.structures, structure_wrangler.structures)]
    )
    assert all(
        [
            s1 == s2
            for s1, s2 in zip(
                sw.refined_structures, structure_wrangler.refined_structures
            )
        ]
    )
    npt.assert_array_equal(sw.feature_matrix, structure_wrangler.feature_matrix)
    assert sw.metadata == structure_wrangler.metadata
    assert all(
        i1["mapping"] == i1["mapping"]
        for i1, i2 in zip(structure_wrangler._items, sw._items)
    )
    assert all(
        np.array_equal(m1, m2)
        for m1, m2 in zip(structure_wrangler.supercell_matrices, sw.supercell_matrices)
    )
    assert_msonable(structure_wrangler)
