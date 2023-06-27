from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.entries.computed_entries import ComputedStructureEntry

from smol.cofe import StructureWrangler
from smol.cofe.extern import EwaldTerm
from tests.utils import (
    assert_msonable,
    gen_fake_training_data,
    gen_random_ordered_structure,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*supercell_structure. Throwing out."),
    pytest.mark.filterwarnings("ignore:.*have duplicated correlation vectors"),
]


def test_add_data(structure_wrangler, rng):
    for entry in gen_fake_training_data(
        structure_wrangler.cluster_subspace.structure, rng=rng
    ):
        structure_wrangler.add_entry(entry, weights={"random": 2.0})
    struct = gen_random_ordered_structure(
        structure_wrangler.cluster_subspace.structure, rng=rng
    )
    energy = -len(struct) * rng.random()
    structure_wrangler.add_entry(
        ComputedStructureEntry(
            struct,
            energy,
            data={"weights": {"random": 3.0}, "properties": {"random": rng.random()}},
        )
    )

    assert all(w == 2.0 for w in structure_wrangler.get_weights("random")[:-1])
    assert (
        len(structure_wrangler.get_weights("random"))
        == structure_wrangler.num_structures
    )
    assert structure_wrangler.get_weights("random")[-1] == 3.0
    assert structure_wrangler.available_properties == ["random"]
    assert structure_wrangler.available_weights == ["random"]

    entries = structure_wrangler._entries
    structure_wrangler.remove_all_data()
    assert len(structure_wrangler.structures) == 0

    # Test passing supercell matrices
    for entry in entries:
        structure_wrangler.add_entry(
            entry,
            entry.data["properties"],
            weights=entry.data["weights"],
            supercell_matrix=entry.data["supercell_matrix"],
        )
    assert len(structure_wrangler.structures) == len(entries)
    structure_wrangler.remove_all_data()
    assert len(structure_wrangler.structures) == 0

    # Test passing site mappings
    for entry in entries:
        structure_wrangler.add_entry(
            entry,
            entry.data["properties"],
            weights=entry.data["weights"],
            site_mapping=entry.data["site_mapping"],
        )

    assert len(structure_wrangler._entries) == len(entries)
    structure_wrangler.remove_all_data()
    # test passing both
    for entry in entries:
        structure_wrangler.add_entry(
            entry,
            entry.data["properties"],
            weights=entry.data["weights"],
            supercell_matrix=entry.data["supercell_matrix"],
            site_mapping=entry.data["site_mapping"],
        )
    assert len(structure_wrangler.structures) == len(entries)

    # Add more properties to test removal
    structure_wrangler.add_properties(
        "normalized", structure_wrangler.get_property_vector("energy", normalize=True)
    )
    structure_wrangler.add_properties(
        "normalized1", structure_wrangler.get_property_vector("energy", normalize=True)
    )
    assert all(
        prop in ["normalized", "normalized1", "random"]
        for prop in structure_wrangler.available_properties
    )
    structure_wrangler.remove_properties("normalized", "normalized1")
    assert structure_wrangler.available_properties == ["random"]

    # heavily distorted structure
    struct = struct.copy()
    struct.apply_strain(0.2)
    entry = ComputedStructureEntry(struct, energy)
    with pytest.raises(Exception):
        structure_wrangler.add_entry(
            entry,
            properties={"random": rng.random()},
            weights={"random": 3.0},
            raise_failed=True,
        )

    with pytest.warns(UserWarning):
        structure_wrangler.add_entry(
            entry,
            properties={"random": rng.random()},
            weights={"random": 3.0},
            raise_failed=False,
        )

    with pytest.raises(AttributeError):
        structure_wrangler.add_properties("test", structure_wrangler.sizes[:-2])

    structure_wrangler.add_properties(
        "normalized_energy",
        structure_wrangler.get_property_vector("energy", normalize=True),
    )

    entry = deepcopy(
        structure_wrangler.entries[rng.choice(range(structure_wrangler.num_structures))]
    )

    with pytest.raises(ValueError):
        entry.data["properties"].update({"foo": 10})
        structure_wrangler.add_entry(
            ComputedStructureEntry(entry.structure, entry.energy),
            entry.data["properties"],
            weights=entry.data["weights"],
            supercell_matrix=entry.data["supercell_matrix"],
            site_mapping=entry.data["site_mapping"],
        )
    with pytest.raises(ValueError):
        entry.data["properties"] = {}
        structure_wrangler.add_entry(
            ComputedStructureEntry(entry.structure, entry.energy),
            entry.data["properties"],
            weights=entry.data["weights"],
            supercell_matrix=entry.data["supercell_matrix"],
            site_mapping=entry.data["site_mapping"],
        )
    with pytest.raises(ValueError):
        entry.data["weights"].update({"foo": 10})
        structure_wrangler.add_entry(
            ComputedStructureEntry(entry.structure, entry.energy),
            entry.data["properties"],
            weights=entry.data["weights"],
            supercell_matrix=entry.data["supercell_matrix"],
            site_mapping=entry.data["site_mapping"],
        )
    with pytest.raises(ValueError):
        _ = entry.data["weights"].pop("random")
        structure_wrangler.add_entry(
            ComputedStructureEntry(entry.structure, entry.energy),
            entry.data["properties"],
            weights=entry.data["weights"],
            supercell_matrix=entry.data["supercell_matrix"],
            site_mapping=entry.data["site_mapping"],
        )

    with pytest.warns(RuntimeWarning):
        structure_wrangler.remove_properties("blab")


def test_add_weights(structure_wrangler, rng):
    sc_matrices = structure_wrangler.supercell_matrices
    num_structs = structure_wrangler.num_structures
    structures = structure_wrangler.structures
    energies = structure_wrangler.get_property_vector("energy")
    weights = rng.random(num_structs)
    structure_wrangler.add_weights("comp", weights)
    structure_wrangler.remove_all_data()
    assert structure_wrangler.num_structures == 0
    for struct, energy, weight, matrix in zip(
        structures, energies, weights, sc_matrices
    ):
        structure_wrangler.add_entry(
            ComputedStructureEntry(struct, energy),
            weights={"comp": weight},
            supercell_matrix=matrix,
        )
    assert num_structs == structure_wrangler.num_structures
    npt.assert_array_almost_equal(
        weights, structure_wrangler.get_weights("comp"), decimal=9
    )
    with pytest.raises(AttributeError):
        structure_wrangler.add_weights("test", weights[:-2])


def test_append_entries(structure_wrangler, rng):
    entries = structure_wrangler._entries
    structure_wrangler.remove_all_data()
    assert len(structure_wrangler.structures) == 0

    processed_entries = []
    for entry in entries:
        processed_entries.append(
            structure_wrangler.process_entry(
                ComputedStructureEntry(entry.structure, entry.energy),
                properties={"random": 1.0},
                weights=entry.data["weights"],
                supercell_matrix=entry.data["supercell_matrix"],
                site_mapping=entry.data["site_mapping"],
            )
        )

    structure_wrangler.append_entries(processed_entries[:-2])
    structure_wrangler.append_entries(processed_entries[-2:])
    assert len(structure_wrangler.entries) == len(entries)
    entry = deepcopy(
        structure_wrangler.entries[rng.choice(range(structure_wrangler.num_structures))]
    )

    with pytest.raises(ValueError):
        entry.data["properties"].update({"foo": 10})
        structure_wrangler.append_entries([entry])
    with pytest.raises(ValueError):
        _ = entry.data["properties"].pop("random")
        structure_wrangler.append_entries([entry])
    with pytest.raises(ValueError):
        entry.data["weights"].update({"foo": 10})
        structure_wrangler.append_entries([entry])
    with pytest.raises(ValueError):
        _ = entry.data["weights"].pop("random")
        structure_wrangler.append_entries([entry])


def test_data_indices(structure_wrangler, rng):
    test = rng.choice(range(structure_wrangler.num_structures), 5)
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


def test_remove_entry(structure_wrangler, rng):
    total = len(structure_wrangler.structures)
    entry = structure_wrangler.entries[rng.integers(0, total)]
    structure_wrangler.remove_entry(entry)
    assert len(structure_wrangler.structures) == total - 1
    with pytest.warns(RuntimeWarning):
        structure_wrangler.remove_entry(entry)


def test_update_features(structure_wrangler):
    shape = structure_wrangler.feature_matrix.shape
    structure_wrangler.cluster_subspace.add_external_term(EwaldTerm())
    structure_wrangler.update_features()
    assert shape[1] + 1 == structure_wrangler.feature_matrix.shape[1]
    structure_wrangler.update_features()


def test_get_gram_matrix(structure_wrangler, rng):
    G = structure_wrangler.get_gram_matrix()
    assert G.shape == 2 * (structure_wrangler.num_features,)
    npt.assert_array_equal(G, G.T)
    npt.assert_array_almost_equal(np.ones(G.shape[0]), G.diagonal())

    rows = rng.choice(
        range(structure_wrangler.num_structures), structure_wrangler.num_structures - 2
    )
    cols = rng.choice(
        range(structure_wrangler.num_features), structure_wrangler.num_features - 4
    )
    G = structure_wrangler.get_gram_matrix(rows=rows, cols=cols, normalize=False)
    assert G.shape == 2 * (structure_wrangler.num_features - 4,)
    npt.assert_array_equal(G, G.T)
    assert not np.allclose(np.ones(G.shape[0]), G.diagonal())


def test_get_similarity_matrix(structure_wrangler, rng):
    S = structure_wrangler.get_similarity_matrix()
    assert S.shape == 2 * (structure_wrangler.num_features,)
    npt.assert_array_equal(S, S.T)
    npt.assert_array_equal(S.diagonal(), np.ones(S.shape[0]))
    rows = rng.choice(
        range(structure_wrangler.num_structures), structure_wrangler.num_structures - 2
    )
    cols = rng.choice(
        range(structure_wrangler.num_features), structure_wrangler.num_features - 4
    )

    S = structure_wrangler.get_similarity_matrix(rows=rows, cols=cols)
    assert S.shape == 2 * (structure_wrangler.num_features - 4,)
    npt.assert_array_equal(S, S.T)
    npt.assert_array_equal(np.ones(S.shape[0]), S.diagonal())


def test_matrix_properties(structure_wrangler, rng):
    assert structure_wrangler.get_condition_number() >= 1
    rows = rng.choice(
        range(structure_wrangler.num_structures), structure_wrangler.num_structures // 3
    )
    cols = rng.choice(
        range(structure_wrangler.num_features), structure_wrangler.num_features // 3
    )
    assert structure_wrangler.get_condition_number() >= 1
    assert structure_wrangler.get_condition_number(rows, cols) >= 1
    assert structure_wrangler.get_feature_matrix_rank(
        rows, cols
    ) >= structure_wrangler.get_feature_matrix_rank(rows=rows, cols=cols[:-3])


def test_get_orbit_rank(structure_wrangler, rng):
    for _ in range(10):
        oid = rng.choice(range(1, len(structure_wrangler.cluster_subspace.orbits) + 1))
        orb_size = structure_wrangler.cluster_subspace.orbits[oid - 1]
        assert structure_wrangler.get_feature_matrix_orbit_rank(oid) <= len(orb_size)


def test_get_duplicate_corr_inds(structure_wrangler, rng):
    ind = rng.integers(structure_wrangler.num_structures)
    dup_item = deepcopy(structure_wrangler.entries[ind])
    with pytest.warns(UserWarning):
        structure_wrangler.add_entry(dup_item, dup_item.data["properties"])

    assert [
        ind,
        structure_wrangler.num_structures - 1,
    ] in structure_wrangler.get_duplicate_corr_indices()


def test_get_matching_corr_duplicate_inds(structure_wrangler, rng):
    ind = rng.integers(structure_wrangler.num_structures)
    dup_entry = deepcopy(structure_wrangler.entries[ind])
    ind2 = rng.integers(structure_wrangler.num_structures)
    # change the structure for this one:

    structs = [s for s in structure_wrangler.structures if s != dup_entry.structure]
    struct = structs[rng.choice(len(structs))]
    dup_entry2 = structure_wrangler.process_entry(
        ComputedStructureEntry(struct, structure_wrangler.entries[ind2].energy),
        properties=structure_wrangler.entries[ind2].data["properties"],
        weights=structure_wrangler.entries[ind2].data["weights"],
    )
    structure_wrangler.append_entries([dup_entry, dup_entry2])
    expected_matches = [ind, structure_wrangler.num_structures - 2]
    assert any(
        expected_matches == sorted(matches)
        for matches in structure_wrangler.get_matching_corr_duplicate_indices()
    )


def test_get_constant_features(structure_wrangler, rng):
    ind = rng.integers(1, structure_wrangler.num_features)
    for entry in structure_wrangler.entries:
        entry.data["correlations"][ind] = 3.0  # make constant
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
        i1.data["site_mapping"] == i1.data["site_mapping"]
        for i1, i2 in zip(structure_wrangler._entries, sw._entries)
    )
    assert all(
        np.array_equal(m1, m2)
        for m1, m2 in zip(structure_wrangler.supercell_matrices, sw.supercell_matrices)
    )
    assert_msonable(structure_wrangler)
