import pytest
from copy import deepcopy
import numpy as np
from smol.cofe import StructureWrangler, ClusterSubspace
from smol.cofe.wrangling import filter_by_ewald_energy, unique_corr_vector_indices
from tests.data import lno_prim, lno_data


# TODO create synthetic dataset and add fixtures in conftest.py
@pytest.fixture(scope="module")
def structure_wrangler():
    wrangler = StructureWrangler(ClusterSubspace.from_cutoffs(
                                                        lno_prim,
                                                        cutoffs={2: 5, 3: 4.1},
                                                        supercell_size='O2-'))
    for struct, energy in lno_data:
        wrangler.add_data(struct, {'energy': energy})
    return wrangler


# TODO improve test by precalculating expected energies and filtered structs
def test_filter_by_ewald(structure_wrangler):
    initial = structure_wrangler.num_structures
    filter_by_ewald_energy(structure_wrangler, max_relative_energy=1.0)
    final = structure_wrangler.num_structures
    assert initial != final
    assert (structure_wrangler.metadata['applied_filters'][0]['filter_by_ewald_energy']['initial_num_items']
            == initial)
    assert (structure_wrangler.metadata['applied_filters'][0]['filter_by_ewald_energy']['final_num_items']
            == final)


def test_filter_duplicate_corr_vectors(structure_wrangler):
    dup_items = []
    for i in range(4):
        ind = np.random.randint(structure_wrangler.num_structures)
        dup_item = deepcopy(structure_wrangler.data_items[ind])
        dup_item['properties']['energy'] = np.inf
        dup_items.append(dup_item)

    final = structure_wrangler.num_structures
    structure_wrangler._items += dup_items
    initial = structure_wrangler.num_structures

    assert structure_wrangler.num_structures == final + len(dup_items)
    assert np.inf in structure_wrangler.get_property_vector('energy')
    unique_corr_vector_indices(structure_wrangler, property_key='energy')
    assert structure_wrangler.num_structures == final
    assert np.inf not in structure_wrangler.get_property_vector('energy')
    assert (structure_wrangler.metadata['applied_filters'][-1]['filter_duplicate_corr_vectors']['initial_num_items']
            == initial)
    assert (structure_wrangler.metadata['applied_filters'][-1]['filter_duplicate_corr_vectors']['final_num_items']
            == final)
