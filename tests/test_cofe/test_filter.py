import pytest
from copy import deepcopy
import numpy as np
from smol.cofe import StructureWrangler, ClusterSubspace
from smol.cofe.wrangling import unique_corr_vector_indices, max_ewald_energy_indices
from tests.data import lno_prim, lno_data


# TODO create synthetic dataset and add fixtures in conftest.py
@pytest.fixture(scope="module")
def structure_wrangler():
    wrangler = StructureWrangler(
        ClusterSubspace.from_cutoffs(lno_prim,
                                     cutoffs={2: 5, 3: 4.1},
                                     supercell_size='O2-'))
    for struct, energy in lno_data:
        wrangler.add_data(struct, {'energy': energy})
    return wrangler


# TODO improve test by precalculating expected energies and filtered structs
def test_filter_by_ewald(structure_wrangler):
    n_structs = structure_wrangler.num_structures
    indices = max_ewald_energy_indices(structure_wrangler,
                                       max_relative_energy=1.0)
    assert len(indices) < n_structs
    indices, compliment = max_ewald_energy_indices(structure_wrangler,
                                                   max_relative_energy=1.0,
                                                   return_compliment=True)
    assert len(indices) + len(compliment) == n_structs


def test_filter_duplicate_corr_vectors(structure_wrangler):
    # add some repeat structures with infinite energy
    dup_items = []
    for i in range(5):
        ind = np.random.randint(structure_wrangler.num_structures)
        dup_item = deepcopy(structure_wrangler.data_items[ind])
        dup_item['properties']['energy'] = np.inf
        dup_items.append(dup_item)

    final = structure_wrangler.num_structures
    structure_wrangler._items += dup_items
    n_structs = structure_wrangler.num_structures

    assert structure_wrangler.num_structures == final + len(dup_items)
    assert np.inf in structure_wrangler.get_property_vector('energy')
    indices = unique_corr_vector_indices(structure_wrangler,
                                         property_key='energy')
    assert len(indices) < n_structs
    assert len(indices) == n_structs - len(dup_items)
    assert np.inf not in structure_wrangler.get_property_vector('energy')[indices]
    indices, compliment = unique_corr_vector_indices(structure_wrangler,
                                                     property_key='energy',
                                                     return_compliment=True)
    assert len(indices) + len(compliment) == n_structs
    indices = unique_corr_vector_indices(structure_wrangler,
                                         property_key='energy',
                                         filter_by='max')
    assert np.inf in structure_wrangler.get_property_vector('energy')[indices]
