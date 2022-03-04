from copy import deepcopy

import numpy as np
import numpy.testing as npt

from smol.cofe.extern import EwaldTerm
from smol.cofe.wrangling import max_ewald_energy_indices, unique_corr_vector_indices

# TODO write these unit-tests


def test_unique_corr_indices(structure_wrangler):
    pass


def test_ewald_energy_indices(structure_wrangler):
    pass


def test_weights_above_composition():
    """
    This is from old tests using StructureWrangler with lno data

    weights = weights_energy_above_composition(self.sw.structures,
                                               self.sw.get_property_vector('energy', False),
                                               temperature=1000)
    self.sw.add_weights('comp', weights)
    expected = np.array([0.85637358, 0.98816678, 1., 0.59209449, 1.,
                0.92882071, 0.87907454, 0.94729315, 0.40490513, 0.82484222,
                0.81578984, 1., 0.89615121, 0.92893004, 0.81650693,
                0.6080223 , 0.94848913, 0.92135297, 0.92326977, 0.83995635,
                1., 0.94663979, 1., 0.9414506, 1.])
    self.assertTrue(np.allclose(expected, self.sw.get_weights('comp')))
    sc_matrices = self.sw.supercell_matrices
    num_structs = self.sw.num_structures
    structures = self.sw.structures
    energies = self.sw.get_property_vector('energy')
    structures = self.sw.structures
    self.sw.remove_all_data()
    self.assertTrue(self.sw.num_structures == 0)
    for struct, energy, weight, matrix in zip(structures, energies,
                                              weights, sc_matrices):
        self.sw.add_data(struct, {'energy': energy},
                         weights={'comp': weight}, supercell_matrix=matrix)
    self.assertEqual(num_structs, self.sw.num_structures)
    self.assertTrue(np.allclose(expected, self.sw.get_weights('comp')))
    """


def test_weights_above_hull():
    """
    This is from old tests using StructureWrangler with lno data

    weights = weights_energy_above_hull(self.sw.structures,
                                            self.sw.get_property_vector('energy', False),
                                            self.cs.structure,
                                            temperature=1000)
    self.sw.add_weights('hull', weights)
    expected = np.array([0.85637358, 0.98816678, 1., 0.56916328, 0.96127103,
       0.89284844, 0.84502889, 0.91060546, 0.40490513, 0.82484222,
       0.81578984, 1., 0.89615121, 0.92893004, 0.81650693,
       0.58819251, 0.91755548, 0.89130433, 0.89315862, 0.81256235,
       0.9673864 , 0.91576647, 1., 0.9414506 , 1])
    self.assertTrue(np.allclose(expected, self.sw.get_weights('hull')))
    sc_matrices = self.sw.supercell_matrices
    num_structs = self.sw.num_structures
    structures = self.sw.structures
    energies = self.sw.get_property_vector('energy')
    structures = self.sw.structures
    self.sw.remove_all_data()
    self.assertTrue(self.sw.num_structures == 0)
    for struct, energy, weight, matrix in zip(structures, energies,
                                              weights, sc_matrices):
        self.sw.add_data(struct, {'energy': energy},
                         weights={'hull': weight}, supercell_matrix=matrix)
    self.assertEqual(num_structs, self.sw.num_structures)
    self.assertTrue(np.allclose(expected, self.sw.get_weights('hull')))
    self.assertRaises(AttributeError, self.sw.add_weights, 'test',
                      weights[:-2])
    """


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
    structure_wrangler.cluster_subspace._external_terms = []


def test_filter_duplicate_corr_vectors(structure_wrangler):
    # add some repeat structures with infinite energy
    dup_items = []
    for i in range(5):
        ind = np.random.randint(structure_wrangler.num_structures)
        dup_item = deepcopy(structure_wrangler.data_items[ind])
        dup_item["properties"]["energy"] = np.inf
        dup_items.append(dup_item)

    final = structure_wrangler.num_structures
    structure_wrangler._items += dup_items
    n_structs = structure_wrangler.num_structures

    assert structure_wrangler.num_structures == final + len(dup_items)
    assert np.inf in structure_wrangler.get_property_vector("energy")
    indices = unique_corr_vector_indices(structure_wrangler, property_key="energy")
    assert len(indices) < n_structs
    assert len(indices) == n_structs - len(dup_items)
    assert np.inf not in structure_wrangler.get_property_vector("energy")[indices]
    indices, compliment = unique_corr_vector_indices(
        structure_wrangler, property_key="energy", return_compliment=True
    )
    assert len(indices) + len(compliment) == n_structs
    indices = unique_corr_vector_indices(
        structure_wrangler, property_key="energy", filter_by="max"
    )
    assert np.inf in structure_wrangler.get_property_vector("energy")[indices]
