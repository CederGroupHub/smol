import unittest
import json
from copy import deepcopy

import numpy as np
import numpy.testing as npt
from smol.cofe import StructureWrangler, ClusterSubspace
from smol.cofe.wrangling import weights_energy_above_hull, \
    weights_energy_above_composition
from smol.cofe.extern import EwaldTerm
from tests.data import lno_prim, lno_data


class TestStructureWrangler(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cs = ClusterSubspace.from_cutoffs(lno_prim, cutoffs={2: 5, 3: 4.1},
                                              ltol=0.15, stol=0.2,
                                              angle_tol=5, supercell_size='O2-')
        cls.sw = StructureWrangler(cls.cs)

    def setUp(self):
        self.sw.remove_all_data()
        for struct, energy in lno_data[:-1]:
            self.sw.add_data(struct, {'energy': energy},
                            weights={'random': 2.0})
        struct, energy = lno_data[-1]
        self.sw.add_data(struct, {'energy': energy}, weights={'random': 3.0})

    def test_data_indices(self):
        test = np.random.choice(range(self.sw.num_structures), 5)
        train = np.setdiff1d(range(self.sw.num_structures), test)
        self.sw.add_data_indices('test', test)
        self.sw.add_data_indices('train', train)
        self.assertTrue(all(key in self.sw.available_indices
                            for key in ['test', 'train']))
        self.assertRaises(ValueError, self.sw.add_data_indices,
                          'bla', [self.sw.num_structures, ])
        self.assertRaises(TypeError, self.sw.add_data_indices, 'foo', 77)

    def test_properties(self):
        self.assertEqual(self.sw.feature_matrix.shape,
                         (self.sw.num_structures, self.cs.num_corr_functions))
        self.assertEqual(len(self.sw.occupancy_strings),
                         self.sw.num_structures)
        num_prim_sits = len(self.cs.structure)
        for struct, occu, size in zip(self.sw.structures,
                                      self.sw.occupancy_strings,
                                      self.sw.sizes):
            self.assertTrue(len(struct) <= len(occu))  # < with vacancies
            self.assertTrue(size*num_prim_sits, len(occu))

    def test_get_gram_matrix(self):
        G = self.sw.get_gram_matrix()
        self.assertEqual(G.shape, 2*(self.sw.num_features, ))
        npt.assert_array_equal(G, G.T)
        npt.assert_array_almost_equal(np.ones(G.shape[0]), G.diagonal())

        rows = np.random.choice(range(self.sw.num_structures),
                                self.sw.num_structures - 2)
        cols = np.random.choice(range(self.sw.num_features),
                                self.sw.num_features - 4)
        G = self.sw.get_gram_matrix(rows=rows, cols=cols, normalize=False)
        self.assertEqual(G.shape, 2 * (self.sw.num_features - 4,))
        npt.assert_array_equal(G, G.T)
        self.assertFalse(np.allclose(np.ones(G.shape[0]), G.diagonal()))

    def test_matrix_properties(self):
        self.assertGreaterEqual(self.sw.get_condition_number(), 1)
        rows = np.random.choice(range(self.sw.num_structures), 16)
        cols = np.random.choice(range(self.sw.num_features), 10)
        self.assertGreaterEqual(self.sw.get_condition_number(), 1)
        self.assertGreaterEqual(self.sw.get_condition_number(rows, cols), 1)
        print(self.sw.feature_matrix.shape)
        self.assertGreaterEqual(self.sw.get_feature_matrix_rank(rows, cols),
                                self.sw.get_feature_matrix_rank(cols=cols[:-3]))

    def test_add_data(self):
        # Check that a structure that does not match raises error.
        self.assertRaises(Exception, self.sw.add_data, lno_data[0][0],
                          {'energy': 0}, raise_failed=True)

        # Check data in setup was added correctly
        self.assertTrue(all(w == 2.0 for w in self.sw.get_weights('random')[:-1]))
        self.assertTrue(len(self.sw.get_weights('random')) == self.sw.num_structures)
        self.assertTrue(self.sw.get_weights('random')[-1] == 3.0)
        self.assertEqual(self.sw.available_properties, ['energy'])
        self.assertEqual(self.sw.available_weights, ['random'])

        # Check adding new properties
        self.assertRaises(AttributeError, self.sw.add_properties, 'test',
                          self.sw.sizes[:-2])
        self.sw.add_properties('normalized_energy',
                               self.sw.get_property_vector('energy', normalize=True))
        items = self.sw._items
        self.sw.remove_all_data()
        self.assertEqual(len(self.sw.structures), 0)

        # Test passing supercell matrices
        for item in items:
            self.sw.add_data(item['structure'], item['properties'],
                             supercell_matrix=item['scmatrix'])
        self.assertEqual(len(self.sw.structures), len(items))
        self.sw.remove_all_data()
        self.assertEqual(len(self.sw.structures), 0)

        # Test passing site mappings
        for item in items:
            self.sw.add_data(item['structure'], item['properties'],
                             site_mapping=item['mapping'])
        self.assertEqual(len(self.sw.structures), len(items))
        self.sw.remove_all_data()
        # test passing both
        for item in items:
            self.sw.add_data(item['structure'], item['properties'],
                             supercell_matrix=item['scmatrix'],
                             site_mapping=item['mapping'])
        self.assertEqual(len(self.sw.structures), len(items))
        
        # Add more properties to test removal
        self.sw.add_properties('normalized',
                               self.sw.get_property_vector('energy', normalize=True))
        self.sw.add_properties('normalized1',
                               self.sw.get_property_vector('energy', normalize=True))
        self.assertTrue(all(prop in ['energy', 'normalized_energy',
                                     'normalized', 'normalized1'] for
                            prop in self.sw.available_properties))
        self.sw.remove_properties('normalized_energy', 'normalized',
                                  'normalized1')
        self.assertEqual(self.sw.available_properties, ['energy'])
        self.assertWarns(RuntimeWarning, self.sw.remove_properties, 'blab')

    def test_append_data_items(self):
        items = self.sw._items
        self.sw.remove_all_data()
        self.assertEqual(len(self.sw.structures), 0)
        self.assertRaises(ValueError, self.sw.append_data_items, [{'b': 1}])
        data_items = []
        for item in items:
            data_items.append(self.sw.process_structure(item['structure'],
                                                        item['properties'],
                             supercell_matrix=item['scmatrix'],
                             site_mapping=item['mapping']))

        self.sw.append_data_items(data_items)
        self.assertEqual(len(self.sw.data_items), len(items))

    def test_remove_structure(self):
        total = len(self.sw.structures)
        s = self.sw.structures[np.random.randint(0, total)]
        self.sw.remove_structure(s)
        self.assertEqual(len(self.sw.structures), total - 1)
        self.assertRaises(ValueError, self.sw.remove_structure, s)

    def test_update_features(self):
        shape = self.sw.feature_matrix.shape
        self.cs.add_external_term(EwaldTerm())
        self.sw.update_features()
        self.assertEqual(shape[1] + 1, self.sw.feature_matrix.shape[1])

    def test_weights_e_above_comp(self):
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

    def test_weights_e_above_hull(self):
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

    def test_get_duplicate_corr_inds(self):
        ind = np.random.randint(self.sw.num_structures)
        dup_item = deepcopy(self.sw.data_items[ind])
        self.assertWarns(UserWarning, self.sw.add_data, dup_item["structure"],
                         dup_item["properties"])
        self.assertEqual(self.sw.get_duplicate_corr_indices(),
                         [[ind, self.sw.num_structures - 1]])

    def test_get_constant_features(self):
        ind = np.random.randint(1, self.sw.num_features)
        for item in self.sw.data_items:
            item["features"][ind] = 3.0  # make constant
        self.assertTrue(ind in self.sw.get_constant_features())
        self.assertTrue(0 not in self.sw.get_constant_features())

    def test_msonable(self):
        self.sw.metadata['key'] = 4
        d = self.sw.as_dict()
        sw = StructureWrangler.from_dict(d)
        self.assertTrue(np.array_equal(sw.get_property_vector('energy'),
                                       self.sw.get_property_vector('energy')))
        self.assertTrue(np.array_equal(sw.get_weights('random'),
                        self.sw.get_weights('random')))
        self.assertTrue(all([s1 == s2 for s1, s2 in zip(sw.structures, self.sw.structures)]))
        self.assertTrue(all([s1 == s2 for s1, s2 in zip(sw.refined_structures, self.sw.refined_structures)]))
        self.assertTrue(np.array_equal(sw.feature_matrix, self.sw.feature_matrix))
        self.assertEqual(sw.metadata, self.sw.metadata)
        self.assertTrue(all(i1['mapping'] == i1['mapping'] for i1, i2 in
                            zip(self.sw._items, sw._items)))
        self.assertTrue(all(np.array_equal(m1, m2) for m1, m2 in
                            zip(self.sw.supercell_matrices,
                                sw.supercell_matrices)))
        j = json.dumps(d)
        json.loads(j)
