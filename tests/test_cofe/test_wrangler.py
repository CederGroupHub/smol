import unittest
import warnings
import numpy as np
from smol.cofe import (StructureWrangler, ClusterSubspace,
                       weights_energy_above_hull,
                       weights_energy_above_composition)
from smol.cofe.configspace import EwaldTerm
from tests.data import lno_prim, lno_data


class TestStructureWrangler(unittest.TestCase):
    def setUp(self) -> None:
        self.cs = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1},
                                             ltol=0.15, stol=0.2,
                                             angle_tol=5, supercell_size='O2-')
        self.sw = StructureWrangler(self.cs)
        for struct, energy in lno_data[:-1]:
            self.sw.add_data(struct, {'energy': energy},
                             weights={'random': 2.0})
        struct, energy = lno_data[-1]
        self.sw.add_data(struct, {'energy': energy}, weights={'random': 3.0})

    def test_add_data(self):
        self.assertTrue(all(w == 2.0 for w in self.sw.get_weights('random')[:-1]))
        self.assertTrue(len(self.sw.get_weights('random')) == self.sw.num_structures)
        self.assertTrue(self.sw.get_weights('random')[-1] == 3.0)
        self.assertRaises(AttributeError, self.sw.add_properties, 'test',
                          self.sw.sizes[:-2])
        self.sw.add_properties('normalized_energy',
                               self.sw.get_property_vector('energy', normalize=True))

    def test_update_features(self):
        shape = self.sw.feature_matrix.shape
        self.cs.add_external_term(EwaldTerm())
        self.sw.update_features()
        self.assertEqual(shape[1] + 1, self.sw.feature_matrix.shape[1])

    def test_weights_e_above_comp(self):
        weights = weights_energy_above_composition(self.sw.structures,
                                                   self.sw.get_property_vector('energy'),
                                                   temperature=1000)
        self.sw.add_weights('comp', weights)
        expected = np.array([0.85637358, 0.98816678, 1., 0.59209449, 1.,
                    0.92882071, 0.87907454, 0.94729315, 0.40490513, 0.82484222,
                    0.81578984, 1., 0.89615121, 0.92893004, 0.81650693,
                    0.6080223 , 0.94848913, 0.92135297, 0.92326977, 0.83995635,
                    1., 0.94663979, 1., 0.9414506 , 1.])
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
                                            self.sw.get_property_vector('energy'),
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

    # TODO write a better test. One that actually checks the structures
    #  expected to be removed are removed
    def test_filter_by_ewald(self):
        len_total = self.sw.num_structures
        self.sw.filter_by_ewald(1)
        len_filtered = self.sw.num_structures
        self.assertNotEqual(len_total, len_filtered)
        self.assertEqual(self.sw.metadata['applied_filters'][0]['Ewald']['nstructs_removed'],
                         len_total - len_filtered)
        self.assertEqual(self.sw.metadata['applied_filters'][0]['Ewald']['nstructs_total'],
                         len_total)

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
