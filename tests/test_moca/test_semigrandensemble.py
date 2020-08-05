from copy import deepcopy
import numpy as np
import numpy.testing as npt
from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion
from smol.cofe.extern import EwaldTerm
from smol.moca import (CEProcessor, EwaldProcessor, CompositeProcessor,
                       MuSemiGrandEnsemble, FuSemiGrandEnsemble)
from tests.data import (synthetic_CE_binary, synthetic_CEewald_binary,
                        lno_data, lno_prim)
import tests.test_moca.base_ensemble_test as be


class TestMuSemiGrandEnsembleSynthBinary(be._EnsembleTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subspace = ClusterSubspace.from_dict(synthetic_CE_binary['cluster_subspace'])
        cls.n_allowed_species = 2
        sw = StructureWrangler(cls.subspace)
        for item in synthetic_CE_binary['data'][:200]:
            sw.add_data(item[0], {'energy': item[1]})
        coefs = np.linalg.lstsq(sw.feature_matrix,
                                sw.get_property_vector('energy', True),
                                rcond=None)[0]
        cls.expansion = ClusterExpansion(cls.subspace, coefs, sw.feature_matrix)
        sc_matrix = np.array([[3, 1, 1], [1, 3, 1], [1, 1, 3]])
        cls.processor = CEProcessor(cls.subspace, sc_matrix, coefs)
        cls.ensemble = MuSemiGrandEnsemble(cls.processor, temperature=500,
                                           chemical_potentials={'Na+': -0.05,
                                                                'Cl-': 0})
        cls.ensemble_kwargs = {'chemical_potentials': cls.ensemble.chemical_potentials}

    def test_compute_feature_vector(self):
        self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                self.ensemble.compute_feature_vector(self.enc_occu)),
                         self.processor.compute_property(self.enc_occu)
                         - self.ensemble.compute_chemical_work(self.enc_occu))
        npt.assert_array_equal(self.ensemble.compute_feature_vector(self.enc_occu)[:-1],
                               self.ensemble.processor.compute_feature_vector(self.enc_occu))
        for _ in range(10):  # test a few flips
            site = np.random.choice(range(self.processor.num_sites))
            spec = np.random.choice(list(set(range(self.n_allowed_species))-{self.enc_occu[site]}))
            flip = [(site, spec)]
            dmu = self.ensemble._mu_table[site][spec] - self.ensemble._mu_table[site][self.enc_occu[site]]
            self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                    self.ensemble.compute_feature_vector_change(self.enc_occu, flip)),
                             self.processor.compute_property_change(self.enc_occu, flip) - dmu)
            npt.assert_array_equal(self.ensemble.compute_feature_vector_change(self.enc_occu, flip),
                                   np.append(self.processor.compute_feature_vector_change(self.enc_occu, flip),
                                             dmu))

    def test_bad_chemical_potentials(self):
        with self.assertRaises(ValueError):
            self.ensemble.chemical_potentials = {'Na+': 0.5}
        with self.assertRaises(ValueError):
            self.ensemble.chemical_potentials = {'A': 0.5, 'D': 0.6}
        self.assertRaises(ValueError, MuSemiGrandEnsemble, self.processor,
                          500, chemical_potentials={'Na+': .3, 'Cl-': 0.5,
                                                    'X': .4})
        self.assertRaises(ValueError, MuSemiGrandEnsemble, self.processor,
                          500, chemical_potentials={'Na+': .3})


    def test_build_mu_table(self):
        table = self.ensemble._build_mu_table(self.ensemble.chemical_potentials)
        for space, row in zip(self.processor.allowed_species, table):
            for i, species in enumerate(space):
                self.assertEqual(self.ensemble.chemical_potentials[species],
                                 row[i])


class TestMuSemiGrandEnsembleSynthBinaryEwald(be._EnsembleTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subspace = ClusterSubspace.from_dict(synthetic_CEewald_binary['cluster_subspace'])
        cls.subspace.add_external_term(EwaldTerm())
        cls.n_allowed_species = 2
        sw = StructureWrangler(cls.subspace)
        for item in synthetic_CEewald_binary['data']:
            sw.add_data(item[0], {'energy': item[1]})
        coefs = np.linalg.lstsq(sw.feature_matrix,
                                sw.get_property_vector('energy', True),
                                rcond=None)[0]
        cls.expansion = ClusterExpansion(cls.subspace, coefs, sw.feature_matrix)
        sc_matrix = np.array([[3, 1, 1], [1, 3, 1], [1, 1, 3]])
        cls.processor = CompositeProcessor(cls.subspace, sc_matrix)
        cls.processor.add_processor(CEProcessor, coefs[:-1])
        cls.processor.add_processor(EwaldProcessor, ewald_term=EwaldTerm(),
                                    coefficient=coefs[-1])
        cls.ensemble = MuSemiGrandEnsemble(cls.processor, temperature=500,
                                           chemical_potentials={'Na+': -0.05,
                                                                'Cl-': 0})
        cls.ensemble_kwargs = {'chemical_potentials': cls.ensemble.chemical_potentials}

    def test_compute_feature_vector(self):
        self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                self.ensemble.compute_feature_vector(self.enc_occu)),
                         self.processor.compute_property(self.enc_occu)
                         - self.ensemble.compute_chemical_work(self.enc_occu))
        npt.assert_array_equal(self.ensemble.compute_feature_vector(self.enc_occu)[:-1],
                               self.ensemble.processor.compute_feature_vector(self.enc_occu))
        for _ in range(10):  # test a few flips
            site = np.random.choice(range(self.processor.num_sites))
            spec = np.random.choice(list(set(range(self.n_allowed_species))-{self.enc_occu[site]}))
            flip = [(site, spec)]
            dmu = self.ensemble._mu_table[site][spec] - self.ensemble._mu_table[site][self.enc_occu[site]]
            self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                    self.ensemble.compute_feature_vector_change(self.enc_occu, flip)),
                             self.processor.compute_property_change(self.enc_occu, flip) - dmu)
            npt.assert_array_equal(self.ensemble.compute_feature_vector_change(self.enc_occu, flip),
                                   np.append(self.processor.compute_feature_vector_change(self.enc_occu, flip),
                                             dmu))

    def test_build_mu_table(self):
        table = self.ensemble._build_mu_table(self.ensemble.chemical_potentials)
        for space, row in zip(self.processor.allowed_species, table):
            for i, species in enumerate(space):
                self.assertEqual(self.ensemble.chemical_potentials[species],
                                 row[i])


class TestMuSemiGrandEnsembleLNO(be._EnsembleTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subspace = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1},
                                                  ltol=0.15, stol=0.2,
                                                  angle_tol=5,
                                                  supercell_size='O2-')
        ewald_term = EwaldTerm()
        cls.subspace.add_external_term(ewald_term)
        cls.n_allowed_species = 2
        sw = StructureWrangler(cls.subspace)
        for struct, energy in lno_data:
            sw.add_data(struct, {'energy': energy})
        coefs = np.linalg.lstsq(sw.feature_matrix,
                                sw.get_property_vector('energy', True),
                                rcond=None)[0]
        cls.expansion = ClusterExpansion(cls.subspace, coefs,
                                         sw.feature_matrix)
        scmatrix = np.array([[3, 0, 0],
                             [0, 2, 0],
                             [0, 0, 1]])
        cls.processor = CompositeProcessor(cls.subspace, scmatrix)

        cls.processor.add_processor(CEProcessor, coefficients=coefs[:-1])
        cls.processor.add_processor(EwaldProcessor, coefficient=coefs[-1],
                                    ewald_term=ewald_term)
        cls.ensemble = MuSemiGrandEnsemble(cls.processor, temperature=500,
                                           chemical_potentials={'Li+': -0.05,
                                                                'Vacancy': 0,
                                                                'Ni3+': 0.5,
                                                                'Ni4+': 0.1})
        cls.ensemble_kwargs = {'chemical_potentials': cls.ensemble.chemical_potentials}

    def setUp(self):
        self.enc_occu = np.array([np.random.randint(len(species))
                                  for species in
                                  self.processor.allowed_species], dtype=int)
        self.init_occu = self.processor.decode_occupancy(self.enc_occu)

    def test_compute_feature_vector(self):
        self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                self.ensemble.compute_feature_vector(self.enc_occu)),
                         self.processor.compute_property(self.enc_occu)
                         - self.ensemble.compute_chemical_work(self.enc_occu))
        npt.assert_array_equal(self.ensemble.compute_feature_vector(self.enc_occu)[:-1],
                               self.ensemble.processor.compute_feature_vector(self.enc_occu))
        for _ in range(10):  # test a few flips
            site = np.random.choice(range(self.processor.num_sites))
            spec = np.random.choice(list(set(range(self.n_allowed_species)) - {self.enc_occu[site]}))
            flip = [(site, spec)]
            dmu = self.ensemble._mu_table[site][spec] - self.ensemble._mu_table[site][self.enc_occu[site]]
            self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                    self.ensemble.compute_feature_vector_change(self.enc_occu, flip)),
                             self.processor.compute_property_change(self.enc_occu, flip) - dmu)
            npt.assert_array_equal(self.ensemble.compute_feature_vector_change(self.enc_occu, flip),
                                   np.append(self.processor.compute_feature_vector_change(self.enc_occu, flip),
                                             dmu))

    def test_build_mu_table(self):
        table = self.ensemble._build_mu_table(self.ensemble.chemical_potentials)
        for space, row in zip(self.processor.allowed_species, table):
            if len(space) == 1:  # skip inactive sites
                continue
            for i, species in enumerate(space):
                self.assertEqual(self.ensemble.chemical_potentials[species],
                                 row[i])


class TestFuSemiGrandEnsembleSynthBinary(be._EnsembleTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subspace = ClusterSubspace.from_dict(synthetic_CE_binary['cluster_subspace'])
        cls.n_allowed_species = 2
        sw = StructureWrangler(cls.subspace)
        for item in synthetic_CE_binary['data'][:200]:
            sw.add_data(item[0], {'energy': item[1]})
        coefs = np.linalg.lstsq(sw.feature_matrix,
                                sw.get_property_vector('energy', True),
                                rcond=None)[0]
        cls.expansion = ClusterExpansion(cls.subspace, coefs, sw.feature_matrix)
        sc_matrix = np.array([[3, 1, 1], [1, 3, 1], [1, 1, 3]])
        cls.processor = CEProcessor(cls.subspace, sc_matrix, coefs)
        cls.ensemble = FuSemiGrandEnsemble(cls.processor, temperature=500)
        cls.ensemble_kwargs = {}

    def test_compute_feature_vector(self):
        self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                self.ensemble.compute_feature_vector(self.enc_occu)),
                         self.processor.compute_property(self.enc_occu)
                         - self.ensemble.compute_chemical_work(self.enc_occu))
        npt.assert_array_equal(self.ensemble.compute_feature_vector(self.enc_occu)[:-1],
                               self.ensemble.processor.compute_feature_vector(self.enc_occu))
        for _ in range(10):  # test a few flips
            site = np.random.choice(range(self.processor.num_sites))
            spec = np.random.choice(list(set(range(self.n_allowed_species))-{self.enc_occu[site]}))
            flip = [(site, spec)]
            dfu = np.log(self.ensemble._fu_table[site][spec]/self.ensemble._fu_table[site][self.enc_occu[site]])
            self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                    self.ensemble.compute_feature_vector_change(self.enc_occu, flip)),
                             self.processor.compute_property_change(self.enc_occu, flip) - dfu)
            npt.assert_array_equal(self.ensemble.compute_feature_vector_change(self.enc_occu, flip),
                                   np.append(self.processor.compute_feature_vector_change(self.enc_occu, flip),
                                             dfu))

    def test_build_fu_table(self):
        table = self.ensemble._build_fu_table(self.ensemble.fugacity_fractions)
        for space, row in zip(self.processor.allowed_species, table):
            fugacity_fractions = None
            for fus in self.ensemble.fugacity_fractions:
                if space == list(fus.keys()):
                    fugacity_fractions = fus
            for i, species in enumerate(space):
                self.assertEqual(fugacity_fractions[species], row[i])

    def test_bad_fugacity_fractions(self):
        with self.assertRaises(ValueError):
            self.ensemble.fugacity_fractions = [{'Na+': 0.6, 'Cl-': 0.2}]
        with self.assertRaises(ValueError):
            self.ensemble.fugacity_fractions = [{'Na+': 0.6, 'Cl-': 0.2,
                                                 'X': 0.2}]
        with self.assertRaises(ValueError):
            self.ensemble.fugacity_fractions = [{'Na+': 0.6, 'X': 0.4}]
        with self.assertRaises(ValueError):
            self.ensemble.fugacity_fractions = [{'Na+': 0.6, 'X': 0.4}]
        fugacity_fractions = deepcopy(self.ensemble.fugacity_fractions)
        fugacity_fractions[0]['Na+'] = 2
        self.assertRaises(ValueError, FuSemiGrandEnsemble, self.processor,
                          temperature=50, fugacity_fractions=fugacity_fractions)
        fugacity_fractions = deepcopy(self.ensemble.fugacity_fractions)
        del fugacity_fractions[0]['Na+']
        self.assertRaises(ValueError, FuSemiGrandEnsemble, self.processor,
                          temperature=50, fugacity_fractions=fugacity_fractions)
        fugacity_fractions = deepcopy(self.ensemble.fugacity_fractions)
        fugacity_fractions[0]['X'] = fugacity_fractions[0].pop('Na+')
        self.assertRaises(ValueError, FuSemiGrandEnsemble, self.processor,
                          temperature=50, fugacity_fractions=fugacity_fractions)


class TestFuSemiGrandEnsembleSynthBinaryEwald(be._EnsembleTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subspace = ClusterSubspace.from_dict(synthetic_CEewald_binary['cluster_subspace'])
        cls.subspace.add_external_term(EwaldTerm())
        cls.n_allowed_species = 2
        sw = StructureWrangler(cls.subspace)
        for item in synthetic_CEewald_binary['data']:
            sw.add_data(item[0], {'energy': item[1]})
        coefs = np.linalg.lstsq(sw.feature_matrix,
                                sw.get_property_vector('energy', True),
                                rcond=None)[0]
        cls.expansion = ClusterExpansion(cls.subspace, coefs, sw.feature_matrix)
        sc_matrix = np.array([[3, 1, 1], [1, 3, 1], [1, 1, 3]])
        cls.processor = CompositeProcessor(cls.subspace, sc_matrix)
        cls.processor.add_processor(CEProcessor, coefs[:-1])
        cls.processor.add_processor(EwaldProcessor, ewald_term=EwaldTerm(),
                                    coefficient=coefs[-1])
        cls.ensemble = FuSemiGrandEnsemble(cls.processor, temperature=500)
        cls.ensemble_kwargs = {}

    def test_compute_feature_vector(self):
        self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                self.ensemble.compute_feature_vector(self.enc_occu)),
                         self.processor.compute_property(self.enc_occu)
                         - self.ensemble.compute_chemical_work(self.enc_occu))
        npt.assert_array_equal(self.ensemble.compute_feature_vector(self.enc_occu)[:-1],
                               self.ensemble.processor.compute_feature_vector(self.enc_occu))
        for _ in range(10):  # test a few flips
            site = np.random.choice(range(self.processor.num_sites))
            spec = np.random.choice(list(set(range(self.n_allowed_species)) - {self.enc_occu[site]}))
            flip = [(site, spec)]
            dfu = np.log(self.ensemble._fu_table[site][spec] / self.ensemble._fu_table[site][self.enc_occu[site]])
            self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                    self.ensemble.compute_feature_vector_change(self.enc_occu, flip)),
                             self.processor.compute_property_change(self.enc_occu, flip) - dfu)
            npt.assert_array_equal(self.ensemble.compute_feature_vector_change(self.enc_occu, flip),
                                   np.append(self.processor.compute_feature_vector_change(self.enc_occu, flip),
                                             dfu))

    def test_build_fu_table(self):
        table = self.ensemble._build_fu_table(self.ensemble.fugacity_fractions)
        for space, row in zip(self.processor.allowed_species, table):
            fugacity_fractions = None
            for fus in self.ensemble.fugacity_fractions:
                if space == list(fus.keys()):
                    fugacity_fractions = fus
            for i, species in enumerate(space):
                self.assertEqual(fugacity_fractions[species], row[i])


class TestFuSemiGrandEnsembleLNO(be._EnsembleTest):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subspace = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1},
                                                  ltol=0.15, stol=0.2,
                                                  angle_tol=5,
                                                  supercell_size='O2-')
        ewald_term = EwaldTerm()
        cls.subspace.add_external_term(ewald_term)
        cls.n_allowed_species = 2
        sw = StructureWrangler(cls.subspace)
        for struct, energy in lno_data:
            sw.add_data(struct, {'energy': energy})
        coefs = np.linalg.lstsq(sw.feature_matrix,
                                sw.get_property_vector('energy', True),
                                rcond=None)[0]
        cls.expansion = ClusterExpansion(cls.subspace, coefs,
                                         sw.feature_matrix)
        scmatrix = np.array([[3, 0, 0],
                             [0, 2, 0],
                             [0, 0, 1]])
        cls.processor = CompositeProcessor(cls.subspace, scmatrix)

        cls.processor.add_processor(CEProcessor, coefficients=coefs[:-1])
        cls.processor.add_processor(EwaldProcessor, coefficient=coefs[-1],
                                    ewald_term=ewald_term)
        cls.ensemble = FuSemiGrandEnsemble(cls.processor, temperature=500)
        cls.ensemble_kwargs = {}

    def setUp(self):
        self.enc_occu = np.array([np.random.randint(len(species))
                                  for species in
                                  self.processor.allowed_species], dtype=int)
        self.init_occu = self.processor.decode_occupancy(self.enc_occu)

    def test_compute_feature_vector(self):
        self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                self.ensemble.compute_feature_vector(self.enc_occu)),
                         self.processor.compute_property(self.enc_occu)
                         - self.ensemble.compute_chemical_work(self.enc_occu))
        npt.assert_array_equal(self.ensemble.compute_feature_vector(self.enc_occu)[:-1],
                               self.ensemble.processor.compute_feature_vector(self.enc_occu))
        for _ in range(10):  # test a few flips
            site = np.random.choice(range(self.processor.num_sites))
            spec = np.random.choice(list(set(range(self.n_allowed_species)) - {self.enc_occu[site]}))
            flip = [(site, spec)]
            dfu = np.log(self.ensemble._fu_table[site][spec] / self.ensemble._fu_table[site][self.enc_occu[site]])
            self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                    self.ensemble.compute_feature_vector_change(self.enc_occu, flip)),
                             self.processor.compute_property_change(self.enc_occu, flip) - dfu)
            npt.assert_array_equal(self.ensemble.compute_feature_vector_change(self.enc_occu, flip),
                                   np.append(self.processor.compute_feature_vector_change(self.enc_occu, flip),
                                             dfu))

    def test_build_fu_table(self):
        table = self.ensemble._build_fu_table(self.ensemble.fugacity_fractions)
        for space, row in zip(self.processor.allowed_species, table):
            if len(space) == 1:  # skip inactive sites
                continue
            fugacity_fractions = None
            for fus in self.ensemble.fugacity_fractions:
                if space == list(fus.keys()):
                    fugacity_fractions = fus
            for i, species in enumerate(space):
                self.assertEqual(fugacity_fractions[species], row[i])
