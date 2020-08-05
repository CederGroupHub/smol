import numpy as np
import numpy.testing as npt
from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion
from smol.cofe.extern import EwaldTerm
from smol.moca import CEProcessor, EwaldProcessor, CompositeProcessor, CanonicalEnsemble
from tests.data import (synthetic_CE_binary, synthetic_CEewald_binary,
                        lno_data, lno_prim)
import tests.test_moca.base_ensemble_test as be


class TestCanonicalEnsembleSynthBinary(be._EnsembleTest):
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
        cls.ensemble = CanonicalEnsemble(cls.processor, temperature=500)
        cls.ensemble_kwargs = {}

    def test_compute_feature_vector(self):
        self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                self.ensemble.compute_feature_vector(self.enc_occu)),
                         self.processor.compute_property(self.enc_occu))
        npt.assert_array_equal(self.ensemble.compute_feature_vector(self.enc_occu),
                               self.ensemble.processor.compute_feature_vector(self.enc_occu))
        for _ in range(10):  # test a few flips
            site = np.random.choice(range(self.processor.num_sites))
            spec = np.random.choice(list(set(range(self.n_allowed_species))-{self.enc_occu[site]}))
            flip = [(site, spec)]
            self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                    self.ensemble.compute_feature_vector_change(self.enc_occu, flip)),
                             self.processor.compute_property_change(self.enc_occu, flip))
            npt.assert_array_equal(self.ensemble.compute_feature_vector_change(self.enc_occu, flip),
                                   self.processor.compute_feature_vector_change(self.enc_occu, flip))


class TestCanonicalEnsembleSynthBinaryEwald(be._EnsembleTest):
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
        cls.ensemble = CanonicalEnsemble(cls.processor, temperature=500)
        cls.ensemble_kwargs = {}

    def test_compute_feature_vector(self):
        self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                self.ensemble.compute_feature_vector(self.enc_occu)),
                         self.processor.compute_property(self.enc_occu))
        npt.assert_array_equal(self.ensemble.compute_feature_vector(self.enc_occu),
                               self.ensemble.processor.compute_feature_vector(self.enc_occu))
        for _ in range(10):  # test a few flips
            site = np.random.choice(range(self.processor.num_sites))
            spec = np.random.choice(list(set(range(self.n_allowed_species))-{self.enc_occu[site]}))
            flip = [(site, spec)]
            self.assertEqual(np.dot(self.ensemble.natural_parameters,
                                    self.ensemble.compute_feature_vector_change(self.enc_occu, flip)),
                             self.processor.compute_property_change(self.enc_occu, flip))
            npt.assert_array_equal(self.ensemble.compute_feature_vector_change(self.enc_occu, flip),
                                   self.processor.compute_feature_vector_change(self.enc_occu, flip))


class TestCanonicalEnsembleLNO(be._EnsembleTest):
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
        cls.ensemble = CanonicalEnsemble(cls.processor, temperature=500)
        cls.ensemble_kwargs = {}

    def setUp(self):
        self.enc_occu = np.array([np.random.randint(len(species))
                                  for species in
                                  self.processor.allowed_species], dtype=int)
        self.init_occu = self.processor.decode_occupancy(self.enc_occu)
