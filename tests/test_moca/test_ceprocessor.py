import numpy as np
from pymatgen.transformations.standard_transformations import \
    OrderDisorderedStructureTransformation
from smol.moca import CEProcessor
from smol.cofe import ClusterExpansion, StructureWrangler, ClusterSubspace
from tests.data import lno_prim, lno_data, synthetic_CE_binary
from tests.test_moca import base_processor_test as bp

# TODO test with all synthetic data binary/ternary/electrostatic
# TODO check that delta_corr gives same values for random sympos shuffles of same structure
# TODO check that delta_corr works for all bases and orthonormal combos


class BaseTest:
    """Wrap this up so it is not run as a test."""
    class TestCEProcessor(bp.TestProcessor):
        """Speficic to test some CEProcessors only."""

        @classmethod
        def setUpClass(cls):
            cls.sw = None  # define this as well

        def test_compute_property(self):
            self.assertTrue(np.isclose(
                np.dot(self.pr.coefs,
                       self.cs.corr_from_structure(self.test_struct, False)),
                self.pr.compute_property(self.enc_occu)))

        def test_compute_feature_vector(self):
            self.assertTrue(np.allclose(
                self.cs.corr_from_structure(self.test_struct, False),
                self.pr.compute_feature_vector(self.enc_occu)))

        def test_feature_update_indicator(self):
            cs = self.cs.copy()
            cs.change_site_bases('indicator')
            sw = StructureWrangler(cs)

            # Need to refit
            for struct, energy, matrix in \
                    zip(self.sw.structures,
                        self.sw.get_property_vector('energy'),
                        self.sw.supercell_matrices):
                sw.add_data(struct, {'energy': energy}, supercell_matrix=matrix)

            coefs = np.linalg.lstsq(sw.feature_matrix,
                                    sw.get_property_vector('energy', True),
                                    rcond=None)[0]
            pr = CEProcessor(cs, self.pr.supercell_matrix, coefs,
                             optimize_indicator=True)
            occu = self.enc_occu.copy()
            for _ in range(50):
                sublatt = np.random.choice(self.sublattices)
                site = np.random.choice(sublatt['sites'])
                new_sp = np.random.choice(sublatt['spaces'])
                new_occu = occu.copy()
                new_occu[site] = new_sp
                # Test forward
                dcorr = pr.compute_feature_vector_change(occu, [(site, new_sp)])
                corr_f = pr.compute_feature_vector(new_occu)
                corr_i = pr.compute_feature_vector(occu)
                self.assertTrue(np.allclose(dcorr, corr_f - corr_i,
                                            rtol=bp.RTOL, atol=self.atol))
                # Test reverse matches forward
                old_sp = occu[site]
                rdcorr = pr.compute_feature_vector_change(new_occu, [(site, old_sp)])
                self.assertTrue(np.allclose(dcorr, -1 * rdcorr,
                                            rtol=bp.RTOL, atol=self.atol))

        def test_constructor_except(self):
            coefs = np.random.random(self.pr.n_orbit_functions + 1)
            self.assertRaises(ValueError, CEProcessor, self.pr.cluster_subspace,
                              self.pr.supercell_matrix, coefs)


class TestCEProcessorSynthBinary(BaseTest.TestCEProcessor):
    """Test on binary synthetic data."""
    atol = 10 * bp.ATOL
    drift_tol = 10 * bp.DRIFT_TOL

    @classmethod
    def setUpClass(cls):
        cls.cs = ClusterSubspace.from_dict(synthetic_CE_binary['cluster_subspace'])
        cls.sw = StructureWrangler(cls.cs)
        data = synthetic_CE_binary['data']
        train_ids = np.random.choice(range(len(data)), 50, replace=False)
        for i in train_ids:
            struct, energy = data[i]
            cls.sw.add_data(struct, {'energy': energy})
        coefs = np.linalg.lstsq(cls.sw.feature_matrix,
                                cls.sw.get_property_vector('energy', True),
                                rcond=None)[0]
        scmatrix = np.array([[6, 0, 0],
                             [0, 6, 0],
                             [0, 0, 6]])
        cls.pr = CEProcessor(cls.cs, scmatrix, coefficients=coefs)

        # create a test structure
        cls.enc_occu = np.random.randint(2, size=cls.pr.size*len(cls.cs.structure))
        cls.test_occu = cls.pr.decode_occupancy(cls.enc_occu)
        cls.test_struct = cls.pr.structure_from_occupancy(cls.enc_occu)

        cls.sublattices = []  # list of dicts of sites and spaces
        cls.sublattices = cls._create_sublattice_dicts(cls.pr.unique_site_spaces,
                                                       cls.pr.allowed_species)


class TestCEProcessorLNO(BaseTest.TestCEProcessor):
    """Test the tutorial LNO to test a real multisublattice binary."""
    @classmethod
    def setUpClass(cls) -> None:
        cls.cs = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1},
                                            ltol=0.15, stol=0.2,
                                            angle_tol=5,
                                            supercell_size='O2-',
                                            basis='sinusoid',
                                            orthonormal=True,
                                            use_concentration=True)
        cls.sw = StructureWrangler(cls.cs)
        for struct, energy in lno_data:
            cls.sw.add_data(struct, {'energy': energy})
        coefs = np.linalg.lstsq(cls.sw.feature_matrix,
                                cls.sw.get_property_vector('energy', True),
                                rcond=None)[0]
        cls.ce = ClusterExpansion(cls.cs, coefs, cls.sw.feature_matrix)
        scmatrix = np.array([[3, 0, 0],
                             [0, 2, 0],
                             [0, 0, 1]])
        cls.pr = CEProcessor(cls.cs, scmatrix, coefficients=coefs)

        # create a test structure
        test_struct = lno_prim.copy()
        test_struct.replace_species({"Li+": {"Li+": 2},
                                     "Ni3+": {"Ni3+": 2},
                                     "Ni4+": {"Ni4+": 0}})
        test_struct.make_supercell(scmatrix)
        ro = {"Li+": {"Li+": 0.5},
              "Ni3+": {"Ni3+": .35, "Ni4+": 1 - .35}}
        test_struct.replace_species(ro)
        order = OrderDisorderedStructureTransformation(algo=2)
        test_struct = order.apply_transformation(test_struct)
        cls.test_struct = test_struct
        cls.test_occu = cls.cs.occupancy_from_structure(test_struct,
                                                        scmatrix=scmatrix)
        cls.enc_occu = cls.pr.occupancy_from_structure(test_struct)
        cls.sublattices = cls._create_sublattice_dicts(cls.pr.unique_site_spaces,
                                                       cls.pr.allowed_species)

