import json
import numpy as np
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.transformations.standard_transformations import \
    OrderDisorderedStructureTransformation
from smol.cofe.extern import EwaldTerm
from smol.moca import EwaldProcessor
from smol.cofe import ClusterSubspace
from tests.data import synthetic_CEewald_binary, lno_prim
from tests.test_moca import _base_processor_test as bp


# Note that in these tests the flips are not constrained to neutral.
# Tolerances need more slack to pass, its still pretty tight though
ATOL = 3 * bp.ATOL
DRIFT_TOL = 3 * bp.DRIFT_TOL


class testEwaldProcessorSynthBinary(bp._TestProcessor):
    atol = ATOL
    drift_tol = DRIFT_TOL

    @classmethod
    def setUpClass(cls):
        cls.cs = ClusterSubspace.from_dict(synthetic_CEewald_binary['cluster_subspace'])
        scmatrix = np.array([[3, 0, 0],
                             [0, 3, 0],
                             [0, 0, 3]])
        cls.pr = EwaldProcessor(cls.cs, supercell_matrix=scmatrix,
                                ewald_term=EwaldTerm())
        cls.enc_occu = np.random.randint(2, size=cls.pr.size * len(cls.cs.structure))
        cls.test_occu = cls.pr.decode_occupancy(cls.enc_occu)
        cls.test_struct = cls.pr.structure_from_occupancy(cls.enc_occu)

        cls.sublattices = []  # list of dicts of sites and spaces
        cls.sublattices = cls._create_sublattice_dicts(cls.pr.unique_site_spaces,
                                                       cls.pr.allowed_species)


class TestEwaldProcessorLNO(bp._TestProcessor):
    atol = ATOL
    drift_tol = DRIFT_TOL

    @classmethod
    def setUpClass(cls):
        cls.cs = ClusterSubspace.from_radii(lno_prim, {2: 0},
                                            supercell_size='O2-')
        # create a test structure
        test_struct = lno_prim.copy()
        test_struct.replace_species({"Li+": {"Li+": 2},
                                     "Ni3+": {"Ni3+": 2},
                                     "Ni4+": {"Ni4+": 0}})
        cls.scmatrix = np.array([[3, 0, 0],
                                 [0, 2, 0],
                                 [0, 0, 1]])
        test_struct.make_supercell(cls.scmatrix)
        ro = {"Li+": {"Li+": 0.5},
              "Ni3+": {"Ni3+": .35, "Ni4+": 1 - .35}}
        test_struct.replace_species(ro)
        order = OrderDisorderedStructureTransformation(algo=2)
        cls.test_struct = order.apply_transformation(test_struct)
        cls.test_occu = cls.cs.occupancy_from_structure(cls.test_struct,
                                                        scmatrix=cls.scmatrix)

        pr = EwaldProcessor(cls.cs, cls.scmatrix, EwaldTerm())
        cls.sublattices = cls._create_sublattice_dicts(pr.unique_site_spaces,
                                                       pr.allowed_species)

    def setUp(self):
        self.setUpProcessor()

    def tearDown(self) -> None:
        self.cs._external_terms = []

    def setUpProcessor(self, term='total') -> None:
        ewald_term = EwaldTerm(use_term=term)
        self.cs.add_external_term(ewald_term)
        self.pr = EwaldProcessor(self.cs, self.scmatrix, ewald_term)
        self.enc_occu = self.pr.occupancy_from_structure(self.test_struct)

class TestEwaldProcessorReal(TestEwaldProcessorLNO):
    def setUp(self):
        self.setUpProcessor(term='real')

    def tearDown(self) -> None:
        self.cs._external_terms = []


class TestEwaldProcessorRecip(TestEwaldProcessorLNO):
    def setUp(self):
        self.setUpProcessor(term='reciprocal')

    def tearDown(self) -> None:
        self.cs._external_terms = []


class TestEwaldProcessorPoint(TestEwaldProcessorLNO):
    def setUp(self):
        self.setUpProcessor(term='point')

    def tearDown(self) -> None:
        self.cs._external_terms = []

