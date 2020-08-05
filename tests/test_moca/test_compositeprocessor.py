import numpy as np
from pymatgen.transformations.standard_transformations import \
    OrderDisorderedStructureTransformation
from smol.moca import CompositeProcessor, CEProcessor, EwaldProcessor
from smol.cofe import ClusterExpansion, StructureWrangler, ClusterSubspace
from smol.cofe.extern import EwaldTerm
from tests.data import lno_prim, lno_data, synthetic_CEewald_binary
from tests.test_moca import base_processor_test as bp


# Note that in these tests the flips are not constrained to neutral.
# Tolerances need more slack to pass, its still pretty tight though
ATOL = 3 * bp.ATOL
DRIFT_TOL = bp.DRIFT_TOL


class TestCompositeProcessorLNO(bp._TestProcessor):
    atol = ATOL
    drift_tol = DRIFT_TOL

    @classmethod
    def setUpClass(cls) -> None:
        cls.cs = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1},
                                            ltol=0.15, stol=0.2,
                                            angle_tol=5,
                                            supercell_size='O2-',
                                            basis='sinusoid',
                                            orthonormal=True,
                                            use_concentration=True)
        ewald_term = EwaldTerm()
        cls.cs.add_external_term(ewald_term)
        cls.sw = StructureWrangler(cls.cs)
        for struct, energy in lno_data:
            cls.sw.add_data(struct, {'energy': energy})
        coefs = np.linalg.lstsq(cls.sw.feature_matrix,
                                cls.sw.get_property_vector('energy', True),
                                rcond=None)[0]
        scmatrix = np.array([[3, 0, 0],
                             [0, 2, 0],
                             [0, 0, 1]])
        cls.pr = CompositeProcessor(cls.cs, scmatrix)

        cls.pr.add_processor(CEProcessor, coefficients=coefs[:-1])
        cls.pr.add_processor(EwaldProcessor, coefficient=coefs[-1],
                              ewald_term=ewald_term)

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
        cls.test_occu = cls.cs.occupancy_from_structure(test_struct, scmatrix=scmatrix)
        cls.enc_occu = cls.pr.occupancy_from_structure(test_struct)
        cls.sublattices = cls._create_sublattice_dicts(cls.pr.unique_site_spaces,
                                                       cls.pr.allowed_species)
