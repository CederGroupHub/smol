import unittest
from pymatgen import Structure, Lattice


#TODO implement these
class TestEstimator(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = Lattice([[3, 3, 0],[0, 3, 3],[3, 0, 3]])
        species = [{'Li+': 0.3333333}] * 3 + ['Br-']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                  (0.5, 0.5, 0.5),  (0, 0, 0))
        self.structure = Structure(self.lattice, species, coords)
        self.structure.make_supercell([[2, 1, 1]])

    def test_constrain_dielectric(self):
        pass
        '''self.ce.cluster_subspace.add_external_term(EwaldTerm)
        ce = ClusterExpansion.from_structure_wrangler(self.sw,
                                                      estimator=CVXEstimator())
        ce.fit()
        constrain_dielectric(ce, 5)
        self.assertEqual(ce.ecis[-1], 1/5)'''

    def test_CVX(self):
        pass
        # This is broken for some reason, No structures are generated
        # This is an issue with pymatgen or installation
        '''
        est = EnumerateStructureTransformation(1, 1, symm_prec=0.01)
        structures = []
        energies = []
        for x in est.apply_transformation(self.structure, 10000):
            structures.append(x['structure'])
            energies.append(x['energy'])

        _subspace = ClusterSubspace.from_radii(self.structure, {2: 2})

        feature_matrix = np.array([_subspace.corr_from_structure(s) for s in structures])
        energies = np.array(energies)

        est = CVXEstimator()
        est.fit(feature_matrix, energies)
        self.assertTrue(np.allclose(est.coef_, [-16.25708172, -0.29365225, 0.51789381], atol=1e-5))
        '''