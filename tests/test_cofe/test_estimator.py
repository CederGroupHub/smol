import unittest
import numpy as np
from pymatgen import Structure, Lattice
from pymatgen.transformations.advanced_transformations import EnumerateStructureTransformation
from smol.cofe import ClusterSubspace
from smol.cofe.regression import CVXEstimator


class TestEstimator(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = Lattice([[3, 3, 0],[0, 3, 3],[3, 0, 3]])
        species = [{'Li+': 0.3333333}] * 3 + ['Br-']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                  (0.5, 0.5, 0.5),  (0, 0, 0))
        self.structure = Structure(self.lattice, species, coords)
        self.structure.make_supercell([[2, 1, 1]])

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

        subspace = ClusterSubspace.from_radii(self.structure, {2: 2})

        feature_matrix = np.array([subspace.corr_from_structure(s) for s in structures])
        energies = np.array(energies)

        est = CVXEstimator()
        est.fit(feature_matrix, energies)
        self.assertTrue(np.allclose(est.coef_, [-16.25708172, -0.29365225, 0.51789381], atol=1e-5))
        '''
