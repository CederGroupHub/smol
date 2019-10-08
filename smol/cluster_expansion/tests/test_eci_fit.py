from __future__ import division
import unittest

from pymatgen import Lattice, Structure
from pyabinitio.cluster_expansion.eci_fit import EciGenerator
from pyabinitio.cluster_expansion.ce import ClusterExpansion
from pymatgen.transformations.advanced_transformations import EnumerateStructureTransformation

import numpy as np

class EciGeneratorTest(unittest.TestCase):
    
    def setUp(self):
        self.lattice = Lattice([[3, 3, 0],[0, 3, 3],[3, 0, 3]])
        species = [{'Li+': 0.3333333}] * 3 + ['Br-']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75), 
                  (0.5, 0.5, 0.5),  (0, 0, 0))
        self.structure = Structure(self.lattice, species, coords)
        self.structure.make_supercell([[2, 1, 1]])
        
    def test_generator(self):
        est = EnumerateStructureTransformation(1, 1, symm_prec=0.01)
        structures = []
        energies = []
        for i, x in enumerate(est.apply_transformation(self.structure, 10000)):
            structures.append(x['structure'])
            energies.append(x['energy'])

        ce = ClusterExpansion.from_radii(self.structure, {2: 2}, use_ewald=False)

        # currently there's a factor of 2 difference between the mu's in cvxopt and bregman implementations
        # maybe should fix this

        eg = EciGenerator.unweighted(ce, structures=structures, energies=energies, mu=100, solver='bregman_l1')
        bregman_fit = np.dot(eg.feature_matrix, eg.ecis)
        self.assertTrue(np.allclose(eg.ecis, [-16.25708172, -0.29365225, 0.51789381], atol=1e-5))

        eg = EciGenerator.unweighted(ce, structures=structures, energies=energies, mu=50, solver='cvxopt_l1')
        cvxopt_fit = np.dot(eg.feature_matrix, eg.ecis)

        self.assertTrue(np.allclose(bregman_fit, cvxopt_fit, atol=1e-5))

        # bregman with mu 100, temperature 2000
        eg = EciGenerator.weight_by_e_above_hull(ce, structures=structures, temperature=2000,
                                                 energies=energies, mu=100, solver='cvxopt_l1')
        self.assertTrue(np.allclose(eg.ecis, [-17.08007087, -1.70603985, 0.53173669], atol=1e-5))

        # e above comp should be the same with only one composition
        eg = EciGenerator.weight_by_e_above_comp(ce, structures=structures, temperature=2000,
                                                 energies=energies, mu=100, solver='cvxopt_l1')
        self.assertTrue(np.allclose(eg.ecis, [-17.08007087, -1.70603985, 0.53173669], atol=1e-5))

        # test max_dielectric
        ce = ClusterExpansion.from_radii(self.structure, {2: 1}, use_ewald=True)
        eg = EciGenerator.unweighted(ce, structures=structures, energies=energies, mu=10000, solver='cvxopt_l1',
                                     max_dielectric=.5)
        self.assertTrue(np.allclose(eg.ecis, [15.69172655, 1.43831153, 0.0384877, 2.], atol=1e-3))


    def test_cv_fitting(self):
        self.structure.make_supercell([1, 2, 1])
        est = EnumerateStructureTransformation(1, 1, symm_prec=0.01)
        structures = []
        energies = []
        for i, x in enumerate(est.apply_transformation(self.structure, 10000)):
            structures.append(x['structure'])
            energies.append(x['energy'])

        ce = ClusterExpansion.from_radii(self.structure, {2: 2}, use_ewald=False)
        eg = EciGenerator.weight_by_e_above_hull(ce, structures=structures, energies=energies, solver='cvxopt_l1')

        plt = eg.get_scatterplot(xaxis='e_above_hull_input', yaxis='normalized_error')
        plt = eg.get_scatterplot(xaxis='e_above_hull_input', yaxis='e_above_hull_ce')

    def test_dict_representation(self):
        est = EnumerateStructureTransformation(1, 1, symm_prec=0.01)
        structures = []
        energies = []
        for i, x in enumerate(est.apply_transformation(self.structure, 10000)):
            structures.append(x['structure'])
            energies.append(x['energy'])

        ce = ClusterExpansion.from_radii(self.structure, {2: 2}, use_ewald=False)

        eg = EciGenerator.unweighted(ce, structures=structures, energies=energies, mu=100)
        d = eg.as_dict()
        generator2 = EciGenerator.from_dict(d)

    def test_dict_gs_preserve(self):
        # this tests a case where gs_preserve doesn't have to change the fit.
        # tests that it gets the same quality of fit
        est = EnumerateStructureTransformation(1, 2, symm_prec=0.01)
        structures = []
        energies = []
        for i, x in enumerate(est.apply_transformation(self.structure, 10000)):
            structures.append(x['structure'])
            energies.append(x['energy'])

        ce = ClusterExpansion.from_radii(self.structure, {2: 3}, use_ewald=False)

        eg_gs = EciGenerator.unweighted(ce, structures=structures, energies=energies,
                                        mu=1000, solver='gs_preserve')

        eg_cvxopt = EciGenerator.unweighted(ce, structures=structures, energies=energies,
                                            mu=1000, solver='cvxopt_l1')

        self.assertAlmostEqual(eg_gs.rmse, eg_cvxopt.rmse)

        #TODO: write a test that actually test the gs preservation part
