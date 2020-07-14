import unittest
import numpy as np
from pymatgen import Structure, Lattice
from pymatgen.analysis.ewald import EwaldSummation
from smol.cofe import ClusterSubspace
from smol.cofe.extern import EwaldTerm


class TestEwald(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = Lattice([[3, 3, 0], [0, 3, 3], [3, 0, 3]])
        species = [{'Li': 0.1, 'Ca': 0.1}] * 3 + ['Br']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                  (0.5, 0.5, 0.5), (0, 0, 0))
        self.structure = Structure(self.lattice, species, coords)

    def test_corr_from_occupancy(self):
        self.structure.add_oxidation_state_by_element({'Br': -1,
                                                       'Ca': 2,
                                                       'Li': 1})
        cs = ClusterSubspace.from_radii(self.structure, {2: 6, 3: 5},
                                        basis='indicator',
                                        supercell_size='volume')
        m = np.array([[2, 0, 0], [0, 2, 0], [0, 1, 1]])
        supercell = cs.structure.copy()
        supercell.make_supercell(m)
        s = Structure(supercell.lattice,
                      ['Ca2+', 'Li+', 'Li+', 'Br-', 'Br-', 'Br-', 'Br-'],
                      [[0.125, 1, 0.25], [0.125, 0.5, 0.25],
                       [0.375, 0.5, 0.75], [0, 0, 0], [0, 0.5, 1],
                       [0.5, 1, 0], [0.5, 0.5, 0]])
        occu = cs.occupancy_from_structure(s, encode=True)
        ew = EwaldTerm(eta=0.15)
        np.testing.assert_almost_equal(ew.corr_from_occupancy(occu, supercell, 1),
                                       EwaldSummation(s, eta=ew.eta).total_energy,
                                       decimal=7)
        ew = EwaldTerm(eta=0.15, use_term='real')
        np.testing.assert_almost_equal(ew.corr_from_occupancy(occu, supercell, 1),
                                       EwaldSummation(s, eta=ew.eta).real_space_energy,
                                       decimal=7)
        ew = EwaldTerm(eta=0.15, use_term='reciprocal')
        np.testing.assert_almost_equal(ew.corr_from_occupancy(occu, supercell, 1),
                                       EwaldSummation(s, eta=ew.eta).reciprocal_space_energy,
                                       decimal=7)
        ew = EwaldTerm(eta=0.15, use_term='point')
        np.testing.assert_almost_equal(ew.corr_from_occupancy(occu, supercell, 1),
                                       EwaldSummation(s, eta=ew.eta).point_energy,
                                       decimal=7)
        # TODO elaborate on this
        _, _ = ew._get_ewald_structure(supercell)

    def test_msonable(self):
        ew = EwaldTerm(eta=0.15, real_space_cut=0.5, use_term='point')
        d = ew.as_dict()
        self.assertEqual(EwaldTerm.from_dict(d).as_dict(), d)