import unittest
import numpy as np
from pymatgen import Structure, Lattice
from pymatgen.analysis.ewald import EwaldSummation
from smol.cofe import ClusterSubspace
from smol.cofe.configspace import ClusterSupercell, EwaldTerm

class TestEwald(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = Lattice([[3, 3, 0], [0, 3, 3], [3, 0, 3]])
        species = [{'Li': 0.1, 'Ca': 0.1}] * 3 + ['Br']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                  (0.5, 0.5, 0.5), (0, 0, 0))
        self.structure = Structure(self.lattice, species, coords)

    def test_ewald_eci(self):
        self.structure.add_oxidation_state_by_element({'Br': -1, 'Ca': 2, 'Li': 1})
        cs = ClusterSubspace.from_radii(self.structure, {2: 6, 3: 5})
        m = np.array([[2, 0, 0], [0, 2, 0], [0, 1, 1]])
        supercell = cs.structure.copy()
        supercell.make_supercell(m)
        sc = cs.supercell_from_structure(supercell)
        s = Structure(supercell.lattice, ['Ca2+', 'Li+', 'Li+', 'Br-', 'Br-', 'Br-', 'Br-'],
                      [[0.125, 1, 0.25], [0.125, 0.5, 0.25], [0.375, 0.5, 0.75], [0, 0, 0], [0, 0.5, 1],
                       [0.5, 1, 0], [0.5, 0.5, 0]])
        occu = sc.occu_from_structure(s)
        ew = EwaldTerm(sc, eta=0.15)
        self.assertAlmostEqual(ew._get_ewald_eci(occu)[0] * sc.size,
                               EwaldSummation(s, eta=ew._ewald._eta).total_energy, places=5)
        self.assertIsNotNone(ew.partial_ems)  # these need to be improved to check actual values
        self.assertIsNotNone(ew.all_ewalds)

    def test_get_ewald_diffs(self):
        # TODO implement a test here for montecarlo
        pass