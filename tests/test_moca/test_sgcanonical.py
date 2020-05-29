
import unittest
import os
import json
import numpy as np
from monty.json import MontyEncoder, MontyDecoder
from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion
from smol.moca import CEProcessor, MuSemiGrandEnsemble, FuSemiGrandEnsemble
from tests.data import synthetic_CE_binary


class TestMuSemiGrandEnsemble(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cs = ClusterSubspace.from_dict(synthetic_CE_binary['cluster_subspace'])
        sw = StructureWrangler(cs)
        for item in synthetic_CE_binary['data'][:200]:
            sw.add_data(item[0], {'energy': item[1]})
        ecis = np.linalg.lstsq(sw.feature_matrix,
                               sw.get_property_vector('energy', True),
                               rcond=None)[0]
        ce = ClusterExpansion(cs, ecis, sw.feature_matrix)
        sc_matrix = np.array([[3, 1, 1], [1, 3, 1], [1, 1, 3]])
        cls.pr = CEProcessor(ce, sc_matrix)
        cls.n_atoms = len(cls.pr.structure)

    def setUp(self):
        self.init_encoc = np.random.randint(0, 2, size=self.n_atoms)
        self.init_occu = self.pr.decode_occupancy(self.init_encoc)
        self.sample_interval = 200
        self.chem_pots = {'Na+': -0.05, 'Cl-': 0}
        self.ensemble = MuSemiGrandEnsemble(self.pr,
                                            temperature=1000,
                                            chemical_potentials=self.chem_pots,
                                            sample_interval=self.sample_interval,
                                            initial_occupancy=self.init_occu)

    def test_run(self):
        energy = self.ensemble.current_energy
        self.assertEqual(self.chem_pots, self.ensemble.chemical_potentials)
        self.assertEqual(self.ensemble.data, [])
        self.ensemble.run(5000)
        self.assertEqual(self.ensemble.current_step, 5000)
        self.assertEqual(len(self.ensemble.data), 5000//self.sample_interval)
        if self.ensemble.accepted_steps > 0:
            self.assertFalse(np.array_equal(self.ensemble.energy_samples,
                             energy*np.ones_like(self.ensemble.energy_samples)))

        # check that extreme values of chem pots give expected compositoins
        chem_pots = {'Na+': 100.0, 'Cl-': 0.0}
        ensemble = MuSemiGrandEnsemble(self.pr,
                                       temperature=1000,
                                       chemical_potentials=chem_pots,
                                       sample_interval=self.sample_interval,
                                       initial_occupancy=self.init_occu)
        ensemble.run(1000)
        expected = {'Na+': 1.0, 'Cl-': 0.0}
        actual = ensemble.average_composition
        for sp, val in actual.items():
            self.assertAlmostEqual(val, expected[sp])
        chem_pots = {'Na+': -100.0, 'Cl-': 0.0}
        ensemble = MuSemiGrandEnsemble(self.pr,
                                       temperature=1000,
                                       chemical_potentials=chem_pots,
                                       sample_interval=self.sample_interval,
                                       initial_occupancy=self.init_occu)
        ensemble.run(1000)
        expected = {'Na+': 0.0, 'Cl-': 1.0}
        actual = ensemble.average_composition
        for sp, val in actual.items():
            self.assertAlmostEqual(val, expected[sp])

    def test_restrict_sites(self):
        restrict = np.random.choice(range(self.ensemble.num_atoms), size=4)
        current_occu = self.ensemble._occupancy.copy()
        self.ensemble.restrict_sites(restrict)
        self.assertEqual(self.ensemble.restricted_sites, list(restrict))
        self.ensemble.run(1000)
        self.assertTrue(np.array_equal(current_occu[restrict],
                                       self.ensemble._occupancy[restrict]))
        self.ensemble.reset_restricted_sites()
        self.assertEqual(self.ensemble.restricted_sites, [])

    def test_bad_species_chem_pots(self):
        chem_pots = {'Blab': -100, 'Cl-': 0}
        self.assertRaises(ValueError, MuSemiGrandEnsemble, self.pr,
                          temperature=1000, chemical_potentials=chem_pots,
                          sample_interval=self.sample_interval,
                          initial_occupancy=self.init_occu)
        chem_pots = {'Na+': -100}
        self.assertRaises(ValueError, MuSemiGrandEnsemble, self.pr,
                          temperature=1000, chemical_potentials=chem_pots,
                          sample_interval=self.sample_interval,
                          initial_occupancy=self.init_occu)

    def test_attempt_step(self):
        for _ in range(100):
            occu = self.ensemble.current_occupancy
            energy = self.ensemble.current_energy
            acc = self.ensemble._attempt_step()
            if acc:
                self.assertNotEqual(occu, self.ensemble.current_occupancy)
                self.assertNotEqual(energy, self.ensemble.current_energy)
            else:
                self.assertEqual(occu, self.ensemble.current_occupancy)
                self.assertEqual(energy, self.ensemble.current_energy)

    def test_get_flips(self):
        flip, sublat, sp_old, sp_new = self.ensemble._get_flips()
        self.assertTrue(sp_old in sublat['domain'])
        self.assertTrue(sp_new in sublat['domain'])

    def test_get_counts_comps(self):
        self.ensemble.run(1000)
        counts = self.ensemble._get_counts()
        for name, count in counts.items():
            self.assertEqual(sum(count),
                             len(self.ensemble._active_sublatts[name]['sites']))
        comps = self.ensemble._get_sublattice_comps()
        for name, comp in comps.items():
            self.assertEqual(sum(c for c in comp.values()), 1.0)

    def test_get_current_data(self):
        self.ensemble.run(100)
        d = self.ensemble._get_current_data()
        self.assertEqual(d['energy'], self.ensemble.current_energy)
        self.assertEqual(d['occupancy'], self.ensemble.current_occupancy)
        self.assertEqual(d['counts'], self.ensemble.current_species_counts)
        self.assertEqual(d['composition'],
                         self.ensemble.current_composition)
        self.assertEqual(d['sublattice_compositions'],
                         self.ensemble.current_sublattice_compositions)

    def test_dump(self):
        self.ensemble.run(1000)
        file_path = './test_file.txt'
        self.ensemble.dump(file_path)
        self.assertEqual(self.ensemble.data, [])
        self.assertTrue(os.path.isfile(file_path))
        os.remove(file_path)

    def test_msnonable(self):
        self.maxDiff = None
        self.ensemble.run(1000)
        d = self.ensemble.as_dict()
        ensemble = MuSemiGrandEnsemble.from_dict(d)
        self.assertEqual(d, ensemble.as_dict())
        s = json.dumps(d, cls=MontyEncoder)
        ensemble1 = json.loads(s, cls=MontyDecoder)
        self.assertEqual(ensemble.as_dict(), d)
        # TODO without MontyEncoder, fails because _sublattices have numpy
        #  arrays
        # json.dumps(d)


class TestFuSemiGrandEnsemble(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cs = ClusterSubspace.from_dict(synthetic_CE_binary['cluster_subspace'])
        sw = StructureWrangler(cs)
        for item in synthetic_CE_binary['data'][:200]:
            sw.add_data(item[0], {'energy': item[1]})
        ecis = np.linalg.lstsq(sw.feature_matrix,
                               sw.get_property_vector('energy', True),
                               rcond=None)[0]
        ce = ClusterExpansion(cs, ecis, sw.feature_matrix)
        sc_matrix = np.array([[3, 1, 1], [1, 3, 1], [1, 1, 3]])
        cls.pr = CEProcessor(ce, sc_matrix)
        cls.n_atoms = len(cls.pr.structure)

    def setUp(self):
        self.init_encoc = np.random.randint(0, 2, size=self.n_atoms)
        self.init_occu = self.pr.decode_occupancy(self.init_encoc)
        self.sample_interval = 200
        self.ensemble = FuSemiGrandEnsemble(self.pr,
                                            temperature=1000,
                                            sample_interval=self.sample_interval,
                                            initial_occupancy=self.init_occu)

    def test_run(self):
        energy = self.ensemble.current_energy
        # self.assertEqual(self.fu add test to get fugacity fracts
        self.assertEqual(self.ensemble.data, [])
        self.ensemble.run(5000)
        self.assertEqual(self.ensemble.current_step, 5000)
        self.assertEqual(len(self.ensemble.data), 5000 // self.sample_interval)
        if self.ensemble.accepted_steps > 0:
            self.assertFalse(np.array_equal(self.ensemble.energy_samples,
                                            energy*np.ones_like(self.ensemble.energy_samples)))

        # check that hight T limit gives correct compositions
        ensemble = FuSemiGrandEnsemble(self.pr,
                                       temperature=1E10,  # ridiculously high?
                                       sample_interval=self.sample_interval,
                                       initial_occupancy=self.init_occu)
        ensemble.run(10000)
        expected = {'Na+': 0.5, 'Cl-': 0.5}
        actual = ensemble.average_composition
        for sp, val in actual.items():
            self.assertAlmostEqual(val, expected[sp], places=1)

        expected = {'Na+': 0.1, 'Cl-': 0.9}
        ensemble = FuSemiGrandEnsemble(self.pr,
                                       temperature=1E10,
                                       fugacity_fractions=[expected],
                                       sample_interval=self.sample_interval,
                                       initial_occupancy=self.init_occu)
        ensemble.run(10000)
        actual = ensemble.average_composition
        for sp, val in actual.items():
            self.assertAlmostEqual(val, expected[sp], places=1)

    def test_restrict_sites(self):
        restrict = np.random.choice(range(self.ensemble.num_atoms), size=4)
        current_occu = self.ensemble._occupancy.copy()
        self.ensemble.restrict_sites(restrict)
        self.assertEqual(self.ensemble.restricted_sites, list(restrict))
        self.ensemble.run(1000)
        self.assertTrue(np.array_equal(current_occu[restrict],
                                       self.ensemble._occupancy[restrict]))
        self.ensemble.reset_restricted_sites()
        self.assertEqual(self.ensemble.restricted_sites, [])

    def test_bad_species_fug_fracts(self):
        fug_fracs = {'Blab': 0.1, 'Cl-': 0.9}
        self.assertRaises(ValueError, FuSemiGrandEnsemble, self.pr,
                          temperature=1000, fugacity_fractions=[fug_fracs],
                          sample_interval=self.sample_interval,
                          initial_occupancy=self.init_occu)
        fug_fracs = {'Na+': 0.1, 'Cl-': 0.8}
        self.assertRaises(ValueError, FuSemiGrandEnsemble, self.pr,
                          temperature=1000, fugacity_fractions=[fug_fracs],
                          sample_interval=self.sample_interval,
                          initial_occupancy=self.init_occu)

    def test_attempt_step(self):
        for _ in range(100):
            occu = self.ensemble.current_occupancy
            energy = self.ensemble.current_energy
            acc = self.ensemble._attempt_step()
            if acc:
                self.assertNotEqual(occu, self.ensemble.current_occupancy)
                self.assertNotEqual(energy, self.ensemble.current_energy)
            else:
                self.assertEqual(occu, self.ensemble.current_occupancy)
                self.assertEqual(energy, self.ensemble.current_energy)

    def test_get_flips(self):
        _, sublat, sp_old, sp_new = self.ensemble._get_flips()
        self.assertTrue(sp_old in sublat['domain'])
        self.assertTrue(sp_new in sublat['domain'])

    def test_get_counts_comps(self):
        self.ensemble.run(1000)
        counts = self.ensemble._get_counts()
        for name, count in counts.items():
            self.assertEqual(sum(count),
                             len(self.ensemble._active_sublatts[name]['sites']))
        comps = self.ensemble._get_sublattice_comps()
        for name, comp in comps.items():
            self.assertEqual(sum(c for c in comp.values()), 1.0)

    def test_get_current_data(self):
        self.ensemble.run(100)
        d = self.ensemble._get_current_data()
        self.assertEqual(d['energy'], self.ensemble.current_energy)
        self.assertEqual(d['occupancy'], self.ensemble.current_occupancy)
        self.assertEqual(d['counts'], self.ensemble.current_species_counts)
        self.assertEqual(d['composition'], self.ensemble.current_composition)
        self.assertEqual(d['sublattice_compositions'],
                         self.ensemble.current_sublattice_compositions)

    def test_dump(self):
        self.ensemble.run(1000)
        file_path = './test_file.txt'
        self.ensemble.dump(file_path)
        self.assertEqual(self.ensemble.data, [])
        self.assertTrue(os.path.isfile(file_path))
        os.remove(file_path)

    def test_msnonable(self):
        self.maxDiff = None
        self.ensemble.run(1000)
        d = self.ensemble.as_dict()
        ensemble = FuSemiGrandEnsemble.from_dict(d)
        self.assertEqual(d, ensemble.as_dict())
        s = json.dumps(d, cls=MontyEncoder)
        ensemble1 = json.loads(s, cls=MontyDecoder)
        self.assertEqual(ensemble.as_dict(), d)
        # TODO without MontyEncoder, fails because _sublattices have numpy
        #  arrays
        # json.dumps(d)
