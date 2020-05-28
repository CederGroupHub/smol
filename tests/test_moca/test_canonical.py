import unittest
import os
import json
import numpy as np
from monty.json import MontyEncoder, MontyDecoder
from smol.cofe import ClusterSubspace, StructureWrangler, ClusterExpansion
from smol.moca import CEProcessor, CanonicalEnsemble
from tests.data import synthetic_CE_binary


class TestCanonicalEnsemble(unittest.TestCase):
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
        self.ensemble = CanonicalEnsemble(self.pr,
                                          temperature=1000,
                                          sample_interval=self.sample_interval,
                                          initial_occupancy=self.init_occu)

    def test_bad_occupancy(self):
        self.assertRaises(ValueError, CanonicalEnsemble, self.pr,
                          temperature=1000,
                          sample_interval=self.sample_interval,
                          initial_occupancy=[0, 1, 1, 0])

    def test_run(self):
        energy = self.ensemble.current_energy
        self.assertEqual(self.ensemble.data, [])
        self.ensemble.run(5000)
        self.assertEqual(self.ensemble.current_step, 5000)
        self.assertEqual(len(self.ensemble.data), 5000//self.sample_interval)
        if self.ensemble.accepted_steps > 0:
            self.assertFalse(np.array_equal(self.ensemble.energy_samples,
                             energy*np.ones_like(self.ensemble.energy_samples)))

    def test_attempt_step(self):
        for _ in range(100):
            occu = self.ensemble.current_occupancy
            energy = self.ensemble.current_energy
            acc = self.ensemble._attempt_step()
            if acc:
                self.assertNotEqual(occu, self.ensemble.current_occupancy)
            else:
                self.assertEqual(occu, self.ensemble.current_occupancy)
                self.assertEqual(energy, self.ensemble.current_energy)

    def test_get_flips(self):
        for _ in range(100):
            flips = self.ensemble._get_flips()
            if flips:
                self.assertEqual(len(flips), 2)
                self.assertEqual(self.init_encoc[flips[0][0]],
                                 flips[1][1])
                self.assertEqual(self.init_encoc[flips[1][0]],
                                 flips[0][1])

    def test_get_current_data(self):
        self.ensemble.run(100)
        d = self.ensemble._get_current_data()
        self.assertEqual(d['energy'], self.ensemble.current_energy)
        self.assertEqual(d['occupancy'], self.ensemble.current_occupancy)

    def test_dump(self):
        self.ensemble.run(1000)
        file_path = './test_file.txt'
        self.ensemble.dump(file_path)
        self.assertEqual(self.ensemble.data, [])
        self.assertTrue(os.path.isfile(file_path))
        os.remove(file_path)

    def test_reset(self):
        self.ensemble.run(1000)
        self.assertTrue(self.ensemble.data)
        self.assertNotEqual(self.ensemble.current_step, 0)
        self.ensemble.reset()
        self.assertEqual(self.ensemble.current_step, 0)
        self.assertEqual(self.ensemble.accepted_steps, 0)
        self.assertTrue(not self.ensemble.data)
        self.assertTrue(np.array_equal(self.init_occu,
                                      self.ensemble.initial_occupancy))
        self.assertTrue(np.array_equal(self.init_occu,
                                       self.ensemble.current_occupancy))

    # TODO implement this test
    def test_anneal(self):
        pass

    def test_msnonable(self):
        self.ensemble.run(1000)
        d = self.ensemble.as_dict()
        ensemble = CanonicalEnsemble.from_dict(d)
        self.assertEqual(d, ensemble.as_dict())
        s = json.dumps(d, cls=MontyEncoder)
        ensemble1 = json.loads(s, cls=MontyDecoder)
        # self.assertEqual(ensemble1.as_dict(), d)
        # TODO without MontyEncoder, fails because something still has an
        #  numpy array in its as_dict
        # json.dumps(d)
