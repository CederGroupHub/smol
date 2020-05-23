import unittest
import os
import numpy as np
from smol.io import save_work, load_work
from smol.cofe import ClusterSubspace, ClusterExpansion, StructureWrangler
from smol.moca import CEProcessor, CanonicalEnsemble
from tests.data import lno_prim, lno_data


class TestSaveLoadWork(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cs = ClusterSubspace.from_radii(lno_prim, {2: 5, 3: 4.1},
                                            ltol=0.15, stol=0.2,
                                            angle_tol=5, supercell_size='O2-')
        cls.sw = StructureWrangler(cls.cs)
        for struct, energy in lno_data:
            cls.sw.add_data(struct, {'energy': energy})

        ecis = np.ones(cls.cs.n_bit_orderings)
        cls.ce = ClusterExpansion(cls.cs, ecis, cls.sw.feature_matrix)
        cls.pr = CEProcessor(cls.ce, 2 * np.eye(3))
        cls.en = CanonicalEnsemble(cls.pr, 500, 100)
        cls.en.run(100)
        cls.file_path = './test_save_work.mson'

    def test_save_load_work(self):
        save_work(self.file_path, self.cs, self.sw, self.ce, self.pr, self.en)
        self.assertTrue(os.path.isfile(self.file_path))

        work_dict = load_work(self.file_path)
        self.assertEqual(5, len(work_dict))

        for name, obj in work_dict.items():
            self.assertEqual(name, obj.__class__.__name__)
            self.assertTrue(type(obj) in (ClusterSubspace, ClusterExpansion,
                                          StructureWrangler, CEProcessor,
                                          CanonicalEnsemble))

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.file_path)
