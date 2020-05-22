"""Test vs results from pyabinitio.cluster_expansion module."""

import unittest
import numpy as np
from smol.cofe import StructureWrangler, ClusterSubspace
from smol.cofe.extern import EwaldTerm
from tests.data import pyabinitio_LiMn3OF_dataset, pyabinitio_LiNi2Mn4OF_dataset


class TestvsPyabinitio(unittest.TestCase):
    def test_ionic_binary(self):
        self._test_dataset(pyabinitio_LiMn3OF_dataset)

    def test_ionic_ternary(self):
        self._test_dataset(pyabinitio_LiNi2Mn4OF_dataset)

    def _test_dataset(self, dataset):
        radii = {int(k): v for k, v in dataset['cutoff_radii'].items()}
        cs = ClusterSubspace.from_radii(dataset['prim'],
                                        radii=radii,
                                        basis='indicator',
                                        **dataset['matcher_kwargs'])

        self.assertEqual(cs.n_orbits, dataset['n_orbits'])
        self.assertEqual(cs.n_bit_orderings, dataset['n_bit_orderings'])

        if dataset['ewald']:
            cs.add_external_term(EwaldTerm())

        sw = StructureWrangler(cs)
        # if ever want to check mapping works remove supercell matrix and
        # mapping, but then adding data takes a very long time.
        for item in dataset['data']:
            sw.add_data(item['structure'], item['properties'],
                        supercell_matrix=item['scmatrix'],
                        site_mapping=item['mapping'])

        self.assertTrue(np.allclose(sw.feature_matrix,
                                    dataset['feature_matrix']))
