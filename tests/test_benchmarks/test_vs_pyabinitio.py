"""Test vs results from pyabinitio.cluster_expansion module."""

import unittest
import numpy as np
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
from smol.cofe import StructureWrangler, ClusterSubspace, ClusterExpansion
from smol.cofe.extern import EwaldTerm
from smol.moca import EwaldCEProcessor, CanonicalEnsemble
from tests.data import (pyabinitio_LiMn3OF_dataset,
                        pyabinitio_LiNi2Mn4OF_dataset, lno_prim, lno_data,
                        lno_gc_run_10000)


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

    def test_canonical_montecarlo(self):
        cs = ClusterSubspace.from_radii(structure=lno_prim,
                                        radii={2: 6, 3: 5.1},
                                        basis='indicator',
                                        orthonormal=False,
                                        use_concentration=False,
                                        ltol=0.15, stol=0.2, angle_tol=5,
                                        supercell_size='O2-')
        cs.add_external_term(EwaldTerm())
        sw = StructureWrangler(cs)
        for s, e in lno_data:
            sw.add_data(s, {'energy': e}, verbose=True)

        ecis = np.linalg.lstsq(sw.feature_matrix,
                               sw.get_property_vector('energy'),
                               rcond=None)[0]
        ce = ClusterExpansion(cs, ecis=ecis, feature_matrix=sw.feature_matrix)

        # make a supercell structure
        test_struct = lno_prim.copy()
        test_struct.replace_species({"Li+": {"Li+": 2}, "Ni3+": {"Ni3+": 2},
                                     "Ni4+": {"Ni4+": 0}})
        Li_c, Ni3_c = 0.8, 0.4
        test_struct.replace_species({"Li+": {"Li+": Li_c},
                                     "Ni3+": {"Ni3+": Ni3_c, "Ni4+": 1 - Ni3_c}})
        test_struct.make_supercell([5, 1, 1])

        order = OrderDisorderedStructureTransformation(algo=0)
        test_struct = order.apply_transformation(test_struct)
        matrix = np.array([[1, 1, 1],
                           [1, -3, 1],
                           [1, 1, -3]])
        test_struct.make_supercell(matrix)
        sc_matrix = cs.scmatrix_from_structure(test_struct)
        n_atoms = len(test_struct)
        processor = EwaldCEProcessor(ce, sc_matrix)
        processor_ind = EwaldCEProcessor(ce, sc_matrix,
                                         optimize_indicator=True)
        init_occu = processor.occupancy_from_structure(test_struct)
        iterations = 10000
        sample_interval = 100
        temp = lno_gc_run_10000['temperature']
        for pr in (processor, processor_ind):
            ens = CanonicalEnsemble(processor_ind, temperature=temp,
                                    sample_interval=sample_interval,
                                    initial_occupancy=init_occu)
            ens.run(iterations)
            self.assertAlmostEqual(lno_gc_run_10000['min_energy']/n_atoms,
                                   ens.minimum_energy/n_atoms, places=1)
            self.assertAlmostEqual(lno_gc_run_10000['average_energy']/n_atoms,
                                   ens.average_energy/n_atoms, places=1)
