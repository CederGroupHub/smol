import pytest
import unittest
import numpy.testing as npt
import random
import numpy as np
from itertools import combinations
import json
from pymatgen.core import Lattice, Structure, Species
from pymatgen.util.coord import is_coord_subset_pbc
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from smol.cofe import ClusterSubspace, PottsSubspace
from smol.cofe.space.clusterspace import invert_mapping,\
                                         get_complete_mapping
from smol.cofe.extern import EwaldTerm
from smol.cofe.space.constants import SITE_TOL
from smol.cofe.space.domain import get_allowed_species, Vacancy
from smol.exceptions import StructureMatchError
from src.mc_utils import corr_from_occupancy
from tests.utils import assert_msonable


# TODO test correlations for ternary and for applications of symops to structure
def test_invert_mapping_table():
    forward = [[], [], [1], [1], [1], [2, 4], [3, 4], [2, 3], [5, 6, 7]]
    backward = [[], [2, 3, 4], [5, 7], [6, 7], [5, 6], [8], [8], [8], []]

    forward_invert = [sorted(sub) for sub in invert_mapping(forward)]
    backward_invert = [sorted(sub) for sub in invert_mapping(backward)]

    assert forward_invert == backward
    assert backward_invert == forward


def test_get_complete_mapping():
    forward = [[], [], [1], [1], [1], [2, 4], [3, 4], [2, 3], [5, 6, 7]]
    backward = [[], [2, 3, 4], [5, 7], [6, 7],[5, 6], [8], [8], [8], []]

    forward_full = [[], [], [1], [1], [1], [1, 2, 4], [1, 3, 4], [1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7]]
    backward_full = [[], [2, 3, 4, 5, 6, 7, 8], [5, 7, 8], [6, 7, 8],
                     [5, 6, 8], [8], [8], [8], []]

    forward_comp = [sorted(sub) for sub in get_complete_mapping(forward)]
    backward_comp = [sorted(sub) for sub in get_complete_mapping(backward)]

    assert forward_comp == forward_full
    assert backward_comp == backward_full


def test_from_cutoffs(structure):
    cutoffs = {2: 5, 3: 4, 4: 4}
    for increment in np.arange(0, 3, 1):
        cutoffs.update(
            {k: v + increment/(n + 1)
             for n, (k, v) in enumerate(cutoffs.items())})
        subspace = ClusterSubspace.from_cutoffs(structure, cutoffs)
        tight_subspace = ClusterSubspace.from_cutoffs(
            structure, subspace.cutoffs)
        assert len(subspace) == len(tight_subspace)
        npt.assert_allclose(
            np.array(list(subspace.cutoffs.values())),
            np.array(list(tight_subspace.cutoffs.values())))


def test_potts_subspace(cluster_subspace):
    potts_subspace = PottsSubspace.from_cutoffs(cluster_subspace.structure,
                                                cluster_subspace.cutoffs)
    assert len(potts_subspace.orbits) == len(cluster_subspace.orbits)

    # check sizes and bits included in each orbit
    for porbit, corbit in zip(potts_subspace.orbits, cluster_subspace.orbits):
        assert len(porbit.site_spaces) == len(corbit.site_spaces)
        assert len(porbit.site_bases) == len(corbit.site_bases)
        assert len(porbit.bit_combos) > len(corbit.bit_combos)

        for i, site_space in enumerate(porbit.site_spaces):
            bits_i = np.concatenate([b[:, i] for b in porbit.bit_combos])
            assert all(j in bits_i for j in site_space.codes)

    # check decorations
    for _ in range(10):
        i = random.choice(range(1, potts_subspace.num_corr_functions))
        o_id = potts_subspace.function_orbit_ids[i]
        orbit = potts_subspace.orbits[o_id - 1]
        fdeco = potts_subspace.get_function_decoration(i)
        odeco = potts_subspace.get_orbit_decorations(o_id)
        assert fdeco == odeco[i - orbit.bit_id]
        assert all(  # all decorations include valid species
            deco[i] in species for deco in fdeco
            for i, species in enumerate(orbit.site_spaces))

    assert_msonable(potts_subspace)


class TestClusterSubSpace(unittest.TestCase):
    def setUp(self) -> None:
        self.lattice = Lattice([[3, 3, 0], [0, 3, 3], [3, 0, 3]])
        self.species = [{'Li+': 0.1, 'Ca+': 0.1}] * 3 + ['Br-']
        self.coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                       (0.5, 0.5, 0.5), (0, 0, 0))
        self.structure = Structure(self.lattice, self.species, self.coords)
        sf = SpacegroupAnalyzer(self.structure)
        self.symops = sf.get_symmetry_operations()
        self.cutoffs = {2: 6, 3: 5}
        self.cs = ClusterSubspace.from_cutoffs(self.structure,
                                               cutoffs=self.cutoffs,
                                               basis='indicator',
                                               orthonormal=False,
                                               supercell_size='volume')
        self.domains = get_allowed_species(self.structure)

    def test_function_hierarchy(self):
        hierarchy = self.cs.function_hierarchy()
        self.assertEqual(sorted(hierarchy[0]), [])
        self.assertEqual(sorted(hierarchy[-1]), [17, 21])
        self.assertEqual(sorted(hierarchy[15]), [])
        self.assertEqual(sorted(hierarchy[35]), [5, 7, 10])
        self.assertEqual(sorted(hierarchy[55]), [6, 8, 13])
        self.assertEqual(sorted(hierarchy[75]), [7, 16, 21])
        self.assertEqual(sorted(hierarchy[95]), [9, 19])
        self.assertEqual(sorted(hierarchy[115]), [13, 19, 21])

    def test_orbit_hierarchy(self):
        hierarchy = self.cs.orbit_hierarchy()
        self.assertEqual(sorted(hierarchy[0]), [])  # empty
        self.assertEqual(sorted(hierarchy[1]), [])  # point
        self.assertEqual(sorted(hierarchy[3]), [1, 2])  # distinct site pair
        self.assertEqual(sorted(hierarchy[4]), [1])  # same site pair
        self.assertEqual(sorted(hierarchy[15]), [3, 5])  # triplet
        self.assertEqual(sorted(hierarchy[-1]), [6, 7])

    def test_numbers(self):
        # Test the total generated orbits, orderings and clusters are
        # as expected.
        self.assertEqual(self.cs.num_orbits, 27)
        self.assertEqual(self.cs.num_corr_functions, 124)
        self.assertEqual(self.cs.num_clusters, 377)

    def test_func_orbit_ids(self):
        self.assertEqual(len(self.cs.function_orbit_ids), 124)
        self.assertEqual(len(set(self.cs.function_orbit_ids)), 27)

    def test_orbits(self):
        self.assertEqual(len(self.cs.orbits) + 1, self.cs.num_orbits)  # +1 for empty cluster
        for o1, o2 in combinations(self.cs.orbits, 2):
            self.assertNotEqual(o1, o2)

    def test_iterorbits(self):
        orbits = [o for o in self.cs.orbits]
        for o1, o2 in zip(orbits, self.cs.orbits):
            self.assertEqual(o1, o2)

    def test_cutoffs(self):
        for s, c in self.cs.cutoffs.items():
            self.assertTrue(self.cutoffs[s] >= c)

    def test_orbits_from_cutoffs(self):
        # Get all of them
        self.assertTrue(
            all(o1 == o2 for o1, o2 in
                zip(self.cs.orbits, self.cs.orbits_from_cutoffs(6))))
        for upper, lower in ((5, 0), (6, 3), (5, 2)):
            orbs = self.cs.orbits_from_cutoffs(upper, lower)
            self.assertTrue(len(orbs) < len(self.cs.orbits))
            self.assertTrue(
                all(lower <= o.base_cluster.diameter <= upper for o in orbs)
            )

        # Test with dict
        upper = {2: 4.5, 3: 3.5}
        orbs = self.cs.orbits_from_cutoffs(upper)
        self.assertTrue(len(orbs) < len(self.cs.orbits))
        self.assertTrue(
            all(o.base_cluster.diameter <= upper[2] for o in orbs
                if len(o.base_cluster) == 2)
        )
        self.assertTrue(
            all(o.base_cluster.diameter <= upper[3] for o in orbs
                if len(o.base_cluster) == 3)
        )

        # Test for only pairs
        upper = {2: 4.5}
        orbs = self.cs.orbits_from_cutoffs(upper)
        self.assertTrue(len(orbs) < len(self.cs.orbits))
        self.assertTrue(
            all(o.base_cluster.diameter <= upper[2] for o in orbs
                if len(o.base_cluster) == 2)
        )
        self.assertTrue(
            all(len(o.base_cluster) == 2 for o in orbs)
        )

        # bad cuttoffs
        self.assertTrue(len(self.cs.orbits_from_cutoffs(2, 4)) == 0)

    def test_functions_inds_by_size(self):
        indices = self.cs.function_inds_by_size
        # check that all orbit functions are in there...
        self.assertTrue(
            sum(len(i) for i in indices.values()) == len(self.cs) - 1)
        fun_orb_ids = self.cs.function_orbit_ids
        # Now check sizes are correct.
        for s, inds in indices.items():
            self.assertTrue(
                all(s == len(self.cs.orbits[fun_orb_ids[i] - 1].base_cluster)
                    for i in inds))

    def test_functions_inds_by_cutoffs(self):
        indices = self.cs.function_inds_from_cutoffs(6)
        # check that all of them are in there.
        self.assertTrue(len(indices) == len(self.cs) - 1)
        fun_orb_ids = self.cs.function_orbit_ids
        for upper, lower in ((4, 0), (5, 3), (3, 1)):
            indices = self.cs.function_inds_from_cutoffs(upper, lower)
            self.assertTrue(len(indices) < len(self.cs))
            self.assertTrue(
                all(lower <= self.cs.orbits[fun_orb_ids[i] - 1].base_cluster.diameter <= upper
                    for i in indices)
            )

    def test_bases_ortho(self):
        # test orthogonality, orthonormality of bases with uniform and
        # concentration measure
        self.assertFalse(self.cs.basis_orthogonal)
        self.assertFalse(self.cs.basis_orthonormal)
        cs = ClusterSubspace.from_cutoffs(self.structure, {2: 6, 3: 5},
                                          basis='Indicator', orthonormal=True)
        self.assertTrue(cs.basis_orthogonal)
        self.assertTrue(cs.basis_orthonormal)
        cs = ClusterSubspace.from_cutoffs(self.structure, {2: 6, 3: 5},
                                          basis='sinusoid')
        self.assertTrue(cs.basis_orthogonal)
        # Not orthonormal w.r.t. to uniform measure...
        self.assertFalse(cs.basis_orthonormal)
        cs = ClusterSubspace.from_cutoffs(self.structure, {2: 6, 3: 5},
                                          basis='sinusoid', orthonormal=True)
        self.assertTrue(cs.basis_orthonormal)
        cs = ClusterSubspace.from_cutoffs(self.structure, {2: 6, 3: 5},
                                          basis='sinusoid',
                                          use_concentration=True)
        # Not orthogonal/normal wrt to concentration measure
        self.assertFalse(cs.basis_orthogonal)
        self.assertFalse(cs.basis_orthonormal)
        cs = ClusterSubspace.from_cutoffs(self.structure, {2: 6, 3: 5},
                                          basis='sinusoid', orthonormal=True,
                                          use_concentration=True)
        self.assertTrue(cs.basis_orthogonal)
        self.assertTrue(cs.basis_orthonormal)
        cs = ClusterSubspace.from_cutoffs(self.structure, {2: 6, 3: 5},
                                          basis='indicator', orthonormal=True,
                                          use_concentration=True)
        self.assertTrue(cs.basis_orthogonal)
        self.assertTrue(cs.basis_orthonormal)

    #  These can probably be improved to check odd and specific cases we want
    #  to watch out for
    def test_supercell_matrix_from_structure(self):
        # Simple scaling
        supercell = self.structure.copy()
        supercell.make_supercell(2)
        sc_matrix = self.cs.scmatrix_from_structure(supercell)
        self.assertAlmostEqual(np.linalg.det(sc_matrix), 8)

        # A more complex supercell_structure
        m = np.array([[ 0,  5,  3],
                      [-2,  0,  2],
                      [-2,  4,  3]])
        supercell = self.structure.copy()
        supercell.make_supercell(m)
        sc_matrix = self.cs.scmatrix_from_structure(supercell)
        self.assertAlmostEqual(np.linalg.det(sc_matrix), abs(np.linalg.det(m)))

        # Test a slightly distorted structure
        lattice = Lattice([[2.95, 3, 0], [0, 3, 2.9], [3, 0, 3]])
        structure = Structure(lattice, self.species, self.coords)
        supercell = structure.copy()
        supercell.make_supercell(2)
        sc_matrix = self.cs.scmatrix_from_structure(supercell)
        self.assertAlmostEqual(np.linalg.det(sc_matrix), 8)

        m = np.array([[0, 5, 3],
                      [-2, 0, 2],
                      [-2, 4, 3]])
        supercell = structure.copy()
        supercell.make_supercell(m)
        sc_matrix = self.cs.scmatrix_from_structure(supercell)
        self.assertAlmostEqual(np.linalg.det(sc_matrix), abs(np.linalg.det(m)))

    def test_refine_structure(self):
        lattice = Lattice([[2.95, 3, 0], [0, 3, 2.9], [3, 0, 3]])
        structure = Structure(lattice, ['Li+', ]*2 + ['Ca+'] + ['Br-'],
                              self.coords)
        structure.make_supercell(2)
        ref_structure = self.cs.refine_structure(structure)
        prim_ref = ref_structure.get_primitive_structure()

        # This is failing in pymatgen 2020.1.10, since lattice matrices are not
        # the same, but still equivalent
        # self.assertEqual(self.lattice, structure.lattice)
        self.assertTrue(np.allclose(self.lattice.parameters,
                                    prim_ref.lattice.parameters))
        self.assertTrue(np.allclose(self.cs.corr_from_structure(structure),
                                    self.cs.corr_from_structure(ref_structure)))

    def test_corr_from_structure(self):
        structure = Structure(self.lattice, ['Li+',] * 2 + ['Ca+'] + ['Br-'],
                              self.coords)
        corr = self.cs.corr_from_structure(structure)
        self.assertEqual(len(corr),
                         self.cs.num_corr_functions + len(self.cs.external_terms))
        self.assertEqual(corr[0], 1)

        cs = ClusterSubspace.from_cutoffs(self.structure, {2: 5},
                                          basis='indicator')

        # make an ordered supercell_structure
        s = self.structure.copy()
        s.make_supercell([2, 1, 1])
        species = ('Li+', 'Ca+', 'Li+', 'Ca+', 'Br-', 'Br-')
        coords = ((0.125, 0.25, 0.25),
                  (0.625, 0.25, 0.25),
                  (0.375, 0.75, 0.75),
                  (0.25, 0.5, 0.5),
                  (0, 0, 0),
                  (0.5, 0, 0))
        s = Structure(s.lattice, species, coords)
        self.assertEqual(len(cs.corr_from_structure(s)), 22)
        expected = [1, 0.5, 0.25, 0, 0.5, 0, 0.375, 0, 0.0625, 0.25, 0.125,
                    0, 0.25, 0.125, 0.125, 0, 0, 0.25, 0, 0.125, 0, 0.1875]
        self.assertTrue(np.allclose(cs.corr_from_structure(s), expected))

        # Test occu_from_structure
        occu = [Vacancy(), Species('Li', 1), Species('Ca', 1),
                Species('Li', 1), Vacancy(), Species('Ca', 1),
                Species('Br', -1), Species('Br', -1)]
        self.assertTrue(all(s1 == s2 for s1, s2
                            in zip(occu, cs.occupancy_from_structure(s))))

        # shuffle sites and check correlation still works
        for _ in range(10):
            random.shuffle(s)
            self.assertTrue(np.allclose(cs.corr_from_structure(s), expected))

    def test_remove_orbits(self):
        cs = ClusterSubspace.from_cutoffs(self.structure, {2: 5},
                                          basis='indicator')
        s = self.structure.copy()
        s.make_supercell([2, 1, 1])
        species = ('Li+', 'Ca+', 'Li+', 'Ca+', 'Br-', 'Br-')
        coords = ((0.125, 0.25, 0.25),
                  (0.625, 0.25, 0.25),
                  (0.375, 0.75, 0.75),
                  (0.25, 0.5, 0.5),
                  (0, 0, 0),
                  (0.5, 0, 0))
        s = Structure(s.lattice, species, coords)
        self.assertRaises(ValueError, cs.remove_orbits, [-1])
        self.assertRaises(ValueError, cs.remove_orbits,
                          [cs.num_orbits + 1])
        self.assertRaises(ValueError, cs.remove_orbits, [0])
        cs.remove_orbits([3, 5, 7])
        expected = [1, 0.5, 0.25, 0, 0.5, 0.25, 0.125, 0, 0, 0, 0.25]
        self.assertEqual(len(cs.corr_from_structure(s)), 11)
        self.assertEqual(cs.num_orbits, 5)
        self.assertEqual(len(set(cs.function_orbit_ids)), 5)
        self.assertTrue(np.allclose(cs.corr_from_structure(s), expected))

    def test_remove_bit_combos(self):
        cs = ClusterSubspace.from_cutoffs(self.structure, {2: 5},
                                          basis='indicator')
        s = self.structure.copy()
        s.make_supercell([2, 1, 1])
        species = ('Li+', 'Ca+', 'Li+', 'Ca+', 'Br-', 'Br-')
        coords = ((0.125, 0.25, 0.25),
                  (0.625, 0.25, 0.25),
                  (0.375, 0.75, 0.75),
                  (0.25, 0.5, 0.5),
                  (0, 0, 0),
                  (0.5, 0, 0))
        s = Structure(s.lattice, species, coords)
        remove = [9, 10, 18] #{4: [[0, 0], [0, 1]], 7: [[0, 0]]}
        new_n_orderings = cs.num_corr_functions - len(remove)

        cs.remove_orbit_bit_combos(remove)
        self.assertEqual(cs.num_corr_functions, new_n_orderings)
        expected = [1, 0.5, 0.25, 0, 0.5, 0, 0.375, 0, 0.0625,
                    0, 0.25, 0.125, 0.125, 0, 0, 0.25, 0.125, 0, 0.1875]
        self.assertTrue(np.allclose(cs.corr_from_structure(s), expected))
        self.assertWarns(UserWarning, cs.remove_orbit_bit_combos, [9])

    def test_orbit_mappings_from_matrix(self):
        # check that all supercell_structure index groups map to the correct
        # primitive cell sites, and check that max distance under supercell
        # structure pbc is less than the max distance without pbc

        m = np.array([[2, 0, 0], [0, 2, 0], [0, 1, 1]])
        supercell_struct = self.structure.copy()
        supercell_struct.make_supercell(m)
        fcoords = np.array(supercell_struct.frac_coords)

        for orb, inds in zip(
                self.cs.orbits, self.cs.supercell_orbit_mappings(m)):
            for x in inds:
                pbc_radius = np.max(supercell_struct.lattice.get_all_distances(
                    fcoords[x], fcoords[x]))
                # primitive cell fractional coordinates
                new_fc = np.dot(fcoords[x], m)
                self.assertGreater(
                    orb.base_cluster.diameter + 1e-7, pbc_radius)
                found = False
                for equiv in orb.clusters:
                    if is_coord_subset_pbc(equiv.sites, new_fc, atol=SITE_TOL):
                        found = True
                        break
                self.assertTrue(found)

        # check that the matrix was cached
        m_hash = tuple(sorted(tuple(s) for s in m))
        self.assertTrue(self.cs._supercell_orb_inds[m_hash] is
                        self.cs.supercell_orbit_mappings(m))

    def test_periodicity(self):
        # Check to see if a supercell of a smaller structure gives the same corr
        m = np.array([[2, 0, 0], [0, 2, 0], [0, 1, 1]])
        supercell = self.structure.copy()
        supercell.make_supercell(m)
        s = Structure(supercell.lattice,
                      ['Ca+', 'Li+', 'Li+', 'Br-', 'Br-', 'Br-', 'Br-'],
                      [[0.125, 1, 0.25], [0.125, 0.5, 0.25],
                       [0.375, 0.5, 0.75], [0, 0, 0], [0, 0.5, 1],
                       [0.5, 1, 0], [0.5, 0.5, 0]])
        cs = ClusterSubspace.from_cutoffs(self.structure, {2: 6, 3: 5},
                                          basis='indicator',
                                          supercell_size='volume')
        a = cs.corr_from_structure(s)
        s.make_supercell([2, 1, 1])
        b = cs.corr_from_structure(s)
        self.assertTrue(np.allclose(a, b))

    @staticmethod
    def _encode_occu(occu, bits):
        return np.array([bit.index(sp) for sp, bit in zip(occu, bits)])

    def test_vs_CASM_pairs(self):
        species = [{'Li+':0.1}] * 3 + ['Br-']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                  (0.5, 0.5, 0.5),  (0, 0, 0))
        structure = Structure(self.lattice, species, coords)
        cs = ClusterSubspace.from_cutoffs(structure, {2: 6},
                                          basis='indicator')
        bits = get_allowed_species(structure)
        m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        orbit_list = [
            (orb.bit_id, orb.bit_combo_array, orb.bit_combo_inds,
             orb.bases_array, inds)
            for orb, inds in zip(cs.orbits, cs.supercell_orbit_mappings(m))]

        # last two clusters are switched from CASM output (occupancy basis)
        # all_li (ignore casm point term)
        occu = self._encode_occu([Species('Li', 1), Species('Li', 1),
                                  Species('Li', 1)], bits)
        corr = corr_from_occupancy(occu, cs.num_corr_functions, orbit_list)
        self.assertTrue(np.allclose(corr, np.array([1]*12)))

        # all_vacancy
        occu = self._encode_occu([Vacancy(),
                                  Vacancy(),
                                  Vacancy()], bits)
        corr = corr_from_occupancy(occu, cs.num_corr_functions, orbit_list)
        self.assertTrue(np.allclose(corr,
                                    np.array([1]+[0]*11)))
        # octahedral
        occu = self._encode_occu([Vacancy(),
                                  Vacancy(),
                                  Species('Li', 1)], bits)
        corr = corr_from_occupancy(occu, cs.num_corr_functions, orbit_list)
        self.assertTrue(np.allclose(corr,
                                    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]))
        # tetrahedral
        occu = self._encode_occu([Species('Li', 1), Species('Li', 1),
                                  Vacancy()], bits)
        corr = corr_from_occupancy(occu, cs.num_corr_functions, orbit_list)
        self.assertTrue(np.allclose(corr,
                                    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0]))
        # mixed
        occu = self._encode_occu([Species('Li', 1), Vacancy(),
                                  Species('Li', 1)], bits)
        corr = corr_from_occupancy(occu, cs.num_corr_functions, orbit_list)
        self.assertTrue(np.allclose(corr,
                                    [1, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0, 0.5, 1]))
        # single_tet
        occu = self._encode_occu([Species('Li', 1), Vacancy(),
                                  Vacancy()], bits)
        corr = corr_from_occupancy(occu, cs.num_corr_functions, orbit_list)
        self.assertTrue(np.allclose(corr,
                                    [1, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0]))

    def test_vs_CASM_triplets(self):
        """
        Test vs casm generated correlation with occupancy basis.
        """
        species = [{'Li+': 0.1}] * 3 + ['Br-']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
                  (0.5, 0.5, 0.5), (0, 0, 0))
        structure = Structure(self.lattice, species, coords)
        cs = ClusterSubspace.from_cutoffs(structure, {2: 6, 3: 4.5},
                                          basis='indicator')
        spaces = get_allowed_species(structure)
        m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        orbit_list = [
            (orb.bit_id, orb.bit_combo_array, orb.bit_combo_inds,
             orb.bases_array, inds)
            for orb, inds in zip(cs.orbits, cs.supercell_orbit_mappings(m))]
        # last two pair terms are switched from CASM output (occupancy basis)
        # all_vacancy (ignore casm point term)
        occu = self._encode_occu([Vacancy(),
                                  Vacancy(),
                                  Vacancy()], spaces)
        corr = corr_from_occupancy(occu, cs.num_corr_functions, orbit_list)
        self.assertTrue(np.allclose(corr, np.array([1] + [0] * 18)))
        # all_li
        occu = self._encode_occu([Species('Li', 1), Species('Li', 1),
                                  Species('Li', 1)], spaces)
        corr = corr_from_occupancy(occu, cs.num_corr_functions, orbit_list)
        self.assertTrue(np.allclose(corr, np.array([1] * 19)))
        # octahedral
        occu = self._encode_occu([Vacancy(),
                                  Vacancy(), Species('Li', 1)],
                                 spaces)
        corr = corr_from_occupancy(occu, cs.num_corr_functions, orbit_list)
        self.assertTrue(np.allclose(corr,
                                    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                     0, 1, 0, 0, 0, 0, 0, 0, 1]))

        # tetrahedral
        occu = self._encode_occu([Species('Li', 1), Species('Li', 1),
                                  Vacancy()], spaces)
        corr = corr_from_occupancy(occu, cs.num_corr_functions, orbit_list)
        self.assertTrue(np.allclose(corr,
                                    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
                                     0, 0, 1, 0, 0, 1, 1, 0]))
        # mixed
        occu = self._encode_occu([Species('Li', 1), Vacancy(),
                                  Species('Li', 1)], spaces)
        corr = corr_from_occupancy(occu, cs.num_corr_functions, orbit_list)
        self.assertTrue(np.allclose(corr,
                                    [1, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0, 0,
                                     0.5, 1, 0, 0, 0.5, 0.5, 0.5, 0.5, 1]))
        # single_tet
        occu = self._encode_occu([Species('Li', 1), Vacancy(),
                                  Vacancy()], spaces)
        corr = corr_from_occupancy(occu, cs.num_corr_functions, orbit_list)
        self.assertTrue(np.allclose(corr,
                                    [1, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0,
                                     0.5, 0, 0, 0, 0, 0, 0.5, 0.5, 0]))

    def test_vs_CASM_multicomp(self):
        cs = ClusterSubspace.from_cutoffs(self.structure, {2: 5},
                                          basis='indicator')
        m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        orbit_list = [
            (orb.bit_id, orb.bit_combo_array, orb.bit_combo_inds,
             orb.bases_array, inds)
            for orb, inds in zip(cs.orbits, cs.supercell_orbit_mappings(m))]
        # mixed
        occu = self._encode_occu([Vacancy(), Species('Li', 1),
                                  Species('Li', 1)], self.domains)
        corr = corr_from_occupancy(occu, cs.num_corr_functions, orbit_list)
        self.assertTrue(np.allclose(corr,
                                    [1, 0.5, 0, 1, 0, 0.5, 0, 0, 0, 0, 0,
                                     0, 0.5, 0, 0, 1, 0, 0, 0.5, 0, 0, 0]))
        # Li_tet_ca_oct
        occu = self._encode_occu([Vacancy(), Species('Li', 1),
                                  Species('Ca', 1)], self.domains)
        corr = corr_from_occupancy(occu, cs.num_corr_functions, orbit_list)
        self.assertTrue(np.allclose(corr,
                                    [1, 0.5, 0, 0, 1, 0, 0.5, 0, 0, 0, 0,
                                     0, 0.5, 0, 0, 0, 0, 1, 0, 0.5, 0, 0]))

    def test_copy(self):
        cs = self.cs.copy()
        self.assertFalse(cs is self.cs)
        self.assertTrue(isinstance(cs, ClusterSubspace))

    def test_change_basis(self):
        # make an ordered supercell_structure
        s = self.structure.copy()
        s.make_supercell([2, 1, 1])
        species = ('Li+', 'Ca+', 'Li+', 'Ca+', 'Br-', 'Br-')
        coords = ((0.125, 0.25, 0.25),
                  (0.625, 0.25, 0.25),
                  (0.375, 0.75, 0.75),
                  (0.25, 0.5, 0.5),
                  (0, 0, 0),
                  (0.5, 0, 0))
        s = Structure(s.lattice, species, coords)

        cs = self.cs.copy()
        for basis in ('sinusoid', 'chebyshev', 'legendre'):
            cs.change_site_bases(basis)
            self.assertFalse(np.allclose(cs.corr_from_structure(s),
                                         self.cs.corr_from_structure(s)))
        cs.change_site_bases('indicator', orthonormal=True)
        self.assertFalse(np.allclose(cs.corr_from_structure(s),
                                     self.cs.corr_from_structure(s)))
        self.assertTrue(cs.basis_orthogonal)
        self.assertTrue(cs.basis_orthonormal)
        cs.change_site_bases('indicator', orthonormal=False)
        self.assertTrue(np.allclose(cs.corr_from_structure(s),
                                    self.cs.corr_from_structure(s)))
        self.assertFalse(cs.basis_orthogonal)
        self.assertFalse(cs.basis_orthonormal)

    def test_exceptions(self):
        self.assertRaises(NotImplementedError, ClusterSubspace.from_cutoffs,
                          self.structure, {2: 5}, basis='blobs')
        cs = ClusterSubspace.from_cutoffs(self.structure, {2: 5})
        s = self.structure.copy()
        s.make_supercell([2, 1, 1])
        species = ('X', 'Ca+', 'Li+', 'Ca+', 'Br-', 'Br-')
        coords = ((0.125, 0.25, 0.25),
                  (0.625, 0.25, 0.25),
                  (0.375, 0.75, 0.75),
                  (0.25, 0.5, 0.5),
                  (0, 0, 0),
                  (0.5, 0, 0))
        s = Structure(s.lattice, species, coords)
        self.assertRaises(StructureMatchError, cs.corr_from_structure, s)

    def test_repr(self):
        repr(self.cs)

    def test_str(self):
        str(self.cs)

    def test_msonable(self):
        # get corr for a few supercells to cache their orbit indices
        struct = Structure(self.lattice,
                           ['Li+', ] * 2 + ['Ca+'] + ['Br-'],
                           self.coords)
        struct1 = struct.copy()
        struct1.make_supercell(2)
        struct2 = struct1.copy()
        # TODO all symops after 1 make finding supercell step fail this needs
        #  to be checked!
        struct2.apply_operation(self.cs.symops[1])
        for s in (struct, struct1, struct2):  # run this to cache orb indices
            _ = self.cs.corr_from_structure(s)
        self.assertNotEqual(len(self.cs._supercell_orb_inds), 0)

        d = self.cs.as_dict()
        cs = ClusterSubspace.from_dict(d)
        self.assertEqual(cs.as_dict(), d)
        self.assertEqual(cs.num_orbits, 27)
        self.assertEqual(cs.num_corr_functions, 124)
        self.assertEqual(cs.num_clusters, 377)
        self.assertEqual(str(cs), str(self.cs))
        # checked that the cached orbit index mappings where properly kept
        for scm, orb_inds in cs._supercell_orb_inds.items():
            self.assertTrue(scm in self.cs._supercell_orb_inds)
            for orb_inds1, orb_inds2 in zip(
                    orb_inds, self.cs._supercell_orb_inds[scm]):
                self.assertTrue(np.array_equal(orb_inds1, orb_inds2))
        self.assertTrue(np.array_equal(self.cs.corr_from_structure(struct2),
                                       cs.corr_from_structure(struct2)))
        # Check orthonormalization is kept
        self.cs.change_site_bases('indicator', orthonormal=True)
        d = self.cs.as_dict()
        cs = ClusterSubspace.from_dict(d)
        self.assertEqual(cs.as_dict(), d)
        self.assertTrue(cs.basis_orthonormal)
        self.assertTrue(np.array_equal(self.cs.corr_from_structure(struct2),
                                       cs.corr_from_structure(struct2)))

        # Check external terms are kept
        self.cs.add_external_term(EwaldTerm(eta=3))
        d = self.cs.as_dict()
        cs = ClusterSubspace.from_dict(d)
        self.assertEqual(cs.as_dict(), d)
        self.assertEqual(len(cs.external_terms), 1)
        self.assertTrue(isinstance(cs.external_terms[0], EwaldTerm))
        self.assertEqual(cs.external_terms[0].eta, 3)
        module = d['external_terms'][0]['@module']
        d['external_terms'][0]['@module'] = 'smol.blab'
        self.assertWarns(ImportWarning, ClusterSubspace.from_dict, d)
        d['external_terms'][0]['@module'] = module
        d['external_terms'][0]['@class'] = 'Blab'
        self.assertWarns(RuntimeWarning, ClusterSubspace.from_dict, d)
        # test if serializable and correctly instantiates for all basis
        for basis in ('indicator', 'sinusoid', 'chebyshev', 'legendre', 'polynomial'):
            cs.change_site_bases(basis)
            d = cs.as_dict()
            self.assertEqual(d, ClusterSubspace.from_dict(d).as_dict())
            j = json.dumps(d)
            json.loads(j)
