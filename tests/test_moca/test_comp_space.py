import unittest
import numpy as np

from smol.moca.comp_space import CompSpace
from pymatgen import Specie,Element,Composition
from smol.cofe.space.domain import Vacancy

vertices_contain = lambda a,b: np.all([np.any(np.all(np.isclose(a_row,b),axis=1)) for a_row in a])
vertices_equal = lambda a,b: vertices_contain(a,b) and vertices_contain(b,a)

#Charged test case, without vacancy
class TestCompSpace1(unittest.TestCase):
    def setUp(self) -> None:
        li = Specie.from_string('Li+')
        mn = Specie.from_string('Mn3+')
        ti = Specie.from_string('Ti4+')
        o = Specie.from_string('O2-')
        p = Specie.from_string('P3-')
        
        self.bits = [[li,mn,ti],[p,o]]
        self.nbits = [[0,1,2],[0,1]]
        self.unit_n_swps = [(0,2,0),(1,2,0),(0,1,1)]
        self.chg_of_swps = [-3,-1,-1]
        self.swp_ids_in_sublat = [[0,1],[2]]
        
        op1 = {'from':{0:{0:1, 2:1}, 1:{1:1}}, \
               'to':{0:{1:2}, 1:{0:1}}}
        op2 = {'from':{0:{1:1}, 1:{1:1}}, \
               'to':{0:{2:1}, 1:{0:1}}}
        self.flip_table = [op1,op2]
        
        visualized_operation_1 = \
        '1 Li+(0) + 1 Ti4+(0) + 1 O2-(1) -> 2 Mn3+(0) + 1 P3-(1)'
        visualized_operation_2 = \
        '1 Mn3+(0) + 1 O2-(1) -> 1 Ti4+(0) + 1 P3-(1)'
        self.visualized_operations = visualized_operation_1 + \
                                    '\n'+ \
                                    visualized_operation_2

        self.A = np.array([[1,1],[-1,1],[0,1],[-1,-2],[1,-1]])
        self.b = np.array([1/3, 1, 2/3, 0, 0])
        self.R = np.array([[0,1,-1],[-1,2,1],[-3,-1,-1]])
        self.t = np.array([2/3,0,0])

        self.vertices = np.array([[1/3,0,1],[0,1,1],\
                                  [1/2,1/2,0],[2/3,0,0]])

        self.min_grid = np.array([[2,0,6],[1,3,6],[0,6,6],\
                                  [2,1,5],[1,4,5],\
                                  [2,2,4],[1,5,4],\
                                  [3,0,3],[2,3,3],\
                                  [3,1,2],[2,4,2],\
                                  [3,2,1],\
                                  [4,0,0],[3,3,0]])
        self.int_grids_5 = np.array([[3,1,0],[3,0,1],[2,3,1],[2,2,2],\
                                    [2,1,3],[1,4,3],[2,0,4],[1,3,4],\
                                    [1,2,5],[0,5,5]])

        self.comp_space = CompSpace(self.bits)

    def test_swps(self):
        self.assertEqual(self.comp_space.unit_n_swps,self.unit_n_swps)
        self.assertEqual(self.comp_space.chg_of_swps,self.chg_of_swps)
        self.assertEqual(self.comp_space.swp_ids_in_sublat,self.swp_ids_in_sublat)

    def test_space_specs(self):
        self.assertEqual(self.comp_space.bkgrnd_chg,2)
        self.assertEqual(self.comp_space.unconstr_dim,3)
        self.assertEqual(self.comp_space.is_charge_constred,True)
        self.assertEqual(self.comp_space.dim,2)
        self.assertEqual(self.comp_space.dim_nondisc,5)

    def test_vertices(self):
        self.assertTrue(vertices_equal(self.comp_space.unit_spc_vertices(),\
                                        self.vertices))

    def test_random_point(self):
        rand_ucoord = self.comp_space.get_random_point_in_unit_spc()
        self.assertTrue(self.comp_space._is_in_subspace(rand_ucoord))

    def test_integer_grids(self):
        self.assertEqual(self.comp_space.min_sc_size,6)
        self.assertTrue(vertices_equal(self.comp_space.min_grid,self.min_grid))
        self.assertTrue(vertices_equal(self.comp_space.int_grids(sc_size=5),\
                                       self.int_grids_5))

    def test_format_translate(self):
        sc_size = 6
        ucoord = [3,1,2]
        comp = [Composition({'Li+':1/2,'Mn3+':1/6,'Ti4+':1/3}),\
                Composition({'P3-':1/3,'O2-':2/3})]
        compstat = [[3,1,2],[2,4]]

        ccoord_t = self.comp_space.translate_format(ucoord,from_format='unconstr',to_format='constr',sc_size=6)
        ucoord_t1 = self.comp_space.translate_format(ccoord_t,from_format='constr',to_format='unconstr',sc_size=6)

        comp_t = self.comp_space.translate_format(ucoord,from_format='unconstr',to_format='composition',sc_size=6)
        ucoord_t2 = self.comp_space.translate_format(comp_t,from_format='composition',to_format='unconstr',sc_size=6)

        compstat_t = self.comp_space.translate_format(ucoord,from_format='unconstr',to_format='compstat',sc_size=6)
        ucoord_t3 = self.comp_space.translate_format(comp_t,from_format='compstat',to_format='unconstr',sc_size=6)

        self.assertTrue(np.allclose(ucoord,ucoord_t1))
        self.assertTrue(np.allclose(ucoord,ucoord_t2))
        self.assertTrue(np.allclose(ucoord,ucoord_t3))
        self.assertEqual(comp,comp_t)
        self.assertEqual(compstat,compstat_t)

#Uncharged, with vacancy.
class TestCompSpace2(unittest.TestCase):
    def setUp(self) -> None:
        li = Element('Li')
        ag = Element('Ag')
        vac = Vacancy()
        
        self.bits = [[li,ag,vac]]
        self.nbits = [[0,1,2]]
        self.unit_n_swps = [(0,2,0),(1,2,0)]
        self.chg_of_swps = [0,0]
        self.swp_ids_in_sublat = [[0,1]]

        op1 = {'from':{0:{2:1}}, \
               'to':{0:{0:1}}}
        op2 = {'from':{0:{2:1}}, \
               'to':{0:{1:1}}}
        self.flip_table = [op1,op2]
        self.comp_space = CompSpace(self.bits)

    def test_swps(self):
        self.assertEqual(self.comp_space.unit_n_swps,self.unit_n_swps)
        self.assertEqual(self.comp_space.chg_of_swps,self.chg_of_swps)
        self.assertEqual(self.comp_space.swp_ids_in_sublat,self.swp_ids_in_sublat)

    def test_space_specs(self):
        self.assertEqual(self.comp_space.bkgrnd_chg,0)
        self.assertEqual(self.comp_space.unconstr_dim,2)
        self.assertEqual(self.comp_space.is_charge_constred,False)
        self.assertEqual(self.comp_space.dim,2)
        self.assertEqual(self.comp_space.dim_nondisc,3)
