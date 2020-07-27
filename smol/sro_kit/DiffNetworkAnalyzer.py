"""
Implementation of analysis module for diffusion network analysis

Currently 0tm percolation analysis is implemented

When you generate results using this code, you may consider citing the following papers:
1. Adv. Energy Mater. 2020, 1903240
2. Chem 6, 1–16,2020
"""

__author__ = "Bin Ouyang"
__date__ = "2020.07.25"
__version__ = "0.1"

from copy import deepcopy
from itertools import combinations as comb
from itertools import combinations_with_replacement as cwr
from itertools import permutations as perm
from pymatgen.util.coord import find_in_coord_list_pbc
from pymatgen.core.structure import Structure
from smol.sro_kit.ionicstructure import PoscarWithChgDecorator
import time
import numpy as np


class PercolationAnalyzer(object):
    """
    Perform 0tm percolation analysis with Dijkstra algorithm

    """

    def __init__(self, structure, diffusor, tm_symbols, d_cut = 2.97, d_tol = 0.3):
        """
        Initialize the class using ChargedStructure defined in ionicstructure.ChargedStructure

        Args:
            structure: Charged structure defined in ionicstructure.ChargedStructure
            diffusor: The Li/Na to be analyzed for 0tm percolation
            tm_symbols: The symbols of other sites rather than diffusor, with charge decorator
            d_cut: The cut off radius for 0tm unit
            d_tol: The tolerance of bond length
        """
        self.structure = deepcopy(structure)
        self.diffusor = diffusor
        self.tm_symbols = tm_symbols
        self.d_max = d_cut + d_tol
        self.d_min = d_cut - d_tol
        if str(self.structure[0].specie) != self.diffusor:
            print('Reorganize the site so that diffsuor comes the first, this can speed up the algorithm  a lot')
            diff_inds = self.structure.get_ion_indices(self.diffusor)
            tm_inds = []
            for tm_sym in tm_symbols:
                tm_inds.extend(list(self.structure.get_ion_indices(tm_sym)))
            anion_inds = []
            for anion_sym in self.structure.anion_symbols:
                anion_inds.extend(list(self.structure.get_ion_indices(anion_sym)))

            diff_sites = [self.structure[ind] for ind in diff_inds]
            tm_sites = [self.structure[ind] for ind in tm_inds]
            anion_sites = [self.structure[ind] for ind in anion_inds]
            self.structure = Structure.from_sites(diff_sites + tm_sites + anion_sites)


    def get_percolating_Li(self, allow_step=0):
        """
        Get list of percolation diffusor using the Dijkstra's algorithm

        Notes:
            Check the percolation in all three periodic boundary directions

        Args:
            allow_step: The tolerance of number of 1tm jumps
        """
        start_time = time.time()

        # Enumerate all the diffusor sites to check if it is percolating
        scmat0 = np.eye(3)
        perco_diffusor_lsts, perco_diff_paths_lst = [[] for _ in range(3)], [[] for _ in range(3)]

        # Check percolation in all three dimension
        for dim in range(3):
            perco_diffusor_lst, perco_diff_paths = [], []
            scmat = deepcopy(scmat0)
            scmat[dim, dim] = 2.0
            sc_structure = deepcopy(self.structure)
            sc_structure.make_supercell(scmat)
            site_maps = self.map_sites(sc_structure, scmat)

            sc_diff_inds = sc_structure.get_ion_indices(self.diffusor)
            equal_site_lst = []
            for ind1, ind2 in site_maps:
                equal_site_lst.append((ind1, ind2))
            nn_tm_chandict = self.get_nn_tm_chandict(sc_structure)
            for diff_ind, diff_im_ind in equal_site_lst:
                min_step, perco_path = self.dijkstra_distance(nn_tm_chandict, sc_diff_inds, diff_ind, diff_im_ind)
                if min_step <= allow_step:
                    perco_diffusor_lst.append(diff_ind)
                    perco_diff_paths.append(perco_path)

            perco_diffusor_lsts.append(perco_diffusor_lst)
            perco_diff_paths_lst.append(perco_diff_paths)

        print("--- percolation analysis takes %s seconds ---" % (time.time() - start_time))
        return perco_diffusor_lst

    def get_percolating_Li_fast(self, allow_step=0):
        """
        Get list of percolation diffusor using the Dijkstra's algorithm (Fast version)

        Note:
            In this version, both the percolating path and the percolating dimensionality will not be tracked

        Args:
            allow_step: The tolerance of number of 1tm jumps
        """
        start_time = time.time()

        # Enumerate all the diffusor sites to check if it is percolating
        scmat0 = np.eye(3)
        perco_diffusor_lst = []
        # Check percolation in all three dimension
        for dim in range(3):
            scmat = deepcopy(scmat0)
            scmat[dim,dim] = 2.0
            sc_structure = deepcopy(self.structure)
            sc_structure.make_supercell(scmat)
            site_maps = self.map_sites(sc_structure,scmat)

            sc_diff_inds = sc_structure.get_ion_indices(self.diffusor)
            equal_site_lst = []
            for ind1, ind2 in site_maps:
                if (ind1 in perco_diffusor_lst) or (ind2 in perco_diffusor_lst):
                    continue
                if ind1 not in sc_diff_inds:
                    continue
                equal_site_lst.append((ind1, ind2))
            nn_tm_chandict = self.get_nn_tm_chandict(sc_structure)
            for diff_ind, diff_im_ind in equal_site_lst:
                min_step = self.dijkstra_distance_fast(nn_tm_chandict, sc_diff_inds, diff_ind, diff_im_ind)
                if min_step <= allow_step:
                    perco_diffusor_lst.append(diff_ind)

        print("--- percolation analysis takes %s seconds ---" % (time.time() - start_time))
        return perco_diffusor_lst

    def map_sites(self, sc_structure, scmat):
        """
        Get the mapping between supercell and original structure

        Notes:
            Can possibly be optimized with pymatmen structure_matcher, however the current
            get_mapping method in structure_matcher cannot support supercell

        Args:
            sc_structure: The structure supercell
            sc_mat: The supercell matrix
        """
        sc_frac_coords = sc_structure.frac_coords
        frac_coords = self.structure.frac_coords
        scaled_frac_coords = np.dot(sc_frac_coords,scmat)
        site_maps = []
        for i, coord in enumerate(frac_coords):
            inds = find_in_coord_list_pbc(scaled_frac_coords, coord)
            site_maps.append(tuple(inds))

        return site_maps

    def get_nn_tm_chandict(self, sc_structure):
        """
        Form a dictionary with first key being the diffusor indice while the second key
        being the indice of neighboring diffusor and the tm distance. The tm distance is
        defined by dist_table dictionary, the default values are:

        0tm: 0; 1tm: 1, 2tm: 1e2

        Please be noted that the above values are arbitrary as we assume Li would only
        diffuse through 0tm and 1tm channels, 3tm and 4tm does not apply here as well

        return:
            nn_tm_chandict:
                    nn_tm_chandict[key1][key2] will return the tm distance between two
                    neighboring diffusor sites with indices being key1 and key2
        """
        start_time = time.time()
        dist_table = {0:0, 1:1, 2:1e2}
        diffinds = sc_structure.get_ion_indices(self.diffusor)
        tminds = []
        for tm_symbol in self.tm_symbols:
            tminds.extend(sc_structure.get_ion_indices(tm_symbol))
        tm_chan_dict, dist_matrix, nn_inds_lst = \
            self.__get_all_chans(sc_structure, diffinds, tminds)

        # Initialize everything assuming they are 2tm channel, 3tm adn 4tm does not apply here
        nn_tm_chandict = {}

        for ind1 in diffinds:
            nn_tm_chandict[ind1] = {}
            for [ind2, NTM] in nn_inds_lst[ind1]:
                if NTM == 1:
                    continue
                tm_nums = [chan[3] for chan in tm_chan_dict[ind1] if ind2 in chan[:3]]
                # print(ind1,ind2,NTM,tm_nums)
                nn_tm_chandict[ind1][ind2] = dist_table[np.min(tm_nums)]
        print('{}s spent on get_nn_tm_chandict'.format(time.time()-start_time))

        return nn_tm_chandict

    def __get_all_chans_slow(self, structure, diffinds, tminds):
        """
        Get all the tetrahedrons in a given structure

        Args:
            structure: The structure to be analyzed
            diffinds: The diffusor indices
            tminds: The indices of other cations

        return:
            tm_chan_lst: A list of tm channels, each member being a list of site indices
            tm_chan_type_inds: A list of tm channel types, indexed the same way as tm_chan_lst
            tm_type_ind_dict: A dictionary taking type as key and values being a list ot tm channels
            tm_num_lst: A list of tm num for each of the tm channels
            tm_tet_types: A list of the symbol list for each type of tet
        """
        start_time = time.time()
        cation_syms = [self.diffusor] + self.tm_symbols
        tm_tet_types = list(cwr(cation_syms,4))
        tm_tet_types = [sorted(list(item)) for item in tm_tet_types]
        tm_tet_type_inds = [sorted([cation_syms.index(sym) for sym in tet]) for tet in tm_tet_types]
        tm_nums = [4-tm_syms.count(self.diffusor) for tm_syms in tm_tet_types]
        cation_inds = diffinds + tminds
        dist_matrix = structure.distance_matrix
        tm_chan_lst, tm_chan_type_inds, tm_num_lst = [], [], []

        # 1. Enumerate all 4 site combination to check if they are tetrahedron
        # 2. Decide the type of the tetrahedrons
        for (ind1,ind2,ind3,ind4) in cwr(cation_inds,4):
            is_tet = True
            for (i1, i2) in cwr([ind1,ind2,ind3,ind4],2):
                if dist_matrix[i1,i2] > self.d_max:
                    is_tet = False
                    break
            if not is_tet:
                continue
            sym1, sym2, sym3, sym4 = str(structure[ind1].specie), str(structure[ind2].specie),\
                                     str(structure[ind3].specie), str(structure[ind4].specie)
            symlst = [sym1,sym2,sym3,sym4]
            tm_type_ind = tm_tet_types.index(sorted(symlst))
            tm_chan_lst.append(sorted([ind1, ind2, ind3, ind4]))
            tm_chan_type_inds.append(tm_type_ind)
            tm_num_lst.append(tm_nums[tm_type_ind])

        tm_type_ind_dict = {}
        for type_ind in range(len(tm_tet_type_inds)):
            tm_type_ind_dict[type_ind] = \
                [i for i, chan_type_ind in enumerate(tm_chan_type_inds) if chan_type_ind==type_ind]
        print('{}s spent on __get_all_chans'.format(time.time()-start_time))

        return tm_chan_lst, tm_chan_type_inds, tm_type_ind_dict, tm_tet_types, tm_num_lst, dist_matrix

    def __get_all_chans(self, structure, diffinds, tminds):
        """
        Get all the tetrahedrons in a given structure

        Args:
            structure: The structure to be analyzed
            diffinds: The diffusor indices
            tminds: The indices of other cations

        return:
            tm_chan_lst: A list of tm channels, each member being a list of site indices
            tm_chan_type_inds: A list of tm channel types, indexed the same way as tm_chan_lst
            tm_type_ind_dict: A dictionary taking type as key and values being a list ot tm channels
            tm_num_lst: A list of tm num for each of the tm channels
            tm_tet_types: A list of the symbol list for each type of tet
        """
        cation_inds = diffinds + tminds
        dist_matrix = structure.distance_matrix
        tm_chan_dict = {}
        nn_inds_lst = []
        nn_lsts = structure.get_all_neighbors(self.d_max, include_index=True)

        # 1. Enumerate all 4 site combination to check if they are tetrahedron
        # 2. Decide the type of the tetrahedrons
        for ind1 in cation_inds:
            NCount = 0 if ind1 in diffinds else 1
            tm_chan_dict[ind1] = []
            nn_inds = []
            for (_, _, index, _) in nn_lsts[ind1]:
                if index in tminds: 
                    nn_inds.append([index,1])
                elif index in diffinds:
                    nn_inds.append([index,0])
            nn_inds_lst.append(nn_inds)
            for ((ind2, NLi2), (ind3, NLi3), (ind4, NLi4)) in comb(nn_inds,3):
                is_tet = True
                for (i1, i2) in cwr((ind2, ind3, ind4), 2):
                    if dist_matrix[i1, i2] > self.d_max:
                        is_tet = False
                        break
                if not is_tet:
                    continue
                tm_chan_dict[ind1].append([ind2, ind3, ind4, NCount+NLi2+NLi3+NLi4])

        return tm_chan_dict, dist_matrix, nn_inds_lst

    def dijkstra_distance(self, nn_tm_chandict, sc_diff_inds, diff_ind, diff_im_ind):
        """
        Searching for the minimum distance between Li with its image using dijkstra algorithm

        Notes:
            This method would keep track of the percolating pathway. If you want to faster verison,
            you should use dijkstra_distance_fast as that method only track the diffusion distance

        Args:
            nn_tm_chandict: Dictionary of d_min for any diffusor pairs
            sc_diff_inds: All diffusor indices in the supercell
            diff_ind, diff_im_ind: The diffusor index and the corresponding image index

        Return:
            perco_dist[fin_ind]: Minimum distance between diffusor index and the corresponding image index
        """
        flag_arry = np.ones(len(sc_diff_inds))
        perco_dist = np.full(len(sc_diff_inds),1e100)   # Distance initialized as a very large number
        perco_dist[diff_ind] = 0
        # perco_dist[sc_diff_inds.index(diff_ind)] = 0  # The distance between site diff_ind with itself is 0
        init_ind, fin_ind = diff_ind, diff_im_ind
        # i_fin_ind = sc_diff_inds.index(fin_ind)
        perco_path = {ind: [(init_ind, 0)] for ind in sc_diff_inds}

        while True:
            if init_ind == fin_ind:
                # return perco_dist[i_fin_ind], perco_path[fin_ind]
                return perco_dist[fin_ind], perco_path[fin_ind]
            # i_init_ind = sc_diff_inds.index(init_ind)
            # flag_arry[i_init_ind] = 0
            flag_arry[init_ind] = 0
            nn_lst = list(nn_tm_chandict[init_ind].keys())
            for nn_ind in nn_lst:    # Enumerate all the nn of init_ind
                if not flag_arry[nn_ind]:
                    continue
                # i_nn_ind = sc_diff_inds.index(nn_ind)
                # if perco_dist[i_init_ind] + nn_tm_chandict[init_ind][nn_ind] < perco_dist[i_nn_ind]:
                #     perco_dist[i_nn_ind] = perco_dist[i_init_ind] + nn_tm_chandict[init_ind][nn_ind]
                #     perco_path[nn_ind] = perco_path[init_ind] + [(nn_ind, nn_tm_chandict[init_ind][nn_ind])]
                if perco_dist[init_ind] + nn_tm_chandict[init_ind][nn_ind] < perco_dist[nn_ind]:
                    perco_dist[nn_ind] = perco_dist[init_ind] + nn_tm_chandict[init_ind][nn_ind]
                    perco_path[nn_ind] = perco_path[init_ind] + [(nn_ind, nn_tm_chandict[init_ind][nn_ind])]

            # For the rest of diffusor sites, remove the one with shortest distance that are still tracked
            i_next = perco_dist[flag_arry.nonzero()].argmin()
            # init_ind = sc_diff_inds[np.where(flag_arry == 1)[0][i_next]]
            init_ind = np.where(flag_arry == 1)[0][i_next]

    def dijkstra_distance_fast(self, nn_tm_chandict, sc_diff_inds, diff_ind, diff_im_ind):
        """
        Searching for the minimum distance between Li with its image using dijkstra algorithm

        Notes:
            This method would not keep track of the percolating pathway.

        Args:
            nn_tm_chandict: Dictionary of d_min for any diffusor pairs
            sc_diff_inds: All diffusor indices in the supercell
            diff_ind, diff_im_ind: The diffusor index and the corresponding image index

        Return:
            perco_dist[fin_ind]: Minimum distance between diffusor index and the corresponding image index
        """

        # start_time = time.time()
        flag_arry = np.ones(len(sc_diff_inds))
        perco_dist = np.full(len(sc_diff_inds),1e10)   # Distance initialized as a very large number
        perco_dist[diff_ind] = 0
        # perco_dist[sc_diff_inds.index(diff_ind)] = 0  # The distance between site diff_ind with itself is 0
        init_ind, fin_ind = diff_ind, diff_im_ind
        # i_fin_ind = sc_diff_inds.index(fin_ind)

        while True:
            if init_ind == fin_ind:
                # return perco_dist[i_fin_ind]
                # print('{}s eplapsed for dijkstr'.format(time.time()-start_time))
                return perco_dist[fin_ind]
            # i_init_ind = sc_diff_inds.index(init_ind)
            # flag_arry[i_init_ind] = 0
            flag_arry[init_ind] = 0
            nn_lst = nn_tm_chandict[init_ind].keys()
            for nn_ind in nn_lst:    # Enumerate all the nn of init_ind
                if not flag_arry[nn_ind]:
                    continue
                # i_nn_ind = sc_diff_inds.index(nn_ind)
                # if perco_dist[i_init_ind] + nn_tm_chandict[init_ind][nn_ind] < perco_dist[i_nn_ind]:
                #    perco_dist[i_nn_ind] = perco_dist[i_init_ind] + nn_tm_chandict[init_ind][nn_ind]
                if perco_dist[init_ind] + nn_tm_chandict[init_ind][nn_ind] < perco_dist[nn_ind]:
                    perco_dist[nn_ind] = perco_dist[init_ind] + nn_tm_chandict[init_ind][nn_ind]

            # For the rest of diffusor sites, remove the one with shortest distance that are still tracked
            i_next = perco_dist[flag_arry.nonzero()].argmin()
            # init_ind = sc_diff_inds[np.where(flag_arry == 1)[0][i_next]]
            init_ind = np.where(flag_arry == 1)[0][i_next]

    @classmethod
    def from_poscar(cls, posname, diffusor='Li+', cations=None, d_cut=2.97, d_tol=0.3):
        """
        Initialize an object from a POSCAR

        Args:
            posname: The name of the POSCAR
            diffusor: The diffusing ion
            cations(default:None): The cation symbols, if not set, will be set as all postively charged ions
            d_cut: The cut off radius for 0tm unit
            d_tol: The tolerance of bond length
        """
        structure = PoscarWithChgDecorator.from_file(posname).structure
        if cations is None:
            cations = structure.cation_symbols
        tm_symbols = deepcopy(cations)
        tm_symbols.remove(diffusor)

        return cls(structure, diffusor, tm_symbols, d_cut=d_cut, d_tol=d_tol)
