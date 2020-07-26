"""
Implementation of analysis module for diffusion network analysis

Currently 0TM percolation analysis is implemented

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
from ionicstructure import PoscarWithChgDecorator
import time
import numpy as np


class PercolationAnalyzer(object):
    """
    Perform 0TM percolation analysis with Dijkstra algorithm

    """

    def __init__(self, structure, diffusor, TM_symbols, d_cut = 2.97, d_tol = 0.3):
        """
        Initialize the class using ChargedStructure defined in ionicstructure.ChargedStructure

        Args:
            structure: Charged structure defined in ionicstructure.ChargedStructure
            diffusor: The Li/Na to be analyzed for 0TM percolation
            TM_symbols: The symbols of other sites rather than diffusor, with charge decorator
            d_cut: The cut off radius for 0TM unit
            d_tol: The tolerance of bond length
        """
        self.structure = deepcopy(structure)
        self.diffusor = diffusor
        self.TM_symbols = TM_symbols
        self.d_max = d_cut + d_tol
        self.d_min = d_cut - d_tol

    def get_percolating_Li(self, allow_step=0):
        """
        Get list of percolation diffusor using the Dijkstra's algorithm

        Notes:
            Check the percolation in all three periodic boundary directions

        Args:
            allow_step: The tolerance of number of 1TM jumps
        """
        start_time = time.time()

        # Enumerate all the diffusor sites to check if it is percolating
        scmat0 = np.eye(3)
        perco_diffusor_lsts, perco_diff_paths_lst = [[] for _ in range(3)], [[] for _ in range(3)]

        # Check percolation in all three dimension
        for dim in range(3):
            perco_diffusor_lst, perco_diff_paths = [], []
            scmat = deepcopy(scmat0)
            scmat[dim,dim] = 2.0
            sc_structure = deepcopy(self.structure)
            sc_structure.make_supercell(scmat)
            site_maps = self.map_sites(sc_structure, scmat)

            sc_diff_inds = sc_structure.get_ion_indices(self.diffusor)
            equal_site_lst = []
            for ind1, ind2 in site_maps:
                equal_site_lst.append((ind1, ind2))
            nn_TM_chandict = self.get_nn_TM_chandict(sc_structure)
            for diff_ind, diff_im_ind in equal_site_lst:
                min_step, perco_path = self.dijkstra_distance(nn_TM_chandict, sc_diff_inds, diff_ind, diff_im_ind)
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
            allow_step: The tolerance of number of 1TM jumps
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
            site_maps = self.map_sites(sc_structure, scmat)

            sc_diff_inds = sc_structure.get_ion_indices(self.diffusor)
            equal_site_lst = []
            for ind1, ind2 in site_maps:
                if (ind1 in perco_diffusor_lst) or (ind2 in perco_diffusor_lst):
                    continue
                if ind1 not in diff_inds:
                    continue
                equal_site_lst.append((ind1, ind2))
            nn_TM_chandict = self.get_nn_TM_chandict(sc_structure)
            for diff_ind, diff_im_ind in equal_site_lst:
                min_step = self.dijkstra_distance(nn_TM_chandict, sc_diff_inds, diff_ind, diff_im_ind)
                if min_step <= allow_step:
                    perco_diffusor_lst.append(diff_ind)

        print("--- percolation analysis takes %s seconds ---" % (time.time() - start_time))
        return perco_diffusor_lst

    def map_sites(self, sc_structure, scmat):
        """
        Get the mapping between supercell and original structure

        Notes:
            Can possibly be optimized with pymatgen structure_matcher, however the current
            get_mapping method in structure_matcher cannot support supercell

        Args:
            sc_structure: The structure supercell
            sc_mat: The supercell matrix
        """
        sc_frac_coords = sc_structure.frac_coords
        frac_coords = self.structure.frac_coords
        scaled_frac_coords = sc_frac_coords * scmat
        site_maps = []
        for i, coord in enumerate(frac_coords):
            inds = find_in_coord_list_pbc(scaled_frac_coords, coord)
            site_maps.append(tuple(inds))

        return site_maps

    def get_nn_TM_chandict(self, sc_structure):
        """
        Form a dictionary with first key being the diffusor indice while the second key
        being the indice of neighboring diffusor and the TM distance. The TM distance is
        defined by dist_table dictionary, the default values are:

        0TM: 0; 1TM: 1, 2TM: 1e2

        Please be noted that the above values are arbitrary as we assume Li would only
        diffuse through 0TM and 1TM channels, 3TM and 4TM does not apply here as well

        return:
            nn_TM_chandict:
                    nn_TM_chandict[key1][key2] will return the TM distance between two
                    neighboring diffusor sites with indices being key1 and key2
        """
        dist_table = {0:0, 1:1, 2:1e2}
        diffinds = sc_structure.get_ion_indices(self.diffusor)
        TMinds = []
        for TM_symbol in self.TM_symbols:
            TMinds.extend(sc_structure.get_ion_indices(TM_symbol))
        TM_chan_lst, TM_chan_type_inds, TM_type_ind_dict, TM_tet_types, TM_num_lst, dist_matrix = \
            self.__get_all_chans(self, sc_structure, diffinds, TMinds)

        # Initialize everything assuming they are 2TM channel, 3TM adn 4TM does not apply here
        nn_TM_chandict = {ind1:{ind2:dist_table[2] for ind2 in diffinds if ind1!=ind2} for ind1 in diffinds}
        pair_optimized = []   # keep track of the optimized pair

        # Get all the 0TM channels and 1TM channels first to speed up the distance calculations
        chan_0TMs, chan_1TMs = [], []
        for chan_ind, chan in enumerate(TM_chan_lst):
            if TM_num_lst[chan_ind] == 0:
                chan_0TMs.append(chan)
            elif TM_num_lst[chan_ind] == 1:
                chan_1TMs.append(chan)

        # Taking out the 0TM pair distance
        for chan_0TM in chan_0TMs:
            for i1, i2 in perm(chan_0TM,2):
                if i1 in pair_optimized or i2 in pair_optimized:
                    continue
                nn_TM_chandict[i1][i2], nn_TM_chandict[i2][i1] = dist_table[0], dist_table[0]
                pair_optimized.append(sorted([i1,i2]))

        # Taking out the 1TM pair distance
        for chan_1TM in chan_1TMs:
            for i1, i2 in perm(chan_0TM,2):
                if sorted([i1,i2]) in pair_optimized:
                    continue
                nn_TM_chandict[i1][i2], nn_TM_chandict[i2][i1] = dist_table[1], dist_table[1]
                pair_optimized.append(sorted([i1,i2]))

        return nn_TM_chandict

    def __get_all_chans(self, structure, diffinds, TMinds):
        """
        Get all the tetrahedrons in a given structure

        Args:
            structure: The structure to be analyzed
            diffinds: The diffusor indices
            TMinds: The indices of other cations

        return:
            TM_chan_lst: A list of TM channels, each member being a list of site indices
            TM_chan_type_inds: A list of TM channel types, indexed the same way as TM_chan_lst
            TM_type_ind_dict: A dictionary taking type as key and values being a list ot TM channels
            TM_num_lst: A list of TM num for each of the TM channels
            TM_tet_types: A list of the symbol list for each type of tet
        """
        cation_syms = [self.diffusor] + self.TM_symbols
        TM_tet_types = list(comb(cation_syms,4))
        TM_tet_type_inds = [sorted([cation_syms.index(sym) for sym in tet]) for tet in TM_tet_types]
        TM_nums = [4-TM_syms.count(self.diffusor) for TM_syms in TM_tet_types]
        cation_inds = diffinds + TMinds
        dist_matrix = structure.distance_matrix()
        TM_chan_lst, TM_chan_type_inds, TM_num_lst = [], [], []

        # 1. Enumerate all 4 site combination to check if they are tetrahedron
        # 2. Decide the type of the tetrahedrons
        for (ind1,ind2,ind3,ind4) in comb(cation_inds,4):
            is_tet = True
            for (i1, i2) in comb([ind1,ind2,ind3,ind4],2):
                if dist_matrix[i1,i2] > self.d_max:
                    is_tet = False
                    break
            if not is_tet:
                continue
            sym1, sym2, sym3, sym4 = str(structure[ind1].specie), str(structure[ind2].specie),\
                                     str(structure[ind3].specie), str(structure[ind4].specie)
            TM_type_ind = TM_tet_type_inds.index(sorted([sym1, sym2, sym3, sym4]))
            TM_chan_lst.append(sorted[ind1, ind2, ind3, ind4])
            TM_chan_type_inds.append(TM_type_ind)
            TM_num_lst.append(TM_nums[TM_type_ind])

        TM_type_ind_dict = {}
        for type_ind in range(len(TM_tet_type_inds)):
            TM_type_ind_dict[type_ind] = \
                [i for i, chan_type_ind in enumerate(TM_chan_type_inds) if chan_type_ind==type_ind]

        return TM_chan_lst, TM_chan_type_inds, TM_type_ind_dict, TM_tet_types, TM_num_lst, dist_matrix

    def dijkstra_distance(self, nn_TM_chandict, sc_diff_inds, diff_ind, diff_im_ind):
        """
        Searching for the minimum distance between Li with its image using dijkstra algorithm

        Notes:
            This method would keep track of the percolating pathway. If you want to faster verison,
            you should use dijkstra_distance_fast as that method only track the diffusion distance

        Args:
            nn_TM_chandict: Dictionary of d_min for any diffusor pairs
            sc_diff_inds: All diffusor indices in the supercell
            diff_ind, diff_im_ind: The diffusor index and the corresponding image index

        Return:
            perco_dist[fin_ind]: Minimum distance between diffusor index and the corresponding image index
        """
        start_time = time.time()
        flag_arry = np.ones(len(sc_diff_inds))
        perco_dist = np.full(len(sc_diff_inds),1e100)   # Distance initialized as a very large number
        perco_dist[sc_diff_inds.index(diff_ind)] = 0  # The distance between site diff_ind with itself is 0
        init_ind, fin_ind = diff_ind, diff_im_ind
        perco_path = {ind: [(init_ind, 0)] for ind in sc_diff_inds}

        while True:
            if init_ind == fin_ind:
                print("--- dijkstra_distance analysis takes %s seconds ---" % (time.time() - start_time))
                return perco_dist_dict[fin_ind], perco_path[fin_ind]
            i_init_ind = sc_diff_inds
            flag_arry[i_init_ind] = 0
            nn_lst = list(nn_TM_chandict[init_ind].keys())
            for nn_ind in nnlst:    # Enumerate all the nn of init_ind
                if not flag_arry[nn_ind]:
                    continue
                i_nn_ind = sc_diff_inds.index(nn_ind)
                if perco_dist[i_init_ind] + nn_TM_chandict[init_ind][nn_ind] < perco_dist[i_nn_ind]:
                    perco_dist[i_nn_ind] = perco_dist[i_init_ind] + nn_TM_chandict[init_ind][nn_ind]
                    perco_path[nn_ind] = perco_path[init_ind] + [(nn_ind, nn_TM_chandict[init_ind][nn_ind])]

            # For the rest of diffusor sites, remove the one with shortest distance that are still tracked
            i_next = perco_dist[flag_arry.nonzero()].argmin()
            init_ind = sc_diff_inds[np.where(flag_arry == 1)[0][i_next]]


    @classmethod
    def from_poscar(cls, posname, diffusor='Li+', cations=None, d_cut=2.97, d_tol=0.3):
        """
        Initialize an object from a POSCAR

        Args:
            posname: The name of the POSCAR
            diffusor: The diffusing ion
            cations(default:None): The cation symbols, if not set, will be set as all postively charged ions
            d_cut: The cut off radius for 0TM unit
            d_tol: The tolerance of bond length
        """
        structure = PoscarWithChgDecorator.from_file(posname).structure
        if cations is None:
            cations = structure.cation_symbols
        TM_symbols = deepcopy(cations)
        TM_symbols.remove(diffusor)

        return cls(structure, diffusor, TM_symbols, d_cut=d_cut, d_tol=d_tol)