from __future__ import division
from __future__ import unicode_literals

import numpy as np
import random
import argparse
import os
import math
import multiprocessing
import logging
import json
import time
import itertools
import re

from numpy import linalg as LA
from collections import defaultdict
from operator import mul
from functools import partial
from copy import deepcopy
from matplotlib import pyplot as plt

from pymatgen import Structure, Composition, PeriodicSite, Specie, Element
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis import local_env
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
from pyabinitio.cluster_expansion.ce import Cluster, SymmetrizedCluster, get_bits, ClusterExpansion
from monty.serialization import dumpfn, loadfn

kB = 8.617385692256675e-05
site_type_key = 'site_type'

class KineticMonteCarlo:
    def __init__(self, init_occu, cluster_supercell, cluster_expansion, ecis, inds, min_dist=1.82, temperature=300):
        print("Initializing parameters")
        self.beta = 1.0/(kB*temperature)
        self.freq  = 1e7
        self.occu = deepcopy(init_occu)
        self.init_occu = deepcopy(init_occu)
        self.cs = cluster_supercell
        self.ce = cluster_expansion
        self.ecis = deepcopy(ecis)
        
        # initiation
        non_zero_eci_data = self._get_non_zero_ecis()
        self.interacting_sites = self._get_interacting_sites(non_zero_eci_data)
        ind_groups_non_empty = []
        for ind_group in deepcopy(inds):
            if len(ind_group) > 0:
                ind_groups_non_empty.append(deepcopy(ind_group))
        self.ind_groups = ind_groups_non_empty
        if len(self.ind_groups) == 0:
            raise ValueError("MC: all index groups are empty, so there are no valid flips!")
        self.min_dist = min_dist
        self.move_list = self._get_move_list()
        self.surrounding_sites = self._get_surrounding_sites()
        
        # Initialize requisite objects
        num_rates = np.sum([len(self.move_list[i]) for i in self.move_list])
        self.num_rates = num_rates
        self.rate_index_to_flip = [] # Rates will change, but indexing should not, as it depends only on move_list
        for i in self.move_list:
            for j in self.move_list[i]:
                self.rate_index_to_flip.append((i, j))
        self.flip_to_rate_index = {}
        rate_counter = 0
        for i in self.move_list:
            for j in self.move_list[i]:
                self.flip_to_rate_index[(i, j)] = rate_counter
                rate_counter += 1
    
    def _get_non_zero_ecis(self):
        """
        Get details on the cluster + bit combos with non-zero ECI
        :param ce: Cluster Expansion
        :param ecis: list of ECIs for the given cluster expansion
        :return: list of dictionaries containing information on the clusters with non-zero ECI. The dictionaries 
        contain the following information:
        sc_b_id: overall bit ordering index, used to access ECI (ecis[sc_b_id])
        eci: ECI of the specific bit combo on the given cluster
        cluster size: number of sites in the cluster
        cluster number: index of the specific cluster in ce.clusters[cluster size]
        cluster counter: overall cluster index, used to access site information of a specific cluster supercell 
        (cs.cluster_indices[cluster counter])
        bit ordering number: index of bit ordering within the cluster 
        (ce.clusters[cluster size][cluster number][bit ordering number])
        bit ordering: specific bit combo within the cluster
        """

        bit_ordering_counter = 0
        cluster_counter = 0
        non_zero_eci_data = []
        for cluster_size in self.ce.clusters:
            for i, cluster in enumerate(self.ce.clusters[cluster_size]):
                for j, bit_ordering in enumerate(cluster.bit_combos):
                    if np.abs(self.ecis[bit_ordering_counter]) > 1e-6:
                        non_zero_eci_data.append({'sc_b_id': bit_ordering_counter, 
                                                  'eci': self.ecis[bit_ordering_counter],
                                                  'cluster size': cluster_size, 'cluster number': i, 
                                                  'cluster counter': cluster_counter,
                                                  'bit ordering number': j, 'bit ordering': bit_ordering})
                    bit_ordering_counter += 1
                cluster_counter += 1
        return non_zero_eci_data
    
    def _get_interacting_sites(self, non_zero_eci_data):
        """
        Find all interacting sites/bit pairs (i.e., a site with a specific bit/species) for a specific site/bit pair. 
        Interacting site/bit pair are defined as site/bit pairs that are in the same cluster+bit combo with non-zero 
        ECI as a given site/bit pair.
        :param non_zero_eci_data: list of dictionaries with information on the non-zero ECI cluster+bit combos,
        returned by function _get_non_zero_ecis
        :return: dictionary of site index/bit combos that are considered to "interact" with a given site index/bit 
        combo
        """
        interaction_dict = defaultdict(list)
        for cluster_data in non_zero_eci_data:
            if cluster_data['cluster size'] > 1:
                # identify unique cluster occurrences and their site indices
                cluster_site_indices = np.unique(self.cs.cluster_indices[cluster_data['cluster counter']][1], 
                                                 axis=0)
                # get bit ordering of this ECI
                bit_ordering = cluster_data['bit ordering']
                for index_group, order in itertools.product(cluster_site_indices, bit_ordering):
                    if len(index_group) != len(order):
                        print(index_group)
                        print(order)
                        raise ValueError("Index group and bit ordering do not have the same length")
                    index_bit_pairs = []
                    for i in range(len(order)):
                        index_bit_pairs.append((index_group[i], order[i]))
                    for index_bit_pair in index_bit_pairs:
                        interacting_sites = deepcopy(index_bit_pairs)
                        interacting_sites.remove(index_bit_pair)
                        for site in interacting_sites:
                            interaction_dict[index_bit_pair].append(site)
        for index_bit_pair in interaction_dict:
            interaction_dict[index_bit_pair] = list(set(interaction_dict[index_bit_pair]))

        return interaction_dict

    def _get_move_list(self):
        """
        For each index in inds, gets list of nearest neighbor site indices which can be flipped with index
        :param inds: site indices which can be flipped
        :param min_dist: distance between sites below which we will consider the sites nearest neighbors
        :return: dict[i]=[j1, j2, j3...], where i is the index of the site that can be flipped and j1, j2, j3, ... 
        are the nearest neighbor sites which can be flipped with i
        """
        move_list = defaultdict(list)
        for i in self.ind_groups[0]:
            for j in self.ind_groups[0]:
                if i != j and \
                abs(self.cs.supercell[i].distance(self.cs.supercell[j]) - self.min_dist) <= 0.2:
                    move_list[i].append(j)
        return move_list
    
    def _get_surrounding_sites(self):
        """
        Finds the relevant surrounding sites for all sites to get the barrier.
        For LTO system, the relevant sites are the occupancies of the 16c adjacent to the 8a site and the 16d 
        adjacent to the 16c site. This may change though and will likely change for the system.
        :param inds: indices of the sites that can flip
        :return: dictionary where keys are indices (inds) and values are the indices of the nearest 16d sites for 16c 
        sites and the indices of the nearest 16c sites for 16d sites
        """

        surrounding_site_map = defaultdict(list)

        for i, other_i in itertools.product(self.ind_groups[0], range(len(self.cs.supercell))):
            if (self.cs.supercell[i].properties[site_type_key] == '8a' and \
            self.cs.supercell[other_i].properties[site_type_key] == '16c'):
                if self.cs.supercell[i].distance(self.cs.supercell[other_i]) < 1.82:
                    surrounding_site_map[i].append(other_i)
            elif self.cs.supercell[i].properties[site_type_key] == '16c' and \
            self.cs.supercell[other_i].properties[site_type_key] == '16d':
                if self.cs.supercell[i].distance(self.cs.supercell[other_i]) < 2.97:
                    surrounding_site_map[i].append(other_i)

        return surrounding_site_map
    
    def initialize_rate_table(self, ekra_map):
        """
        Updates the rate table for all possible moves for each index that can be flipped
        :param move_list: dictionary of indices of nearest neighbors which can be flipped with the key (also an index)
        :param occu: current site occupation
        :param rate_table: dictionary of lists for site i of move rates in the same order as the moves in move_list[i]
        """

        rate_table = defaultdict(list)
        for i in self.move_list:
            rate_table = self.update_rate_table(i, self.move_list[i], self.init_occu, rate_table, ekra_map)

        return rate_table
    
    # hash table to hold rates for all possible moves
    # function to update rate table near initial site and final site after move is selected
    def update_rate_table(self, i, moves, occu, rate_table, ekra_map):
        """
        Update the rate table for a single site and its rates
        :param i: index of site whose rates are to be updated
        :param moves: list of sites indices that a site i can jump to (i.e. nearest neighbor)
        :param occu: current site occupation
        :param rate_table: dictionary of current move rates for each site
        :param ecis: all ecis
        :param surrounding_sites_map: dictionary of the nearby 16c sites if site is 8a or nearby 16d sites if site is 
        16c
        :param ekra_map: dictionary of ekra's for given tet with nearby 16c sites and given oct with nearby 16d sites
        :return: updated rate_table dictionary object
        """

        
        
        # some guidelines:
        # 1. rate is 0 if occupancy of site i is Vacancy
        # 2. rate is 0 if site we want to move to is not Vacancy
        # 3. else, consider occupancy of surrounding sites to obtain Ekra
        # 4. then calculate difference in energies via pointenergy to obtain full E_activation
        # Note: what is considered "surrounding sites" may differ among different systems

        # for site i, for each potential move, calculate the energy difference -- use pointenergy in the future!
        new_rates = []
        for j in moves:
            try:
                if self.cs.bits[i][occu[i]] == 'Vacancy':
                    new_rates.append(0.0)
                elif self.cs.bits[j][occu[j]] != 'Vacancy':
                    new_rates.append(0.0)
                else:
                    if self.cs.supercell[i].properties[site_type_key] == 'tet':
                        tet_surrounding_sites = self.surrounding_sites[i]
                        oct_surrounding_sites = self.surrounding_sites[j]
                    else:
                        tet_surrounding_sites = self.surrounding_sites[j]
                        oct_surrounding_sites = self.surrounding_sites[i]

                    neighbors_8a = 0
                    neighbors_16c = 0
                    for site_index in tet_surrounding_sites:
                        if self.cs.bits[site_index][occu[site_index]] == 'Li':
                            neighbors_8a += 1
                    for site_index in oct_surrounding_sites:
                        if self.cs.bits[site_index][occu[site_index]] == 'Li':
                            neighbors_16c += 1
                    #ekra = ekra_map[(neighbors_8a, neighbors_16c)]
                    ekra = 0.150
                    delta_corr, new_occu = self.cs.delta_corr([(i, occu[j]), (j, occu[i])], occu)
                    de = np.dot(delta_corr, self.ecis)*self.cs.size
                    if ekra is None:
                        barrier = de
                    else:
                        barrier = ekra + de*0.5
                    new_rates.append(self.rate(barrier))
            except:
                print(i)
                print(occu[i])
                print(j)
                print(occu[j])
                print(self.cs.bits[i])
                print(self.cs.bits[i][occu[i]])
                raise ValueError("Cannot calculate rate")
        rate_table[i] = new_rates

        return rate_table
    
    def run_kmc(self, nloops, seeds=None, ekra_map=None, data_outfile=None, occupancies_outfile=None):

        # Need to instantiate ekra_map for barrier data
        if ekra_map is None:
            raise ValueError("Must provide barrier in the form of a dictionary")
            
        if data_outfile is None:
            data_outfile = 'kmc_data.json'
        if occupancies_outfile is None:
            occupancies_outfile = 'kmc_occupancies.json'
            
        if seeds is None:
            np.random.seed()
            randnum_gen_1 = iter(np.random.rand(nloops))
            np.random.seed()
            randnum_gen_2 = iter(np.random.rand(nloops))
        else:
            np.random.seed(seeds[0])
            randnum_gen_1 = iter(np.random.rand(nloops))
            np.random.seed(seeds[1])
            randnum_gen_2 = iter(np.random.rand(nloops))
        
        occu = deepcopy(self.init_occu)

        # Initialization of rate table and related objects
        rate_table = self.initialize_rate_table(ekra_map)
        list_of_rates = list(itertools.chain.from_iterable(rate_table.values()))
        bittree = construct_bittree(list_of_rates, len(list_of_rates)) # Initialize BITTree

        # Initialization of time
        mc_time = 0.0

        # Initialization of other quantities of interest
        mc_time_list = [None]*(nloops)
        mc_time_list[0] = mc_time
        energy = 0.0
        energy_list = [None]*(nloops)
        energy_list[0] = energy
        occu_list = [None]*(nloops) # Keep track of occupancies over KMC
        occu_list[0] = deepcopy(occu).tolist()
        # Note it may be less memory-intensive to directly write structures to a folder or write occupancies to a file
        # Keep track of ion (non-Vacancy) trajectories, including periodic boundary conditions, for MSD
        sc = self.cs.supercell
        ion_trajectories_init = []
        # Convenience dictionary to quickly obtain index of migrating ion
        current_ion_indices = dict.fromkeys(range(len(occu)))
        migrating_ion_counter = 0
        for i, occu_i in enumerate(occu):
            if self.cs.bits[i][occu_i] != 'Vacancy' and self.cs.bits[i][occu_i] != 'O2-' \
            and self.cs.bits[i][occu_i] != 'O' and self.cs.bits[i][occu_i] != 'Ti':
                ion_trajectories_init.append({'sc index': i, 'frac coords': sc[i].frac_coords, 
                                              'specie': self.cs.bits[i][occu_i]}) 
                current_ion_indices[i] = migrating_ion_counter
                migrating_ion_counter += 1
        ion_trajectories = [[[None] for i in range(len(ion_trajectories_init))] for i in range(nloops)]
        ion_trajectories[0] = ion_trajectories_init
        picked_flip_types = [] # record which flip types occur
        picked_flips = [] # record which flips occur
        # Keep track of MSD
        msd_list = [[dict.fromkeys(['a^2', 'b^2', 'c^2', 'total']) for i in range(len(ion_trajectories_init))] \
                    for i in range(nloops)]
        for i in range(len(ion_trajectories_init)):
            msd_list[0][i]['a^2'] = 0.0
            msd_list[0][i]['b^2'] = 0.0
            msd_list[0][i]['c^2'] = 0.0
            msd_list[0][i]['total'] = 0.0

        # What else needs to be initialized?

        # Perform KMC loops
        # time the loop
        print("Beginning KMC loops")
        initial_time = time.clock()
        for n in range(1, nloops):
            
            if n % 1000 == 0:
                print("%s/%s"%(n, nloops))

            # Get cumulative rate using BITTree
            total_rate = getsum(bittree, self.num_rates-1)

            # Pick random number for escaping time
            mc_time = mc_time - np.log(next(randnum_gen_1))/total_rate

            # Pick another random number to pick event from rate table
            randnum = next(randnum_gen_2)*total_rate

            # Based on random number, get corresponding move via binary search+evaluating cumulative sums using 
            #bittree
            ir = self.num_rates-1; il = -1
            while (ir - il > 1):
                interv = ir - il
                mid = int(np.ceil(interv/2.0))+il
                if getsum(bittree, mid) > randnum:
                    ir = mid
                else:
                    il = mid
            flip = self.rate_index_to_flip[ir]

            # Update energy -- should change this later so that delta energy is calculated using a pointenergy
            # function
            delta_corr, new_occu = self.cs.delta_corr([(flip[0], occu[flip[1]]), (flip[1], occu[flip[0]])], occu)
            old_occu = deepcopy(occu)
            delta_energy = np.dot(delta_corr, self.ecis) * self.cs.size
            # Perform actual move/flip
            occu = new_occu

            # Identify list of relevant sites to update their rates
            sites_bits_to_update = []
            # 1. sites that interact (in same non-zero ECI cluster+bit combo) with initial site/bit combo of site 1
            sites_bits_to_update = sites_bits_to_update + self.interacting_sites[(flip[0], occu[flip[0]])]
            # 2. sites that interact with final site/bit combo of site 1
            sites_bits_to_update = sites_bits_to_update + self.interacting_sites[(flip[0], occu[flip[1]])]
            # 3. sites that interact with initial site/bit combo of site 2
            sites_bits_to_update = sites_bits_to_update + self.interacting_sites[(flip[1], occu[flip[1]])]
            # 4. sites that interact with final site/bit combo of site 2
            sites_bits_to_update = sites_bits_to_update + self.interacting_sites[(flip[1], occu[flip[0]])]
            sites_bits_to_update = list(set(sites_bits_to_update))

            sites_to_update = []
            for site_index, bit in sites_bits_to_update:
                if occu[site_index] == bit:
                    sites_to_update.append(site_index)
            # 5. sites whose potential move list includes site 1
            for j in self.move_list[flip[0]]:
                sites_to_update.append(j)
            # 6. sites whose potential move list includes site 2
            for j in self.move_list[flip[1]]:
                sites_to_update.append(j)
            # Delete duplicate site indices
            sites_to_update = list(set(sites_to_update))

            # Delete duplicate site-bit pairs if site is already in list of sites_to_update
            sites_bits_to_update_deduped = []
            for site_bit_pair in sites_bits_to_update:
                already_to_update = False
                if site_bit_pair[0] in sites_to_update:
                    already_to_update = True
                if not already_to_update:
                    sites_bits_to_update_deduped.append(site_bit_pair)
    
            # Because we are only shallow-copying, update_rate_table shouldn't directly change lists in rate_table
            # Instead, update_rate_table should make new lists and replace the lists in rate_table
            old_rate_table = rate_table.copy() 

            # Update rate table for relevant sites, also updating accumulated rate in bittree along the way
            for site_index in sites_to_update:
                rate_table = self.update_rate_table(site_index, self.move_list[site_index], occu, 
                                                    rate_table, ekra_map)
                for j, mj in enumerate(self.move_list[site_index]):
                    updatebit(bittree, len(list_of_rates), self.flip_to_rate_index[(site_index, mj)], 
                              rate_table[site_index][j] - old_rate_table[site_index][j])

            # Also update site-bit pairs, but only need to perform update if the other site has the specific given bit
            for site_bit_pair in sites_bits_to_update_deduped:
                site_index = site_bit_pair[0]
                bit = site_bit_pair[1]
                if occu[site_index] == bit:
                    rate_table = self.update_rate_table(site_index, self.move_list[site_index], occu,
                                                        rate_table, ekra_map)
                    for j, mj in enumerate(self.move_list[site_index]):
                        updatebit(bittree, len(list_of_rates), self.flip_to_rate_index[(site_index, mj)], 
                                  rate_table[site_index][j] - old_rate_table[site_index][j])

            # Update relevant variables (namely thermodynamic averages -- note that time has already been updated in 
            # the beginning of the loop)
            mc_time_list[n] = mc_time
            energy += delta_energy
            energy_list[n] = energy
            occu_list[n] = deepcopy(occu).tolist()
            # update ion trajectories
            (i, j) = flip
            if current_ion_indices[i] is None:
                (j, i) = flip
            picked_flips.append((i, j))
            migrating_ion_index = current_ion_indices[i] # index of migrating ion in ion_trajectories
            
            current_site = PeriodicSite(Specie(ion_trajectories[n-1][migrating_ion_index]['specie']), 
                                        ion_trajectories[n-1][migrating_ion_index]['frac coords'], 
                                        sc.lattice)
            dist, jimage = current_site.distance_and_image(sc[j])
            ion_trajectories[n] = deepcopy(ion_trajectories[n-1]) # copy ion_trajectories from previous index
            ion_trajectories[n][migrating_ion_index]['sc index'] = j # update the ion_trajectory of moved Li
            ion_trajectories[n][migrating_ion_index]['frac coords'] = sc[j].frac_coords+jimage
            # update migrating_ion_counter
            current_ion_indices[i] = None
            current_ion_indices[j] = migrating_ion_index
            # update flip type history
            if self.cs.supercell[i].properties[site_type_key] == 'tet':
                tet_surrounding_sites = self.surrounding_sites[i]
                oct_surrounding_sites = self.surrounding_sites[j]
            else:
                tet_surrounding_sites = self.surrounding_sites[j]
                oct_surrounding_sites = self.surrounding_sites[i]

            neighbors_8a = 0
            neighbors_16c = 0
            for site_index in tet_surrounding_sites:
                if self.cs.bits[site_index][old_occu[site_index]] == 'Li+':
                    neighbors_8a += 1
            for site_index in oct_surrounding_sites:
                if self.cs.bits[site_index][old_occu[site_index]] == 'Li+':
                    neighbors_16c += 1
                    
            picked_flip_types.append((neighbors_8a, neighbors_16c))
            
            # update MSD
            for i, migrating_ion_data in enumerate(ion_trajectories[n]):
                msd_list[n][i]['a^2'] = (migrating_ion_data['frac coords'][0]-
                                         ion_trajectories[0][i]['frac coords'][0])**2
                msd_list[n][i]['b^2'] = (migrating_ion_data['frac coords'][1]-
                                         ion_trajectories[0][i]['frac coords'][1])**2
                msd_list[n][i]['c^2'] = (migrating_ion_data['frac coords'][2]-
                                         ion_trajectories[0][i]['frac coords'][2])**2
                msd_list[n][i]['total'] = msd_list[n][i]['a^2'] + msd_list[n][i]['b^2'] + msd_list[n][i]['c^2']
            

            if n == 1:
                print("Time for first loop: %s"%(time.clock() - initial_time))

        print("Time per loop: %s"%((time.clock() - initial_time)/float(n)))

        # Output relevant files
        # nloops (range(0, nloops)), mc time (mc_time_list), energy (energy_list), MSD in single file
        print("Converting trajectory data to jsonable format")
        for n_traject_list in ion_trajectories:
            for traject in n_traject_list:
                traject['frac coords'] = list(traject['frac coords'])
        output_dict = {'nloop': range(0, nloops), 'mc time': mc_time_list, 'energy': energy_list, 
                       'msd_list': msd_list, 'flip_list': picked_flips, 'ion_trajectories': ion_trajectories, 
                       'flip_type_list': picked_flip_types}
        with open(data_outfile, 'w') as fout: json.dump(output_dict, fout)
        # Occupancies in another file
        with open(occupancies_outfile, 'w') as fout: json.dump(occu_list, fout)
    
    def rate(self, Ea, A=1e13):
        # dividing the prefactor, A, by 1e12 gives self.time in the unit of 1e-12s (ps)
        return np.exp(-Ea * self.beta) * A


# BiTTree functions

# Python implementation of Binary Indexed Tree 

#************************************************************
# Routines for operations of Binary Index Tree (Fenwick Tree)
# http://www.geeksforgeeks.org/binary-indexed-tree-or-fenwick-tree-2/
#    n  --> No. of elements present in input array.
#    BITree[0..n] --> Array that represents Binary Indexed Tree.
#    arr[0..n-1]  --> Input array for whic prefix sum is evaluated.

# Returns sum of arr[0..index]. This function assumes 
# that the array is preprocessed and partial sums of 
# array elements are stored in BITree[]. 

def getsum(BITTree,i): 
    s = 0 #initialize result 

    # index in BITree[] is 1 more than the index in arr[] 
    i = i+1

    # Traverse ancestors of BITree[index] 
    while i > 0: 

        # Add current element of BITree to sum 
        s += BITTree[i] 

        # Move index to parent node in getSum View 
        i -= i & (-i) 
    return s

# Updates a node in Binary Index Tree (BITree) at given index 
# in BITree. The given value 'val' is added to BITree[i] and 
# all of its ancestors in tree. 
def updatebit(BITTree , n , i ,v): 

    # index in BITree[] is 1 more than the index in arr[] 
    i += 1

    # Traverse all ancestors and add 'val' 
    while i <= n: 

        # Add 'val' to current node of BI Tree 
        BITTree[i] += v 

        # Update index to that of parent in update View 
        i += i & (-i)

# Constructs and returns a Binary Indexed Tree for given 
# array of size n. 
def construct_bittree(arr, n): 

    # Create and initialize BITree[] as 0 
    BITTree = [0]*(n+1) 

    # Store the actual values in BITree[] using update() 
    for i in range(n): 
        updatebit(BITTree, n, i, arr[i]) 

    return BITTree
