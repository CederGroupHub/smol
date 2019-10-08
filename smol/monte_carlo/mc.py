########
###Updated on 2019.02.27 to be compatible with python3
########

from __future__ import division

from pymatgen import Composition
from collections import defaultdict
import numpy as np
import random
import argparse
from pymatgen import Structure
from monty.serialization import dumpfn, loadfn
from copy import deepcopy
import os

from pymatgen.analysis.structure_matcher import StructureMatcher
from pyabinitio.cluster_expansion.ce import Cluster, SymmetrizedCluster, get_bits
import logging

from copy import deepcopy
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation



def get_flips(inds, occu):
    """
    :param inds: site indicies
    :param occu: current site occupation
    :return: List of potential species flips
    """
    i = random.choice(inds)
    all_j = [j for j in inds if occu[i] != occu[j]]
    if not all_j: return []
    j = random.choice(all_j)
    return [(i, occu[j]), (j, occu[i])]

def run_T(ecis, cluster_supercell, occu, T, n_loops, ind_groups, n_rand=0, check_unique=True, sample_energies=0):
    """
    Run a production run at a single temperature

    :param eg: ECI Generator Object
    :param cluster_supercell:  Cluster Supercell Obkect
    :param occu: Current occupation
    :param T: Current Temperature
    :param n_loops: Number of loops to perform
    :param n_rand: How many random samples structures to return
    :param ind_groups: Indicies corresponding to valid flips
    :param sample_energies: how many sampled energies to return
    :return: current occupation, minimum energy occupation observed, minimum energy, list of random structures, list of sampled energies OR
             current occupation, minimum energy occupation observed, minimum energy, list of sampled energies
    """

    ind_groups_non_empty = []
    for ind_group in ind_groups:
        if len(ind_group) > 0:
            ind_groups_non_empty.append(deepcopy(ind_group))
    ind_groups = ind_groups_non_empty
    if len(ind_groups) == 0:
        raise ValueError("MC: all index groups are empty, so there are no valid flips!")

    k = 8.617332e-5
    kt = k * T
    kt_inv = 1.0 / kt

    corr = cluster_supercell.corr_from_occupancy(occu)
    e = np.dot(corr, ecis) * cluster_supercell.size

    min_occu = occu
    min_e = e

    rand_occu = []
    if n_rand > 0:
        rand_step = int(n_loops/n_rand)
    else:
        rand_step = None
    save_rand = False

    energies = []
    ###Set up the step when energy is taken
    if sample_energies > 0:
        en_step = int(n_loops/sample_energies)
    else:
        en_step = None
    for i, loop in enumerate(range(n_loops)):
        flips = get_flips(random.choice(ind_groups), occu)
        d_corr, new_occu = cluster_supercell.delta_corr(flips, occu, debug=False)
        de = np.dot(d_corr, ecis) * cluster_supercell.size
        p = np.exp(-de * kt_inv)

        if np.random.rand() <= p:
            corr += d_corr
            occu = new_occu
            e += de

        # Save min structure
        if e < min_e:
            min_e = e
            min_occu = deepcopy(occu)

        # Save random structures that have different energies that existing saved structures
        if rand_step and not save_rand and (loop + 1) % rand_step == 0:
            save_rand = True

        if rand_step and save_rand:
            new_rand = True
            if check_unique:
                for ex_occu, ex_e in rand_occu:
                    if e == ex_e:
                        new_rand = False
                        break
            if new_rand:
                rand_occu.append((deepcopy(occu), e))
                save_rand = False
            else:
                save_rand = True

        # Save energies
        if en_step and (loop + 1) % en_step == 0:
            energies.append(e)

    if rand_step and len(rand_occu) > n_rand:
        rand_occu = rand_occu[0:n_rand]

    if en_step:
        return occu, min_occu, min_e, energies, rand_occu
    else:
        return occu, min_occu, min_e, rand_occu

def simulated_anneal(ecis, cluster_supercell, occu, ind_groups, n_loops, init_T, final_T, n_steps, return_E=False):
    #print(init_T,final_T)
    assert final_T < init_T
    T_step = int((init_T - final_T) / n_steps)
    T = init_T
    min_occu = occu
    for i in range(n_steps):
        T = T - T_step
        occu, min_occu_step, min_e, rand_occu = run_T(ecis, cluster_supercell, deepcopy(min_occu), T, n_loops, ind_groups, n_rand=0, check_unique=True)

        corr_min = cluster_supercell.corr_from_occupancy(min_occu)
        corr_min_step = cluster_supercell.corr_from_occupancy(min_occu_step)
        if np.dot(corr_min_step, ecis) < np.dot(corr_min, ecis):
            min_occu = deepcopy(min_occu_step)
    if not return_E:
        return min_occu
    else:
        corr_min = cluster_supercell.corr_from_occupancy(min_occu)
        min_energy = np.dot(corr_min, ecis)
        return min_occu, min_energy



