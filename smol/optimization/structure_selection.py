from __future__ import division
from __future__ import unicode_literals
import os
import sys
import argparse
import json
from copy import deepcopy
import numpy as np
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import *
from pymatgen.io.vasp.inputs import *
from pymatgen.core.structure import Structure
from itertools import permutations
from monty.serialization import loadfn, dumpfn
from mc_old import *
from math import gcd
from functools import reduce
import random
from itertools import permutations
from operator import mul
from functools import partial, reduce
import multiprocessing as mp
import pickle
import scipy.io as sio



def find_indices(A, init_A):
    indices = []
    for i in range(init_A.shape[0]):
        index = np.argwhere(np.all(A == init_A[i], axis=1))
        indices.append(index[0][0])
    indices = np.array(indices)
    return indices

def leverage_score(A, k=0):
    m, d = A.shape
    U, _, VT =np.linalg.svd(A)
    
    if k==0:
        k = d
    
    U = U[:,:k]
    
    L = np.linalg.norm(U, axis = 1)
   
    return L**2



def CX_decompose(A, C):
    """R is a matrix of selected rows of A"""
    
    X = np.dot(np.linalg.pinv(C), A)
    
    return X

def CUR_decompose(G, C, R):
    """ calcualate U s.t. G = CUR """
    
    C_inv = np.linalg.pinv(C)
    R_inv = np.linalg.pinv(R)
    U = np.dot(np.dot(C_inv, G), R_inv)
    
    return U

def Nystrom_k(A ,indices, k = 5, full_rank = True):
    
    G = np.dot(np.transpose(A), A)
    C = G[:, indices]
    R = G[indices, :]

    skeleton = np.dot(np.transpose(A[:,indices]), A[:,indices])

    u, s, vT = np.linalg.svd(skeleton)

    s_inv = np.zeros(s.shape)
    
    rank = np.linalg.matrix_rank(skeleton)
    
    if full_rank:
    
        s_inv[:rank] = 1 / s[:rank]
        G_app = np.dot(np.dot(C, np.dot(u * s_inv, vT)), R) 
    else:
        s_inv[:k] = 1 / s[:k]
        G_app = np.dot(np.dot(C, np.dot(u * s_inv, vT)), R) 
    

    return G_app, rank




class StructureSelector():
    def __init__(self, s_dicts, feature_matrix, use_Ewald = False):
        """
        Args:
            s_dicts: a list of dictionary of structure as pymatgen.Structures.as_dict()
            feature_matrix: a matrix of correlation vector of size m*d (m: number of structures, d: number of features)
        
        Notes:
            feature_matrix contains no Ewald term. Only cluster features included

            structures: a pool of structures to be selected
            feature_matrix: the corresponding feature matrix of structures
            selected_structures: structures after selection algorithms
            selected_feature_matrix: feature matrix corresponds to the selected structures

        Algorithms:
            Consider two regional problem, undetermined and determined:
            1. In undertermined problem, use CX or CUR matrix decomposition method to minimize the reconstruction error to find
            the mose informative feature matrix and structures;
            2. In determined problem, use interpolation to minimize the high leverage point in regression, making the prediction more robust.
        
        """
        
        assert len(s_dicts) == feature_matrix.shape[0], "Size mismatch! Please check the inputs."
        
        self.structures = np.asarray(s_dicts)
        self.feature_matrix = feature_matrix
        self.selected_structures = []
        self.selected_feature_matrix = []
        
        
    def initialization(self, num_init, num_iter = 1000, num_step = 5, solver = 'CUR'):
        """
        :param num_init: number of structures to be initialized, typically should be ~ number of ECIs
        :param num_iter: number of iteration for finding the local minimus solution, depending on calculating time, can vary from 100 to 10^4
        :param num_step: number of structures to be selected in each searching step, 5 in default.
        e.g. add 5 structures each time to minimize the CUR or CX error
        e.g. if num_step = 1, it's a special case of gready algorithm
        :param solver: can be CUR or CX,
        CUR decomposition: G=A^TA = CUR (C, R are real columns/rows of matrix G)
        CX decomposition: A = CX (C are real columns of A)
        :return:
        """
        if solver == 'CUR':
            self.Nystrom_selection(num_init= num_init, num_iter = num_iter, num_step= num_step)
        elif solver == 'CX':
            self.CX_selection(num_init= num_init, num_iter = num_iter, num_step= num_step)
    
    def interpolation(self, num_structures = 100, num_probe = 5):
        """
        :param num_structures: number of structures to be added in interpolation step
        :param num_probe:
        :return:
        """
        n = int(np.ceil(num_structures / num_probe))
        for i in range(n):
            ss.variance_selection(num_probe= num_probe)
        
    def variance_selection(self, num_probe = 5):
        """
        selection method in determined region
        :param num_probe:
        :return:
        """
        
        assert (len(self.selected_structures) != 0), "Initialization is NOT completed"
        
        d = self.feature_matrix.shape[1]
        domain = np.eye(d)
        
        init_A = self.selected_feature_matrix.copy()
        init_s = self.selected_structures.copy()
        
        pool_A = self.feature_matrix.copy()
        pool_s = self.structures.copy()
        
        
        old_kernal = np.dot(np.transpose(init_A), init_A)
        old_inv = np.linalg.pinv(old_kernal)
        
        num_pool = pool_A.shape[0] # number of structures in the pool
        reduction = np.zeros(num_pool)

        for i in range(num_pool):
            trial_A = np.concatenate((init_A, pool_A[i].reshape(1, d)), axis =0)

            kernal = np.dot(np.transpose(trial_A), trial_A)
            inv = np.linalg.pinv(kernal)

            reduction[i] = np.sum(np.multiply( (inv-old_inv), domain))

        indices = np.argsort(reduction)
#         print(indices)


        self.selected_feature_matrix = np.concatenate((init_A, pool_A[indices[:num_probe]]), axis = 0)
        self.selected_structures = np.concatenate((init_s, pool_s[indices[:num_probe]]), axis = 0)
            
        ### delete from the pool ###
        self.feature_matrix = np.delete(pool_A, indices[:num_probe], axis=0)
        self.structures = np.delete(pool_s, indices[:num_probe], axis=0)

    def Nystrom_selection(self, num_init = 10, num_iter = 100, num_step = 5):
        
        """
        selection method in undetermined region
        :param num_init:
        :param num_iter:
        :param num_step:
        :return:
        """
        assert len(self.selected_structures) == 0, "Initial selected is NOT empty"
        
        origin_A = self.feature_matrix.copy()
        A = self.feature_matrix.copy()
        s = self.structures.copy()

        G = np.dot(origin_A, np.transpose(origin_A))
        total_indices_inall = [i for i in range(origin_A.shape[0])]
        L = leverage_score(A=A[:,1:])

        init_A = []
        init_s = []


        for n in range(num_step,num_init,num_step):

            error = np.linalg.norm(origin_A)*10000

            for i in range(num_iter):
                Pr = L / np.sum(L)

                total_indices = [i for i in range(A.shape[0])]

                trial_indices = np.random.choice(total_indices, size=num_step, replace= True)
                if n == num_step:
                    C = G[:, trial_indices]
                    R = G[trial_indices, :]
                else:
                    trial_A = np.array(init_A + A[trial_indices].tolist())
                    trial_indices_inall = find_indices(A=origin_A, init_A=trial_A)
                    remain_indices_inall = list(set(total_indices_inall) - set(trial_indices_inall))

                    C = G[:, trial_indices_inall]
                    R = G[trial_indices_inall, :]

                U = CUR_decompose(G, C, R)

                re = (np.linalg.norm(G - np.dot(np.dot(C, U),R)))

                if re < error:
                    return_indices = trial_indices
                    error = re
            print(error)

            init_A += A[return_indices,:].tolist()
            init_s += s[return_indices].tolist()

            A = np.delete(A, return_indices, axis = 0)
            s = np.delete(s, return_indices, axis= 0)
            L = np.delete(L, return_indices, axis = 0 )
            

        self.selected_feature_matrix = np.array(init_A)
        self.selected_structures = np.asarray(init_s)
        
        self.feature_matrix = np.array(A)
        self.structures = s
        

