from __future__ import division
from __future__ import unicode_literals
import os
import sys
import argparse
import json
from copy import deepcopy
import numpy as np
import numpy.linalg as la
from itertools import permutations
from monty.serialization import loadfn, dumpfn
from math import gcd
from functools import reduce
import random
from itertools import permutations
from operator import mul
from functools import partial, reduce
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min



def mu2_sampling(X, normalized_E, num_init= 10):
    cf = X[:,1:-1].copy()
    num_inputs, num_features = cf.shape
    norm_cf = np.linalg.norm(cf, axis=1).reshape(num_inputs, 1)
#     X = eg.feature_matrix
    indices = np.argsort(norm_cf)
    init_indices = indices[num_init]
    init_A = X[init_indices,:]
    init_E = normalized_E[init_indices]
    print(init_indices)

    total_indices = [i for i in range(num_inputs)]
    remain_indices = list(set(total_indices) - set(init_indices))

    remain_A = X[remain_indices,:]
    remain_E = normalized_E[remain_indices]

    return init_A, init_E, remain_A, remain_E

def kmeans_sampling(feature_matrix, normalized_E, num_init):
    A = feature_matrix[:,0:-1].copy()
    m, d = A.shape
    A /= np.linalg.norm(A, axis=1).reshape(m, 1)

    km = KMeans(n_clusters= num_init, random_state=0).fit(A)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, A)
    init_A = feature_matrix[closest]
    init_E = normalized_E[closest]
    return init_A, init_E


def hat_matrix(X):
    inv = np.linalg.pinv(np.dot(np.transpose(X), X))
#     print(inv)
    H = np.dot(np.dot(X, inv), np.transpose(X))
    return H

def leverage_sampling(X, normalized_E, num_init= 10):
    num_inputs, num_features = X.shape
#     X = eg.feature_matrix
    for i in range(num_inputs,num_init,-1):
#         print(i)
        H = hat_matrix(X)
        Hii = np.diag(hat_matrix(X))
        lev_indices = np.flip(np.argsort(Hii), axis = 0)

        X = np.delete(X, lev_indices[-1], axis = 0)
        normalized_E = np.delete(normalized_E, lev_indices[-1], axis=0)
    return X, normalized_E

def leverage_skeleton(A, r, c):
    u, sigma, vT = np.linalg.svd(np.transpose(A))
    leverage = np.zeros(vT.shape[0])
#     r = 50
#     r= np.linalg.matrix_rank(A)

    for j in range(vT.shape[1]):
        for i in range(r):
            leverage[j] += vT[i,j] **2

        leverage[j] = np.min([leverage[j]*c / r, 1])
    sort_indices = np.flip(np.argsort(leverage), axis = 0)# descent sorting, the first is the largest
    # sort_indices = (np.argsort(leverage))

    return sort_indices, leverage[sort_indices]


def coherence_leverage(A,r):
    u, sigma, vT = np.linalg.svd(np.transpose(A))
    leverage = np.zeros(vT.shape[0])
#     r = 50
#     r= np.linalg.matrix_rank(A)

    for j in range(vT.shape[1]):
        max = 0
        # for i in range(r):
        leverage[j] = np.max(np.abs(vT[:r, j]))
    sort_indices = (np.argsort(leverage))
    return sort_indices, leverage[sort_indices]

def ecis_fit(feature_matrix, normalized_E, mu, weights = None):

    A = feature_matrix.copy()
    f = normalized_E.copy()

#     A = A[0:150,:]
#     f = f[0:150]

    if weights is None:
        weights = np.ones(len(f))


    A_w = A * weights[:, None] ** 0.5
    f_w = f * weights ** 0.5

    from l1regls import l1regls, solvers
    solvers.options['show_progress'] = False
    from cvxopt import matrix
    A1 = matrix(A)
    b = matrix(f * mu)
    ecis = (np.array(l1regls(A1, b)) / mu).flatten()

    ce_energies = np.dot(A, ecis)

    rmse = np.average((ce_energies - f)**2)**0.5
#     print('RMSE = (in meV)', rmse*1000/2)
    return ecis, rmse


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(npoints, ndim)
    vec /= np.linalg.norm(vec, axis=1).reshape(npoints, 1)
    return vec

def find_nearest_structures(feature_matrix, num_init = 10):
    """
    valid cluster: clusters without void one and ewald term

    """
    N, M = feature_matrix[:,1:-1].shape
    # M: number of valid cluster (void and ewald term is deleted)
    # N: number of structures in reservoir

    random_matrix = sample_spherical(npoints= num_init, ndim= M)
    cluster_feature = feature_matrix[:, 1:-1] # feature matrix for valid clusters
    cluster_feature /= np.linalg.norm(cluster_feature, axis=1).reshape(N, 1)

    res = 10000
    corr_init = []
    for i in range(num_init):
        corr_idx = 0
        for j in range(cluster_feature.shape[0]):
            new_res = np.linalg.norm( cluster_feature[j,:])
            print(new_res)
            if new_res < res:
                res = new_res
                corr_idx = j
#         print(random_matrix[i,:])
#         print(cluster_feature[corr_idx,:])
        print('res = {}, idx = {}'.format(res, corr_idx))
        cluster_feature = np.delete(cluster_feature, corr_idx, axis=0)
        feature_matrix = np.delete(feature_matrix, corr_idx, axis= 0)
        corr_init.append(feature_matrix[corr_idx])

    return None

def rmse_set(feature_matrix, ecis, normalized_energies):
    return np.average((np.dot(feature_matrix, ecis)- normalized_energies)**2) **0.5


def initial_structure_PCA(feature_matrix, normalized_E, num_init= 10):
    """
    void cluster is always 1, contibute nothing in PCA analysis
    ewald term is highyly correlated, we treat it as an offset and it's not considered in structure selection

    """
    N, M = feature_matrix[:,0:-1].shape
    cluster_feature = feature_matrix[:,1:-1]
    normalized_E = np.array(normalized_E)
    # M: number of valid cluster (void and ewald term is deleted)
    # N: number of structures in reservoir
    init_corr = []
    init_cf = []
    init_normalized_E = []

    init_corr.append(feature_matrix[0,:])
    init_cf.append(cluster_feature[0,:])
    init_normalized_E.append(normalized_E[0])


    feature_matrix = np.delete(feature_matrix, 0, axis=0)
    cluster_feature = np.delete(cluster_feature, 0, axis=0)
    normalized_E = np.delete(normalized_E, 0, axis = 0)

    main_factor = []

    for i in range(num_init):
        return_idx = 0
        main= 1e30
        for j in range(cluster_feature.shape[0]):
            temp_cf = init_cf.copy()
            temp_cf.append(cluster_feature[j,:])
            w, _ = np.linalg.eig(np.dot(np.transpose(temp_cf), temp_cf))


            new_main = w[0]
            if new_main < main:
                return_idx = j
                main = new_main
        init_corr.append(feature_matrix[return_idx, :])
        init_cf.append(cluster_feature[return_idx, :])
        init_normalized_E.append(normalized_E[return_idx])

        feature_matrix = np.delete(feature_matrix, return_idx, axis=0)
        cluster_feature = np.delete(cluster_feature, return_idx, axis=0)
        normalized_E = np.delete(normalized_E, return_idx, axis = 0)

        main_factor.append(main)
#         print("main factor = {}, index = {}".format(main, return_idx))


    remain_feature = feature_matrix
    remain_normalized_E = normalized_E
    return np.array(init_corr), np.array(init_cf), init_normalized_E, remain_feature, np.array(cluster_feature),remain_normalized_E



def initial_structure_selection(feature_matrix, normalized_E, num_init= 10, n_components= 10,pca_cut = 3):
    """
    void cluster is always 1, contibute nothing in PCA analysis
    ewald term is highyly correlated, we treat it as an offset and it's not considered in structure selection

    """
    N, M = feature_matrix[:,0:-1].shape
    cluster_feature = feature_matrix[:,0:-1]
    normalized_E = np.array(normalized_E)
    # M: number of valid cluster (void and ewald term is deleted)
    # N: number of structures in reservoir
    init_corr = []
    init_cf = []
    init_normalized_E = []

    init_corr.append(feature_matrix[0,:])
    init_cf.append(cluster_feature[0,:])
    init_normalized_E.append(normalized_E[0])


    feature_matrix = np.delete(feature_matrix, 0, axis=0)
    cluster_feature = np.delete(cluster_feature, 0, axis=0)
    normalized_E = np.delete(normalized_E, 0, axis = 0)

    main_factor = []

    for i in range(num_init):
        return_idx = 0
        main= 100
        for j in range(cluster_feature.shape[0]):
            temp_cf = init_cf.copy()
            temp_cf.append(cluster_feature[j,:])

            pca = PCA(n_components= n_components)
            pca.fit(temp_cf)

            new_main = np.sum(pca.explained_variance_ratio_[0:pca_cut])
            if new_main < main:
                return_idx = j
                main = new_main
        init_corr.append(feature_matrix[return_idx, :])
        init_cf.append(cluster_feature[return_idx, :])
        init_normalized_E.append(normalized_E[return_idx])

        feature_matrix = np.delete(feature_matrix, return_idx, axis=0)
        cluster_feature = np.delete(cluster_feature, return_idx, axis=0)
        normalized_E = np.delete(normalized_E, return_idx, axis = 0)

        main_factor.append(main)
#         print("main factor = {}, index = {}".format(main, return_idx))


    remain_feature = feature_matrix
    remain_normalized_E = normalized_E
    return np.array(init_corr), init_normalized_E, remain_feature, remain_normalized_E, main_factor



def Mutual_coherence(A):
    A = np.array(A)
    m, d = A.shape
#     print(A.shape)
    A /= np.linalg.norm(A, axis=1).reshape(m, 1)
    G = np.dot(np.transpose(A), A)
    AAT = np.dot(A, np.transpose(A))
    # ATA = np.dot(np.transpose(A), A)
    # coherence = 1000
    for i in range(m):
        AAT[i,i] = 0
        # G[i,i]= 0
    #
    # for i in range(d):
    #     ATA[i,i] = 0
        # G[i,i]= 0

    coherence = np.max(np.abs(AAT))
    # coherence = np.max(np.abs(ATA))


    return coherence


def mutual_coherence_sampling(feature_matrix, normalized_E, num_init= 10):
    """
    void cluster is always 1, contibute nothing in PCA analysis
    ewald term is highyly correlated, we treat it as an offset and it's not considered in structure selection

    """
    N, M = feature_matrix[:,0:-1].shape
    cluster_feature = feature_matrix[:,1:-1]
    # cluster_feature = feature_matrix[:,0:]
    normalized_E = np.array(normalized_E)
    total_indices = [i for i in range(N)]
    # M: number of valid cluster (void and ewald term is deleted)
    # N: number of structures in reservoir

    init_corr = []
    init_cf = []
    init_E = []

    init_corr.append(feature_matrix[0,:])
    init_cf.append(cluster_feature[0,:])
    init_E.append(normalized_E[0])

    feature_matrix = np.delete(feature_matrix, 0, axis=0)
    cluster_feature = np.delete(cluster_feature, 0, axis=0)
    normalized_E = np.delete(normalized_E, 0, axis = 0)

    for i in range(num_init):
        return_idx = 0
        coherence = 100
        for j in range(feature_matrix.shape[0]):
            temp_cf = init_cf.copy()
            temp_cf.append(cluster_feature[j,:])
            temp_coherence = Mutual_coherence(temp_cf)
            if temp_coherence < coherence:
                return_idx = j
                coherence = temp_coherence

        init_corr.append(feature_matrix[return_idx, :])
        init_cf.append(cluster_feature[return_idx, :])
        init_E.append(normalized_E[return_idx])


        feature_matrix = np.delete(feature_matrix, return_idx, axis=0)
        cluster_feature = np.delete(cluster_feature, return_idx, axis=0)
        normalized_E = np.delete(normalized_E, return_idx, axis = 0)
        # print('coherence', coherence)


    remain_corr = feature_matrix
    remain_E = normalized_E
    remain_cf = cluster_feature

    return np.array(init_corr), np.array(init_cf), init_E, np.array(remain_corr), np.array(remain_cf),remain_E

def RIPless_coherence(A):
    A = np.array(A)
    m, d = A.shape
    A /= np.linalg.norm(A, axis=1).reshape(m, 1)

    ATA = np.dot(np.transpose(A), A)
    mus = np.diag(ATA)

    coherence = np.max(mus)
    # coherence = np.max(np.abs(ATA))


    return coherence


def RIPless_sampling(feature_matrix, normalized_E, num_init= 10):
    """
    void cluster is always 1, contibute nothing in PCA analysis
    ewald term is highyly correlated, we treat it as an offset and it's not considered in structure selection

    """
    N, M = feature_matrix[:,0:-1].shape
    cluster_feature = feature_matrix[:,1:-1]
    # cluster_feature = feature_matrix[:,0:]
    normalized_E = np.array(normalized_E)
    total_indices = [i for i in range(N)]
    # M: number of valid cluster (void and ewald term is deleted)
    # N: number of structures in reservoir

    init_corr = []
    init_cf = []
    init_E = []

    init_corr.append(feature_matrix[0,:])
    init_cf.append(cluster_feature[0,:])
    init_E.append(normalized_E[0])

    feature_matrix = np.delete(feature_matrix, 0, axis=0)
    cluster_feature = np.delete(cluster_feature, 0, axis=0)
    normalized_E = np.delete(normalized_E, 0, axis = 0)

    for i in range(num_init):
        return_idx = 0
        coherence = 10000
        for j in range(feature_matrix.shape[0]):
            temp_cf = init_cf.copy()
            temp_cf.append(cluster_feature[j,:])
            temp_coherence = RIPless_coherence(temp_cf)
            if temp_coherence < coherence:
                return_idx = j
                coherence = temp_coherence

        init_corr.append(feature_matrix[return_idx, :])
        init_cf.append(cluster_feature[return_idx, :])
        init_E.append(normalized_E[return_idx])


        feature_matrix = np.delete(feature_matrix, return_idx, axis=0)
        cluster_feature = np.delete(cluster_feature, return_idx, axis=0)
        normalized_E = np.delete(normalized_E, return_idx, axis = 0)
        # print('coherence', coherence)


    remain_corr = feature_matrix
    remain_E = normalized_E

    return np.array(init_corr), np.array(init_E), np.array(remain_corr), np.array(remain_E)


def global_coherence(A):
    A = np.array(A)
    m, d = A.shape
#     print(A.shape)
    A /= np.linalg.norm(A, axis=1).reshape(m, 1)
    AAT = np.dot(A, np.transpose(A))
    # coherence = 1000
    for i in range(m):
        AAT[i,i] = 0

    coherence = np.sum(AAT) / (m**2 - m)
    return coherence

def global_coherence_sampling(feature_matrix, normalized_E, num_init= 10):
    """
    void cluster is always 1, contibute nothing in PCA analysis
    ewald term is highyly correlated, we treat it as an offset and it's not considered in structure selection

    """
    N, M = feature_matrix[:,0:-1].shape
    cluster_feature = feature_matrix[:,0:-1]
    # cluster_feature = feature_matrix[:,0:]
    normalized_E = np.array(normalized_E)
    total_indices = [i for i in range(N)]
    # M: number of valid cluster (void and ewald term is deleted)
    # N: number of structures in reservoir

    init_corr = []
    init_cf = []
    init_E = []

    init_corr.append(feature_matrix[0,:])
    init_cf.append(cluster_feature[0,:])
    init_E.append(normalized_E[0])

    feature_matrix = np.delete(feature_matrix, 0, axis=0)
    cluster_feature = np.delete(cluster_feature, 0, axis=0)
    normalized_E = np.delete(normalized_E, 0, axis = 0)

    for i in range(num_init):
        return_idx = 0
        coherence = 100
        for j in range(feature_matrix.shape[0]):
            temp_cf = init_cf.copy()
            temp_cf.append(cluster_feature[j,:])
            temp_coherence = global_coherence(temp_cf)
            if temp_coherence < coherence:
                return_idx = j
                coherence = temp_coherence

        init_corr.append(feature_matrix[return_idx, :])
        init_cf.append(cluster_feature[return_idx, :])
        init_E.append(normalized_E[return_idx])


        feature_matrix = np.delete(feature_matrix, return_idx, axis=0)
        cluster_feature = np.delete(cluster_feature, return_idx, axis=0)
        normalized_E = np.delete(normalized_E, return_idx, axis = 0)
        # print('coherence', coherence)


    remain_corr = feature_matrix
    remain_E = normalized_E


    return np.array(init_corr), init_E, np.array(remain_corr), remain_E



def residue_coherence(A):
    A = np.array(A)
    m, d = A.shape
#     print(A.shape)
    A /= np.linalg.norm(A, axis=1).reshape(m, 1)

    cohe  = 1000
    for i in range(m):
        for j in range(m):
            new_cohe = np.linalg.norm(A[i,:] - A[j,:])
            if new_cohe < cohe:
                cohe = new_cohe
    return cohe


def residue_coherence_sampling(feature_matrix, normalized_E, num_init= 10):
    """
    void cluster is always 1, contibute nothing in PCA analysis
    ewald term is highyly correlated, we treat it as an offset and it's not considered in structure selection

    """
    N, M = feature_matrix[:,0:-1].shape
    cluster_feature = feature_matrix[:,0:-1]
    # cluster_feature = feature_matrix[:,0:]
    normalized_E = np.array(normalized_E)
    total_indices = [i for i in range(N)]
    # M: number of valid cluster (void and ewald term is deleted)
    # N: number of structures in reservoir

    init_corr = []
    init_cf = []
    init_E = []

    init_corr.append(feature_matrix[0,:])
    init_cf.append(cluster_feature[0,:])
    init_E.append(normalized_E[0])

    feature_matrix = np.delete(feature_matrix, 0, axis=0)
    cluster_feature = np.delete(cluster_feature, 0, axis=0)
    normalized_E = np.delete(normalized_E, 0, axis = 0)

    for i in range(num_init):
        return_idx = 0
        coherence = 100
        for j in range(feature_matrix.shape[0]):
            temp_cf = init_cf.copy()
            temp_cf.append(cluster_feature[j,:])
            temp_coherence = residue_coherence(temp_cf)
            if temp_coherence < coherence:
                return_idx = j
                coherence = temp_coherence

        init_corr.append(feature_matrix[return_idx, :])
        init_cf.append(cluster_feature[return_idx, :])
        init_E.append(normalized_E[return_idx])


        feature_matrix = np.delete(feature_matrix, return_idx, axis=0)
        cluster_feature = np.delete(cluster_feature, return_idx, axis=0)
        normalized_E = np.delete(normalized_E, return_idx, axis = 0)
        # print('coherence', coherence)


    remain_corr = feature_matrix
    remain_E = normalized_E


    return np.array(init_corr), init_E, np.array(remain_corr), remain_E
