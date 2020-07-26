#!/usr/bin/env python

"""
Script for parallelization of percolation analysis
"""

__author__ = "Bin Ouyang"
__date__ = "2020.07.25"
__version__ = "0.1"


import os
import numpy as np
from smol.sro_kit.DiffNetworkAnalyzer import PercolationAnalyzer as PA
import multiprocessing
from functools import partial
import time
import argparse


def parse_percolation(root, allow_step=1, diffusor='Li', cations=None):
    """
    Function to parse percolation
    """
    print('allow_step = {}, diffusor = {}, cations = {}'.format(allow_step, diffusor, cations))
    start_time = time.time()
    gzipped_pos = '{}/POSCAR.gz'.format(root)
    bzipped_pos = '{}/POSCAR.bz2'.format(root)
    is_gzip, is_bzip = False, False
    if os.path.isfile(gzipped_pos):
        os.system('gunzip {}'.format(gzipped_pos))
        is_gzip = True
        print('gunzip structure file: {}'.format(gzipped_pos))
    if os.path.isfile(bzipped_pos):
        os.system('bunzip2 {}'.format(bzipped_pos))
        IsBzip = True
        print('bunzip structure file: {}'.format(bzipped_pos))

    try:
        PA0 = PA.from_poscar('{}/POSCAR'.format(root))
    except:
        raise('Cannot parse {}/POSCAR'.format(root))
    perco_diffusor_lst = PA0.get_percolating_Li_fast(allow_step=allow_step)
    n_diffusor = len(PA0.structure.get_ion_indices(diffusor))
    with open('{}/perco_li_inds.dat'.format(root),'w') as fid:
        fid.write(str(perco_diffusor_lst))
    perco_li_percent = 1.0 * len(perco_diffusor_lst) / n_diffusor
    print('Time spent on path {} is {}'.format(root,time.time()-start_time))
    print('Percolating amount of Li is {}'.format(perco_li_percent))
    if is_bzip:
        print('bzip {}/POSCAR'.format(root))
        flag = os.popen('bzip2 {}/POSCAR'.format(root)).read()
    elif is_gzip:
        print('gzip {}/POSCAR'.format(root))
        flag = os.popen('gzip {}/POSCAR'.format(root)).read()

    return perco_li_percent


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Perform percolation analysis in parallel")
    parser.add_argument('-conf', help="Configuration Folder Name", type=str, default=None)
    parser.add_argument('--ncpu', help="Number of CPU used (default: 16)", type=int, default=16)
    parser.add_argument('--ncut', help="Cutoff distance 1 means only 0-TM (default: 1)", type=int, default=1)
    parser.add_argument('--cations', help="List of cations (default: None)", type=str, nargs='+', default=None)
    parser.add_argument('--diffusor',help='The symbol of diffusor (default: Li)', type=str, default='Li+')
    args = parser.parse_args()

    start_time = time.time()
    dirs_to_parse = []
    print('Performing percolation analysis on directory: {}'.format(args.conf))

    for root, dir, files in os.walk(args.conf):
        if ('POSCAR' in files) or ('POSCAR.gz' in files) or ('POSCAR.bz2' in files) :
            dirs_to_parse.append(root)

    print('{} structures to be analyzed'.format(len(dirs_to_parse)))

    pool = multiprocessing.Pool(args.ncpu)
    print('Pooling the supercell size for data extraction')
    perco_runner = partial(parse_percolation, allow_step=args.ncut, diffusor=args.diffusor, cations=args.cations)
    perco_li_percents = pool.map(perco_runner, dirs_to_parse)
    pool.close()
    pool.join()

    avg_perco_percent = np.mean(perco_li_percents)
    print('Average portion of percolating Li is {}'.format(avg_perco_percent))
    print("--- Overall time is %s seconds ---" % (time.time() - start_time))


