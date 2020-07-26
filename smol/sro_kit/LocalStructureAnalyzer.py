"""
Modules for performing SRO analysis for ionic compound
"""


__author__ = "Bin Ouyang"
__date__ = "2020.07.26"
__version__ = "0.1"

import os
from ionicstructure import PoscarWithChgDecorator
from DiffNetworkAnalyzer import PercolationAnalyzer as PA
import numpy as np
from pprint import pprint


class SROAnalyzer(object):
    """
    Perform short range order related analysis of structures
    """

    def __init__(self, structure):
        self.structure = structure

    def get_sro_nn(self, atomsym, nnsyms, dcut):
        """
        Get the short range order parameter for given structure

        Args:
            atomsym: Host atom symbol, SRO will be on all bonds with this atom(with charge decorator)
            nnsyms: List of neighboring site symbols to be tracked (usually use this to opt-out Cations/Anions)
            dcut: Distance Cutoff

        Returns:
            pair_info:
                Collection of all bond information (site, dist, index, image) for each pair.
                Two levels of key being host atom indices and neighboring atom type
            pair_inds:
                All indices that form bond with the host atom,
                Two levels of key being the atom indices and neighboring atom type
        """
        nn_lsts = self.structure.get_all_neighbors(dcut, include_index=True)
        atom_inds = self.structure.get_ion_indices(atomsym)
        # Gather the collection of SRO dictionary
        pair_info = {}
        pair_inds = {}
        for i, ind in enumerate(atom_inds):
            pair_info[ind], pair_inds[ind] = {}, {}
            nnlst = nn_lsts[ind]
            for siteind, (site, dist, index, image) in enumerate(nnlst):  # Enumerate all the sites
                ionsym = str(site.specie)
                if ionsym not in nnsyms:
                    continue
                if ionsym not in pair_info[ind]:
                    pair_info[ind][ionsym], pair_inds[ind][ionsym] = [], []
                pair_info[ind][ionsym].append((site, dist, index, image))
                pair_inds[ind][ionsym].append(siteind)
        return pair_info, pair_inds

    def get_sro_para(self, atomsym, nnsyms, pair_inds):
        """
        Get the Warren-Cowley SRO parameter
            1. The random limit is calculated from N(Host)/N(Host)+N(Neighboring sites considered)

        Args:
            atomsym: Host atom symbol, SRO will be on all bonds with this atom(with charge decorator)
            nnsyms: List of neighboring site symbols to be tracked (usually use this to opt-out Cations/Anions)
            pair_info, pair_inds: obtained from get_sro_nn and get_sro_shell

        Return:
            sro_para_avg: short range order parameter dictionary with key being bond type (averaged among all pairs)
            sro_para_all: short range order parameter dictionary, the dictionary has key being bond type and member
                          being a list of sro parameter
        """
        # Obtain random limit pair correlation
        valid_ion_inds=[]
        for nnsym in nnsyms:
            valid_ion_inds = valid_ion_inds + self.structure.get_ion_indices(nnsym)
        random_limit_pair_correlation = {}
        for elesym in nnsyms:
            random_limit_pair_correlation['{}{}'.format(atomsym,elesym)] = \
                len(self.structure.get_ion_indices(elesym))/len(valid_ion_inds)
        #pprint(random_limit_pair_correlation)

        # Evaluate the SRO parameter
        sro_para_all = {}
        for atomind, eledictlst in pair_inds.items():
            nnsum = 0
            for elesym in eledictlst:
                nnsum += len(eledictlst[elesym])
            sro_para_atom_dict = {}
            for elesym, symindlst in eledictlst.items():
                bondname = '{}{}'.format(atomsym, elesym)
                if bondname not in sro_para_all:
                    sro_para_all[bondname] = []
                sro_value = 1.0-1.0*len(symindlst)/nnsum/random_limit_pair_correlation[bondname]
                sro_para_all[bondname].append(sro_value)
        sro_para_avg = {bondname: np.average(sro_values) for bondname, sro_values in sro_para_all.items()}
        return sro_para_avg, sro_para_all

    def get_sro_shell(self, atomsym, nnsyms, dmax, dmin=0.5):
        """
        ***More generalized method compared with get_sro_nn***
        Get the short range order parameter for given structure with a shell from DMin to DMax

        Args:
            atomsym: Host atom symbol, SRO will be on all bonds with this atom(with charge decorator)
            nnsyms: List of neighboring site symbols to be tracked (usually use this to opt-out Cations/Anions)
            dmin,dmax: The inner and outer radius

        Returns:
            pair_info:
                Collection of all bond information (site, dist, index, image) for each pair.
                Two levels of key being host atom indices and neighboring atom type
            pair_inds:
                All indices that form bond with the host atom,
                Two levels of key being the atom indices and neighboring atom type
        """
        rmean = (dmin + dmax) / 2
        dr = (dmax - dmin) / 2
        atominds = self.structure.get_ion_indices(atomsym)
        # Gather the SRO dictionary
        pair_info = {}
        pair_inds = {}
        for i, ind in enumerate(atominds):
            pair_info[ind], pair_inds[ind] = {}, {}
            nnlst = self.structure.get_neighbors_in_shell(self.structure[ind].coords, rmean,
                                                          dr, include_index=True, include_image=True);
            for siteind, (site, dist, index, image) in enumerate(nnlst):  # Enumerate all the sites
                ionsym = str(site.specie)
                if ionsym not in nnsyms:
                    continue
                if ionsym not in pair_info[ind]:
                    pair_info[ind][ionsym], pair_inds[ind][ionsym] = [], []
                pair_info[ind][ionsym].append((site, dist, index, image))
                pair_inds[ind][ionsym].append(index)
            # Make sure all nnsyms are considered
            for nnsym in nnsyms:
                if nnsym not in pair_info[ind]:
                    pair_inds[ind][nnsym] = []
                    pair_info[ind][nnsym] = []
        return pair_info, pair_inds

    @staticmethod
    def from_file(posname):
        """
        Initialize the SRO analyzer from POSCAR
        """
        chargedstructure = PoscarWithChgDecorator.from_file(posname).structure
        return SROAnalyzer(chargedstructure)


class LiExcessSROAnalyzer(SROAnalyzer):
    """
    SRO analysis specifically in lithium excess FCC based materials
    """
    def __init__(self, structure):
        super(LiExcessSROAnalyzer, self).__init__(structure)

    def get_tetinfo(self, cations, is_return_structure):
        """
        get cation tet coordination environment
        """
        PA0 = PA(self.structure, cations)
        countlst, Litet_inds = PA0.getChanNums(self.structure, cations)

        if is_return_structure:
            site_list = []
            for ind in Litet_inds:
                site_list.append(self.structure[ind])
            perco_structure = Structure.from_sites(site_list)
            return countlst, Litet_inds, perco_structure
        return countlst, Litet_inds

    @staticmethod
    def from_file(posname):
        '''
        Initialize the SRO analyzer from POSCAR

        '''
        chargedstructure = PoscarWithChgDecorator.from_file(posname).structure
        return LiExcessSROAnalyzer(chargedstructure)


