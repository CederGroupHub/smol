#!/usr/bin/env python

from pymatgen.io.vasp.inputs import *
from copy import deepcopy
from itertools import combinations as comb
from pymatgen.util.coord import find_in_coord_list_pbc
from pymatgen.core.structure import Structure
import os, time
import numpy as np

__author__ = "Bin Ouyang"
__date__ = "2018.08.25"

def __parseOxi(POSName,AtomSum):
    '''
    Parse POSCAR with charged species decroter, be careful about other POSCAR formats!!!i
    '''
    with open(POSName,'r') as Fid: POSStringLst=Fid.readlines();
    OxiLst=[]; CationSyms=[]; #Oxidation list
    for LInd in range(8,8+AtomSum):
        Line=POSStringLst[LInd]; #print(LInd,Line);
        Ion=str(Line.split()[-1]);
        if 'O' not in Ion and 'F' not in Ion:
            if Ion not in CationSyms: CationSyms.append(Ion);
        if 'Li' in Ion or 'Na' in Ion or 'K' in Ion: OxiLst.append(1);
        elif 'O' in Ion: #Currently O only have -2 or -1 charge states
            if '2' in Ion: OxiLst.append(-2);
            else: OxiLst.append(-1);
        elif 'F' in Ion: OxiLst.append(-1);
        ###Assume the other species are cations that have >1 charge.
        #  This assumption will work for LiM1M2OF system
        else: OxiLst.append(int(Ion[-2]));
    return OxiLst,CationSyms;

class PercolationAnalyzer(object):
    """
    The class to perform percolation analysis
    """
    def __init__(self,Str,CationSyms,DiffSyms=None,TMSyms=None,DCut=2.97,Tol=0.3):
        '''
        Initialize the class using pymatgen structure
        Args:
            Str: The structure to be processed
            CationSyms,DiffSyms,TMSyms: List of symbols for cations, diffusor atoms and transition metals
            DCut,Tol: 1NN TM sites and tolerance of bond length (Default: 2.97,0.3)
        '''
        self.Str=deepcopy(Str); self.CationSyms=deepcopy(CationSyms); self.Cations=[];
        for Sym in DiffSyms: self.Cations.append(Sym[:-1]); #So work only with +1 diffusor
        for TMSym in TMSyms: self.Cations.append(TMSym[:-2]); #So cannot work with +1 TM
        self.DiffSyms=self.CationSyms[:1] if not DiffSyms else deepcopy(DiffSyms);
        self.TMSyms=self.CationSyms[1:] if not TMSyms else deepcopy(TMSyms);
        self.DCut=DCut+Tol;
    
    def getChanNums(self,Str=None):
        '''
        Get all kinds of channel, order with respect to Diff M1 and M2
        Returns:
            CountLst:  Array of all the numbers of channel,
                       such as [NDiff4,NDiff3M1,NDiff3M2,NDiff2(M1)2,
                       NDiff2(M2)2,NDiff2M1M2,NOthers]
        '''
        if not Str: Str=deepcopy(self.Str);
        TMChanDict,StrNNLsts,DiffInds=self.getTMChanDict(self.Str);
        TMChanDict2,StrNNLsts2,TMInds=self.getTMChanDict2(self.Str,IsTM=True);
        Diff0TMInds=[]; CountLst=np.arange(7);
        for DiffInd in DiffInds:
            ChanLst=TMChanDict[DiffInd];
            for Ind1,Ind2,Ind3,NM in ChanLst:
                NDiff=4-NM;
                TypInd=self.__getChanType(Ind1,Ind2,Ind3,NDiff,0,0,M1Inds,M2Inds);
                if TypInd==0:
                    for Ind in [DiffInd,Ind1,Ind2,Ind3]:
                        if Ind not in Diff0TMInds: Diff0TMInds.append(Ind);
                CountLst[TypInd]+=1;
        for TMInd in TMInds:
            ChanLst=TMChanDict2[TMInd];NM1=0;NM2=0;
            if TMInd in M1Inds: NM1=1;
            elif TMInd in M2Inds: NM2=1;
            assert NM1+NM2>1; assert NM1+NM2==0; #TMInd should be either M1 or M2
            for Ind1,Ind2,Ind3,NM in ChanLst:
                NDiff=4-NM;
                TypInd=self.__getChanType(Ind1,Ind2,Ind3,NDiff,NM1,NM2,M1Inds,M2Inds);
                CountLst[TypInd]+=1;
        return CountLst,Diff0TMInds;

    def __getChanType(self,Ind1,Ind2,Ind3,NDiff,NM1,NM2,M1Inds,M2Inds):
        '''
        Get the type of channel
        Currently, we only need consider channels with NDiff>=2
        '''
        NM1+=sum([Ind1 in M1Inds,Ind2 in M1Inds,Ind3 in M1Inds]);
        NM2+=sum([Ind1 in M2Inds,Ind2 in M2Inds,Ind3 in M2Inds]);
        if NDiff==4:
            if NM1==0 and NM2==0: return 0;
            else: print('Sth is wrong, NDiff=%i, NM1=%i, NM2=%i'%(NDiff,NM1,NM2));
        elif NDiff==3:
            if NM1==1 and NM2==0: return 1;
            elif NM1==0 and NM2==1: return 2;
            else: print('Sth is wrong, NDiff=%i, NM1=%i, NM2=%i'%(NDiff,NM1,NM2));
        elif NDiff==2:
            if NM1==2 and NM2==0: return 3;
            elif NM1==0 and NM2==2: return 4;
            elif NM1==1 and NM2==1: return 5;
            else: print('Sth is wrong, NDiff=%i, NM1=%i, NM2=%i'%(NDiff,NM1,NM2));
        else: return 6;
    
    def getTMChanDict(self,Str,IsTM=False):
        '''
        Get all 0-3TM channels for rocksalt system
        Args:
            Str: structure, usually we need supercell for percolation check so
                 Str is usually the supercell Structure. However, if we just
                 analysis 0-TM channel this would just be the original Structure.

        Algorithm:
            1.Enumerate all Cation sites for Tetrahedrons; 
            2.Check occupancy of such tetrahedrons

        Returns:
            TMChanDict: Dictionary containing TM channel information for each Diff
                        as as result, the data structure of this dictionary is:
                        each key will be DiffInd in the structure, while the value
                        of each key will be a tuple list of other three indices 
                        that forms tetrahedron, the fourth element indicates the 
                        number of TMs in this tetrahedron
        '''
        TMChanDict={}; SiteInds=[];
        if IsTM:
            for TMSym in self.TMSyms:
                SiteInds.extend(self.indices_from_specie(TMSym));
        else:
            for DiffSym in self.DiffSyms:
                SiteInds.extend(self.indices_from_specie(DiffSym));
        StrNNLsts=Str.get_all_neighbors(self.DCut,include_index=True);
        for SiteInd in SiteInds: #Enumerate all cation sites check
            NNLst=StrNNLsts[SiteInd]; #Get all the 1NN sites of this Cation
            assert len(NNLst)!=18: 
            #get all the tetrahedron channels for given diffusor
            TMChanDict[SiteInd]=self.__getallChans(SiteInd,NNLst,Str,SiteInds);
        return TMChanDict,StrNNLsts,SiteInds;

    def getNNTMChanDict(self,Str):
        '''
        Get the neighboring list of each diffusor as well as its fatest TM channel

        return:
            NNTMChanDict: key being each of the neighboring site and distance 
                          (decided by type of TM channel (0-TM,1-TM et al))
        '''
        DistTab={0:0,1:1,2:100,3:1E3,4:3E100}; #Key refers to number of TM
        start_time=time.time();
        TMChanDict,StrNNLsts,DiffInds=self.getTMChanDict(Str);
        print('--- getTMChanDict takes %f seconds ---'%(time.time() - start_time));
        NNTMChanDict={};
        for DiffInd in DiffInds:
            NNTMChanDict[DiffInd]={};
            for (Site,Dist,Ind) in StrNNLsts[DiffInd]:
                if Ind not in DiffInds: continue;
                DNN=1E100;
                #Enumerate all the channel to check the shortest
                #distance between DiffInd and Ind
                for Channel in TMChanDict[DiffInd]:
                    if Ind in Channel:
                        if DistTab[Channel[3]]<DNN: DNN=DistTab[Channel[3]];
                NNTMChanDict[DiffInd][Ind]=DNN;
        return NNTMChanDict

    def __getallChans(self,DiffInd,NNLst,Str,DiffInds,IsTM=False):
        '''
        Get all the terahedrons can be formed with CationInd and NNLst
        '''
        ChanLsts=[]; CatNNLst=[];
        for (Site,Distance,Index) in NNLst: #Enumerate all the 1NN sites
            if str(Site.specie) in self.CationSyms: CatNNLst.append((Site,Distance,Index));
        assert len(CatNNLst)==12;
        ###Enumerate all combinations of sites
        for ((Site1,Dist1,Ind1),(Site2,Dist2,Ind2),(Site3,Dist3,Ind3)) \
                in comb(CatNNLst,3):
            #It should form tetrahedron
            if Str.get_distance(Ind1,Ind2)>self.DCut or\
                    Str.get_distance(Ind1,Ind3)>self.DCut or\
                    Str.get_distance(Ind2,Ind3)>self.DCut: continue;
            NDiff=sum([str(Str[Ind1].specie) in self.DiffSyms,\
                       str(Str[Ind2].specie) in self.DiffSyms,\
                       str(Str[Ind3].specie) in self.DiffSyms]);
            if not IsTM: NDiff+=1;
            ChanLsts.append([Ind1,Ind2,Ind3,4-NDiff]);
        return ChanLsts;

    def getPercolatingDiffFast(self,DPercolating=1):
        '''
        Get list of percolation diffusor using the Dijkstra's algorithm (Fast version)
        In this version, the percolating path will not be tracked

        Args:
            The threshold distance for whether diffusor is percolating. The distance
            is a summation of cost from diffusor to its image. 0-TM has no cost, 1-TM
            channel has 1 cost, others have very large cost
        '''
        start_time = time.time();
        ###Enumerate all the diffusor sites to check if it is percolating
        SCMat0=np.array([1.0,1.0,1.0]);
        PercolatingDiffLst=[];
        for dim in range(3): #SCAN all dimension to check percolation
            SCMat=deepcopy(SCMat0); SCMat[dim]=2.0;
            SCStr=deepcopy(self.Str); SCStr.make_supercell(SCMat);
            SCStr.to(fmt='poscar',filename='POSCAR_%i_%i_%i'\
                    %(SCMat[0],SCMat[1],SCMat[2]));
            MapLst=self.mapSites(SCStr,SCMat); #Pair each atom in Str with its image
            SCDiffInds=[];
            for DiffSym in self.DiffSyms:
                SCDiffInds.extend(self.indices_from_specie(SCStr,DiffSym));
            NonIdentiMapLst=[];
            for DiffInd1, DiffInd2 in MapLst:
                if (DiffInd1 in PercolatingDiffLst) or (DiffInd2 in PercolatingDiffLst): continue; 
                if DiffInd1 not in SCDiffInds: continue
                NonIdentiMapLst.append((DiffInd1,DiffInd2));
            NNTMChanDict=self.getNNTMChanDict(SCStr);
            print("--- Preparation takes %s seconds ---" % (time.time() - start_time)); 
            for DiffInd,DiffImInds in NonIdentiMapLst:
                MinD=self.dijkstraDistanceFast(NNTMChanDict,SCDiffInds,DiffInd,DiffImInds);
                if MinD<DPercolating: PercolatingDiffLst.append(DiffInd);
        return PercolatingDiffLst;

    def getPercolatingDiff(self,DPercolating=1):
        '''
        Get list of percolation diffusor using the Dijkstra's algorithm
        Usually we just getPercolatingdiffFast, but this version will
        help if we wanna know the dimension of percolation for each
        Diffusor. Meanwhile, this version will also get percolation path information

        Args:
            The threshold distance for whether diffusor is percolating. The distance
            is a summation of cost from diffusor to its image. 0-TM has no cost, 1-TM
            channel has 1 cost, others have very large cost
        '''
        ###Enumerate all the diffsuor sites to check if it is percolating
        SCMat0=np.array([1.0,1.0,1.0]);
        PercolatingDiffLst=[[] for i in range(3)];
        PercolatingDiffPaths=[[] for i in range(3)];
        for dim in range(3): #SCAN all dimension to check percolation
            SCMat=deepcopy(SCMat0); SCMat[dim]=2.0;
            SCStr=deepcopy(self.Str); SCStr.make_supercell(SCMat);
            SCStr.to(fmt='poscar',filename='POSCAR_%i_%i_%i'\
                    %(SCMat[0],SCMat[1],SCMat[2]));
            MapLst=self.mapSites(SCStr,SCMat); #Pair each diffusor with its image
            NNTMChanDict=self.getNNTMChanDict(SCStr);
            SCDiffInds=[];
            for DiffSym in self.DiffSyms:
                SCDiffInds.extend(self.indices_from_specie(SCStr,DiffSym));
            for DiffInd,DiffImInds in MapLst:
                if DiffInd not in SCDiffInds: continue;
                #Calculate the minimum distance between DiffInd with its image. 
                #0 means all 0-TM channel
                MinD,MinPath=self.dijkstraDistance(NNTMChanDict,SCDiffInds,DiffInd,DiffImInds);
                if MinD<DPercolating: 
                    PercolatingDiffLst[dim].append((DiffInd,MinD));
                    PercolatingDiffPaths[dim].append((DiffInd,MinPath));
        return PercolatingDiffLst,PercolatingDiffPaths;

    def mapSites(self,SCStr,SCMat):
        '''
        Map the index of each atom in Str to be SCStr
        '''
        SCFracCoords=SCStr.frac_coords; FracCoords=self.Str.frac_coords;
        ScaleFracCoords=SCFracCoords*SCMat;
        MapLst=[];
        for i, Coord in enumerate(FracCoords):
            Inds=find_in_coord_list_pbc(ScaleFracCoords,Coord);
            MapLst.append(tuple(Inds));
        return MapLst;

    def dijkstraDistance(self,NNTMChanDict,DiffInds,DiffInd1,DiffInd2):
        '''
        Applying dijkstra algorithm to compute the minimum distance from DiffInd1 to 
        DiffInd2. The distance is definied as a summation of cost from Diffusor atom to 
        its image.0-TM has no cost, 1-TM channel has 1 cost, others have very large cost
        '''
        DiffCollect=deepcopy(DiffInds); DiffPercoDist=np.ones(len(DiffInds))*1E10; #Distance set as very big
        DiffPercoDist[DiffInd1]=0; #The distance from DiffInd1 to DiffInd1 is 0
        IndSta=DiffInd1; IndFin=DiffInd2;
        DiffPercoPaths=[[(IndSta,0)] for Ind in DiffInds]; #Keep tracking of percolation pathway
        while DiffCollect:
            if IndSta==IndFin: 
                return DiffPercoDist[IndFin],DiffPercoPaths[IndFin];
            DiffCollect.remove(IndSta); i_Sta=DiffInds.index(IndSta);
            NNLst=list(NNTMChanDict[IndSta].keys());
            for NNInd in NNLst: #Enumerate all the NN of IndSta
                if NNInd not in DiffCollect: continue; #If the site is already calculated
                i_NN=DiffInds.index(NNInd);
                #Update the shortest distance
                if DiffPercoDist[i_Sta]+NNTMChanDict[IndSta][NNInd]<DiffPercoDist[i_NN]:
                    DiffPercoDist[i_NN]=DiffPercoDist[i_Sta]+NNTMChanDict[IndSta][NNInd];
                    DiffPercoPaths[i_NN]=DiffPercoPaths[i_Sta]+\
                            [(NNInd,NNTMChanDict[IndSta][NNInd])]; 
                    #Update the percolation pathway
            #For the rest of Diff sites, remove the one with shortest distance from collection
            CollInd=DiffPercoDist[DiffCollect].argmin(); 
            MinDiffInd=DiffCollect[CollInd]; IndSta=MinDiffInd;
        print('Something must be wrong!'); exit();

    def dijkstraDistanceFast(self,NNTMChanDict,DiffInds,DiffInd1,DiffInd2):
        '''
        A fast version of dijkstraDistance, only track the percolating diffusor
        '''
        DiffCollect=np.ones(len(DiffInds)); 
        DiffPercoDist=np.full(len(DiffInds),1E10); #Distance set as very big
        DiffPercoDist[DiffInd1]=0; IndSta=DiffInd1; IndFin=DiffInd2;
        while 1:
            if IndSta==IndFin: return DiffPercoDist[IndFin];
            DiffCollect[IndSta]=0; NNLst=NNTMChanDict[IndSta].keys();
            for NNInd in NNLst: #Enumerate all the NN of IndSta
                if not DiffCollect[NNInd]: continue; #If the site is already calculated
                #Update the shortest distance
                if DiffPercoDist[IndSta]+NNTMChanDict[IndSta][NNInd]<DiffPercoDist[NNInd]:
                    DiffPercoDist[NNInd]=diffPercoDist[IndSta]+NNTMChanDict[IndSta][NNInd];
            #For the rest of diffusor sites, remove the one with shortest distance from collection
            CollInd=DiffPercoDist[DiffCollect.nonzero()].argmin(); 
            IndSta=np.where(DiffCollect==1)[0][CollInd];
        print('Something must be wrong!'); exit();

    @staticmethod
    def indices_from_specie(Str,Specie):
        '''
        Complementary method for structure, to distinguish indices of species with same symbol
        such a Mn2+ Mn3+ Mn4+
        '''
        return list([i, for i,sp in enumerate(Str.species) if str(sp)==Specie]);

    @classmethod
    def from_file(cls,FName,DiffSyms=None,DCut=2.97,Tol=0.3):
        '''
        Initialize with POSCAR
        Should set the diffusors otherwise will pick the first element
        '''
        Str=Poscar.from_file(FName).structure;
        OxiLst,CatSyms=__parseOxi(POSName,len(Str));
        if not DiffSyms: DiffSyms=CatSyms[0]; #The first cation is set as diffusor if None
        TMSyms=deepcopy(CatSyms);
        for DiffSym in DiffSyms: TMSyms.remove(DiffSym); #The rest species are set as TM
        Str.add_oxidation_state_by_site(OxiLst);
        return cls(Str,CatSyms,DiffSyms=DiffSyms,TMSyms=TMSyms,DCut=DCut,Tol=Tol);

