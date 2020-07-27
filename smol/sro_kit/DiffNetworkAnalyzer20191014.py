#!/usr/bin/env python

from pymatgen.io.vasp.inputs import *
from copy import deepcopy
from itertools import combinations as comb
from DRXTools import loadPOSwithCS, getCSIndices
from pymatgen.util.coord import find_in_coord_list_pbc
import os, time
import numpy as np
from numpy import array
from itertools import combinations_with_replacement as cwr

__author__ = "Bin Ouyang"
__date__ = "2019.10.14"
####2018.11.05,There is multiple counting issue in the getChanNums, may not affect
###the current analysis, each TM/Li is associated with 8 tetrahedron
####2018.12.12,Generalize the channel counting to 1TM and 2TM systems with
####charge decorator
####For future coding
####1. Supporting differnt amount of Cations
####2019.01.08,Further simplify all the methods, do not distinguish TM indices and Li indices in some process
####Can be further optimized by eliminate all Li indices that are already considered in several methods
####2019.03.19,Now can deal with all possible cation amounts
####2019.10.14,Speed up the following methods: getTetraDict, getChanDistDict

class PercolationAnalyzer(object):
    """
    The class to perform percolation analysis
    """
    def __init__(self,OStr,Cations,DCut=2.97,Tol=0.3):
        '''
        Initialize the class using pymatgen structure
        Args:
            Str: The structure to be processed
            Cations: The list of cations with charge decorator
            DCut,Tol: 1NN TM sites and tolerance of bond length (Default: 2.97,0.3)
        
        Class attributes:
            Str: The input structure
            Cations: The list of cations with charge decorator
        '''
        self.OStr=deepcopy(OStr); self.Cations=deepcopy(Cations);
        self.DCut=DCut+Tol;

        TMOccuLst=list(cwr(range(self.CationNum),4));
        print(self.Cations);
        print('TMOccuLst={}'.format(str(TMOccuLst)));
    
    @property
    def CationNum(self):
        return len(self.Cations);
    
    def getCationInds(self,Str):
        '''
        Get the indices for all Cations, Str can be primary cell or supercell
        '''
        Inds=[];
        for Cation in self.Cations: Inds.extend(getCSIndices(Str,Cation));
        return Inds;
    
    def getCationIndLsts(self,Str):
        '''
        Get list of list of indices for each Cation
        '''
        return [getCSIndices(Str,Cation) for Cation in self.Cations];

    def classify0TMLi(self,Str,DPercolating=1):
        '''
        Divide all the 0TM Li into different categories

        1. 0TM no percolating Li
        2. Shared by 1 0TM, 2 0TM, 3 0TM to 8 0TM
        '''
        PercolatingLiLst=self.getPercolatingLi(DPercolating=DPercolating);
        PercoLiLst=list(array(PercolatingLiLst)/2);
        CountLst,Li0TMInds,TM0Chans=self.getChanNums(Str,IsTrack0TM=True);
        self.chkPercoConsistent(array(Li0TMInds),PercoLiLst,TM0Chans);
        NoPercoLi=[]; Share0TMNum=np.zeros(8);
        for Ind in Li0TMInds:
            if Ind not in PercoLiLst: NoPercoLi.append(Ind);
            Share0TMNum[TM0Chans[Ind]//4-1]+=1; #Should divide quadro counting
        return NoPercoLi,Share0TMNum,TM0Chans,Li0TMInds,PercoLiLst,CountLst;

    def chkPercoConsistent(self,TMLiArry,PercoLiLst,TM0Chans):
        '''
        Check for the following criterion:
        1. PercoLi has to be subset of TMLi.
        2. All TMLiArry should have nonzero value in TM0Chans.
        3. All nonzero value should be TMLiArry.
        4. TM0Chans should be no more than 8*4, and should be integer times of 4
        '''
        ###Check criterion 1
        CompResult=all(Ind in TMLiArry for Ind in PercoLiLst);
        if not CompResult: print('PercoLi has to be subset of TMLi!!!');
        ###Check criterion 2 and 3,4 
        NonZeroLiInds=list(TMLiArry);
        for Ind in TM0Chans:
            if TM0Chans[Ind]>0:
                if Ind not in TMLiArry: 
                    print('All nonzero value should be TMLiArry');
                else: NonZeroLiInds.remove(Ind);
            if TM0Chans[Ind]>32: print('TM0Chans should be no more than 32');
            if TM0Chans[Ind]%4!=0: print('TM0Chans should be integer times of 4');
        if len(NonZeroLiInds)!=0:
            print('There has been some Inds in TMLiArry have nonzero value in TM0Chans');
            print(NonZeroLiInds);

    def getChanNums(self,Str,IsTrack0TM=False):
        '''
        Get all kinds of channel, order with respect to Li M1 and M2
        Returns:
            CountLst:  Array of all the numbers of channel, take the format of
                       [NLi4,all the possible NLi3 in the order of Cations,
                       all the possible NLi2,all the possible NLi1, 
                       all the possible NLi0]
            Li0TMInds: The Li indice that get involved into 0TM channel
                       can be further developed to keep track of
                       network connectivity.
        '''
        ChanDict,CationInds,StrNNLsts=self.getTetraDict(Str);
        Li0TMInds=[]; TM0Chans={};
        LiInds=getCSIndices(Str,'Li+');
        for LiInd in LiInds: TM0Chans[LiInd]=0;
        TMOccuLst=list(cwr(range(self.CationNum),4)); CountLstSize=len(TMOccuLst);
        CountLst=np.zeros(CountLstSize);
        IndLsts=self.getCationIndLsts(Str);
        for Ind in CationInds:
            ChanLst=ChanDict[Ind];
            #print(len(ChanLst));
            if len(ChanLst)!=8: print('each Cation should have 8 tetrahedron');
            for Ind1,Ind2,Ind3,NTM in ChanLst:
                TypeInds=[];
                for i in [Ind,Ind1,Ind2,Ind3]:
                    for ii,Inds in enumerate(IndLsts):
                        if i in Inds: TypeInds.append(ii);
                if len(TypeInds)!=4: print('TypeInds should have only 4 element');
                ChanType=self.__getChanType(tuple(sorted(TypeInds)));
                if ChanType==0:
                    for i in [Ind,Ind1,Ind2,Ind3]:
                        if i not in Li0TMInds: Li0TMInds.append(i);
                    if IsTrack0TM:
                        for i in [Ind,Ind1,Ind2,Ind3]:
                            TM0Chans[i]+=1;
                CountLst[ChanType]+=1;
        return CountLst,Li0TMInds,TM0Chans;

    def __getChanType(self,TypeInds):
        '''
            Get the type of channel
            TypeInds: Four members tuple with each member being atom type
                      indice, shoud be sorted!!!
            Cations:  All the cations taken into consideration
        '''
        TMOccuLst=list(cwr(range(self.CationNum),4));
        for Ind,TMOccu in enumerate(TMOccuLst):
            if TMOccu==TypeInds: return Ind;
        print('Really? Did not find aything in TMOccuLst=%s'%str(TMOccuLst));
   
    def getTetraDictOld(self,Str):
        '''Enumerate all cation site and get all tetrahedrons'''
        ChanDict={};
        start_time=time.time();
        StrNNLsts=Str.get_all_neighbors(self.DCut,include_index=True);
        CationInds=self.getCationInds(Str); LiInds=getCSIndices(Str,'Li+');
        #print(CationInds,Str.composition); 
        for Ind in CationInds:
            NNLst=StrNNLsts[Ind]; #Get all 1NN sites of this Cation
            #print(len(NNLst),Ind)
            ChanDict[Ind]=self.__getallChans(Ind,NNLst,Str,LiInds);
            #print(len(ChanDict[Ind]));
        print('getTetraDict takes {} s'.format(time.time()-start_time))
        return ChanDict,CationInds,StrNNLsts

    def __getallChans(self,Ind,NNLst,Str,LiInds):
        '''
        Get all tetrahedrons
        only classify channels according to amount of Li (0TM,1TM,2TM,3TM,4TM)
        '''
        Debug=True; ChanLsts=[]; CatNNLst=[];
        #print(len(NNLst[0]))
        #print(NNLst[0])
        for (Site,Distance,Index,Image) in NNLst: #Enumerate all the 1NN sites
            if str(Site.specie) in self.Cations:
                CatNNLst.append((Site,Distance,Index));
        if Debug:
            if len(CatNNLst)!=12:
                print('Should have only 12 1NN cation sites but has %i'%len(CatNNLst));
        for ((Site1,Dist1,Ind1),(Site2,Dist2,Ind2),(Site3,Dist3,Ind3)) \
                in comb(CatNNLst,3):
            #It should form tetrahedron
            if Str.get_distance(Ind1,Ind2)>self.DCut or\
                    Str.get_distance(Ind1,Ind3)>self.DCut or\
                    Str.get_distance(Ind2,Ind3)>self.DCut: continue;
            NLi=0;
            for i in [Ind,Ind1,Ind2,Ind3]:
                if i in LiInds:NLi+=1;
            ChanLsts.append([Ind1,Ind2,Ind3,4-NLi]);

        return ChanLsts;

    def getTetraDict(self,Str):
        '''Get all tetrahedrons made from cation sites'''
        ChanDict={}; Debug=True;
        start_time=time.time();
        StrNNLsts=Str.get_all_neighbors(self.DCut,include_index=True);
        StaTime=time.time();
        DistMat=deepcopy(Str.distance_matrix);
        print('{}s elapsed for distance matrix'.format(time.time()-StaTime));
        CationInds=self.getCationInds(Str); LiInds=getCSIndices(Str,'Li+');
        TMInds=[Ind for Ind in CationInds if Ind not in LiInds];
        for Ind in CationInds:
            NCount=1 if Ind in LiInds else 0;
            NNLst=StrNNLsts[Ind];
            ChanLsts=[]; CatNNLst=[];
            for (Site,Distance,Index,Image) in NNLst:
                if Index in TMInds: CatNNLst.append([Index,0]);
                elif Index in LiInds: CatNNLst.append([Index,1])
            if Debug:
                if len(CatNNLst)!=12:
                    print('Should have only 12 1NN cation sites but has %i'%len(CatNNLst));
            for ((Ind1,NLi1),(Ind2,NLi2),(Ind3,NLi3)) in comb(CatNNLst,3):
                if DistMat[Ind1,Ind2]>self.DCut or DistMat[Ind1,Ind3]>self.DCut or \
                        DistMat[Ind2,Ind3]>self.DCut: continue;
                ChanLsts.append([Ind1,Ind2,Ind3,4-NCount-NLi1-NLi2-NLi3]);
            ChanDict[Ind]=deepcopy(ChanLsts);
        print('getTetraDict takes {}s'.format(time.time()-start_time));
        return ChanDict,CationInds,StrNNLsts;

    def getPercolatingLi(self,DPercolating=1):
        '''
        Get list of percolating Li using the Dijkstra's algorithm. This is a fast
        version of which the exact percolating path will not be tracked

        Args:
            DPercolating:
                The threshold distance for whether Li is percolating. The distance
                is a summation of cost from Li to its image. 0-TM has no cost,
                1-TM channel has 1 cost, others have very large cost
        Return:
            PercolatingLiLst:
                The list of Li indices that percolating
        '''
        StaTime=time.time();
        SCMat0=np.array([1.0,1.0,1.0]); PercolatingLiLst=[];
        for Dim in range(3): #Check percolation in all three dimension
            SCMat=deepcopy(SCMat0); SCMat[Dim]=2.0;
            SCStr=deepcopy(self.OStr); SCStr.make_supercell(SCMat);
            SCStr.to(fmt='poscar',filename='POSCAR_{}_{}_{}'.\
                    format(SCMat[0],SCMat[1],SCMat[2]));
            MapLst=self.mapSites(SCStr,SCMat); #Pair each Li with its image
            SCLiInds=getCSIndices(SCStr,'Li+');
            NonIdentiMapLst=[];
            #print(MapLst);
            #print(SCLiInds);
            for LiInd1, LiInd2 in MapLst:
                if (LiInd1 in PercolatingLiLst) or (LiInd2 in PercolatingLiLst): continue;
                if LiInd1 not in SCLiInds: continue;
                NonIdentiMapLst.append((LiInd1,LiInd2));
            #StaTime=time.time();
            ChanDistDict=self.getChanDistDict(SCStr);
            #print("--- %s seconds spent on getChanDistDict---"%\
            #    (time.time() - StaTime));
            #print(NonIdentiMapLst)
            #StaTime=time.time();
            for LiInd,LiImInds in NonIdentiMapLst:
                MinD=self.dijkstraDistance(ChanDistDict,SCLiInds,LiInd,LiImInds);
                #print(LiInd,MinD);
                if MinD<DPercolating: PercolatingLiLst.append(LiInd);
            #print("--- %s seconds spent on dijkstra---"%\
            #    (time.time() - StaTime));
        print("--- %s seconds spent on running---"%\
                (time.time() - StaTime));
        #print(PercolatingLiLst);
        return PercolatingLiLst;
    
    def mapSites(self,SCStr,SCMat):
        '''
        Map the index of each atom in Str to be SCStr
        '''
        SCFracCoords=SCStr.frac_coords; FracCoords=self.OStr.frac_coords;
        ScaleFracCoords=SCFracCoords*SCMat;
        MapLst=[];
        for i, Coord in enumerate(FracCoords):
            Inds=find_in_coord_list_pbc(ScaleFracCoords,Coord);
            MapLst.append(tuple(Inds));
        return MapLst;

    def dijkstraDistance(self,ChanDistDict,LiInds,LiInd1,LiInd2):
        '''
        A fast version of dijkstraDistance, only track the percolating Li
        '''
        start_time = time.time()
        LiCollect=np.ones(len(LiInds));
        LiPercoDist=np.full(len(LiInds),1E10); #Distance set as very big
        LiPercoDist[LiInd1]=0; IndSta=LiInd1; IndFin=LiInd2;
        while 1:
            if IndSta==IndFin:
                print('{}s spent on dijkstr'.format(time.time()-start_time))
                return LiPercoDist[IndFin];
            LiCollect[IndSta]=0; NNLst=ChanDistDict[IndSta].keys();
            for NNInd in NNLst: #Enumerate all the NN of IndSta
                if not LiCollect[NNInd]: continue; #If the site is already calculated
                #Update the shortest distance
                if LiPercoDist[IndSta]+ChanDistDict[IndSta][NNInd]<LiPercoDist[NNInd]:
                    LiPercoDist[NNInd]=LiPercoDist[IndSta]+ChanDistDict[IndSta][NNInd];
            #For the rest of Li sites, remove the one with shortest distance from collection
            #IndSta=LiPercoDist[LiCollect==1].argmin();
            CollInd=LiPercoDist[LiCollect.nonzero()].argmin(); IndSta=np.where(LiCollect==1)[0][CollInd];
            #print('IndSta=%i'%IndSta,LiCollect[IndSta],LiPercoDist[LiCollect.nonzero()].min())
        print('Something must be wrong!'); exit();

    def getChanDistDictOld(self,Str):
        '''
        Get the neighboring list of each Li as well as its fatest TM channel

        return:
            ChanDistDict: key being each of the neighboring site and distance
                          (decided by type of TM channel (0-TM,1-TM et al))
        '''
        StaTime=time.time();
        DistTab={0:0,1:1,2:100,3:1E3,4:3E100}; #Key refers to number of TM
        ChanDict,CationInds,StrNNLsts=self.getTetraDict(Str);
        LiInds=getCSIndices(Str,'Li+');
        ChanDistDict={};
        for LiInd in LiInds:
            ChanDistDict[LiInd]={};
            for (Site,Dist,Ind,Image) in StrNNLsts[LiInd]:
                if Ind not in LiInds: continue;
                DNN=1E100;
                #Enumerate all the channel to check the shortest
                #distance between LiInd and Ind
                for Channel in ChanDict[LiInd]:
                    if Ind in Channel:
                        if DistTab[Channel[3]]<DNN: DNN=DistTab[Channel[3]];
                ChanDistDict[LiInd][Ind]=DNN;
        print('getChanDistDict method takes {}s'.format(time.time()-StaTime));
        return ChanDistDict;

    def getChanDistDict(self,Str):
        '''
        Get the neighboring list of each Li as well as its fatest TM channel

        return:
            ChanDistDict: key being each of the neighboring site and distance
                          (decided by type of TM channel (0-TM,1-TM et al))
        '''
        #StaTime=time.time();
        DistTab={0:0,1:1,2:100,3:1E3,4:3E100}; #Key refers to number of TM
        ChanDict,CationInds,StrNNLsts=self.getTetraDict(Str);
        LiInds=getCSIndices(Str,'Li+');
        ChanDistDict={};
        for LiInd in LiInds:
            ChanDistDict[LiInd]={};
            for (Site,Dist,Ind,Image) in StrNNLsts[LiInd]:
                if Ind not in LiInds: continue;
                DNN=1E100;
                #Enumerate all the channel to check the shortest
                #distance between LiInd and Ind
                DistLst=[Channel[3] for Channel in ChanDict[LiInd] if Ind in Channel[:3]];
                ChanDistDict[LiInd][Ind]=DistTab[np.min(DistLst)];
        #print('getChanDistDict method takes {}s'.format(time.time()-StaTime));
        return ChanDistDict;


