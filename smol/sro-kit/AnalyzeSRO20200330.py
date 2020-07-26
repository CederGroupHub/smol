#!/usr/bin/env python

###Update from the last version
###Editted in 2019.03.29
###1. Works for multi-component systems
###2. Improve a bit of the readability and python3 compatability

import os,json
import numpy as np
from IonicToolSet import PoscarWithChgDecorator as PoscarFromChg
from LocalStrAnalyzer20200330 import get_SRO_Shell, get_SRO_NN
import multiprocessing
from functools import partial
import time
import argparse
from pprint import pprint

def ParseSRO(Root,HostIon,NNSyms,DIon):
    '''Function to parse percolation'''
    start_time=time.time();
    print('Parsing the directory:\n%s'%Root);
    CoordInfo={}; SRODict={};
    ZippedPOS='%s/POSCAR.gz'%Root; BzippedPOS='%s/POSCAR.bz2'%Root;
    IsZip=False; IsBzip=False;
    if os.path.isfile(ZippedPOS):
        os.system('gunzip %s'%ZippedPOS); IsZip=True; print('gunzip structure in \n%s'%ZippedPOS);
    if os.path.isfile(BzippedPOS):
        os.system('bunzip2 %s'%BzippedPOS); IsBzip=True; print('bunzip2 structure in \n%s'%BzippedPOS);
    POSName='%s/POSCAR'%Root; Str=PoscarFromChg.from_file(POSName).structure;
    print('Host Ion is {}'.format(HostIon));
    HostNum=len(Str.getIonIndices(HostIon));
    if len(DIon)==1:
        print('Parse DCut={}'.format(DIon))
        SROColl,NColl=get_SRO_NN(Str,HostIon,NNSyms,DIon[0]);
    elif len(DIon)==2:
        print('Parse DMin={} and DMax={}'.format(DIon[0],DIon[1]))
        SROColl,NColl=get_SRO_Shell(Str,HostIon,NNSyms,DIon[0],DIon[1]);
    else:
        print('DIon has to go with two arguments but get {}'.format(DIon))
    for KeyInd in SROColl:
        NCoord=len(SROColl[KeyInd]); #print(NCoord);
        break;
    for Ion in NColl:
        BondName='%s%s'%(Ion,HostIon); CoordInfo[BondName]=len(NColl[Ion]);
    for BondName in CoordInfo:
        SRODict[BondName]=1.0*CoordInfo[BondName]/HostNum/NCoord;
    print(CoordInfo);
    print('Time spent on path\n%s\nis %s seconds'%(Root,time.time()-start_time));
    if IsBzip: os.system('bzip2 %s/POSCAR'%Root); print('bzip2 structure in \n%s'%Root);
    else: os.system('gzip %s/POSCAR'%Root); print('gzip structure in \n%s'%Root);
    return [CoordInfo,SRODict,HostNum]; #Number of Bonds, normalized CoordInfo,Number of Host atoms

if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument('--WorkingDir',help="Direcotry to work on (default: cwd)",type=str,default=os.getcwd());
    parser.add_argument('-HostIon',help="Host atom for SRO (with charge decorator)",type=str,required=True);
    parser.add_argument('-NNSyms',help="A list of NN specie symbols (with charge decorator) to be traced",nargs='+',type=str,required=True);
    parser.add_argument('-DIon',help="Ion distance (if get two, then read as DMin and DMax)",type=float,required=True,nargs='+');
    parser.add_argument('--tag',help="The tag for outputfile",type=str,default='');
    parser.add_argument('--NCPU',help="Number of CPU used",type=int,default=16);
    args=parser.parse_args();

    start_time = time.time(); 
    CalcDirs = [];
    #Usually the configuration name is 
    for Root, Dir, Files in os.walk(args.WorkingDir):
        #if '_199' not in Root: continue;
        if (('POSCAR' in Files) or ('POSCAR.gz' in Files) or \
                ('POSCAR.bz2' in Files)) and 'TMM' not in Root and 'CationGS' not in Root:
            CalcDirs.append(Root);
    print('Number of dirs: %i'%len(CalcDirs));
    pool=multiprocessing.Pool(args.NCPU);
    print('Pooling the supercell size for data extraction');
    ParseRunner=partial(ParseSRO,HostIon=args.HostIon,NNSyms=args.NNSyms,DIon=args.DIon);
    SROInfo=pool.map(ParseRunner,CalcDirs); pool.close(); pool.join();
    AllInfo={};
    for [CoordInfo,SRODict,HostNum], CalcDir in zip(SROInfo,CalcDirs):
        AllInfo[CalcDir]=[CoordInfo,SRODict,HostNum];
    OutName='{}/SRO_{}_{}.json'.format(args.WorkingDir,args.HostIon,args.tag);
    print('Results will be written to {}'.format(OutName));
    with open(OutName,'w') as Fid: 
        Fid.write(json.dumps(AllInfo));
    print("--- Overall time is %s seconds ---" % (time.time() - start_time));


