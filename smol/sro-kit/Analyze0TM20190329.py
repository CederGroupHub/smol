#!/usr/bin/env python

###Update from the last version
###Editted in 2019.03.29
###1. Works for multi-component systems
###2. Improve a bit of the readability and python3 compatability

import os
import numpy as np
from IonicToolSet import PoscarWithChgDecorator as PosChg
from DiffNetworkAnalyzer20191014 import PercolationAnalyzer as PA
from pymatgen.core.structure import Structure
import multiprocessing
from functools import partial
from DRXTools import loadPOSwithCS
import time
import argparse

def Parse0TM(Root,Cations=None):
    '''Function to parse percolation'''
    print('Parsing the directory:\n%s'%Root);
    #print('Cations=%s'%str(Cations));
    start_time=time.time();
    ZippedPOS='{}/POSCAR.gz'.format(Root); BzippedPOS='{}/POSCAR.bz2'.format(Root);
    IsZip=False; IsBzip=False;
    if os.path.isfile(ZippedPOS):
        os.system('gunzip %s'%ZippedPOS); IsZip=True; print('gunzip structure in \n%s'%ZippedPOS);
    if os.path.isfile(BzippedPOS):
        os.system('bunzip2 %s'%BzippedPOS); IsBzip=True; print('bunzip2 structure in \n%s'%BzippedPOS);
    Str=PosChg.from_file('{}/POSCAR'.format(Root)).structure;
    PA0=PA(Str,Cations);
    NoPercoLi,Share0TMNum,TM0Chans,Li0TMInds,PercoLiLst,CountLst=PA0.classify0TMLi(Str);
    print('CountLst=%s'%str(CountLst));
    ##Dump number of channels and percolating Li network
    with open('%s/ChanCount.dat'%Root,'w') as Fid: Fid.write(str(CountLst)); 
    with open('%s/ChanClassify.dat'%Root,'w') as Fid:
        Fid.write(str([NoPercoLi,Share0TMNum,Li0TMInds,PercoLiLst]));
    SiteLst=[];
    for Ind in Li0TMInds: SiteLst.append(Str[Ind]);
    NewStr=Structure.from_sites(SiteLst);
    NewStr.to(fmt='poscar',filename='%s/POSCAR_0TM'%Root);
    print('Time spent on path\n%s\nis %s seconds'%(Root,time.time()-start_time));
    if IsBzip: os.system('bzip2 %s/POSCAR'%Root); print('bzip2 structure in \n%s'%Root);
    else: os.system('gzip %s/POSCAR'%Root); print('gzip structure in \n%s'%Root);
    return CountLst, [NoPercoLi,Share0TMNum,Li0TMInds,PercoLiLst];

if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument('-WorkingDir',help="Configuration Folder Name",type=str,required=True);
    parser.add_argument('-Cations',help="List of cations with charge decorator",type=str,nargs='+');
    parser.add_argument('--NCPU',help="Number of CPU used",type=int,default=16);
    args=parser.parse_args();

    start_time = time.time()
    Prefix=os.getcwd();
    CalcDirs=[];
    for Root, Dir, Files in os.walk(args.WorkingDir):
        if (('POSCAR' in Files) or ('POSCAR.gz' in Files) or ('POSCAR.bz2' in Files)) \
                and 'TMM' not in Root and 'xLi120' not in Root and 'CationGS' not in Root:
                    CalcDirs.append(Root);
    print(CalcDirs);
    pool=multiprocessing.Pool(args.NCPU);
    print('Pooling the supercell size for data extraction');
    ParseRunner=partial(Parse0TM,Cations=args.Cations);
    InfoLsts=pool.map(ParseRunner,CalcDirs); pool.close(); pool.join();
    #Since we do not need to average in the script, so no need to remove None values like
    #what has been on in GetPerco.py
    with open('{}/TMInfo.dat'.format(args.WorkingDir),'w') as Fid: Fid.write(str(InfoLsts));
    print("--- Overall time is %s seconds ---" % (time.time() - start_time));


