#!/usr/bin/env python

import os
import numpy as np
from pymatgen.io.vasp.inputs import Poscar
#from DiffNetworkAnalyzer20190108 import PercolationAnalyzer as PA
from DiffNetworkAnalyzer20190319 import PercolationAnalyzer as PA
from DRXTools import loadPOSwithCS
import multiprocessing
from functools import partial
import time
import argparse

def ParsePercolation(Root,NCut=1,Cations=None):
    '''
    Function to parse percolation
    '''
    print('Parsing the directory:\n%s'%Root);
    print('NCut=%i, Cations=%s'%(NCut,str(Cations)));
    start_time=time.time();
    ZippedPOS='%s/POSCAR.gz'%Root; BzippedPOS='%s/POSCAR.bz2'%Root;
    IsZip=False; IsBzip=False;
    if os.path.isfile(ZippedPOS): 
        os.system('gunzip %s'%ZippedPOS); IsZip=True;
        print('gunzip structure in \n%s'%ZippedPOS);
    if os.path.isfile(BzippedPOS):
        os.system('bunzip2 %s'%BzippedPOS); IsBzip=True;
        print('bunzip2 structure in \n%s'%BzippedPOS);
    #try: Str=Poscar.from_file('%s/POSCAR'%Root).structure;
    try: Str,_=loadPOSwithCS('%s/POSCAR'%Root);
    except: return None;
    PA0=PA(Str,Cations); NLi=len(list(Str.indices_from_symbol('Li')));
    PercoLiLst=PA0.getPercolatingLi(DPercolating=NCut);
    ###Dump analysis results
    with open('%s/PercoLiLst.dat'%Root,'w') as Fid: Fid.write(str(PercoLiLst)); 
    #print(PercoLiLst)
    LiPercent=1.0*len(PercoLiLst)/NLi;
    print('Time spent on path\n%s\nis %s seconds'%(Root,time.time()-start_time));
    print('LiPercent=%.3f'%LiPercent);
    if IsBzip: os.system('bzip2 %s/POSCAR'%Root); print('bzip2 \n%s'%Root);
    else: os.system('gzip %s/POSCAR'%Root); print('gzip \n%s'%Root);
    return LiPercent;

if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument('-Conf',help="Configuration Folder Name",type=str,default=None);
    parser.add_argument('--NCPU',help="Number of CPU used (default: 16)",type=int,default=16);
    parser.add_argument('--NCut',help="Cutoff distance 1 means only 0-TM (default: 1)",type=int,default=1);
    parser.add_argument('-Cations',help="List of cations",type=str,nargs='+');
    args=parser.parse_args();

    start_time = time.time()
    Prefix=os.getcwd(); print(Prefix);
    CalcDirs=[];
    print('%s/%s'%(Prefix,args.Conf))
    for Root, Dir, Files in os.walk('%s/%s'%(Prefix,args.Conf)):
        if ('POSCAR' in Files or 'POSCAR.gz' in Files) and ('TMM' not in Root) and \
                ('xLi' not in Root) and ('CationGS' in Root): CalcDirs.append(Root);
    
    pool=multiprocessing.Pool(args.NCPU);
    print('Pooling the supercell size for data extraction');
    PercoRunner=partial(ParsePercolation,NCut=args.NCut,Cations=args.Cations);
    LiPercentLst=pool.map(PercoRunner,CalcDirs); pool.close(); pool.join();
    #Delete None element
    Count=0; NewLiPercentLst=[];
    for Item in LiPercentLst:
        if Item!=None: NewLiPercentLst.append(Item);
        else: Count+=1;
    print('%i None value found'%Count);
    NLiAvg=np.mean(NewLiPercentLst);
    print('Average portion of percolating Li is %.3f'%NLiAvg);
    with open('LiPercent.dat','w') as Fid: 
        Fid.write('Percolation analysis for %s'%args.Conf);
        Fid.write('Average portion of percolating Li is %.3f'%NLiAvg);
        Fid.write('%s'%str(LiPercentLst));
    print("--- Overall time is %s seconds ---" % (time.time() - start_time));


