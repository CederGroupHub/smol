import pytest
import numpy as np

from smol.moca.sampler.mcusher import Tableflipper
from pymatgen import Specie
from smol.moca.comp_space import *
from pymatgen import Composition
from smol.moca.ensemble.sublattice import Sublattice
from smol.cofe.space.domain import SiteSpace

li = Specie.from_string('Li+')
mn = Specie.from_string('Mn3+')
ti = Specie.from_string('Ti4+')
p = Specie.from_string('P3-')
o = Specie.from_string('O2-')
bits = [[li,mn,ti],[p,o]]
occu = np.array([0,2,0,1,1,0,1,1,1,0,1,1])
space1 = SiteSpace(Composition({'Li+':0.5,'Mn3+':0.3333333,'Ti4+':0.1666667}))
space2 = SiteSpace(Composition({'O2-':0.8333333,'P3-':0.1666667}))
sl1 = Sublattice(space1,np.array([0,1,2,3,4,5]))
sl2 = Sublattice(space2,np.array([6,7,8,9,10,11]))

cs = CompSpace(bits)
flip_table = cs.min_flips

def test_table_flip():
    tf = Tableflipper([sl1,sl2],flip_table)
    
    tf = Tableflipper(flip_table,[sl1,sl2])
    count_directions = {-1:0,1:0,-2:0,2:0}
    for i in range(28000):
        step,direction = tf.propose_step(occu)
        #print('Proposal number {}:'.format(i),step)
        count_directions[direction]+=1
    
    true_directions = {-1:10000,1:1000,-2:2000,2:15000}
    
    for d in count_directions:
        assert(abs(true_directions[d]-count_directions[d])/true_directions[d]<0.05)

def test_cn_flip():
    cf = Cnflipper([sl1,sl2])
    assert cf.n_links == 180
