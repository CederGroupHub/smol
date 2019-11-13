from __future__ import division
import unittest

from pymatgen import Lattice, Structure
from ..ce import ClusterExpansion
from ...cluster_expansion.monte_carlo import ClusterMonteCarlo

import numpy as np

class MCTest(unittest.TestCase):
    
    def test(self):
        lattice = Lattice.rhombohedral(3.96, 60)
        species = [{'Ca':0.333, 'Li':0.333}] * 3 + ['Br']
        coords = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.75), 
                  (0.5, 0.5, 0.5),  (0, 0, 0))
        ce = ClusterExpansion(Structure(lattice, species, coords), {2:5})
        
        ecis = np.random.rand(ce.n_bit_orderings) - 0.5
        
        sc_matrix = [[2,1,0],[1,2,0],[0,0,1]]
        
        cmc = ClusterMonteCarlo(ce, ecis, sc_matrix)
        
        occu = np.zeros(len(cmc.cluster_supercell.bits))
        
        occu[0] = 1
        
        i = 2
        b = 0
        flip = cmc.flip_energy(i, b, occu)
        
        preflip = np.dot(cmc.cluster_supercell.corr_from_occupancy(occu), ecis)
        occu[i]=b
        postflip = np.dot(cmc.cluster_supercell.corr_from_occupancy(occu), ecis)
        
        assert postflip - preflip - flip == 0
        
        pre_anneal = np.dot(cmc.cluster_supercell.corr_from_occupancy(occu), ecis)
        
        best_occu, best_energy, energies = cmc.simulated_anneal(occu, 0.5, 0, 200)
        
        print(np.dot(cmc.cluster_supercell.corr_from_occupancy(best_occu), ecis), best_energy)
        print(best_occu)


        import matplotlib.pylab as plt
        plt.scatter(range(len(energies)), energies)
        plt.show()