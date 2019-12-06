from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmOp 
from pymatgen.io.cif import CifParser 
from pymatgen.core.structure import Structure 
import numpy as np 
import json 
from pyabinitio.cluster_expansion.eci_fit import EciGenerator 
from pyabinitio.cluster_expansion.ce import ClusterExpansion  

# Load and prep prim structure
prim = CifParser('/home/lbluque/Develop/daniil_CEMC_workshop/lno_prim.cif') 
prim = prim.get_structures()[0] 

#print(prim)
# Create old ClusterExpansion behemoth
ce = ClusterExpansion.from_radii(structure=prim, 
                                 radii={2: 5, 3: 4.1}, 
                                 ltol=0.15, stol=0.9, angle_tol=5, 
                                 supercell_size='O2-', 
                                 use_ewald=True, 
                                 use_inv_r=False, eta=None) 

print('Here is the cluster expansion object: \n', ce)

# Open and clean fitting data
with open('/home/lbluque/Develop/daniil_CEMC_workshop/lno_fitting_data.json', 'r') as fin: calc_data = json.loads(fin.read()) 

valid_structs = [] 
for calc_i, calc in enumerate(calc_data): 
    #print('{}/{}'.format(calc_i, len(calc_data))) 
    try: 
        struct = Structure.from_dict(calc['s']) 
        ce.corr_from_structure(struct) 
        valid_structs.append((struct, calc['toten'])) 
    except Exception as e:
        msg = f"Unable to match {struct.composition} with energy {calc['toten']} to supercell. Throwing out. "
        print(msg + f'Error Message: {str(e)}.') 
        continue 
        #raise
 
print("{}/{} structures map to the lattice".format(len(valid_structs), len(calc_data))) 

print('Also here is a random corr_vector:\n', ce.corr_from_structure(valid_structs[10][0]))


# Fit the cluster expansion 
eg = EciGenerator.unweighted(cluster_expansion=ce, 
                             structures=[struct for struct, toten in valid_structs], 
                             energies=[toten for struct, toten in valid_structs], 
                             max_dielectric=100, 
                             max_ewald=3) 

print(f"ECIS: {eg.ecis}")
print("RMSE: {} eV/prim".format(eg.rmse))
print("Number non zero ECIs: {}".format(len([eci for eci in eg.ecis if np.abs(eci) > 1e-3])))    