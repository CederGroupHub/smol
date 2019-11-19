from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmOp
from smol.clex.cluster import Cluster
from smol.clex.orbit import Orbit
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure
import numpy as np
from smol.clex import ClusterSubspace, StructureWrangler, ClusterExpansion

import json

# Load and prep prim structure
prim = CifParser('/home/lbluque/Develop/daniil_CEMC_workshop/lno_prim.cif')
prim = prim.get_structures()[0]

# Create new ClusterSubspace :)
cs = ClusterSubspace.from_radii(structure=prim,
                                 radii={2: 5, 3: 4.1},
                                 ltol=0.15, stol=0.2, angle_tol=5,
                                 supercell_size='O2-',
                                 use_ewald=True,
                                 use_inv_r=False, eta=None)
print('Here is the cluster subspace object: \n', cs)


# Open and clean fitting data
with open('/home/lbluque/Develop/daniil_CEMC_workshop/lno_fitting_data.json', 'r') as fin: calc_data = json.loads(fin.read())

valid_structs = [] 
for calc_i, calc in enumerate(calc_data): 
    #print('{}/{}'.format(calc_i, len(calc_data))) 
    try: 
        struct = Structure.from_dict(calc['s']) 
        cs.corr_from_structure(struct) 
        valid_structs.append((struct, calc['toten'])) 
    except ValueError: 
        #print("\tToo far off lattice, throwing out.") 
        continue
    except AttributeError as e:
        continue
 
print("{}/{} structures map to the lattice".format(len(valid_structs), len(calc_data))) 

print('Also here is a random corr_vector:\n', cs.corr_from_structure(valid_structs[0][0]))

# Create the data wrangler.
sw = StructureWrangler(cs, [struct for struct, _ in valid_structs],
					   [e for _, e in valid_structs], max_ewald=3)


# Create a ClusterExpansion Object
ce = ClusterExpansion(sw, max_dielectric=100)


# Finally need to fit it!