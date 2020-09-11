import os
import pytest
from monty.serialization import loadfn
from smol.cofe import ClusterSubspace

# load test data files and set them up as fixtures
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# some test structures to use in tests
files = ['AuPd_prim.json', 'CrFeW_prim.json', 'LiCaBr_prim.json',
         'LiMOF_prim.json']
test_structures = [loadfn(os.path.join(DATA_DIR, file)) for file in files]
ionic_test_structures = test_structures[2:]


@pytest.fixture(params=test_structures, scope='module')
def structure(request):
    return request.param


@pytest.fixture(params=test_structures)
def cluster_subspace(request):
    return ClusterSubspace.from_cutoffs(request.param,
                                        cutoffs={2: 6, 3: 5, 4: 4},
                                        supercell_size='volume')
