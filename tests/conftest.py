import os
import pytest
import numpy as np
from monty.serialization import loadfn
from smol.cofe import ClusterSubspace
from smol.moca import CanonicalEnsemble, MuSemiGrandEnsemble, \
    FuSemiGrandEnsemble
from smol.moca.processor import CEProcessor

# load test data files and set them up as fixtures
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# some test structures to use in tests
files = ['AuPd_prim.json', 'CrFeW_prim.json', 'LiCaBr_prim.json',
         'LiMOF_prim.json', 'LiMnTiVOF_prim.json']
test_structures = [loadfn(os.path.join(DATA_DIR, file)) for file in files]
ensembles = [CanonicalEnsemble, MuSemiGrandEnsemble, FuSemiGrandEnsemble]


@pytest.fixture(params=test_structures, scope='module')
def structure(request):
    return request.param


@pytest.fixture(params=test_structures)
def cluster_subspace(request):
    subspace = ClusterSubspace.from_cutoffs(
        request.param, cutoffs={2: 6, 3: 5, 4: 4}, supercell_size='volume')
    return subspace


@pytest.fixture(scope='module')
def single_subspace():
    subspace = ClusterSubspace.from_cutoffs(
        test_structures[2], cutoffs={2: 6, 3: 5, 4: 4}, supercell_size='volume')
    return subspace


@pytest.fixture(params=ensembles)
def ensemble(cluster_subspace, request):
    coefs = np.random.random(cluster_subspace.num_corr_functions)
    proc = CEProcessor(cluster_subspace, 4 * np.eye(3), coefs)
    if request.param is MuSemiGrandEnsemble:
        kwargs = {'chemical_potentials':
                  {sp: 0.3 for space in proc.unique_site_spaces
                   for sp in space.keys()}}
    else:
        kwargs = {}
    return request.param(proc, **kwargs)


@pytest.fixture
def single_canonical_ensemble(single_subspace):
    coefs = np.random.random(single_subspace.num_corr_functions)
    proc = CEProcessor(single_subspace, 4 * np.eye(3), coefs)
    return CanonicalEnsemble(proc)
