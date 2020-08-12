import os
from copy import deepcopy
import pytest
from monty.serialization import loadfn
from smol.cofe import ClusterSubspace
from smol.cofe.extern import EwaldTerm

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


@pytest.fixture(params=ionic_test_structures, scope='module')
def ionic_structure(request):
    return request.param


@pytest.fixture(params=test_structures)
def cluster_subspace(request):
    return ClusterSubspace.from_radii(request.param, radii={2: 6, 3: 5, 4: 4})


# Synthetic ClusterExpansion FCC binary data
file_name = 'toyCEFCC2.json'
path = os.path.join(DATA_DIR, file_name)
toyCEFCC2 = loadfn(path)
toyCEFCC2['dataset'] = [(d['structure'], d['energy'])
                        for d in toyCEFCC2['dataset']]

# Synthetic ClusterExpansion with Electrostatics FCC binary data
file_name = 'toyCEFCC2_estat.json'
path = os.path.join(DATA_DIR, file_name)
toyCEFCC2e = loadfn(path)
EwaldFCC2 = deepcopy(toyCEFCC2e)
dset = toyCEFCC2e['dataset']
toyCEFCC2e['dataset'] = [(d['structure'], d['energy']) for d in dset]
# Only the electrostatic energy
EwaldFCC2['dataset'] = [(d['structure'], d['ewald_energy'])
                        for d in dset]

# Sythentic Cluster Expansion FCC ternary data
# TODO get this done

# Simple LNO real dataset
file_name = 'lno31.json'
path = os.path.join(DATA_DIR, file_name)
lno31 = loadfn(path)
lno31['dataset'] = [(d['structure'], d['energy']) for d in lno31['dataset']]

# TODO add a few more real datasets?
# Define some pytests fixtures to be used in tests
synthetic_datasets = [toyCEFCC2, toyCEFCC2e]
real_datasets = [lno31, ]  # real datasets without external terms
composite_datasets = [toyCEFCC2e, lno31]  # datasets with extern ewald term coefs
electrostatic_datasets = [EwaldFCC2, ]  # datasets with ewald energy only


@pytest.fixture(scope='module', params=synthetic_datasets)
def synthetic_system(request):
    dataset = request.param
    radii = {int(key): val for key, val in dataset['radii_cutoffs'].items()}
    subspace = ClusterSubspace.from_radii(structure=dataset['prim'],
                                          radii=radii,
                                          **dataset['subspace_kwargs'])
    return subspace, dataset


@pytest.fixture(scope='module', params=real_datasets)
def real_system(request):
    dataset = request.param
    radii = {int(key): val for key, val in dataset['radii_cutoffs'].items()}
    subspace = ClusterSubspace.from_radii(structure=dataset['prim'],
                                          radii=radii,
                                          **dataset['subspace_kwargs'])
    return subspace, dataset


@pytest.fixture(scope='module', params=synthetic_datasets + real_datasets)
def ce_system(request):
    dataset = request.param
    radii = {int(key): val for key, val in dataset['radii_cutoffs'].items()}
    subspace = ClusterSubspace.from_radii(structure=dataset['prim'],
                                          radii=radii,
                                          **dataset['subspace_kwargs'])
    return subspace, dataset


@pytest.fixture(scope='module', params=composite_datasets)
def composite_system(request):
    dataset = request.param
    radii = {int(key): val for key, val in dataset['radii_cutoffs'].items()}
    subspace = ClusterSubspace.from_radii(structure=dataset['prim'],
                                          radii=radii,
                                          **dataset['subspace_kwargs'])
    subspace.add_external_term(EwaldTerm())
    return subspace, dataset


@pytest.fixture(scope='module', params=electrostatic_datasets)
def electrostatic_system(request):
    dataset = request.param
    radii = {int(key): val for key, val in dataset['radii_cutoffs'].items()}
    subspace = ClusterSubspace.from_radii(structure=dataset['prim'],
                                          radii=radii,
                                          **dataset['subspace_kwargs'])
    return subspace, dataset
