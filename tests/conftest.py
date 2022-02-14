import os
import pytest
import numpy as np
from monty.serialization import loadfn
from smol.cofe import ClusterSubspace
from smol.moca import CanonicalEnsemble, SemiGrandEnsemble, \
    CEProcessor, EwaldProcessor, CompositeProcessor
from smol.cofe.extern import EwaldTerm

# load test data files and set them up as fixtures
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# some test structures to use in tests
files = ['AuPd_prim.json', 'CrFeW_prim.json', 'LiCaBr_prim.json',
         'LiMOF_prim.json', 'LiMnTiVOF_prim.json']
test_structures = [loadfn(os.path.join(DATA_DIR, file)) for file in files]
ensembles = [CanonicalEnsemble, SemiGrandEnsemble]


@pytest.fixture(params=test_structures, scope='package')
def structure(request):
    return request.param


@pytest.fixture(params=test_structures, scope='module')
def cluster_subspace(request):
    subspace = ClusterSubspace.from_cutoffs(
        request.param, cutoffs={2: 6, 3: 5, 4: 4}, supercell_size='volume')
    return subspace


@pytest.fixture(params=test_structures, scope='module')
def cluster_subspace_ewald(request):
    subspace = ClusterSubspace.from_cutoffs(
        request.param, cutoffs={2: 6, 3: 5, 4: 4}, supercell_size='volume')
    subspace.add_external_term(EwaldTerm())
    return subspace


@pytest.fixture(scope='module')
def single_subspace():
    subspace = ClusterSubspace.from_cutoffs(
        test_structures[2], cutoffs={2: 6, 3: 5, 4: 4}, supercell_size='volume')
    return subspace


@pytest.fixture(scope='module')
def ce_processor(cluster_subspace):
    coefs = 2 * np.random.random(cluster_subspace.num_corr_functions)
    scmatrix = 3 * np.eye(3)
    return CEProcessor(cluster_subspace, supercell_matrix=scmatrix,
                       coefficients=coefs)


@pytest.fixture(scope='module')
def composite_processor(cluster_subspace):
    coefs = 2 * np.random.random(cluster_subspace.num_corr_functions + 1)
    scmatrix = 3 * np.eye(3)
    ewald_term = EwaldTerm()
    cluster_subspace.add_external_term(ewald_term)
    proc = CompositeProcessor(cluster_subspace, supercell_matrix=scmatrix)
    proc.add_processor(CEProcessor(cluster_subspace, scmatrix,
                                   coefficients=coefs[:-1]))
    proc.add_processor(EwaldProcessor(cluster_subspace, scmatrix, ewald_term,
                                      coefficient=coefs[-1]))
    return proc


@pytest.fixture(params=ensembles, scope='module')
def ensemble(composite_processor, request):
    if request.param is SemiGrandEnsemble:
        kwargs = {'chemical_potentials':
                  {sp: 0.3 for space in composite_processor.unique_site_spaces
                   for sp in space.keys()}}
    else:
        kwargs = {}
    return request.param(composite_processor, **kwargs)


@pytest.fixture(scope='module')
def single_canonical_ensemble(single_subspace):
    coefs = np.random.random(single_subspace.num_corr_functions)
    proc = CEProcessor(single_subspace, 4 * np.eye(3), coefs)
    return CanonicalEnsemble(proc)
