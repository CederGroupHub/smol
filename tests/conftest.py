import os

import numpy as np
import pytest
from monty.serialization import loadfn
from pymatgen.core import Structure

from smol.cofe import ClusterExpansion, ClusterSubspace, StructureWrangler
from smol.cofe.extern import EwaldTerm
from smol.cofe.space.basis import BasisIterator
from smol.moca import Ensemble
from smol.moca.processor import (
    ClusterDecompositionProcessor,
    ClusterExpansionProcessor,
    CompositeProcessor,
    EwaldProcessor,
)
from smol.moca.processor.distance import (
    ClusterInteractionDistanceProcessor,
    CorrelationDistanceProcessor,
)
from smol.utils.class_utils import get_subclasses
from tests.utils import gen_fake_training_data

# uncomment below to show HDF5 C traceback
# import h5py
# h5py._errors.unsilence_errors()


SEED = None

# load test data files and set them up as fixtures
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# some test structures and other parameters for creating fixtures
files = [
    "AuPd_prim.json",
    "CrFeW_prim.json",
    "LiCaBr_prim.json",
    "LiMOF_prim.json",
    "LiMnTiVOF_prim.json",
]

test_structures = [loadfn(os.path.join(DATA_DIR, file)) for file in files]
basis_iterator_names = list(get_subclasses(BasisIterator))


@pytest.fixture(scope="module")
def rng():
    """Seed and return an RNG for test reproducibility"""
    return np.random.default_rng(SEED)


@pytest.fixture(scope="package")
def cluster_cutoffs():
    # parametrize this if we ever want to test on different cutoffs
    return {2: 6, 3: 5, 4: 4}


@pytest.fixture(params=test_structures, scope="package")
def structure(request):
    return request.param


@pytest.fixture(scope="package")
def expansion_structure(structure):
    sites = [
        site
        for site in structure
        if site.species.num_atoms < 0.99 or len(site.species) > 1
    ]
    return Structure.from_sites(sites)


@pytest.fixture(scope="package")
def single_structure():
    # this is the LiCaBr structure used for some fixed tests
    return test_structures[2]


@pytest.fixture(params=test_structures, scope="package")
def cluster_subspace(cluster_cutoffs, request):
    subspace = ClusterSubspace.from_cutoffs(
        request.param, cutoffs=cluster_cutoffs, supercell_size="volume"
    )
    return subspace


@pytest.fixture(params=test_structures, scope="package")
def cluster_subspace_ewald(cluster_cutoffs, request):
    subspace = ClusterSubspace.from_cutoffs(
        request.param, cutoffs=cluster_cutoffs, supercell_size="volume"
    )
    subspace.add_external_term(EwaldTerm())
    return subspace


@pytest.fixture(scope="package")
def single_subspace(single_structure):
    # this is a subspace with the LiCaBr structure for some fixed tests
    subspace = ClusterSubspace.from_cutoffs(
        single_structure, cutoffs={2: 6, 3: 5}, supercell_size="volume"
    )
    return subspace


# fixture for all processors
@pytest.fixture(
    params=[
        "expansion",
        "decomposition",
        "ewald",
        "composite",
        "corr_distance",
        "int_distance",
    ],
    scope="module",
)
def processor(cluster_subspace, rng, request):
    coefs = 2 * np.random.random(cluster_subspace.num_corr_functions)
    scmatrix = 3 * np.eye(3)

    if request.param == "expansion":
        proc = ClusterExpansionProcessor(cluster_subspace, scmatrix, coefficients=coefs)
    elif request.param == "decomposition":
        expansion = ClusterExpansion(cluster_subspace, coefs)
        proc = ClusterDecompositionProcessor(
            cluster_subspace, scmatrix, expansion.cluster_interaction_tensors
        )
    elif request.param == "ewald":
        proc = EwaldProcessor(cluster_subspace, scmatrix, EwaldTerm(), coefficient=1.0)
    elif request.param == "composite":
        proc = CompositeProcessor(cluster_subspace, supercell_matrix=scmatrix)
        proc.add_processor(
            ClusterExpansionProcessor(cluster_subspace, scmatrix, coefficients=coefs)
        )
        proc.add_processor(
            EwaldProcessor(cluster_subspace, scmatrix, EwaldTerm(), coefficient=1.0)
        )
    elif request.param == "corr_distance":
        proc = CorrelationDistanceProcessor(cluster_subspace, scmatrix)
    else:
        expansion = ClusterExpansion(cluster_subspace, coefs)
        proc = ClusterInteractionDistanceProcessor(
            cluster_subspace, scmatrix, expansion.cluster_interaction_tensors
        )

    yield proc
    cluster_subspace._external_terms = []  # Ewald processor will add one..


@pytest.fixture(scope="module")
def ce_processor(cluster_subspace, rng):
    coefs = 2 * rng.random(cluster_subspace.num_corr_functions)
    scmatrix = 3 * np.eye(3)
    return ClusterExpansionProcessor(
        cluster_subspace, supercell_matrix=scmatrix, coefficients=coefs
    )


@pytest.fixture(params=["expansion", "decomposition"], scope="module")
def composite_processor(cluster_subspace_ewald, rng, request):
    coefs = 2 * np.random.random(cluster_subspace_ewald.num_corr_functions + 1)
    scmatrix = 3 * np.eye(3)
    proc = CompositeProcessor(cluster_subspace_ewald, supercell_matrix=scmatrix)
    if request.param == "expansion":
        proc.add_processor(
            ClusterExpansionProcessor(
                cluster_subspace_ewald, scmatrix, coefficients=coefs[:-1]
            )
        )
    else:  # elif request.param == "decomposition":
        expansion = ClusterExpansion(cluster_subspace_ewald, coefs)
        proc.add_processor(
            ClusterDecompositionProcessor(
                cluster_subspace_ewald, scmatrix, expansion.cluster_interaction_tensors
            )
        )

    proc.add_processor(
        EwaldProcessor(
            cluster_subspace_ewald, scmatrix, EwaldTerm(), coefficient=coefs[-1]
        )
    )
    # bind raw coefficients since OD processors do not store them
    # and be able to test computing properties, hacky but oh well
    proc.raw_coefs = coefs
    return proc


@pytest.fixture(params=["canonical", "semigrand"], scope="module")
def ensemble(composite_processor, request):
    if request.param == "semigrand":
        species = {
            sp
            for space in composite_processor.active_site_spaces
            for sp in space.keys()
        }
        kwargs = {"chemical_potentials": {sp: 0.3 for sp in species}}
    else:
        kwargs = {}
    return Ensemble(composite_processor, **kwargs)


@pytest.fixture(scope="module")
def single_sgc_ensemble(rng):
    # a single sgc ensemble using the LMOF test structures
    subspace = ClusterSubspace.from_cutoffs(
        test_structures[3], cutoffs={2: 6, 3: 4}, supercell_size="volume"
    )
    coefs = rng.random(subspace.num_corr_functions)
    coefs[0] = -1.0
    proc = ClusterExpansionProcessor(subspace, 6 * np.eye(3), coefs)
    species = {sp for space in proc.active_site_spaces for sp in space.keys()}
    return Ensemble(proc, chemical_potentials={sp: 1.0 for sp in species})


@pytest.fixture(scope="module")
def single_canonical_ensemble(single_subspace, rng):
    coefs = rng.random(single_subspace.num_corr_functions)
    proc = ClusterExpansionProcessor(single_subspace, 4 * np.eye(3), coefs)
    return Ensemble(proc)


@pytest.fixture(params=basis_iterator_names, scope="package")
def basis_name(request):
    return request.param.split("Iterator")[0]


@pytest.fixture
def supercell_matrix(rng):
    def skewed(scm):  # check if any angle is less than 45 degrees
        normed_scm = scm / np.linalg.norm(scm, axis=0)
        off_diag = (np.array([0, 0, 1]), np.array([1, 2, 2]))
        return any(abs(normed_scm @ normed_scm)[off_diag] > 0.5)

    # make sure not singular and not overly skewed
    while abs(np.linalg.det(m := rng.integers(-3, 3, size=(3, 3)))) < 1e-4 or skewed(m):
        pass

    return m


@pytest.fixture
def structure_wrangler(single_subspace, rng):
    wrangler = StructureWrangler(single_subspace)
    for entry in gen_fake_training_data(single_subspace.structure, n=10, rng=rng):
        wrangler.add_entry(entry, weights={"random": 2.0})
    yield wrangler
    # force remove any external terms added in tests
    wrangler.cluster_subspace._external_terms = []
