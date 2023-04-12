"""Test solver class construction and usage."""
from itertools import product

import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

from smol.cofe import ClusterExpansion, ClusterSubspace
from smol.moca import Ensemble
from smol.moca.utils.occu import get_dim_ids_table, occu_to_counts
from smol.solver.upper_bound.solver import UpperboundSolver
from smol.solver.upper_bound.variables import get_variable_values_from_occupancy

from ..utils import assert_msonable


# Only SCIP tried on this instance.
@pytest.fixture(params=["SCIP"])
def exotic_solver(exotic_ensemble, exotic_initial_occupancy, request):
    return UpperboundSolver(
        exotic_ensemble, exotic_initial_occupancy, solver=request.param
    )


def test_msonable(exotic_solver, exotic_initial_occupancy):
    assert_msonable(exotic_solver)
    solver_dict = exotic_solver.as_dict()
    solver_reload = UpperboundSolver.from_dict(solver_dict)
    with pytest.raises(RuntimeError):
        _ = solver_reload.ground_state_solution
    with pytest.raises(RuntimeError):
        _ = solver_reload.ground_state_occupancy
    with pytest.raises(RuntimeError):
        _ = solver_reload.ground_state_structure
    with pytest.raises(RuntimeError):
        _ = solver_reload.ground_state_energy


def test_setting_results(exotic_solver):
    exotic_solver._ground_state_solution = get_variable_values_from_occupancy(
        exotic_solver._ensemble.sublattices,
        exotic_solver._initial_occupancy,
        exotic_solver._canonicals.variable_indices,
    )
    with pytest.raises(RuntimeError):
        _ = exotic_solver.ground_state_energy

    exotic_solver._ground_state_energy = 0
    assert exotic_solver.ground_state_energy == 0
    npt.assert_array_equal(
        exotic_solver.ground_state_occupancy, exotic_solver._initial_occupancy
    )
    assert StructureMatcher().fit(
        exotic_solver.ground_state_structure,
        exotic_solver._ensemble.processor.structure_from_occupancy(
            exotic_solver._initial_occupancy
        ),
    )
    # reset.
    exotic_solver.reset()
    with pytest.raises(RuntimeError):
        _ = exotic_solver.ground_state_structure
    with pytest.raises(RuntimeError):
        _ = exotic_solver.ground_state_energy


@pytest.fixture(scope="module")
def simple_prim():
    lat = Lattice.cubic(3.0)
    return Structure(
        lat,
        [{"Li": 0.5, "Ag": 0.5}],
        [[0, 0, 0]],
    )


@pytest.fixture(scope="module", params=["sinusoid"])
def simple_subspace(simple_prim, request):
    # Use sinusoid basis to test if useful.
    space = ClusterSubspace.from_cutoffs(
        simple_prim, {2: 4.5, 3: 3.1}, basis=request.param
    )
    return space


@pytest.fixture(scope="module")
def simple_coefs(simple_subspace):
    simple_coefs = np.empty(simple_subspace.num_corr_functions)
    simple_coefs[0] = -10
    n_pair = len(simple_subspace.function_inds_by_size[2])
    n_tri = len(simple_subspace.function_inds_by_size[3])
    n_quad = 0
    i = 1
    simple_coefs[i : i + n_pair] = np.random.random(size=n_pair)
    i += n_pair
    simple_coefs[i : i + n_tri] = np.random.random(size=n_tri) * 0.4
    i += n_tri
    simple_coefs[i : i + n_quad] = np.random.random(size=n_quad) * 0.1
    return simple_coefs


@pytest.fixture(scope="module")
def simple_expansion(simple_subspace, simple_coefs):
    return ClusterExpansion(simple_subspace, simple_coefs)


@pytest.fixture(
    scope="module",
    params=list(product(["canonical", "semigrand"], ["expansion", "decomposition"])),
)
def simple_ensemble(simple_expansion, request):
    if request.param[0] == "semigrand":
        chemical_potentials = {
            "Li": np.random.normal(),
            "Ag": np.random.normal(),
        }
    else:
        chemical_potentials = None
    return Ensemble.from_cluster_expansion(
        simple_expansion,
        np.diag([2, 2, 2]),  # 8 sites, 8 variables.
        request.param[1],
        chemical_potentials=chemical_potentials,
    )


# Only SCIP tried on this instance.
@pytest.fixture(params=["SCIP"])
def simple_solver(simple_ensemble, request):
    if simple_ensemble.chemical_potentials is not None:
        return UpperboundSolver(simple_ensemble, solver=request.param)
    else:
        fixed_composition = np.array([4, 4])
        return UpperboundSolver(
            simple_ensemble, fixed_composition=fixed_composition, solver=request.param
        )


# Do a small scale solving test.
def test_solve(simple_solver):
    _, energy = simple_solver.solve()

    occu = simple_solver._ensemble.processor.occupancy_from_structure(
        simple_solver.ground_state_structure
    )

    n_dims = sum([len(s.species) for s in simple_solver._ensemble.sublattices])
    assert n_dims == 2
    table = get_dim_ids_table(simple_solver._ensemble.sublattices)
    counts = occu_to_counts(occu, n_dims, dim_ids_table=table)
    if simple_solver._ensemble.chemical_potentials is None:
        # Canonical ensemble, should assume same composition.
        npt.assert_array_equal(counts, simple_solver._fixed_composition)

    features = simple_solver._ensemble.compute_feature_vector(occu)
    true_energy = np.dot(features, simple_solver._ensemble.natural_parameters)

    assert np.isclose(energy, true_energy)
