"""Test groundstate class construction and usage."""
from itertools import permutations, product

import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

from smol.capp.generate.groundstate.upper_bound.solver import PeriodicGroundStateSolver
from smol.capp.generate.groundstate.upper_bound.variables import (
    get_variable_values_from_occupancy,
)
from smol.cofe import ClusterExpansion, ClusterSubspace
from smol.moca import Ensemble
from smol.moca.occu_utils import get_dim_ids_table, occu_to_counts
from tests.utils import assert_pickles


# Only SCIP tried on this instance.
@pytest.fixture
def solver_test_solver(solver_test_ensemble, solver_test_initial_occupancy):
    return PeriodicGroundStateSolver(
        solver_test_ensemble, initial_occupancy=solver_test_initial_occupancy
    )


def test_pickles(solver_test_solver):
    assert_pickles(solver_test_solver)


def test_setting_results(solver_test_solver):
    solver_test_solver._ground_state_solution = get_variable_values_from_occupancy(
        solver_test_solver.ensemble.sublattices,
        solver_test_solver.initial_occupancy,
        solver_test_solver._canonicals.variable_indices,
    )
    with pytest.raises(RuntimeError):
        _ = solver_test_solver.ground_state_energy

    solver_test_solver._ground_state_energy = 0
    assert solver_test_solver.ground_state_energy == 0
    npt.assert_array_equal(
        solver_test_solver.ground_state_occupancy, solver_test_solver.initial_occupancy
    )
    assert StructureMatcher().fit(
        solver_test_solver.ground_state_structure,
        solver_test_solver.ensemble.processor.structure_from_occupancy(
            solver_test_solver.initial_occupancy
        ),
    )
    # reset.
    solver_test_solver.reset()
    with pytest.raises(RuntimeError):
        _ = solver_test_solver.ground_state_structure
    with pytest.raises(RuntimeError):
        _ = solver_test_solver.ground_state_energy


@pytest.fixture(scope="module")
def simple_prim():
    return Structure(
        Lattice.cubic(3.0),
        [{"Li": 0.5, "Ag": 0.5}],
        [[0, 0, 0]],
    )


@pytest.fixture(scope="module", params=["sinusoid"])
def simple_subspace(simple_prim, request):
    # Use sinusoid basis to test if useful.
    space = ClusterSubspace.from_cutoffs(
        simple_prim, {2: 4.5, 3: 4.5}, basis=request.param
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


# @pytest.fixture(params=["SCIP", "GUROBI"])
# SCIP has a very small probability of failing to satisfy constraints.
# Now only use GUROBI for testing.
@pytest.fixture(
    params=[
        pytest.param(
            "SCIP",
            marks=pytest.mark.xfail(
                reason="SCIP has a small probability of failing to satisfy constraints.",
                raises=IndexError,
            ),
        ),
        "GUROBI",
    ]
)
def simple_solver(simple_ensemble, request):
    if simple_ensemble.chemical_potentials is not None:
        return PeriodicGroundStateSolver(simple_ensemble, solver=request.param)
    else:
        fixed_composition = np.array([4, 4])
        return PeriodicGroundStateSolver(
            simple_ensemble, fixed_composition=fixed_composition, solver=request.param
        )


# Do a small scale solving test.
def test_solve(simple_solver):
    n_prims = simple_solver.ensemble.system_size
    n_aux = 0
    for inds in simple_solver._canonicals.indices_in_auxiliary_products:
        n_aux += len(inds) + 1
    # Test number of auxiliary constraints is correct.
    assert simple_solver._canonicals.num_auxiliary_constraints == n_aux
    # No composition space constraints, only normalization constraints.
    if simple_solver.ensemble.chemical_potentials is not None:
        # print("Grand ensemble!")
        n_normalization = len(simple_solver._canonicals.constraints) - n_aux
    # Have 2 additional canonical constraints.
    else:
        # print("Canonical ensemble!")
        n_normalization = len(simple_solver._canonicals.constraints) - n_aux - 2
    assert n_normalization == n_prims

    # print(simple_solver.ensemble.sublattices)
    # print(simple_solver._canonicals.variable_indices)
    # n_constraints = len(simple_solver._canonicals.constraints)
    # for c in simple_solver._canonicals.constraints[: n_constraints - n_aux]:
    #     print(c)
    # assert False

    simple_solver.solve()
    solution = simple_solver.ground_state_solution
    energy = simple_solver.ground_state_energy

    occu = simple_solver.ensemble.processor.occupancy_from_structure(
        simple_solver.ground_state_structure
    )
    sol_occu = get_variable_values_from_occupancy(
        simple_solver.sublattices, occu, simple_solver.variable_indices
    )
    npt.assert_array_equal(solution, sol_occu)

    n_dims = sum([len(s.species) for s in simple_solver.sublattices])
    assert n_dims == 2
    table = get_dim_ids_table(simple_solver.sublattices)
    counts = occu_to_counts(occu, n_dims, dim_ids_table=table)
    if simple_solver.ensemble.chemical_potentials is None:
        # Canonical ensemble, should assume same composition.
        npt.assert_array_equal(counts, simple_solver.fixed_composition)

    features = simple_solver.ensemble.compute_feature_vector(occu)
    true_energy = np.dot(features, simple_solver.ensemble.natural_parameters)

    assert np.isclose(energy, true_energy)

    # Exhaust all other configurations. None should have higher energy than optimal.
    if simple_solver.ensemble.chemical_potentials is not None:
        other_states = list(product(range(2), repeat=8))
    else:
        other_states = set(permutations([0] * 4 + [1] * 4))
    for other_state in other_states:
        other_state = np.array(list(other_state), dtype=int)
        other_feats = simple_solver.ensemble.compute_feature_vector(other_state)
        other_energy = np.dot(other_feats, simple_solver.ensemble.natural_parameters)
        # allow just a tiny slack.
        assert other_energy >= energy - 1e-6
        if np.allclose(other_state, occu):
            assert np.isclose(energy, other_energy)
