"""Test solver class construction and usage."""
import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.moca.utils.occu import get_dim_ids_table, occu_to_counts
from smol.solver.upper_bound.solver import UpperboundSolver
from smol.solver.upper_bound.variables import get_variable_values_from_occupancy

from ..utils import assert_msonable


# Both SCIP, GUROBI tried on this instance.
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


# Do a small scale solving test.
def test_solve(exotic_solver):
    _, energy = exotic_solver.solve()
    # Optimality not tested.
    assert exotic_solver.ground_state_structure.charge == 0
    occu = exotic_solver._ensemble.processor.occupancy_from_structure(
        exotic_solver.ground_state_structure
    )

    n_dims = sum([len(s.species) for s in exotic_solver._ensemble.sublattices])
    table = get_dim_ids_table(exotic_solver._ensemble.sublattices)
    counts = occu_to_counts(occu, n_dims, dim_ids_table=table)
    if exotic_solver._ensemble.chemical_potentials is None:
        # Canonical ensemble, should assume same composition.
        npt.assert_array_equal(counts, exotic_solver._fixed_composition)

    features = exotic_solver._ensemble.compute_feature_vector(occu)
    true_energy = np.dot(features, exotic_solver._ensemble.natural_parameters)

    assert np.isclose(energy, true_energy)
