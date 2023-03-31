"""Test solver class construction and usage."""

import numpy.testing as npt
import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.solver.upper_bound.solver import UpperboundSolver
from smol.solver.upper_bound.variables import get_variable_values_from_occupancy

from ..utils import assert_msonable


@pytest.fixture(params=["SCIP", "GUROBI"])
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


# Do a small scale test.
