"""Solver class for the ground state problem of cluster expansion. SCIP only."""
from types import SimpleNamespace
from typing import Any, List, NamedTuple, Union

import cvxpy as cp
from cvxpy.constraints.constraint import Constraint
from monty.json import MSONable
from numpy.typing import ArrayLike

from smol.moca.composition import CompositionSpace
from smol.moca.ensemble import Ensemble
from smol.solver.upper_bound.utils.constraints import (
    get_upper_bound_composition_space_constraints,
)
from smol.solver.upper_bound.utils.variables import (
    get_upper_bound_variables_from_sublattices,
)


class ProblemCanonicals(NamedTuple):
    """CVXpy Canonical objects representing the underlying optimization problem.

    Attributes:
        problem (cp.Problem):
            The cvxpy problem to solve.
        objective (cp.Expression):
            Objective function.
        variables (cp.Variable):
            Variables corresponding to each species on each site to be optimized.
        auxiliaries (SimpleNamespace of cp.Variable or cp.Expression):
            SimpleNamespace with auxiliary cp.Variable or cp.Expression objects.
            The namespace should be defined by the Regressor generating it.
        constraints (list of cp.Constraint):
            List of constraints.
    """

    problem: cp.Problem
    objective: cp.Expression
    variables: cp.Variable
    # Index of variables on each active site.
    variable_indices: List[List[int]]
    auxiliaries: Union[SimpleNamespace, None]
    constraints: Union[List[Constraint], None]


class UpperboundSolver(MSONable):
    """A solver for the upper-bound problem of cluster expansion."""

    def __init__(
        self,
        ensemble: Ensemble,
        initial_occupancy: ArrayLike = None,
        other_equality_constraints: List[Any] = None,
        other_leq_constraints: List[Any] = None,
        other_geq_constraints: List[Any] = None,
    ):
        """Initialize UpperboundSolver.

        Args:
            ensemble(Ensemble):
                An ensemble to initialize the problem with. If you want to modify
                sub-lattices, please do that in ensemble before initializing the
                solver. Sub-lattices can not be modified after giving the ensemble!
            initial_occupancy(ArrayLike): optional
                An initial occupancy used to set the occupancy of manually restricted
                sites that may have more than one allowed species. Also used to set up
                the initial composition when solving in a canonical ensemble.
                Must be provided if any site has been manually restricted, or solving
                canonical ensemble.
            other_equality_constraints(list[tuple[dict|list[dict], float]]): optional
                Representation of other equality constraints to add in the
                composition space. Should be provided as a list of tuples, with
                the first element in the tuple to specify the left side of the
                equality, and the second element to specify the right side of
                the equality. Meanwhile, the first element in the tuple can
                either be a single dictionary with species as keys and float as
                values, to specify coefficients to the left side of the equation
                corresponding to the particular species on all sub-lattices; or
                be provided as a list of dictionaries, each gives the coefficient in
                the constraint to the amount of species in each sub-lattice.
            other_leq_constraints(list[tuple[dict|list[dict], float]]): optional
                Representation of less-or-equals constraints. Same format as
                other_equality_constraints.
            other_geq_constraints(list[tuple[dict|list[dict], float]]): optional
                Representation of greater-or-equals constraints. Same format as
                other_equality_constraints.
        """
        self._ensemble = ensemble

        bits = [sublattice.species for sublattice in ensemble.sublattices]
        prim_sublattice_sizes = [
            len(sublattice.sites) // ensemble.system_size
            for sublattice in ensemble.sublattices
        ]
        self.other_equality_constraints = other_equality_constraints
        self.other_geq_constraints = other_geq_constraints
        self.other_leq_constraints = other_leq_constraints
        self._composition_space = CompositionSpace(
            bits,
            prim_sublattice_sizes,
            charge_balanced=True,
            other_constraints=self.other_equality_constraints,
            leq_constraints=self.other_leq_constraints,
            geq_constraints=self.other_geq_constraints,
        )

        self._canonical = self._initialize_problem()

    def _initialize_problem(self):
        """Generate variables, objective and constraints."""
        variables, variable_indices = get_upper_bound_variables_from_sublattices(
            self._ensemble.sublattices, self._ensemble.num_sites
        )
        get_upper_bound_composition_space_constraints()
