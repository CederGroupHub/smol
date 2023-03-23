"""Solver class for the ground state problem of cluster expansion. SCIP only."""
from typing import List, NamedTuple, Union

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint
from monty.json import MSONable, jsanitize
from numpy.typing import ArrayLike

from smol.moca.ensemble import Ensemble
from smol.moca.processor import (
    ClusterDecompositionProcessor,
    ClusterExpansionProcessor,
    CompositeProcessor,
    EwaldProcessor,
)
from smol.moca.utils.occu import get_dim_ids_table, occu_to_counts
from smol.solver.upper_bound.utils.constraints import (
    get_upper_bound_composition_space_constraints,
    get_upper_bound_fixed_composition_constraints,
    get_upper_bound_normalization_constraints,
)
from smol.solver.upper_bound.utils.objectives import (
    get_upper_bound_objective_from_chemical_potentials,
    get_upper_bound_objective_from_decomposition_processor,
    get_upper_bound_objective_from_ewald_processor,
    get_upper_bound_objective_from_expansion_processor,
)
from smol.solver.upper_bound.utils.variables import (
    get_occupancy_from_variables,
    get_upper_bound_variables_from_sublattices,
)


class ProblemCanonicals(NamedTuple):
    """CVXpy Canonical objects representing the underlying optimization problem.

    Attributes:
        problem (cp.Problem):
            The cvxpy problem to solve.
        variables (cp.Variable):
            Variables corresponding to each species on each site to be optimized.
        objective_function (cp.Expression):
            Objective function to be minimized. Usually energy per super-cell.
        auxiliaries (SimpleNamespace of cp.Variable or cp.Expression):
            SimpleNamespace with auxiliary cp.Variable or cp.Expression objects.
            The namespace should be defined by the Regressor generating it.
        constraints (list of cp.Constraint):
            List of constraints.
    """

    problem: cp.Problem
    variables: cp.Variable
    # Index of variables on each active site.
    variable_indices: List[List[int]]
    objective_function: cp.Expression
    constraints: Union[List[Constraint], None]


class UpperboundSolver(MSONable):
    """A solver for the upper-bound problem of cluster expansion."""

    def __init__(
        self,
        ensemble: Ensemble,
        initial_occupancy: ArrayLike = None,
        fixed_composition: ArrayLike = None,
        other_constraints: list = None,
        warm_start: bool = False,
        solver: str = "SCIP",
        solver_options: dict = None,
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
            fixed_composition(ArrayLike): optional
                An array of floats corresponding to the "counts" format of composition
                in CompositionSpace. See moca.composition.
                If the ensemble is canonical (no chemical potential has been set), will
                need to constrain the ground-state problem to a fixed composition.
                If initial_occupancy is provided, fixed_composition is not used, and the
                composition will be fixed to that of initial_occupancy.
                Will be ignored in a semi-grand canonical ensemble.
                If any of initial_occupancy or fixed_composition is provided, you are
                fully responsible to ensure that they satisfy charge balance and other
                constraints.
            other_constraints(list): optional
                Other constraints to set in composition space. See moca.composition
                for detailed description.
                Note that constraints are now to be satisfied with the number of sites
                in sub-lattices of the supercell instead of the primitive cell.
            solver(str): optional
               Specify the solver used to solve the problem.
               SCIP is default because it does not require
               to linearize the problem before solving a polynomial
               form pseudo-boolean optimization.
               See cvxpy and PySciOpt documentations on how to correctly
               configure SCIP for cvxpy.
            warm_start(bool): optional
               Whether to use the previous solution as an initialization
               of the current problem solve. Default is False.
            solver_options(dict): optional
               Options to be passed into the solver when calling Problem.solve().
               See cvxpy and the specific solver documentations for detail.
        """
        self._ensemble = ensemble

        self._other_constraints = other_constraints
        if initial_occupancy is not None:
            self._initial_occupancy = np.array(initial_occupancy, dtype=int)
        else:
            self._initial_occupancy = None

        if fixed_composition is not None:
            self._fixed_composition = np.array(fixed_composition, dtype=int)
        elif self._initial_occupancy is not None:
            n_dims = sum(len(s.species) for s in self._ensemble.sublattices)
            dim_ids_table = get_dim_ids_table(
                self._ensemble.sublattices, active_only=False
            )
            self._fixed_composition = occu_to_counts(
                self._initial_occupancy, n_dims, dim_ids_table
            )
        else:
            self._fixed_composition = None

        self.warm_start = warm_start
        self.solver = solver
        self.solver_options = solver_options or {}

        self._ground_state_solution = None
        self._ground_state_energy = None
        self._ground_state_occupancy = None
        self._ground_state_structure = None

        self._initialize_problem()

    def _initialize_problem(self):
        """Generate variables, objective and constraints."""
        sublattices = self._ensemble.sublattices
        variables, variable_indices = get_upper_bound_variables_from_sublattices(
            sublattices, self._ensemble.num_sites
        )

        # Add constraints.
        constraints = get_upper_bound_normalization_constraints(
            variables, variable_indices
        )
        # Canonical ensemble.
        if self._ensemble.chemical_potentials is None:
            constraints.extend(
                get_upper_bound_fixed_composition_constraints(
                    sublattices,
                    variables,
                    variable_indices,
                    self._fixed_composition,
                    self._initial_occupancy,
                )
            )
        # Semi-grand ensemble.
        else:
            constraints.extend(
                get_upper_bound_composition_space_constraints(
                    sublattices,
                    variables,
                    variable_indices,
                    self._other_constraints,
                    self._initial_occupancy,
                )
            )

        def _handle_single_processor(proc):
            if isinstance(proc, ClusterExpansionProcessor):
                return get_upper_bound_objective_from_expansion_processor(
                    sublattices,
                    variables,
                    variable_indices,
                    expansion_processor=proc,
                    initial_occupancy=self._initial_occupancy,
                )
            if isinstance(proc, ClusterDecompositionProcessor):
                return get_upper_bound_objective_from_decomposition_processor(
                    sublattices,
                    variables,
                    variable_indices,
                    decomposition_processor=proc,
                    initial_occupancy=self._initial_occupancy,
                )
            if isinstance(proc, EwaldProcessor):
                return get_upper_bound_objective_from_ewald_processor(
                    sublattices,
                    variables,
                    variable_indices,
                    ewald_processor=proc,
                    initial_occupancy=self._initial_occupancy,
                )
            raise NotImplementedError(
                f"Ground state upper-bound objective function"
                f" not implemented for processor type:"
                f" {type(proc)}"
            )

        # Objective energy function.
        processor = self._ensemble.processor
        if isinstance(processor, CompositeProcessor):
            objective = 0
            for p in processor.processors:
                objective += _handle_single_processor(p)
        else:
            objective = _handle_single_processor(processor)
        # E - mu N for semi-grand ensemble.
        if self._ensemble.chemical_potentials is not None:
            # Chemical potential term already includes the "-" before mu N.
            objective += get_upper_bound_objective_from_chemical_potentials(
                sublattices,
                variables,
                variable_indices,
                chemical_table=self._ensemble._chemical_potentials["table"],
                initial_occupancy=self._initial_occupancy,
            )

        problem = cp.Problem(cp.Minimize(objective), constraints)

        self._canonicals = ProblemCanonicals(
            problem=problem,
            variables=variables,
            variable_indices=variable_indices,
            objective_function=objective,
            constraints=constraints,
        )

        # Clear solution.
        self._ground_state_solution = None
        self._ground_state_energy = None
        self._ground_state_occupancy = None
        self._ground_state_structure = None

    @property
    def problem(self):
        """The cvxpy problem for solving upper-bound of ground-state.

        Returns:
            cvxpy.Problem.
        """
        return self._canonicals.problem

    @property
    def variables(self):
        """CVXPY variables for species on each active site.

        Returns:
            cvxpy.Variable.
        """
        return self._canonicals.variables

    @property
    def variable_indices(self):
        """Indices of variables on each site if site is active.

        Returns:
            list[list[int]]: each sub-list contains variable
            indices on an active site.
        """
        return self._canonicals.variable_indices

    @property
    def objective_function(self):
        """The objective function to be minimized.

        Usually total energy per super-cell.

        Returns:
            cvxpy.Expression.
        """
        return self._canonicals.objective_function

    @property
    def constraints(self):
        """Constraints to be satisfied in minimization.

        Must include normalization per site. May include
        charge balance and other constraints to composition.
        In canonical ensemble, must include a constraint to
        fix composition.

        Returns:
            List[cvxpy.Constraint]
        """
        return self._canonicals.constraints

    def solve(self):
        """Solve the MIP problem.

        Returns:
            np.ndarray, float:
                values of boolean variables in the ground-state solution, and
                the minimized ground-state energy.
        """
        self.problem.solve(
            solver=self.solver, warm_start=self.warm_start, **self.solver_options
        )
        self._ground_state_solution = self.variables.value
        self._ground_state_energy = self.problem.value
        return self._ground_state_solution, self._ground_state_energy

    def reset(self):
        """Clear previous solutions and reinitialize the problem."""
        self._initialize_problem()

    def _raise_unsolved(self):
        if self._ground_state_solution is None:
            raise RuntimeError(
                "The ground state of the current system is not"
                " solved before or was not successful. Call"
                " self.solve()!"
            )

    @property
    def ground_state_solution(self):
        """Boolean variable values at the ground state.

        Returns:
            np.ndarray: the solution array in 0 and 1.
        """
        self._raise_unsolved()
        return self._ground_state_solution

    @property
    def ground_state_energy(self):
        """Energy per super-cell in the ground-state.

        Returns:
            float.
        """
        self._raise_unsolved()
        return self._ground_state_energy

    @property
    def ground_state_occupancy(self):
        """Encoded occupancy string of the ground-state.

        Returns:
            np.ndarray
        """
        self._raise_unsolved()
        if self._ground_state_occupancy is None:
            self._ground_state_occupancy = get_occupancy_from_variables(
                self._ensemble.sublattices,
                self._ground_state_solution,
                self.variable_indices,
                self._initial_occupancy,
            )
        return self._ground_state_occupancy

    @property
    def ground_state_structure(self):
        """The ground state structure.

        Returns:
            Structure.
        """
        self._raise_unsolved()
        if self._ground_state_structure is None:
            self._ground_state_structure = (
                self._ensemble.processor.structure_from_occupancy(
                    self.ground_state_occupancy
                )
            )
        return self._ground_state_structure

    def as_dict(self):
        """Serialize UpperboundSolver as a dictionary.

        Returns:
            dict.
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "ensemble": self._ensemble.as_dict(),
            "initial_occupancy": (
                self._initial_occupancy.tolist()
                if self._initial_occupancy is not None
                else None
            ),
            "fixed_composition": (
                self._fixed_composition.tolist()
                if self._fixed_composition is not None
                else None
            ),
            # Convert species object to strings, if any.
            "other_constraints": jsanitize(self._other_constraints),
            "warm_start": self.warm_start,
            "solver": self.solver,
            "solver_options": self.solver_options,
            "_ground_state_solution": (
                self._ground_state_solution.tolist()
                if self._ground_state_solution is not None
                else None
            ),
            "_ground_state_energy": (
                self._ground_state_energy.tolist()
                if self._ground_state_energy is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d):
        """Initialize UpperboundSolver from dictionary.

        Args:
            d(dict):
                Serialized dictionary, may contain previous solutions.
        Returns:
            UpperboundSolver.
        """
        socket = cls(
            ensemble=Ensemble.from_dict(d["ensemble"]),
            initial_occupancy=d.get("initial_occupancy"),
            fixed_composition=d.get("fixed_composition"),
            other_constraints=d.get("other_constraints"),
            warm_start=d.get("warm_start", False),
            solver=d.get("solver", "SCIP"),
            solver_options=d.get("solver_options"),
        )
        solution = d.get("_ground_state_solution")
        if solution is not None:
            solution = np.array(solution).astype(int)  # Save as 0 and 1.
        socket._ground_state_solution = solution
        socket._ground_state_energy = d.get("_ground_state_energy")

        return socket
