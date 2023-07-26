"""Solver class for the ground state problem of cluster expansion."""
from typing import List, NamedTuple, Union
from warnings import warn

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint
from numpy.typing import ArrayLike

from smol.capp.generate.groundstate.upper_bound.constraints import (
    get_composition_space_constraints,
    get_fixed_composition_constraints,
    get_normalization_constraints,
)
from smol.capp.generate.groundstate.upper_bound.objectives import (
    get_terms_from_chemical_potentials,
    get_terms_from_decomposition_processor,
    get_terms_from_ewald_processor,
    get_terms_from_expansion_processor,
)
from smol.capp.generate.groundstate.upper_bound.terms import (
    get_expression_and_auxiliary_from_terms,
)
from smol.capp.generate.groundstate.upper_bound.variables import (
    get_occupancy_from_variables,
    get_variables_from_sublattices,
)
from smol.moca.ensemble import Ensemble
from smol.moca.occu_utils import get_dim_ids_table, occu_to_counts
from smol.moca.processor import (
    ClusterDecompositionProcessor,
    ClusterExpansionProcessor,
    CompositeProcessor,
    EwaldProcessor,
)

__author__ = "Fengyu Xie"


class ProblemCanonicals(NamedTuple):
    """CVXpy Canonical objects representing the underlying optimization problem.

    Attributes:
        problem (cp.Problem):
            The cvxpy problem to solve.
        variables (cp.Variable):
            Variables corresponding to each species on each site to be optimized.
        variable_indices (cp.Variable):
            List of variable index corresponding to each active site index and species
            indices in its site space. Inactive or restricted sites will not have any
            variable, but be marked as -1 or -2. See groundstate.upper_bound.variables.
        auxiliary_variables (cp.Variable):
            Slack variables used to linearize polynomial pseudo-boolean objective
            terms.
        indices_in_auxiliary_products (list of lists of int):
            A list containing the indices of variables whose product equals to the
            corresponding auxiliary slack variable.
        objective_function (cp.Expression):
            Objective function to be minimized. Usually energy per super-cell.
        constraints (list of cp.Constraint):
            List of constraints.
        num_auxiliary_constraints(int):
            Number of slack linearize constraints. Slack constraints are always
            the last constraints in the list.
    """

    problem: cp.Problem
    variables: cp.Variable
    # Index of variables on each active site.
    variable_indices: List[List[int]]
    auxiliary_variables: cp.Variable
    indices_in_auxiliary_products: List[List[int]]
    objective_function: cp.Expression
    constraints: Union[List[Constraint], None]
    num_auxiliary_constraints: int


class PeriodicGroundStateSolver:
    """Class to solve for the ground state occupancy/structure of a given ensemble.

    This implementation solves for the periodic ground-state configuration for a
    fixed super-cell size stored as the supercell matrix in the ensemble's processor.
    Meaning that the solution corresponds only to an upper bound of the global ground
    state of the corresponding infinitely sized system.

    For more details on the nature of the global problem, i.e. finding the upper and the
    lower bounds, please see:

    https://doi.org/10.1103/PhysRevB.94.134424

    .. warning::
        The number of terms and constraints in underlying mixed-integer problem grow
        very rapidly with the size of the supercell, the number of allowed species,
        and the number of terms in the cluster expansion. As a result, solving for
        ground-states of large supercells and complex systems can be very
        time-consuming or even impossible, even using proprietary solvers and many
        cores. Make sure to check the number of terms and constraints in the problem
        and benchmark on a smaller supercell.

    """

    def __init__(
        self,
        ensemble: Ensemble,
        charge_balanced: bool = True,
        initial_occupancy: ArrayLike = None,
        fixed_composition: ArrayLike = None,
        other_constraints: list = None,
        term_coefficients_cutoff: float = 0.0,
        warm_start: bool = False,
        solver: str = "SCIP",
        solver_options: dict = None,
    ):
        """Initialize PeriodicGroundStateSolver.

        Args:
            ensemble (Ensemble):
                An ensemble to initialize the problem with. If you want to modify
                sub-lattices, please do that in ensemble before initializing the
                groundstate. Sub-lattices can not be modified after giving the ensemble!
            charge_balanced (bool): optional
                Whether to enforce the charge-balance constraint. Default is True.
            initial_occupancy (ArrayLike): optional
                An initial occupancy used to set the occupancy of manually restricted
                sites that may have more than one allowed species. Also used to set up
                the initial composition when solving in a canonical ensemble.
                Must be provided if any site has been manually restricted, or solving
                canonical ensemble.
            fixed_composition (ArrayLike): optional
                An array of floats corresponding to the "counts" format of composition
                in CompositionSpace. See moca.composition. Unit is per super-cell.
                If the ensemble is canonical (no chemical potential has been set), will
                need to constrain the ground-state problem to a fixed composition.
                If initial_occupancy is provided while fixed_composition is None,
                composition will be fixed to that of initial_occupancy.
                Will be ignored in a semi-grand ensemble.
                If any of initial_occupancy or fixed_composition is provided, you are
                fully responsible to ensure that they satisfy charge balance and other
                constraints.
            other_constraints (list): optional
                Other constraints to set in composition space. See moca.composition
                for detailed description.
                Note that constraints are now to be satisfied with the number of sites
                in sub-lattices of the supercell instead of the primitive cell.
            term_coefficients_cutoff (float): optional
                Minimum cutoff to the coefficient of terms in the final polynomial Boolean
                objective function.
                If the absolute value of a term is less than this cutoff, it will not be
                included in the optimization. If no cutoff is given, will include every term
                in the optimization.
            solver (str): optional
               Specify the groundstate used to solve the problem.
               SCIP is default because it does not require
               to linearize the problem before solving a polynomial
               form pseudo-boolean optimization.
               See cvxpy and PySciOpt documentations on how to correctly
               configure SCIP for cvxpy.
            warm_start (bool): optional
               Whether to use the previous solution as an initialization
               of the current problem solve. Default is False.
            solver_options (dict): optional
               Options to be passed into the groundstate when calling Problem.solve().
               See cvxpy and the specific groundstate documentations for detail.
        """
        self.ensemble = ensemble
        self.charge_balanced = charge_balanced
        self._structure = self.ensemble.processor.structure
        self._sublattices = self.ensemble.sublattices

        self._other_constraints = other_constraints
        if initial_occupancy is not None:
            self.initial_occupancy = np.array(initial_occupancy, dtype=int)
        else:
            self.initial_occupancy = None

        # Can always be set, but will only be used in a canonical ensemble.
        if fixed_composition is not None:
            self.fixed_composition = np.array(fixed_composition, dtype=int)
        elif self.initial_occupancy is not None:
            n_dims = sum(len(s.species) for s in self.sublattices)
            dim_ids_table = get_dim_ids_table(self.sublattices, active_only=False)
            self.fixed_composition = occu_to_counts(
                self.initial_occupancy, n_dims, dim_ids_table
            )
        else:
            self.fixed_composition = None

        self.cutoff = term_coefficients_cutoff
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
        sublattices = self.sublattices
        variables, variable_indices = get_variables_from_sublattices(
            sublattices, self.structure, self.initial_occupancy
        )

        # Add constraints.
        constraints = get_normalization_constraints(variables, variable_indices)
        # Canonical ensemble.
        if self.ensemble.chemical_potentials is None:
            constraints.extend(
                get_fixed_composition_constraints(
                    sublattices,
                    variables,
                    variable_indices,
                    self.structure,
                    self.fixed_composition,
                )
            )
        # Semi-grand ensemble.
        else:
            constraints.extend(
                get_composition_space_constraints(
                    sublattices,
                    variables,
                    variable_indices,
                    self.structure,
                    charge_balanced=self.charge_balanced,
                    other_constraints=self._other_constraints,
                )
            )

        def _handle_single_processor(proc):
            # Give energy terms in the expression.
            if isinstance(proc, ClusterExpansionProcessor):
                return get_terms_from_expansion_processor(
                    variable_indices,
                    expansion_processor=proc,
                )
            if isinstance(proc, ClusterDecompositionProcessor):
                return get_terms_from_decomposition_processor(
                    variable_indices,
                    decomposition_processor=proc,
                )
            if isinstance(proc, EwaldProcessor):
                return get_terms_from_ewald_processor(
                    variable_indices,
                    ewald_processor=proc,
                )
            raise NotImplementedError(
                f"Ground state upper-bound objective function"
                f" not implemented for processor type:"
                f" {type(proc)}!"
            )

        # Objective energy function.
        processor = self.ensemble.processor
        if isinstance(processor, CompositeProcessor):
            terms = []
            for p in processor.processors:
                terms.extend(_handle_single_processor(p))
        else:
            terms = _handle_single_processor(processor)
        # E - mu N for semi-grand ensemble.
        if self.ensemble.chemical_potentials is not None:
            # Chemical potential term already includes the "-" before mu N.
            terms.extend(
                get_terms_from_chemical_potentials(
                    variable_indices,
                    chemical_table=self.ensemble._chemical_potentials["table"],
                )
            )

        (
            objective_func,
            aux_variables,
            indices_in_aux_products,
            aux_constraints,
        ) = get_expression_and_auxiliary_from_terms(terms, variables, self.cutoff)
        n_aux_constraints = len(aux_constraints)
        constraints.extend(aux_constraints)

        problem = cp.Problem(cp.Minimize(objective_func), constraints)

        self._canonicals = ProblemCanonicals(
            problem=problem,
            variables=variables,
            variable_indices=variable_indices,
            auxiliary_variables=aux_variables,
            indices_in_auxiliary_products=indices_in_aux_products,
            objective_function=objective_func,
            constraints=constraints,
            num_auxiliary_constraints=n_aux_constraints,
        )

    @property
    def structure(self):
        """Supercell structure of the ensemble without any split.

        Returns:
            Structure.
        """
        return self._structure

    @property
    def sublattices(self):
        """Sub-lattices to build the problem on.

        Potentially contains split sub-lattices or manually restricted sites.
        Returns:
            List of Sublattice.
        """
        return self._sublattices

    @property
    def canonicals(self):
        """The named tuple containing cvxpy problem details.

        Returns:
            ProblemCanonicals.
        """
        return self._canonicals

    @property
    def problem(self):
        """The cvxpy problem for solving upper-bound of ground-state.

        Returns:
            cvxpy.Problem.
        """
        return self.canonicals.problem

    @property
    def variables(self):
        """CVXPY variables for species on each active site.

        Returns:
            cvxpy.Variable.
        """
        return self.canonicals.variables

    @property
    def variable_indices(self):
        """Indices of variables on each site if site is active.

        Returns:
            list of lists of int: each sub-list contains variable
            indices on an active site.
        """
        return self.canonicals.variable_indices

    @property
    def objective_function(self):
        """The objective function to be minimized.

        Usually total energy per super-cell.

        Returns:
            cvxpy.Expression.
        """
        return self.canonicals.objective_function

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
        return self.canonicals.constraints

    @property
    def auxiliary_variables(self):
        """Auxiliary variables for linearizing the problem.

        Returns:
            cp.Variable.
        """
        return self.canonicals.auxiliary_variables

    @property
    def indices_in_auxiliary_products(self):
        """Indices of variables in cluster terms corresponding to auxiliary variables.

        Returns:
            list of lists of int
        """
        return self.canonicals.indices_in_auxiliary_products

    def solve(self):
        """Solve the MIP problem.

        Returns:
            np.ndarray, float:
                values of boolean variables in the ground-state solution, and
                the minimized ground-state energy.
        """
        if (
            self._ground_state_solution is not None
            and self._ground_state_energy is not None
        ):
            warn("Ground state already solved before. Overwriting previous result!")
        self.problem.solve(
            solver=self.solver, warm_start=self.warm_start, **self.solver_options
        )
        if (
            self.variables.value is not None
            and self.objective_function.value is not None
        ):
            self._ground_state_solution = self.variables.value.astype(int)
            self._ground_state_energy = self.objective_function.value
        else:
            warn(
                "Ground state could not be solved! Try using another solver, or"
                " fine tune solver options!"
            )
            self._ground_state_solution = None
            self._ground_state_energy = None
        # Return Nothing.

    def reset(self):
        """Clear previous solutions and reinitialize the problem."""
        self._ground_state_solution = None
        self._ground_state_energy = None
        self._ground_state_occupancy = None
        self._ground_state_structure = None

        self._initialize_problem()

    def _raise_unsolved(self):
        if self._ground_state_solution is None or self._ground_state_energy is None:
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
                self.sublattices,
                self.ground_state_solution,
                self.variable_indices,
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
                self.ensemble.processor.structure_from_occupancy(
                    self.ground_state_occupancy
                )
            )
        return self._ground_state_structure
