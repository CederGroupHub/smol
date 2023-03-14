"""Solver class for the ground state problem of cluster expansion. SCIP only."""
from collections import Counter
from types import SimpleNamespace
from typing import Any, List, NamedTuple, Union

import cvxpy as cp

from smol.cofe.space.domain import get_species
from smol.moca.ensemble import Ensemble
from smol.moca.processor.base import Processor
from smol.moca.utils.occu import get_dim_ids_by_sublattice


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
    auxiliaries: Union[SimpleNamespace, None]
    constraints: Union[List[cp.Expression], None]


class CompositionConstraintsManager:
    """A descriptor class that manages setting composition constraints."""

    def __set_name__(self, owner, name):
        """Set the private variable names."""
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        """Return the chemical potentials if set None otherwise."""
        value = getattr(obj, self.private_name, None)
        return value if value is None else value["value"]

    @staticmethod
    def _check_single_dict(d):
        for spec, count in Counter(map(get_species, d.keys())).items():
            if count > 1:
                raise ValueError(
                    f"{count} values of the constraint coefficient for the same "
                    f"species {spec} were provided.\n Make sure the dictionary "
                    "you are using has only string keys or only Species "
                    "objects as keys."
                )

    @staticmethod
    def _convert_single_dict(left, bits):
        # Set a constraint with only one dictionary.
        CompositionConstraintsManager._check_single_dict(left)
        n_dims = sum([len(sublattice_bits) for sublattice_bits in bits])
        dim_ids = get_dim_ids_by_sublattice(bits)
        left_list = [0 for _ in range(n_dims)]
        for spec, coef in left.items():
            spec = get_species(spec)
            for sl_dim_ids, sl_bits in zip(dim_ids, bits):
                dim_id = sl_dim_ids[sl_bits.index(spec)]
                left_list[dim_id] = coef
        return left_list

    @staticmethod
    def _convert_sublattice_dicts(left, bits):
        # Set a constraint with one dict per sub-lattice.
        n_dims = sum([len(sublattice_bits) for sublattice_bits in bits])
        dim_ids = get_dim_ids_by_sublattice(bits)
        left_list = [0 for _ in range(n_dims)]
        for sl_dict, sl_bits, sl_dim_ids in zip(left, bits, dim_ids):
            CompositionConstraintsManager._check_single_dict(sl_dict)
            for spec, coef in sl_dict.items():
                spec = get_species(spec)
                dim_id = sl_dim_ids[sl_bits.index(spec)]
                left_list[dim_id] = coef
        return left_list

    def __set__(self, obj, value):
        """Set the table given the owner and value."""
        if value is None or len(value) == 0:  # call delete if set to None
            self.__delete__(obj)
            return

        # value must be list of tuples, each with a list and a number.
        # No scaling would be done. Take care when filling in!
        a_matrix = []
        b_array = []
        bits = [sublattice.species for sublattice in obj.sublattices]
        for left, right in value:
            if isinstance(left, dict):
                a_matrix.append(self._convert_single_dict(left, bits))
            else:
                a_matrix.append(self._convert_sublattice_dicts(left, bits))
            b_array.append(right)

        # if first instantiation concatenate the natural parameter
        if not hasattr(obj, self.private_name):
            setattr(
                obj,
                self.private_name,
                {"value": (a_matrix, b_array)},
            )

    def __delete__(self, obj):
        """Delete the boundary condition."""
        if hasattr(obj, self.private_name):
            del obj.__dict__[self.private_name]


class UpperboundSolver(Ensemble):
    """A solver for the upper-bound problem of cluster expansion."""

    def __init__(
        self,
        processor: Processor,
        chemical_potentials: Any = None,
        other_equality_constraints: List[Any] = None,
        other_leq_constraints: List[Any] = None,
        other_geq_constraints: List[Any] = None,
    ):
        """Initialize UpperboundSolver.

        Note: splitting sub-lattices or fixing sub-lattice sites are not supported!
        Args:
            processor(Processor):
                A processor.

        """
        super().__init__(processor, chemical_potentials=chemical_potentials)
