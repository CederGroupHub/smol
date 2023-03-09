"""Generate cvxpy upper-bound problem from an ensemble object."""
from numbers import Number
from typing import List, Tuple

import cvxpy as cp

from smol.moca import Ensemble

__author__ = "Fengyu Xie"

# TODO: 1, define cvxpy problem canonicals as in sparse-lm;
#  2, Implement this.


def generate_problem_from_ensemble(
    ensemble: Ensemble,
    additional_equality_constraints: List[Tuple[List[Number], Number]],
    additional_leq_constraints: List[Tuple[List[Number], Number]],
    additional_geq_constraints: List[Tuple[List[Number], Number]],
) -> cp.Problem:
    """Generate cvxpy problem from an Ensemble object."""
    return
