"""Ground state solver functionalities."""
from importlib.util import find_spec
from .upper_bound.solver import UpperBoundSolver

# Check cvxpy installation.
if find_spec("cvxpy") is None:
    raise ImportError(
        "Ground state utilities require cvxpy and integer-programming solvers!"
    )

__all__ = ["UpperBoundSolver"]
