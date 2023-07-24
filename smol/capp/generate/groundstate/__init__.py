"""Ground state solver functionalities."""

from monty.dev import requires

try:
    from smol.capp.generate.groundstate.upper_bound.solver import UpperBoundSolver
except ImportError:

    @requires(
        False,
        "Ground state solver functionality requires cvxpy to be installed, please install it.",
    )
    class UpperBoundSolver:
        """Dummy class to fail gracefully when cvxpy is not installed."""



__all__ = ["UpperBoundSolver"]
