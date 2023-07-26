"""Ground state solver functionalities."""

from monty.dev import requires

try:
    from smol.capp.generate.groundstate.upper_bound.solver import (
        PeriodicGroundStateSolver,
    )
except ImportError:

    @requires(
        False,
        "Ground state solver functionality requires cvxpy to be installed, please install it.",
    )
    class PeriodicGroundStateSolver:
        """Dummy class to fail gracefully when cvxpy is not installed."""


__all__ = ["PeriodicGroundStateSolver"]
