"""Ground state groundstate functionalities."""

try:
    import cvxpy
except ModuleNotFoundError:
    raise ImportError(
        "Ground state utilities require cvxpy and integer-programming solvers!"
    )
