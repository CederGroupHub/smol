"""Ground state groundstate functionalities."""
from importlib.util import find_spec

# Check cvxpy installation.
if find_spec("cvxpy") is None:
    raise ImportError(
        "Ground state utilities require cvxpy and integer-programming solvers!"
    )
