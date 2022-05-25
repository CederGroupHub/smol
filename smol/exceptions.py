"""Definitions of specific exceptions raised elsewhere."""


SYMMETRY_ERROR_MESSAGE = (
    "Error in calculating symmetry operations."
    "Try using a more symmetrically refined input"
    "structure. "
    "SpacegroupAnalyzer(s).get_refined_structure()"
    ".get_primitive_structure() "
    "usually results in a safe choice"
)


class NotFittedError(ValueError, AttributeError):
    """Exception if sklearn regression function is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


class SymmetryError(ValueError):
    """Exception for incompatibility between structure and given symops.

    Exception to raise when symmetry of a structure is not compatible with a
    set of given symops.
    """


class StructureMatchError(RuntimeError):
    """Raised when a pymatgen StructureMatcher returns None."""
