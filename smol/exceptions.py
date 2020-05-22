"""
Definitions of specific exceptions raised elsewhere in module
"""


SYMMETRY_ERROR_MESSAGE = ("Error in calculating symmetry operations."
                          "Try using a more symmetrically refined input"
                          "structure. "
                          "SpacegroupAnalyzer(s).get_refined_structure()"
                          ".get_primitive_structure() "
                          "usually results in a safe choice")


class NotFittedError(ValueError, AttributeError):
    """
    Exception class to raise if learn is used before fitting.
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """
    pass


class SymmetryError(ValueError):
    """
    Exception class to raise when symmetry of a structure are not compatible
    with a set of given symops.
    """
    pass


class StructureMatchError(RuntimeError):
    """
    Raised when a pymatgen StructureMatcher returns None.
    """
    pass
