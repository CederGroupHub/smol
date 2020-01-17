"""
A lot of random utilies that have no place to go :(
"""
from typing import Dict, Any
SITE_TOL = 1e-6


SYMMETRY_ERROR_MESSAGE = ("Error in calculating symmetry operations."
                          "Try using a more symmetrically refined input"
                          "structure. "
                          "SpacegroupAnalyzer(s).get_refined_structure()"
                          ".get_primitive_structure() "
                          "usually results in a safe choice")


class SymmetryError(ValueError):
    """
    Exception class to raise when symmetry of a structure are not compatible
    with a set of given symops
    """


class NotFittedError(ValueError, AttributeError):
    """
    Exception class to raise if regression is used before fitting.
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


class StructureMatchError(RuntimeError):
    """
    Raised when a pymatgen StructureMatcher returns None
    """


def _repr(object: object, **fields: Dict[str, Any]) -> str:
    """
    A helper function for repr overloading in classes
    """
    attrs = []

    for key, field in fields.items():
        attrs.append(f'{key}={field!r}')

    if len(attrs) == 0:
        return f"<{object.__class__.__name__}" \
               f"{hex(id(object))}({','.join(attrs)})>"
    else:
        return f"<{object.__class__.__name__} {hex(id(object))}>"
