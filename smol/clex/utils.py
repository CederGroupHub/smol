
SITE_TOL = 1e-6


#TODO better error messages and descriptions of exceptions
SYMMETRY_ERROR_MESSAGE = ValueError("Error in calculating symmetry operations. Try using a "
                            "more symmetrically refined input structure. "
                            "SpacegroupAnalyzer(s).get_refined_structure().get_primitive_structure() "
                            "usually results in a safe choice")

class SymmetryError(ValueError):
    """Exception class to raise when symmetry of a structure are not compatible with a set of given
    symops
    """

class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """