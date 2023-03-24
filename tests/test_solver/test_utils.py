"""Test utility functions for upper-bound."""

from smol.solver.upper_bound.variables import get_upper_bound_variables_from_sublattices


# TODO: finish this.
def test_variable_indices_for_components(ensemble):
    sublattices = ensemble.sublattices
    get_upper_bound_variables_from_sublattices(sublattices, ensemble.num_sites)
