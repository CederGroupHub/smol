"""Test boolean variables' generation for upper-bound groundstate."""
import numpy as np
import numpy.testing as npt
import pytest

from smol.capp.generate.groundstate.upper_bound.indices import (
    get_sublattice_indices_by_site,
)
from smol.capp.generate.groundstate.upper_bound.variables import (
    get_occupancy_from_variables,
    get_variable_values_from_occupancy,
    get_variables_from_sublattices,
)
from smol.cofe.space.domain import get_allowed_species

from .utils import get_random_variable_values


def test_upper_variables_and_indices(
    solver_test_ensemble, solver_test_initial_occupancy
):
    processor_structure = solver_test_ensemble.processor.structure
    orig_site_spaces = get_allowed_species(processor_structure)
    # If not given initial occupancy, will throw an error.
    with pytest.raises(ValueError):
        _ = get_variables_from_sublattices(
            solver_test_ensemble.sublattices, processor_structure
        )
    variables, variable_indices = get_variables_from_sublattices(
        solver_test_ensemble.sublattices,
        processor_structure,
        solver_test_initial_occupancy,
    )

    assert len(variable_indices) == solver_test_ensemble.num_sites
    # Variable indices must be continuous.
    assert variables.size == max(max(site_vars) for site_vars in variable_indices) + 1
    flatten_indices = sorted(
        [ind for site_inds in variable_indices for ind in site_inds if ind >= 0]
    )
    assert list(range(variables.size)) == flatten_indices

    # Check for each site.
    for site_id, site_vars in enumerate(variable_indices):
        site_var_inds = np.array(site_vars, dtype=int)
        sublattice = None
        for sl in solver_test_ensemble.sublattices:
            if site_id in sl.sites:
                sublattice = sl
        # Contains inactive site or impossible species.
        if np.any(site_var_inds < 0):
            if np.any(site_var_inds >= 0):
                assert np.all(site_var_inds != -1)
            # Only one species can be fixed true.
            else:
                assert np.sum(site_var_inds == -1) == 1
                assert site_id in solver_test_ensemble.restricted_sites
                # And this fixed true species must match that in initial occupancy.
                # Inactive sub-lattice.
                if len(sublattice.species) == 1:
                    expected_sp_id = orig_site_spaces[site_id].index(
                        sublattice.species[0]
                    )
                # Manually restricted site.
                else:
                    expected_code = solver_test_initial_occupancy[site_id]
                    # Assume split from a fresh sub-lattice (code continuous from 0~n).
                    expected_sp_id = expected_code
                expected_site_vars = np.zeros(len(site_vars), dtype=int) - 2
                expected_site_vars[expected_sp_id] = -1
                npt.assert_array_equal(expected_site_vars, site_vars)
        else:
            assert len(site_vars) == len(sublattice.encoding)
            assert list(range(min(site_vars), max(site_vars) + 1)) == site_vars


def test_occupancy_from_variables(solver_test_ensemble, solver_test_initial_occupancy):
    _, variable_indices = get_variables_from_sublattices(
        solver_test_ensemble.sublattices,
        solver_test_ensemble.processor.structure,
        solver_test_initial_occupancy,
    )

    site_sublattice_ids = get_sublattice_indices_by_site(
        solver_test_ensemble.sublattices
    )

    for _ in range(20):
        rand_vals = get_random_variable_values(solver_test_ensemble.sublattices)
        rand_occu = get_occupancy_from_variables(
            solver_test_ensemble.sublattices,
            rand_vals,
            variable_indices,
        )
        # Constrained sites must be the same.
        npt.assert_array_equal(
            rand_occu[solver_test_ensemble.restricted_sites],
            solver_test_initial_occupancy[solver_test_ensemble.restricted_sites],
        )
        # Check active sites match.
        for site_id in np.setdiff1d(
            np.arange(solver_test_ensemble.num_sites, dtype=int),
            solver_test_ensemble.restricted_sites,
        ):
            sublattice = solver_test_ensemble.sublattices[site_sublattice_ids[site_id]]
            expected_vals = np.array(sublattice.encoding == rand_occu[site_id]).astype(
                int
            )
            active_variable_indices = np.array(variable_indices[site_id], dtype=int)
            active_variable_indices = active_variable_indices[
                active_variable_indices >= 0
            ]
            input_vals = rand_vals[active_variable_indices].astype(int)
            npt.assert_array_equal(expected_vals, input_vals)


def test_variables_from_occupancy(solver_test_ensemble, solver_test_initial_occupancy):
    _, variable_indices = get_variables_from_sublattices(
        solver_test_ensemble.sublattices,
        solver_test_ensemble.processor.structure,
        solver_test_initial_occupancy,
    )
    # Check conversions are correct for random variables.
    for _ in range(20):
        rand_val = get_random_variable_values(solver_test_ensemble.sublattices).astype(
            int
        )
        rand_occu = get_occupancy_from_variables(
            solver_test_ensemble.sublattices,
            rand_val,
            variable_indices,
        )

        test_val = get_variable_values_from_occupancy(
            solver_test_ensemble.sublattices, rand_occu, variable_indices
        ).astype(int)
        npt.assert_array_equal(rand_val, test_val)
        # print("Variable indices:\n", variable_indices)
        # print("rand_occu:\n", rand_occu)
        # print("rand_val:\n", rand_val)
        # assert False
