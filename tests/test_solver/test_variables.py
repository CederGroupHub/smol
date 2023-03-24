"""Test boolean variables' generation for upper-bound solver."""
import itertools

import numpy as np
import numpy.testing as npt

from smol.solver.upper_bound.variables import (
    get_occupancy_from_variables,
    get_upper_bound_variables_from_sublattices,
    get_variable_values_from_occupancy,
)

from .utils import get_random_variable_values


def test_upper_variables_and_indices(exotic_ensemble):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices, exotic_ensemble.num_sites
    )

    assert len(variable_indices) == exotic_ensemble.num_sites
    # Variable indices must be continuous.
    assert (
        variables.shape
        == max(
            (max(site_vars) if len(site_vars) > 0 else -1)
            for site_vars in variable_indices
        )
        + 1
    )
    assert list(range(variables.shape)) == sorted(
        set(itertools.chain(*variable_indices))
    )

    # Check for each site.
    for site_id, site_vars in enumerate(variable_indices):
        if len(site_vars) == 0:
            assert site_id in exotic_ensemble.restricted_sites
        else:
            sublattice = None
            for sl in exotic_ensemble.sublattices:
                if site_id in sl.sites:
                    sublattice = sl
            assert len(site_vars) == len(sublattice.encoding)
            assert list(range(min(site_vars), max(site_vars) + 1)) == site_vars


def test_occupancy_from_variables(exotic_ensemble, exotic_initial_occupancy):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices, exotic_ensemble.num_sites
    )

    num_sites = len(variable_indices)
    site_sublattice_ids = np.zeros(num_sites, dtype=int) - 1
    for sublattice_id, sublattice in enumerate(exotic_ensemble.sublattices):
        site_sublattice_ids[sublattice.sites] = sublattice_id

    for _ in range(20):
        rand_vals = get_random_variable_values(exotic_ensemble.sublattices)
        rand_occu = get_occupancy_from_variables(
            exotic_ensemble.sublattices,
            rand_vals,
            variable_indices,
        )
        # Constrained sites must be the same.
        npt.assert_array_equal(
            rand_occu[exotic_ensemble.restricted_sites],
            exotic_initial_occupancy[exotic_ensemble.restricted_sites],
        )
        # Check active sites match.
        for site_id in np.setdiff1d(
            np.arange(exotic_ensemble.num_sites, dtype=int),
            exotic_ensemble.restricted_sites,
        ):
            sublattice = exotic_ensemble.sublattices[site_sublattice_ids[site_id]]
            expected_vals = (sublattice.encoding == rand_occu[site_id]).astype(int)
            input_vals = rand_vals[variable_indices[site_id]].astype(int)
            npt.assert_array_equal(expected_vals, input_vals)


def test_variables_from_occupancy(exotic_ensemble, exotic_initial_occupancy):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices, exotic_ensemble.num_sites
    )
    # Check conversions are correct for random variables.
    for _ in range(20):
        rand_val = get_random_variable_values(exotic_ensemble.sublattices).astype(int)
        rand_occu = get_occupancy_from_variables(
            exotic_ensemble.sublattices,
            rand_val,
            variable_indices,
            exotic_initial_occupancy,
        )

        test_val = get_variable_values_from_occupancy(
            exotic_ensemble.sublattices, rand_occu, variable_indices
        ).astype(int)
        npt.assert_array_equal(rand_val, test_val)
