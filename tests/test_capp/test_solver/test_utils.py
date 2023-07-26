"""Test utility functions for upper-bound."""
from itertools import product

import cvxpy as cp
import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.core import Species

from smol.capp.generate.groundstate.upper_bound.indices import (
    get_sublattice_indices_by_site,
    get_variable_indices_for_each_composition_component,
    map_ewald_indices_to_variable_indices,
)
from smol.capp.generate.groundstate.upper_bound.terms import (
    get_auxiliary_variable_values,
    get_expression_and_auxiliary_from_terms,
)
from smol.capp.generate.groundstate.upper_bound.variables import (
    get_variables_from_sublattices,
)
from smol.cofe.space.domain import Vacancy, get_allowed_species


def test_variable_indices_for_components(
    solver_test_ensemble, solver_test_initial_occupancy
):
    _, variable_indices = get_variables_from_sublattices(
        solver_test_ensemble.sublattices,
        solver_test_ensemble.processor.structure,
        solver_test_initial_occupancy,
    )

    var_inds_for_components = get_variable_indices_for_each_composition_component(
        solver_test_ensemble.sublattices,
        variable_indices,
        solver_test_ensemble.processor.structure,
    )
    # Total 9 species on all sub-lattices.
    assert len(var_inds_for_components) == 9
    dim_id = 0
    for sublattice in solver_test_ensemble.sublattices:
        sub_bits = sublattice.species
        sl_active_variables = np.array(
            [
                [v for v in variable_indices[i] if v >= 0]
                for i in range(solver_test_ensemble.num_sites)
                if i in sublattice.active_sites
            ],
            dtype=int,
        )
        if Species("Li", 1) in sub_bits:
            # 2 li sites are restricted.
            li_code = sublattice.encoding[sub_bits.index(Species("Li", 1))]
            va_code = sublattice.encoding[sub_bits.index(Vacancy())]
            restricted_all_sites = sublattice.restricted_sites
            assert len(restricted_all_sites) == 4
            restricted_li_sites = restricted_all_sites[
                np.where(
                    solver_test_initial_occupancy[restricted_all_sites] == li_code
                )[0]
            ]
            restricted_vac_sites = np.setdiff1d(
                restricted_all_sites, restricted_li_sites
            )
            assert len(restricted_li_sites) == 3
            assert len(restricted_vac_sites) == 1
            # Only li sites restricted.
            npt.assert_array_equal(
                solver_test_initial_occupancy[restricted_li_sites], li_code
            )
            npt.assert_array_equal(
                solver_test_initial_occupancy[restricted_vac_sites], va_code
            )
            for sp_id, species in enumerate(sub_bits):
                var_ids, n_fix = var_inds_for_components[dim_id]
                if species == Species("Li", 1):
                    assert n_fix == 3  # 3 li sites manually restricted
                elif species == Vacancy():
                    assert n_fix == 1
                else:
                    raise ValueError(
                        "Li/Vac sub-lattice was not correctly partitioned!"
                        f" Extra species {species}."
                    )
                # 6 unrestricted li sites + 5 unrestricted vac sites.
                assert len(var_ids) == 10
                # Check indices are correct.
                npt.assert_array_equal(sl_active_variables[:, sp_id], var_ids)
                dim_id += 1
        elif Species("Mn", 2) in sub_bits:
            # No restricted site, all 6 sites should be active.
            for sp_id in range(len(sub_bits)):
                var_ids, n_fix = var_inds_for_components[dim_id]
                assert n_fix == 0
                assert len(var_ids) == 6
                npt.assert_array_equal(sl_active_variables[:, sp_id], var_ids)
                dim_id += 1
        elif Species("O", -2) in sub_bits:
            for sp_id, species in enumerate(sub_bits):
                var_ids, n_fix = var_inds_for_components[dim_id]
                if species == Species("O", -2):
                    # 2 restricted o2- sites.
                    assert n_fix == 2
                else:
                    assert n_fix == 0
                # 6 unrestricted o2- sites, 2 unrestricted o- sites.
                assert len(var_ids) == 8
                npt.assert_array_equal(sl_active_variables[:, sp_id], var_ids)
                dim_id += 1
        else:  # F sub-lattice totally inactive.
            assert list(sub_bits) == [Species("F", -1)]
            var_ids, n_fix = var_inds_for_components[dim_id]
            assert n_fix == 10
            assert len(var_ids) == 0
            dim_id += 1


def test_ewald_indices(solver_test_ensemble, solver_test_initial_occupancy):
    _, variable_indices = get_variables_from_sublattices(
        solver_test_ensemble.sublattices,
        solver_test_ensemble.processor.structure,
        solver_test_initial_occupancy,
    )
    assert len(variable_indices) == solver_test_ensemble.num_sites
    ew_processor = solver_test_ensemble.processor.processors[-1]
    site_sublattice_ids = get_sublattice_indices_by_site(
        solver_test_ensemble.sublattices
    )

    ew_to_var_id = map_ewald_indices_to_variable_indices(
        ew_processor.structure,
        variable_indices,
    )

    n_ew_rows = len(solver_test_ensemble.processor.processors[-1]._ewald_structure)
    site_spaces = get_allowed_species(solver_test_ensemble.processor.structure)
    assert len(ew_to_var_id) == n_ew_rows
    ew_id = 0
    var_id = 0

    restricted_vac_sites = solver_test_ensemble.restricted_sites[
        np.where(
            solver_test_initial_occupancy[solver_test_ensemble.restricted_sites] == 5
        )[0]
    ]
    # print("supercell:\n", ew_processor.structure)
    # print("Initial occu:", solver_test_initial_occupancy)
    # print("sub-lattices:", solver_test_ensemble.sublattices)
    # print("site sub-lattice indices:", site_sublattice_ids)
    # print("variable indices:", variable_indices)
    # print("restricted_sites:", solver_test_ensemble.restricted_sites)
    # print("restricted_vac_sites:", restricted_vac_sites)
    # print("ew_to_var_id:\n", ew_to_var_id)

    # This test is only applied to the solver_test ensemble in conftest.
    expects = []
    for site_id, site_space in enumerate(site_spaces):
        sublattice_id = site_sublattice_ids[site_id]
        sublattice = solver_test_ensemble.sublattices[sublattice_id]
        for spec in site_space:
            if isinstance(spec, Vacancy):
                # In the Li/Vac sub-lattice.
                if Species("Li", 1) in sublattice.species:
                    # Vacancy in an active site also occupies a variable.
                    # Vacancy in an inactive site does not.
                    if site_id in sublattice.active_sites:
                        var_id += 1
                continue
            if site_id in solver_test_ensemble.restricted_sites:
                # Not the inactive F sub-lattice, just manually restricted.
                if Species("F", -1) not in sublattice.species:
                    # Always occupied by one non-vacancy species.
                    if spec == Species("O", -2) or (
                        spec == Species("Li", 1) and site_id not in restricted_vac_sites
                    ):
                        expected = -1
                    # Always occupied by other species than this one.
                    else:
                        expected = -2
                # Inactive F sub-lattice.
                else:
                    if spec == Species("F", -1):
                        expected = -1
                    else:
                        assert spec.symbol == "O"
                        expected = -2
            # Active site.
            else:
                # In the new sub-lattice.
                if spec in sublattice.species:
                    expected = var_id
                    var_id += 1
                # Not in the new sub-lattice.
                else:
                    expected = -2

            ew_id += 1
            expects.append(expected)

    assert ew_id == n_ew_rows
    # print("expected:\n", expects)
    npt.assert_array_equal(expects, ew_to_var_id)


def test_expression_from_terms():
    x = cp.Variable(2, boolean=True)
    # Empty expression.
    terms = []
    with pytest.raises(RuntimeError):
        _ = get_expression_and_auxiliary_from_terms(terms, x)
    # Expression with only constant.
    terms = [([], -1, 1.0), ([], 100, 1.0)]
    with pytest.raises(RuntimeError):
        _ = get_expression_and_auxiliary_from_terms(terms, x)
    # Correct expression, which is to be evaluated.
    # A very simple test case.
    # 0 + 2 x0 -3 x1 + x0 * x1.
    terms = [
        ([], -1, 1),
        ([], 1, 1),
        ([0], 2, 1),
        ([1], -3, 1),
        ([0, 1], -1, 1),
        ([1, 0], 2, 1),
    ]
    func, y, indices, aux_cons = get_expression_and_auxiliary_from_terms(terms, x)
    assert y.size == 1
    assert len(indices) == 1
    assert list(indices[0]) == [0, 1]
    assert len(aux_cons) == 3
    assert isinstance(func, cp.Expression)
    for val0, val1 in product(*[[0, 1], [0, 1]]):
        true_val = 2 * val0 - 3 * val1 + val0 * val1
        x.value = np.array([val0, val1], dtype=int)
        y.value = [val0 * val1]
        assert func.value == true_val
        # Auxiliary constraints should always be satisfied.
        for con in aux_cons:
            assert con.value()

    for rand_vals in product(*[[0, 1], [0, 1]]):
        rand_vals = np.array(rand_vals)
        rand_aux_vals = get_auxiliary_variable_values(rand_vals, indices)
        for ii, inds in enumerate(indices):
            assert len(inds) > 1
            assert np.product(rand_vals[inds]) == rand_aux_vals[ii]
