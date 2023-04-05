"""Test utility functions for upper-bound."""
import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.core import Species

from smol.cofe.space.domain import Vacancy, get_allowed_species
from smol.solver.upper_bound.utils.indices import (
    get_variable_indices_for_each_composition_component,
    map_ewald_indices_to_variable_indices,
)
from smol.solver.upper_bound.variables import get_upper_bound_variables_from_sublattices


def test_variable_indices_for_components(exotic_ensemble, exotic_initial_occupancy):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices, exotic_ensemble.num_sites
    )

    # When initial occupancy not given, expect an error.
    with pytest.raises(ValueError):
        _ = get_variable_indices_for_each_composition_component(
            exotic_ensemble.sublattices, variable_indices
        )

    var_inds_for_components = get_variable_indices_for_each_composition_component(
        exotic_ensemble.sublattices, variable_indices, exotic_initial_occupancy
    )
    # Total 9 species on all sub-lattices.
    assert len(var_inds_for_components) == 9
    dim_id = 0
    for sublattice in exotic_ensemble.sublattices:
        sub_bits = sublattice.species
        sl_active_variables = np.array(
            [
                variable_indices[i]
                for i in range(exotic_ensemble.num_sites)
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
                np.where(exotic_initial_occupancy[restricted_all_sites] == li_code)[0]
            ]
            restricted_vac_sites = np.setdiff1d(
                restricted_all_sites, restricted_li_sites
            )
            assert len(restricted_li_sites) == 3
            assert len(restricted_vac_sites) == 1
            # Only li sites restricted.
            npt.assert_array_equal(
                exotic_initial_occupancy[restricted_li_sites], li_code
            )
            npt.assert_array_equal(
                exotic_initial_occupancy[restricted_vac_sites], va_code
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


def test_ewald_indices(exotic_ensemble, exotic_initial_occupancy):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices, exotic_ensemble.num_sites
    )
    assert len(variable_indices) == exotic_ensemble.num_sites
    ew_processor = exotic_ensemble.processor.processors[-1]
    num_sites = len(variable_indices)
    site_sublattice_ids = np.zeros(num_sites, dtype=int) - 1
    for sublattice_id, sublattice in enumerate(exotic_ensemble.sublattices):
        site_sublattice_ids[sublattice.sites] = sublattice_id

    # When initial occupancy not given, expect an error.
    with pytest.raises(ValueError):
        _ = map_ewald_indices_to_variable_indices(
            exotic_ensemble.sublattices, ew_processor.structure, variable_indices
        )

    ew_to_var_id = map_ewald_indices_to_variable_indices(
        exotic_ensemble.sublattices,
        ew_processor.structure,
        variable_indices,
        exotic_initial_occupancy,
    )

    n_ew_rows = len(exotic_ensemble.processor.processors[-1]._ewald_structure)
    site_spaces = get_allowed_species(exotic_ensemble.processor.structure)
    assert len(ew_to_var_id) == n_ew_rows
    ew_id = 0
    var_id = 0

    restricted_vac_sites = exotic_ensemble.restricted_sites[
        np.where(exotic_initial_occupancy[exotic_ensemble.restricted_sites] == 5)[0]
    ]
    # print("supercell:\n", ew_processor.structure)
    # print("Initial occu:", exotic_initial_occupancy)
    # print("sub-lattices:", exotic_ensemble.sublattices)
    # print("site sub-lattice indices:", site_sublattice_ids)
    # print("variable indices:", variable_indices)
    # print("restricted_sites:", exotic_ensemble.restricted_sites)
    # print("restricted_vac_sites:", restricted_vac_sites)
    # print("ew_to_var_id:\n", ew_to_var_id)

    # This test is only applied to the exotic ensemble in conftest.
    expects = []
    for site_id, site_space in enumerate(site_spaces):
        sublattice_id = site_sublattice_ids[site_id]
        sublattice = exotic_ensemble.sublattices[sublattice_id]
        for sp_id, spec in enumerate(site_space):
            if isinstance(spec, Vacancy):
                # In the Li/Vac sub-lattice.
                if Species("Li", 1) in sublattice.species:
                    # Vacancy in an active site also occupies a variable.
                    # Vacancy in an inactive site does not.
                    if site_id in sublattice.active_sites:
                        var_id += 1
                continue
            if site_id in exotic_ensemble.restricted_sites:
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
