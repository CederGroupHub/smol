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
            restricted_li_sites = sublattice.restricted_sites
            assert len(restricted_li_sites) == 3
            # Only li sites restricted.
            npt.assert_array_equal(
                exotic_initial_occupancy[restricted_li_sites], li_code
            )
            for sp_id, species in enumerate(sub_bits):
                var_ids, n_fix = var_inds_for_components[dim_id]
                if species == Species("Li", 1):
                    assert n_fix == 3  # 3 li sites manually restricted
                elif species == Vacancy():
                    assert n_fix == 0
                else:
                    raise ValueError(
                        "Li/Vac sub-lattice was not correctly partitioned!"
                        f" Extra species {species}."
                    )
                # 6 unrestricted li sites + 5 unrestricted vac sites.
                assert len(var_ids) == 11
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
            assert sub_bits == [Species("F", -1)]
            var_ids, n_fix = var_inds_for_components[dim_id]
            assert n_fix == 10
            assert len(var_ids) == 0
            dim_id += 1


def test_ewald_indices(exotic_ensemble, exotic_initial_occupancy):
    variables, variable_indices = get_upper_bound_variables_from_sublattices(
        exotic_ensemble.sublattices, exotic_ensemble.num_sites
    )

    # When initial occupancy not given, expect an error.
    with pytest.raises(ValueError):
        _ = map_ewald_indices_to_variable_indices(
            exotic_ensemble.sublattices, variable_indices
        )

    ew_to_var_id = map_ewald_indices_to_variable_indices(
        exotic_ensemble.sublattices, variable_indices, exotic_initial_occupancy
    )

    n_ew_rows = len(exotic_ensemble.processor.processors[-1]._ewald_structure)
    site_spaces = get_allowed_species(exotic_ensemble.structure)
    assert len(ew_to_var_id) == n_ew_rows
    ew_id = 0
    var_id = 0

    # This test is only applied to the exotic ensemble in conftest.
    for site_id, site_space in enumerate(site_spaces):
        for sp_id, spec in enumerate(site_space):
            if isinstance(spec, Vacancy):
                continue
            if site_id in exotic_ensemble.restrict_sites:
                # Not inactive, just manually restricted.
                if spec != Species("F", -1):
                    # Always occupied by one species.
                    if spec == Species("O", -2) or spec == Species("Li", 1):
                        assert ew_to_var_id[ew_id] == -1
                    # Always occupied by other species than this one.
                    else:
                        assert ew_to_var_id[ew_id] == -2
                # Inactive sub-lattice.
                else:
                    assert ew_to_var_id == -1
            # Active site.
            else:
                assert ew_to_var_id[ew_id] == var_id
                var_id += 1

            ew_id += 1
