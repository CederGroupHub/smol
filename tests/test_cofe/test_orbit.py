from copy import deepcopy
from functools import reduce
from itertools import product
from random import choices

import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.core import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from smol.cofe.space import Cluster, Orbit
from smol.cofe.space.basis import basis_factory
from smol.cofe.space.domain import SiteSpace, get_site_spaces
from tests.utils import assert_msonable


@pytest.fixture(params=[(1, 2), (2, 8)])
def orbit(expansion_structure, rng, request):
    num_sites = rng.integers(*request.param)
    site_inds = choices(range(len(expansion_structure)), k=num_sites)
    coords = [expansion_structure.frac_coords[i] for i in site_inds]
    # add random integer multiples
    n = 0
    while n < request.param[0]:
        for coord in coords:
            coord += rng.integers(-4, 5)
        frac_coords, inds = np.unique(coords, axis=0, return_index=True)
        n = len(frac_coords)

    sg_analyzer = SpacegroupAnalyzer(expansion_structure)
    spaces = get_site_spaces([expansion_structure[i] for i in site_inds])
    spaces = [spaces[i] for i in inds]
    bases = [basis_factory("indicator", bit) for bit in spaces]
    orbit = Orbit(
        frac_coords,
        expansion_structure.lattice,
        [np.arange(len(space) - 1) for space in spaces],
        bases,
        sg_analyzer.get_symmetry_operations(),
    )
    orbit.assign_ids(1, 1, 1)
    return orbit


def test_constructor(expansion_structure):
    num_sites = len(expansion_structure) + 2
    site_inds = choices(range(len(expansion_structure)), k=num_sites)
    coords = [expansion_structure.frac_coords[i] for i in site_inds]
    sg_analyzer = SpacegroupAnalyzer(expansion_structure)
    spaces = get_site_spaces([expansion_structure[i] for i in site_inds])
    bases = [basis_factory("indicator", bit) for bit in spaces]
    with pytest.raises(AttributeError):
        Orbit(
            coords,
            expansion_structure.lattice,
            [np.arange(len(space) - 1) for space in spaces[:-1]],
            bases,
            sg_analyzer.get_symmetry_operations(),
        )
    with pytest.raises(AttributeError):
        Orbit(
            coords,
            expansion_structure.lattice,
            [np.arange(len(space) - 1) for space in spaces[:-1]],
            bases[:-1],
            sg_analyzer.get_symmetry_operations(),
        )


def test_cluster(orbit):
    base_cluster = Cluster(
        orbit.site_spaces, orbit.base_cluster.frac_coords, orbit.base_cluster.lattice
    )
    assert orbit.base_cluster == base_cluster
    assert base_cluster in orbit.clusters
    for cluster in orbit.clusters[1:]:
        assert base_cluster != cluster


def test_cluster_symops(orbit):
    assert orbit.multiplicity * len(orbit.cluster_symops) == len(orbit.structure_symops)

    for symop in orbit.cluster_symops:
        cluster = Cluster(
            orbit.site_spaces,
            symop.operate_multi(orbit.base_cluster.frac_coords),
            orbit.base_cluster.lattice,
        )
        assert cluster == orbit.base_cluster


def test_cluster_permutations(orbit):
    for permutation in orbit.cluster_permutations:
        cluster = Cluster(
            orbit.site_spaces,
            orbit.base_cluster.frac_coords[permutation],
            orbit.base_cluster.lattice,
        )
        assert cluster == orbit.base_cluster


def test_equality(orbit, rng):
    for _ in range(3):
        frac_coords = orbit.base_cluster.frac_coords.copy()
        other_coords = frac_coords.copy()
        other_coords[0] += rng.random()
        frac_coords += rng.integers(-4, 4)
        orbit1 = Orbit(
            frac_coords,
            orbit.base_cluster.lattice,
            [np.arange(len(space) - 1) for space in orbit.site_spaces],
            orbit.site_bases,
            orbit.structure_symops,
        )
        orbit2 = Orbit(
            other_coords,
            orbit.base_cluster.lattice,
            [np.arange(len(space) - 1) for space in orbit.site_spaces],
            orbit.site_bases,
            orbit.structure_symops,
        )

        assert orbit1 == orbit
        assert orbit2 != orbit
        if len(frac_coords) > 1:
            orbit3 = Orbit(
                frac_coords[:-1],
                orbit.base_cluster.lattice,
                [np.arange(len(space) - 1) for space in orbit.site_spaces[:-1]],
                orbit.site_bases[:-1],
                orbit.structure_symops,
            )
            assert orbit3 != orbit


def test_contains(orbit, rng):
    for cluster in orbit.clusters:
        assert cluster in orbit
    new_coords = orbit.base_cluster.frac_coords.copy()
    new_coords += 2 * rng.random(new_coords.shape) - 1
    cluster = Cluster(orbit.site_spaces, new_coords, orbit.base_cluster.lattice)
    assert cluster not in orbit


def test_bit_combos(orbit):
    if len(orbit.base_cluster) == 1:
        assert len(orbit.site_spaces) == 1
        nspecies = len(orbit.site_spaces[0])
        assert len(orbit.bit_combos) == nspecies - 1
        for i in range(nspecies - 1):
            assert orbit.bit_combos[i][0] == i
    else:
        nfunctions = reduce(
            lambda x, y: x * (y - 1), [len(space) for space in orbit.site_spaces], 1
        )
        if len(orbit.cluster_permutations) > 1:
            assert len(orbit) <= nfunctions
        else:
            assert len(orbit) == nfunctions

        assert all(
            combo[i] in np.arange(len(orbit.site_spaces[i]))
            for bit_combo in orbit.bit_combos
            for combo in bit_combo
            for i in range(len(orbit.base_cluster))
        )


def test_remove_bit_combos(orbit):
    nbits = len(orbit)
    with pytest.raises(ValueError):
        orbit.remove_bit_combos_by_inds([nbits + 1])

    if nbits > 1:
        bit_combo = orbit.bit_combos[0]
        orbit.remove_bit_combos_by_inds([0])
        assert not any(
            any(np.array_equal(bit_combo, b) for b in b_c) for b_c in orbit.bit_combos
        )
        assert len(orbit) == nbits - 1
        nbits -= 1

    # remove all bit combos
    with pytest.raises(RuntimeError):
        orbit.remove_bit_combos_by_inds(range(nbits))


def test_is_sub_orbit(expansion_structure, rng):
    num_sites = rng.integers(6, 10)
    site_inds = choices(range(len(expansion_structure)), k=num_sites)
    frac_coords = [expansion_structure.frac_coords[i] for i in site_inds]
    # add random integer multiples
    for coord in frac_coords:
        coord += rng.integers(-4, 5)
    frac_coords, inds = np.unique(frac_coords, axis=0, return_index=True)

    sg_analyzer = SpacegroupAnalyzer(expansion_structure)
    spaces = get_site_spaces([expansion_structure[i] for i in site_inds])
    spaces = [spaces[i] for i in inds]
    bases = [basis_factory("indicator", bit) for bit in spaces]
    orbit = Orbit(
        frac_coords[:-1],
        expansion_structure.lattice,
        [np.arange(len(space) - 1) for space in spaces[:-1]],
        bases[:-1],
        sg_analyzer.get_symmetry_operations(),
    )
    # with itself
    assert not orbit.is_sub_orbit(orbit)

    for _ in range(3):
        new_frac_coords = frac_coords.copy()
        new_frac_coords += rng.integers(-4, 5)

        # same orbit but shifted sites
        orbit1 = Orbit(
            new_frac_coords[:-1],
            orbit.base_cluster.lattice,
            [np.arange(len(space) - 1) for space in spaces[:-1]],
            bases[:-1],
            sg_analyzer.get_symmetry_operations(),
        )
        assert not orbit.is_sub_orbit(orbit1)

        if len(set(spaces)) > len(set(orbit.site_spaces)):
            # singles site orbit with site not in original
            orbit1 = Orbit(
                [new_frac_coords[-1]],
                orbit.base_cluster.lattice,
                [np.arange(len(spaces[-1]))],
                [bases[-1]],
                sg_analyzer.get_symmetry_operations(),
            )
            assert not orbit.is_sub_orbit(orbit1)

        # pair orbit with site not in original
        orbit1 = Orbit(
            [new_frac_coords[0], new_frac_coords[-1]],
            orbit.base_cluster.lattice,
            [np.arange(len(spaces[i])) for i in [0, -1]],
            [bases[0], bases[-1]],
            sg_analyzer.get_symmetry_operations(),
        )
        assert not orbit.is_sub_orbit(orbit1)

        # point suborbit
        i = rng.choice(range(len(orbit.base_cluster.frac_coords)))
        orbit1 = Orbit(
            [new_frac_coords[i]],
            orbit.base_cluster.lattice,
            [np.arange(len(spaces[i]))],
            [bases[i]],
            orbit.structure_symops,
        )
        assert orbit.is_sub_orbit(orbit1)

        # larger suborbit
        orbit1 = Orbit(
            new_frac_coords[:-2],
            orbit.base_cluster.lattice,
            [np.arange(len(space) - 1) for space in spaces[:-2]],
            bases[:-2],
            sg_analyzer.get_symmetry_operations(),
        )
        assert orbit.is_sub_orbit(orbit1)

        # shifted site
        new_frac_coords[i] += rng.random()
        orbit1 = Orbit(
            [new_frac_coords[i]],
            orbit.base_cluster.lattice,
            [np.arange(len(spaces[i]))],
            [bases[i]],
            orbit.structure_symops,
        )
        assert not orbit.is_sub_orbit(orbit1)


def test_sub_orbit_mappings(orbit, rng):
    frac_coords = orbit.base_cluster.frac_coords.copy()
    frac_coords[0] += rng.random()
    orbit1 = Orbit(
        frac_coords,
        orbit.base_cluster.lattice,
        [np.arange(len(sp)) for sp in orbit.site_spaces],
        orbit.site_bases,
        orbit.structure_symops,
    )
    assert len(orbit.sub_orbit_mappings(orbit1)) == 0

    nsites = len(orbit.base_cluster.frac_coords)
    # choose all but one for orbits with 2 or more sites
    size = nsites if nsites == 1 else nsites - 1
    for _ in range(3):
        inds = rng.choice(range(nsites), size=size, replace=False)
        orbit1 = Orbit(
            orbit.base_cluster.frac_coords[inds],
            orbit.base_cluster.lattice,
            [np.arange(len(orbit.site_spaces[i])) for i in inds],
            [orbit.site_bases[i] for i in inds],
            orbit.structure_symops,
        )
        mappings = orbit.sub_orbit_mappings(orbit1)
        cluster = Cluster(
            orbit.site_spaces,
            orbit.base_cluster.frac_coords[mappings][0],
            orbit.base_cluster.lattice,
        )
        assert cluster in orbit1.clusters

    for i in range(nsites):
        orbit1 = Orbit(
            [orbit.base_cluster.frac_coords[i]],
            orbit.base_cluster.lattice,
            [np.arange(len(orbit.site_spaces[i]))],
            [orbit.site_bases[i]],
            orbit.structure_symops,
        )
        mappings = orbit.sub_orbit_mappings(orbit1)
        cluster = Cluster(
            orbit.site_spaces,
            orbit.base_cluster.frac_coords[mappings][0],
            orbit.base_cluster.lattice,
        )
        assert cluster in orbit1.clusters


def test_basis_orthonormal(orbit):
    assert not orbit.basis_orthogonal
    assert not orbit.basis_orthonormal
    bases = deepcopy(orbit.site_bases)
    for b in bases:
        b.orthonormalize()
        assert b.is_orthonormal
    orbit1 = Orbit(
        orbit.base_cluster.frac_coords,
        orbit.base_cluster.lattice,
        orbit.bits,
        bases,
        orbit.structure_symops,
    )
    assert orbit1.basis_orthogonal
    assert orbit1.basis_orthonormal


def _test_basis_arrs(bases, orbit):
    for occu in product(*[range(arr.shape[1]) for arr in orbit.basis_arrays]):
        for bit_combo in orbit.bit_combos:
            for combo in bit_combo:
                value1 = np.prod(
                    [
                        bases[i].function_array[b, s]
                        for i, (b, s) in enumerate(zip(combo, occu))
                    ]
                )
                value2 = np.prod(
                    [
                        orbit.basis_arrays[i][b, s]
                        for i, (b, s) in enumerate(zip(combo, occu))
                    ]
                )
            assert value1 == value2


def test_basis_array(orbit):
    bases = [
        basis_factory(basis.flavor, basis.site_space) for basis in orbit.site_bases
    ]
    _test_basis_arrs(bases, orbit)


def test_transform_basis(orbit, basis_name):
    orbit.transform_site_bases(basis_name)
    bases = [
        basis_factory(basis.flavor, basis.site_space) for basis in orbit.site_bases
    ]
    _test_basis_arrs(bases, orbit)

    orbit.transform_site_bases(basis_name, orthonormal=True)
    bases = [
        basis_factory(basis.flavor, basis.site_space) for basis in orbit.site_bases
    ]
    for b in bases:
        b.orthonormalize()
    assert orbit.basis_orthonormal
    _test_basis_arrs(bases, orbit)


def test_msonable(orbit, rng):
    _ = repr(orbit), str(orbit)
    assert_msonable(orbit)
    orbit1 = Orbit.from_dict(orbit.as_dict())

    # test remove bit combos are properly reconstructed
    if len(orbit.bit_combos) - 1 > 0:
        orbit1.remove_bit_combos_by_inds(
            rng.integers(len(orbit1.bit_combos), size=len(orbit.bit_combos) - 1)
        )
        orbit2 = Orbit.from_dict(orbit1.as_dict())
        assert all(
            all(np.array_equal(b1, b2) for b1, b2 in zip(c1, c2))
            for c1, c2 in zip(orbit1.bit_combos, orbit2.bit_combos)
        )


# hard-coded specific tests for LiCaBr
@pytest.fixture(scope="module")
def licabr_orbit(single_structure):
    sf = SpacegroupAnalyzer(single_structure)
    symops = sf.get_symmetry_operations()
    spaces = [
        SiteSpace(Composition({"Li": 1.0 / 3.0, "Ca": 1.0 / 3.0})),
        SiteSpace(Composition({"Li": 1.0 / 3.0, "Ca": 1.0 / 3.0})),
        SiteSpace(Composition({"Li": 1.0 / 3.0, "Ca": 1.0 / 3.0})),
    ]
    bases = [basis_factory("indicator", bit) for bit in spaces]
    orbit = Orbit(
        single_structure.frac_coords[:3],
        single_structure.lattice,
        [[0, 1], [0, 1], [0, 1]],
        bases,
        symops,
    )
    orbit.assign_ids(1, 1, 1)
    return orbit


def test_fixed_values(licabr_orbit):
    assert licabr_orbit.multiplicity == 4
    assert len(licabr_orbit.cluster_symops) == 12


def test_equal_fixed(licabr_orbit):
    orbit1 = Orbit(
        licabr_orbit.base_cluster.frac_coords[:3],
        licabr_orbit.base_cluster.lattice,
        [[0, 1], [0, 1], [0, 1]],
        licabr_orbit.site_bases,
        licabr_orbit.structure_symops,
    )
    orbit2 = Orbit(
        licabr_orbit.base_cluster.frac_coords[:2],
        licabr_orbit.base_cluster.lattice,
        [[0, 1], [0, 1]],
        licabr_orbit.site_bases[:2],
        licabr_orbit.structure_symops,
    )
    assert orbit1 == licabr_orbit
    assert orbit2 != licabr_orbit


def test_bit_combos_fixed(licabr_orbit):
    # orbit with 3 sites and two symmetrically equivalent sites
    assert len(licabr_orbit) == 6

    orbit1 = Orbit(
        licabr_orbit.base_cluster.frac_coords[1:3],
        licabr_orbit.base_cluster.lattice,
        [[0, 1], [0, 1]],
        licabr_orbit.site_bases[:2],
        licabr_orbit.structure_symops,
    )
    # orbit with only two symmetrically distinct sites
    assert len(orbit1) == 4


def test_suborbits_fixed(licabr_orbit):
    # same orbit
    orbit = Orbit(
        licabr_orbit.base_cluster.frac_coords[:3],
        licabr_orbit.base_cluster.lattice,
        [[0, 1], [0, 1], [0, 1]],
        licabr_orbit.site_bases,
        licabr_orbit.structure_symops,
    )
    assert not licabr_orbit.is_sub_orbit(orbit)
    # point orbits with site in original
    orbit = Orbit(
        [licabr_orbit.base_cluster.frac_coords[0]],
        licabr_orbit.base_cluster.lattice,
        [[0, 1]],
        [licabr_orbit.site_bases[0]],
        licabr_orbit.structure_symops,
    )
    assert licabr_orbit.is_sub_orbit(orbit)
    orbit = Orbit(
        [licabr_orbit.base_cluster.frac_coords[1]],
        licabr_orbit.base_cluster.lattice,
        [[0, 1]],
        [licabr_orbit.site_bases[1]],
        licabr_orbit.structure_symops,
    )
    assert licabr_orbit.is_sub_orbit(orbit)
    # pair suborbit
    orbit = Orbit(
        licabr_orbit.base_cluster.frac_coords[:2],
        licabr_orbit.base_cluster.lattice,
        [[0, 1], [0, 1]],
        licabr_orbit.site_bases[:2],
        licabr_orbit.structure_symops,
    )
    assert licabr_orbit.is_sub_orbit(orbit)
    # point not in original
    orbit = Orbit(
        [[0, 0, 0]],
        licabr_orbit.base_cluster.lattice,
        [[0, 1]],
        [licabr_orbit.site_bases[0]],
        licabr_orbit.structure_symops,
    )
    assert not licabr_orbit.is_sub_orbit(orbit)
    # pair orbit with site not in original
    orbit = Orbit(
        [licabr_orbit.base_cluster.frac_coords[0], [0, 0, 0]],
        licabr_orbit.base_cluster.lattice,
        [[0, 1], [0, 1]],
        licabr_orbit.site_bases[:2],
        licabr_orbit.structure_symops,
    )
    assert not licabr_orbit.is_sub_orbit(orbit)


def test_suborbit_mappings_fixed(licabr_orbit):
    orbit = Orbit(
        np.vstack([licabr_orbit.base_cluster.frac_coords[1:], [0, 0, 0]]),
        licabr_orbit.base_cluster.lattice,
        [[0, 1], [0, 1], [0, 1]],
        licabr_orbit.site_bases,
        licabr_orbit.structure_symops,
    )
    # zero mappings since not suborbit
    assert len(licabr_orbit.sub_orbit_mappings(orbit)) == 0

    orbit = Orbit(
        licabr_orbit.base_cluster.frac_coords[:2],
        licabr_orbit.base_cluster.lattice,
        [[0, 1], [0, 1]],
        licabr_orbit.site_bases[:2],
        licabr_orbit.structure_symops,
    )
    npt.assert_array_equal(licabr_orbit.sub_orbit_mappings(orbit), [[0, 1]])

    orbit = Orbit(
        [licabr_orbit.base_cluster.frac_coords[2]],
        licabr_orbit.base_cluster.lattice,
        [[0, 1]],
        [licabr_orbit.site_bases[2]],
        licabr_orbit.structure_symops,
    )
    npt.assert_array_equal(licabr_orbit.sub_orbit_mappings(orbit), [[2]])
