import itertools
import re
from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.core import Composition, Species

from smol.cofe.space.domain import Vacancy
from smol.moca.composition import CompositionSpace
from smol.moca.occu_utils import get_dim_ids_by_sublattice
from smol.utils.math import NUM_TOL, compute_snf, integerize_vector
from tests.utils import assert_msonable, assert_pickles, assert_table_set_equal


@pytest.fixture(scope="module")
def comp_space(ensemble):
    bits = [s.species for s in ensemble.sublattices]
    sublattice_sizes = np.array([len(s.sites) for s in ensemble.sublattices])
    gcd = np.gcd.reduce(sublattice_sizes)
    sublattice_sizes = sublattice_sizes // gcd
    return CompositionSpace(bits, sublattice_sizes)  # Charge balanced = True, default.


@pytest.fixture(scope="module")
def comp_space_lmtpo():
    li = Species.from_string("Li+")
    mn = Species.from_string("Mn3+")
    ti = Species.from_string("Ti4+")
    o = Species.from_string("O2-")
    p = Species.from_string("P3-")

    bits = [[li, mn, ti], [p, o]]
    sublattice_sizes = [1, 1]
    return CompositionSpace(
        bits,
        sublattice_sizes,
        charge_balanced=True,
        optimize_basis=True,
        table_ergodic=True,
    )


@pytest.fixture(scope="module")
def comp_space_lmtpo2():
    li = Species.from_string("Li+")
    mn = Species.from_string("Mn3+")
    ti = Species.from_string("Ti4+")
    o = Species.from_string("O2-")
    p = Species.from_string("P3-")

    bits = [[li, mn, ti], [p, o]]
    sublattice_sizes = [1, 1]
    return CompositionSpace(
        bits,
        sublattice_sizes,
        charge_balanced=True,
        optimize_basis=True,
        table_ergodic=True,
        other_constraints=[([2, 1, 0, 0, 0], 7 / 6)],
    )


@pytest.fixture(scope="module")
def comp_space_lmtpo3():
    li = Species.from_string("Li+")
    mn = Species.from_string("Mn3+")
    ti = Species.from_string("Ti4+")
    o = Species.from_string("O2-")
    p = Species.from_string("P3-")

    bits = [[li, mn, ti], [p, o]]
    sublattice_sizes = [1, 1]
    return CompositionSpace(
        bits,
        sublattice_sizes,
        charge_balanced=True,
        optimize_basis=True,
        table_ergodic=True,
        leq_constraints=[
            ([0, 1, 0, 0, 0], 5 / 6),
            ([0, 0, 0, 1, 0], 5 / 6),
            ([2, 1, 0, 0, 0], 8 / 6),
        ],
        geq_constraints=[
            ([0, 1, 0, 0, 0], 1 / 6),
            ([0, 0, 0, 1, 0], 1 / 6),
            ([2, 1, 0, 0, 0], 5 / 6),
        ],
    )


# A test example with extra constraints.
@pytest.fixture(scope="module")
def comp_space_lmntof():
    li = Species.from_string("Li+")
    ni = Species.from_string("Ni2+")
    mn = Species.from_string("Mn3+")
    ti = Species.from_string("Ti4+")
    f = Species.from_string("F-")
    o = Species.from_string("O2-")

    bits = [[li, ni, mn, ti], [o, f]]
    sublattice_sizes = [1, 1]
    other_constraints = [([0, 1, -1, 0, 0, 0], 0), ([0, 0, 1, -1, 0, 0], 0)]
    return CompositionSpace(
        bits,
        sublattice_sizes,
        other_constraints=other_constraints,
        charge_balanced=True,
        optimize_basis=True,
        table_ergodic=True,
    )


# Generic attributes test.
def test_generic_attributes(comp_space):
    dim_ids = comp_space.dim_ids
    assert dim_ids == get_dim_ids_by_sublattice(comp_space.bits)
    species_set = set(itertools.chain(*comp_space.bits))
    vac_count = 0
    for sp in comp_space.species:
        assert sp in species_set
        if isinstance(sp, Vacancy):
            vac_count += 1
    assert vac_count <= 1
    assert species_set & set(comp_space.species) == set(comp_space.species)
    assert sorted(comp_space.species) == comp_space.species

    for sl_id, dim_ids_sl in enumerate(comp_space.species_ids):
        for sp_id, d in enumerate(dim_ids_sl):
            bit = comp_space.bits[sl_id][sp_id]
            sp = comp_space.species[d]
            if not isinstance(sp, Vacancy):
                assert bit == sp
            else:
                assert isinstance(bit, Vacancy)

    A = comp_space._A
    b = comp_space._b
    assert len(A) == 1 + len(comp_space.bits)  # No other constraints.
    assert len(b) == len(A)
    assert b[0] == 0
    assert np.all(A[1:, :] <= 1)
    npt.assert_array_equal(b[1:], comp_space.sublattice_sizes)
    min_sc = comp_space.min_supercell_size

    prim_verts = comp_space.prim_vertices
    assert prim_verts.shape[1] == A.shape[1]
    npt.assert_almost_equal(A @ prim_verts.T - b[:, None], 0, decimal=6)
    assert np.all(np.any(np.isclose(prim_verts, 0, atol=NUM_TOL), axis=-1))

    if comp_space.num_unconstrained_compositions < 10**6:
        scs = [1, min_sc, min_sc * 2] + ([min_sc // 2] if min_sc >= 2 else [])
        for sc in scs:
            if not np.allclose(sc * comp_space._b, np.round(sc * comp_space._b)):
                with pytest.raises(ValueError):
                    grid = comp_space.get_composition_grid(sc, step=1)
            else:
                grid = comp_space.get_composition_grid(sc, step=1)
            ns = grid @ comp_space.basis + comp_space.get_supercell_base_solution(sc)
            npt.assert_array_equal(grid, comp_space._comp_grids[(sc, 1)])
            # x-format
            npt.assert_array_equal(A @ ns.T - b[:, None] * sc, 0)
            assert np.all(ns >= 0)

    # Flip table not optimized.
    npt.assert_array_equal(comp_space.flip_table, comp_space.basis)

    reactions = comp_space.flip_reactions
    flips = []
    bits_str = [[str(sp) for sp in sl_sps] for sl_sps in comp_space.bits]
    for r in reactions:
        sps = []
        sl_ids = []
        nums = []
        n_from = np.inf
        tub_id = 0
        for t in r.split():
            if t not in ["->", "+"]:
                if t[-1] == ")":
                    sp_str, sl_id = re.search(r"(.+)\((\d+)\)", t).groups()
                    sps.append(sp_str)
                    sl_ids.append(int(sl_id))
                    tub_id += 1
                else:
                    if tub_id < n_from:
                        nums.append(-int(t))
                    else:
                        nums.append(int(t))
            elif t == "->":
                n_from = tub_id
        flip = np.zeros(comp_space.n_dims, dtype=int)

        for sl_id, sp_str, n in zip(sl_ids, sps, nums):
            sp_id = bits_str[sl_id].index(sp_str)
            ii = comp_space.dim_ids[sl_id][sp_id]
            flip[ii] += n
        flips.append(flip)
    flips = np.array(flips, dtype=int)
    npt.assert_array_equal(flips, comp_space.flip_table)


def test_serialize(comp_space, comp_space_lmtpo3, comp_space_lmntof):
    _ = comp_space.flip_table
    if comp_space.num_unconstrained_compositions < 10**6:
        _ = comp_space.min_supercell_grid
    assert_msonable(comp_space)
    assert_msonable(comp_space_lmtpo3)
    assert_msonable(comp_space_lmntof)

    assert_pickles(comp_space)
    assert_pickles(comp_space_lmtpo3)
    assert_pickles(comp_space_lmntof)

    comp_space_reload = CompositionSpace.from_dict(comp_space.as_dict())
    npt.assert_array_equal(comp_space_reload._flip_table, comp_space._flip_table)
    npt.assert_array_equal(comp_space_reload._vs, comp_space._vs)

    npt.assert_array_equal(comp_space_reload._n0, comp_space._n0)

    npt.assert_array_equal(
        comp_space_reload._min_supercell_size, comp_space._min_supercell_size
    )
    assert set(list(comp_space._comp_grids.keys())) == set(
        list(comp_space_reload._comp_grids.keys())
    )
    for k in comp_space._comp_grids.keys():
        npt.assert_array_equal(
            comp_space._comp_grids[k], comp_space_reload._comp_grids[k]
        )


# Pre-computed data test.
def test_supercell_size(
    comp_space_lmtpo, comp_space_lmtpo2, comp_space_lmtpo3, comp_space_lmntof
):
    assert comp_space_lmtpo.min_supercell_size == 6
    assert comp_space_lmtpo2.min_supercell_size == 12
    assert comp_space_lmtpo3.min_supercell_size == 6
    assert comp_space_lmntof.min_supercell_size == 6


def test_n_comps_unconstrained(comp_space_lmtpo, comp_space_lmntof):
    assert comp_space_lmtpo.num_unconstrained_compositions == 7776
    assert comp_space_lmntof.num_unconstrained_compositions == 46656


def test_basis(
    comp_space_lmtpo, comp_space_lmtpo2, comp_space_lmtpo3, comp_space_lmntof
):
    vs1 = comp_space_lmtpo.basis
    vs2 = comp_space_lmntof.basis
    vs3 = comp_space_lmtpo2.basis
    vs4 = comp_space_lmtpo3.basis
    ts1 = comp_space_lmtpo.flip_table
    ts2 = comp_space_lmntof.flip_table
    ts3 = comp_space_lmtpo2.flip_table
    ts4 = comp_space_lmtpo3.flip_table
    # Pre-computed optimal basis
    std1 = np.array([[0, -1, 1, 1, -1], [-1, 1, 0, 2, -2]], dtype=int)
    std2 = np.array([[-3, 1, 1, 1, 6, -6]], dtype=int)
    std3 = np.array([[-1, 2, -1, 1, -1]], dtype=int)

    assert_table_set_equal(vs1, std1)
    assert_table_set_equal(vs2, std2)
    assert_table_set_equal(ts1, std1)
    assert_table_set_equal(ts2, std2)
    assert_table_set_equal(vs3, std3)
    assert_table_set_equal(vs4, std1)
    assert_table_set_equal(ts3, std3)
    assert_table_set_equal(ts4, std1)


def test_get_supercell_base_solution(comp_space):
    comp_space._n0 = None  # Clear up.
    min_supercell_size = comp_space.min_supercell_size
    n00 = comp_space.get_supercell_base_solution(supercell_size=min_supercell_size * 2)
    _, min_feasible_size = integerize_vector(comp_space._b)
    assert min_supercell_size % min_feasible_size == 0
    scale = min_supercell_size * 2 // min_feasible_size
    n01 = comp_space.get_supercell_base_solution(supercell_size=min_feasible_size)
    npt.assert_array_equal(n01 * scale, n00)


def test_enumerate_grid(comp_space_lmtpo, comp_space_lmtpo2, comp_space_lmtpo3):
    # Convert x-format to n-format
    grid2 = (
        comp_space_lmtpo2.min_supercell_grid @ comp_space_lmtpo2.basis
        + comp_space_lmtpo2.get_supercell_base_solution()
    )
    grid3 = (
        comp_space_lmtpo3.min_supercell_grid @ comp_space_lmtpo3.basis
        + comp_space_lmtpo3.get_supercell_base_solution()
    )
    std2 = np.array(
        [
            [2, 10, 0, 8, 4],
            [3, 8, 1, 7, 5],
            [4, 6, 2, 6, 6],
            [5, 4, 3, 5, 7],
            [6, 2, 4, 4, 8],
            [7, 0, 5, 3, 9],
        ],
        dtype=int,
    )
    std3 = np.array(
        [
            [3, 2, 1, 1, 5],
            [3, 1, 2, 2, 4],
            [2, 4, 0, 2, 4],
            [2, 3, 1, 3, 3],
            [2, 2, 2, 4, 2],
            [1, 5, 0, 4, 2],
            [2, 1, 3, 5, 1],
            [1, 4, 1, 5, 1],
        ],
        dtype=int,
    )
    grid2 = np.array(sorted(grid2.tolist()), dtype=int)
    grid3 = np.array(sorted(grid3.tolist()), dtype=int)
    std2 = np.array(sorted(std2.tolist()), dtype=int)
    std3 = np.array(sorted(std3.tolist()), dtype=int)
    npt.assert_array_equal(grid2, std2)
    npt.assert_array_equal(grid3, std3)

    grid = comp_space_lmtpo.get_composition_grid(
        supercell_size=4
    ) @ comp_space_lmtpo.basis + comp_space_lmtpo.get_supercell_base_solution(
        supercell_size=4
    )
    std = np.array(
        [
            [0, 4, 0, 4, 0],
            [1, 2, 1, 3, 1],
            [2, 0, 2, 2, 2],
            [1, 3, 0, 2, 2],
            [1, 1, 2, 4, 0],
            [2, 1, 1, 1, 3],
            [2, 2, 0, 0, 4],
        ],
        dtype=int,
    )
    grid = np.array(sorted(grid.tolist()), dtype=int)
    std = np.array(sorted(std.tolist()), dtype=int)
    npt.assert_array_equal(grid, std)

    # Scalability guaranteed in comp_space, which means when supercell_size / step
    # is the same value, the enumerated grid will always be original_grid
    # * step.
    min_supercell_size = comp_space_lmtpo.min_supercell_size
    grid = comp_space_lmtpo.get_composition_grid(
        supercell_size=2 * min_supercell_size, step=2
    )
    std = comp_space_lmtpo.min_supercell_grid * 2
    grid = np.array(sorted(grid.tolist()), dtype=int)
    std = np.array(sorted(std.tolist()), dtype=int)
    npt.assert_array_equal(grid, std)

    grid = comp_space_lmtpo.get_composition_grid(supercell_size=8, step=2)
    std = comp_space_lmtpo.get_composition_grid(supercell_size=4, step=1) * 2
    std2 = comp_space_lmtpo.get_composition_grid(supercell_size=4, step=1) * 2
    grid = np.array(sorted(grid.tolist()), dtype=int)
    std = np.array(sorted(std.tolist()), dtype=int)
    std2 = np.array(sorted(std2.tolist()), dtype=int)
    npt.assert_array_equal(grid, std)
    npt.assert_array_equal(grid, std2)

    grid = comp_space_lmtpo2.get_composition_grid(supercell_size=12, step=2)
    std = comp_space_lmtpo2.get_composition_grid(supercell_size=6, step=1) * 2
    std2 = comp_space_lmtpo2.get_composition_grid(supercell_size=6, step=1) * 2
    grid = np.array(sorted(grid.tolist()), dtype=int)
    std = np.array(sorted(std.tolist()), dtype=int)
    std2 = np.array(sorted(std2.tolist()), dtype=int)
    npt.assert_array_equal(grid, std)
    npt.assert_array_equal(grid, std2)

    grid = comp_space_lmtpo3.get_composition_grid(supercell_size=8, step=2)
    std = comp_space_lmtpo3.get_composition_grid(supercell_size=4, step=1) * 2
    std2 = comp_space_lmtpo3.get_composition_grid(supercell_size=4, step=1) * 2
    grid = np.array(sorted(grid.tolist()), dtype=int)
    std = np.array(sorted(std.tolist()), dtype=int)
    std2 = np.array(sorted(std2.tolist()), dtype=int)
    npt.assert_array_equal(grid, std)
    npt.assert_array_equal(grid, std2)

    grid1 = comp_space_lmtpo.get_composition_grid(supercell_size=10, step=2)
    grid2 = comp_space_lmtpo.get_composition_grid(supercell_size=5, step=1) * 2
    grid1 = np.array(sorted(grid1.tolist()), dtype=int)
    grid2 = np.array(sorted(grid2.tolist()), dtype=int)
    npt.assert_array_equal(grid1, grid2)


def test_grid_storage(comp_space_lmtpo):
    comp_space_lmtpo._comp_grids = {}  # Clear up for new enumeration
    assert len(comp_space_lmtpo._comp_grids) == 0
    _ = comp_space_lmtpo.get_composition_grid(supercell_size=10, step=2)
    _ = comp_space_lmtpo.get_composition_grid()
    _ = comp_space_lmtpo.get_composition_grid(supercell_size=24, step=4)
    _ = comp_space_lmtpo.get_composition_grid(supercell_size=16, step=6)
    _ = comp_space_lmtpo.get_composition_grid(supercell_size=8, step=3)
    _ = comp_space_lmtpo.get_composition_grid(supercell_size=12, step=1)
    _ = comp_space_lmtpo.get_composition_grid(supercell_size=24, step=2)
    keyset = {(5, 1), (6, 1), (8, 3), (12, 1), (1, 1)}
    assert keyset == set(list(comp_space_lmtpo._comp_grids.keys()))


def test_convert_formats(comp_space):
    s, m, t = compute_snf(comp_space._A)
    null_space = t[:, : comp_space.n_dims - len(comp_space.basis)].T
    for _ in range(5):
        # Test good cases.
        supercell_size = (
            20 // comp_space.min_supercell_size * comp_space.min_supercell_size
        )
        a = comp_space._A
        b = comp_space._b * supercell_size
        x_std = comp_space.get_centroid_composition(supercell_size)
        # print("basis:", comp_space.basis)
        # print("n0:", comp_space.get_supercell_base_solution(supercell_size))
        # print("x_std:", x_std)
        n = comp_space.basis.T @ x_std + comp_space.get_supercell_base_solution(
            supercell_size
        )
        npt.assert_almost_equal(a @ n - b, 0, decimal=6)
        assert np.all(n >= -NUM_TOL)

        n2 = comp_space.translate_format(
            n, supercell_size, from_format="counts", to_format="counts"
        )
        npt.assert_almost_equal(n2, n, decimal=6)
        x = comp_space.translate_format(
            n, supercell_size, from_format="counts", to_format="coordinates"
        )
        # This x format must be integers.
        npt.assert_almost_equal(x, np.round(x), decimal=6)
        n0 = comp_space.get_supercell_base_solution(supercell_size)
        npt.assert_almost_equal(comp_space.basis.T @ x + n0, n, decimal=6)
        npt.assert_almost_equal(x, x_std, decimal=6)
        npt.assert_almost_equal(
            comp_space.translate_format(
                x, supercell_size, from_format="coordinates", to_format="counts"
            ),
            n,
        )
        c = comp_space.translate_format(
            n, supercell_size, from_format="counts", to_format="compositions"
        )
        assert len(c) == len(comp_space.bits)
        assert all(isinstance(sl_c, Composition) for sl_c in c)
        assert all(0 <= sl_c.num_atoms <= 1 for sl_c in c)
        npt.assert_almost_equal(
            comp_space.translate_format(
                c, supercell_size, from_format="compositions", to_format="counts"
            ),
            n,
            decimal=6,
        )
        nd = comp_space.translate_format(
            n, supercell_size, from_format="counts", to_format="species-counts"
        )
        for sp_id, sp in enumerate(comp_space.species):
            dim_id = []
            for dim_ids, sl_sps in zip(comp_space.dim_ids, comp_space.bits):
                for d, sp2 in zip(dim_ids, sl_sps):
                    if not isinstance(sp, Vacancy):
                        if sp == sp2:
                            dim_id.append(d)
                    elif isinstance(sp2, Vacancy):
                        dim_id.append(d)
            assert np.sum(n[dim_id]) == nd[sp_id]
        # Nondisc format can not be converted to anything else.
        with pytest.raises(ValueError):
            _ = comp_space.translate_format(
                nd, supercell_size, from_format="species-counts", to_format="counts"
            )
        with pytest.raises(ValueError):
            _ = comp_space.translate_format(
                n, supercell_size, from_format="counts", to_format="whatever"
            )
        with pytest.raises(ValueError):
            _ = comp_space.translate_format(
                n, supercell_size, from_format="whatever", to_format="counts"
            )
        # Test bad cases.
        # Test negative error.
        n_bad = n - 30
        with pytest.raises(ValueError):
            _ = comp_space.translate_format(
                n_bad, supercell_size, from_format="counts", to_format="counts"
            )
        # Test constraints violation
        null_vec = np.random.rand(len(null_space)) * 0.01
        dn = null_space.T @ null_vec
        n_bad = n + dn
        if len(null_space) > 0 and not np.allclose(
            np.abs(comp_space._A @ (n / supercell_size) - comp_space._b),
            0,
            atol=NUM_TOL,
        ):
            if np.all(n_bad >= 0):
                # print("bad composition:", n_bad)
                # print("A:", comp_space._A)
                # print("b:", comp_space._b * supercell_size)
                with pytest.raises(ValueError):
                    _ = comp_space.translate_format(
                        n_bad, supercell_size, from_format="counts", to_format="counts"
                    )
        # Test rounding fails
        dx = np.random.rand(len(comp_space.basis)) + 1e-5
        x_bad = x + dx
        n_bad = comp_space.basis.T @ x_bad + n0
        if np.all(n_bad >= 0):
            with pytest.raises(ValueError):
                _ = comp_space.translate_format(
                    n_bad,
                    supercell_size,
                    from_format="counts",
                    to_format="counts",
                    rounding=True,
                )
        # Test un-normalized compositions.
        c_bad = deepcopy(c)
        c0_bad = {k: v + 100 for k, v in c_bad[0].items()}
        c_bad[0] = Composition(c0_bad)
        with pytest.raises(ValueError):
            _ = comp_space.translate_format(
                c_bad, supercell_size, from_format="compositions", to_format="counts"
            )
