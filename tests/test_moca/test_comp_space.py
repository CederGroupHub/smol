import pytest
import numpy.testing as npt
import itertools
import re
from copy import deepcopy

from smol.moca.comp_space import *
from smol.cofe.space.domain import Vacancy
from pymatgen.core import Species, Composition

from smol.moca.utils.math_utils import (gcd_list, NUM_TOL,
                                        compute_snf)
from smol.moca.utils.occu_utils import get_dim_ids_by_sublattice

from tests.utils import assert_msonable, assert_table_set_equal


@pytest.fixture(scope="module")
def comp_space(ensemble):
    bits = [s.species for s in ensemble.sublattices]
    sl_sizes = np.array([len(s.sites) for s in ensemble.sublattices])
    gcd = gcd_list(sl_sizes)
    sl_sizes = sl_sizes // gcd
    return CompSpace(bits, sl_sizes)  # Charge balanced = True, default.


@pytest.fixture(scope="module")
def comp_space_lmtpo():
    li = Species.from_string('Li+')
    mn = Species.from_string('Mn3+')
    ti = Species.from_string('Ti4+')
    o = Species.from_string('O2-')
    p = Species.from_string('P3-')

    bits = [[li, mn, ti], [p, o]]
    sl_sizes = [1, 1]
    return CompSpace(bits, sl_sizes,
                     charge_balanced=True,
                     optimize_basis=True,
                     table_ergodic=True)


# A test example with extra constraints.
@pytest.fixture(scope="module")
def comp_space_lmntof():
    li = Species.from_string('Li+')
    ni = Species.from_string('Ni2+')
    mn = Species.from_string('Mn3+')
    ti = Species.from_string('Ti4+')
    f = Species.from_string('F-')
    o = Species.from_string('O2-')

    bits = [[li, ni, mn, ti], [o, f]]
    sl_sizes = [1, 1]
    other_constraints = [([0, 1, -1, 0, 0, 0], 0),
                         ([0, 0, 1, -1, 0, 0], 0)]
    return CompSpace(bits, sl_sizes,
                     other_constraints=other_constraints,
                     charge_balanced=True,
                     optimize_basis=True,
                     table_ergodic=True)


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

    for sl_id, dim_ids_sl in enumerate(comp_space.dim_ids_nondisc):
        for sp_id, d in enumerate(dim_ids_sl):
            bit = comp_space.bits[sl_id][sp_id]
            sp = comp_space.species[d]
            if not isinstance(sp, Vacancy):
                assert bit == sp
            else:
                assert isinstance(bit, Vacancy)

    A = comp_space.A
    b = comp_space.b
    assert len(A) == 1 + len(comp_space.bits)  # No other constraints.
    assert len(b) == len(A)
    assert b[0] == 0
    assert np.all(A[1:, :] <= 1)
    npt.assert_array_equal(b[1:], comp_space.sl_sizes)
    min_sc = comp_space.min_sc_size

    prim_verts = comp_space.prim_vertices
    assert prim_verts.shape[1] == A.shape[1]
    npt.assert_almost_equal(A @ prim_verts.T - b[:, None],
                            0, decimal=6)
    assert np.all(np.any(np.isclose(prim_verts, 0, atol=NUM_TOL), axis=-1))

    if comp_space.n_comps_estimate < 10 ** 6:
        scs = [1, min_sc // 2, min_sc]
        for sc in scs:
            grid = comp_space.get_comp_grid(sc)
            ns = grid @ comp_space.basis + comp_space.n0 * sc
            npt.assert_array_equal(grid, comp_space._comp_grids[sc])
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


def test_serialize(comp_space):
    _ = comp_space.flip_table
    if comp_space.n_comps_estimate < 10 ** 6:
        _ = comp_space.min_sc_grid
    assert_msonable(comp_space)
    comp_space_reload = CompSpace.from_dict(comp_space.as_dict())
    npt.assert_array_equal(comp_space_reload._flip_table,
                           comp_space._flip_table)
    npt.assert_array_equal(comp_space_reload._vs,
                           comp_space._vs)
    npt.assert_array_equal(comp_space_reload._n0,
                           comp_space._n0)
    npt.assert_array_equal(comp_space_reload._min_sc_size,
                           comp_space._min_sc_size)
    assert (set(list(comp_space._comp_grids.keys()))
            == set(list(comp_space_reload._comp_grids.keys())))
    for k in comp_space._comp_grids.keys():
        npt.assert_array_equal(comp_space._comp_grids[k],
                               comp_space_reload._comp_grids[k])


# Pre-computed data test.
def test_sc_size(comp_space_lmtpo, comp_space_lmntof):
    assert comp_space_lmtpo.min_sc_size == 6
    assert comp_space_lmntof.min_sc_size == 6


def test_n_comps_est(comp_space_lmtpo, comp_space_lmntof):
    assert comp_space_lmtpo.n_comps_estimate == 7776
    assert comp_space_lmntof.n_comps_estimate == 46656


def test_basis(comp_space_lmtpo, comp_space_lmntof):
    vs1 = comp_space_lmtpo.basis
    vs2 = comp_space_lmntof.basis
    ts1 = comp_space_lmtpo.flip_table
    ts2 = comp_space_lmntof.flip_table
    # Pre-computed optimal basis
    std1 = np.array([[0, -1, 1, 1, -1],
                     [-1, 1, 0, 2, -2]], dtype=int)
    std2 = np.array([[-3, 1, 1, 1, 6, -6]], dtype=int)

    assert_table_set_equal(vs1, std1)
    assert_table_set_equal(vs2, std2)
    assert_table_set_equal(ts1, std1)
    assert_table_set_equal(ts2, std2)


def test_convert_formats(comp_space):
    s, m, t = compute_snf(comp_space.A)
    null_space = t[:, : comp_space.n_dims - len(comp_space.basis)].T
    for _ in range(5):
        # Test good cases.
        sc_size = 20
        a = comp_space.A
        b = comp_space.b * sc_size
        x_std = comp_space.get_centroid_composition(sc_size)
        print("basis:", comp_space.basis)
        print("n0:", comp_space.n0)
        print("x_std:", x_std)
        n = comp_space.basis.T @ x_std + comp_space.n0 * sc_size
        npt.assert_almost_equal(a @ n - b, 0, decimal=6)
        assert np.all(n >= -NUM_TOL)

        n2 = comp_space.translate_format(n, sc_size,
                                         from_format="n", to_format="n")
        npt.assert_almost_equal(n2, n, decimal=6)
        x = comp_space.translate_format(n, sc_size,
                                        from_format="n", to_format="x")
        # This x format must be integers.
        npt.assert_almost_equal(x, np.round(x), decimal=6)
        n0 = comp_space.n0 * sc_size
        npt.assert_almost_equal(comp_space.basis.T @ x + n0, n,
                                decimal=6)
        npt.assert_almost_equal(x, x_std, decimal=6)
        npt.assert_almost_equal(comp_space.translate_format(x, sc_size,
                                                            from_format="x",
                                                            to_format="n"),
                                n)
        c = comp_space.translate_format(n, sc_size,
                                        from_format="n",
                                        to_format="comp")
        assert len(c) == len(comp_space.bits)
        assert all(isinstance(sl_c, Composition) for sl_c in c)
        assert all(0 <= sl_c.num_atoms <= 1 for sl_c in c)
        npt.assert_almost_equal(comp_space.translate_format(c, sc_size,
                                                           from_format="comp",
                                                           to_format="n"),
                               n, decimal=6)
        nd = comp_space.translate_format(n, sc_size,
                                         from_format="n",
                                         to_format="nondisc")
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
            _ = comp_space.translate_format(nd, sc_size,
                                            from_format="nondisc",
                                            to_format="n")
        with pytest.raises(ValueError):
            _ = comp_space.translate_format(n, sc_size,
                                            from_format="n",
                                            to_format="whatever")
        with pytest.raises(ValueError):
            _ = comp_space.translate_format(n, sc_size,
                                            from_format="whatever",
                                            to_format="n")
        # Test bad cases.
        # Test negative error.
        n_bad = n - 30
        with pytest.raises(NegativeSpeciesError):
            _ = comp_space.translate_format(n_bad, sc_size,
                                            from_format="n",
                                            to_format="n")
        # Test constraints violation
        null_vec = np.random.rand(len(null_space)) * 0.01
        if len(null_space) > 0 and not np.allclose(np.abs(null_vec),
                                                   0, atol=NUM_TOL):
            dn = null_space.T @ null_vec
            n_bad = n + dn
            if np.all(n_bad >= 0):
                with pytest.raises(ConstraintViolationError):
                    _ = comp_space.translate_format(n_bad, sc_size,
                                                    from_format="n",
                                                    to_format="n")
        # Test rounding fails
        dx = np.random.rand(len(comp_space.basis)) + 1E-5
        x_bad = x + dx
        n_bad = comp_space.basis.T @ x_bad + n0
        if np.all(n_bad >= 0):
            with pytest.raises(RoundingError):
                _ = comp_space.translate_format(n_bad, sc_size,
                                                from_format="n",
                                                to_format="n",
                                                rounding=True)
        # Test un-normalized compositions.
        c_bad = deepcopy(c)
        c0_bad = {k: v + 100 for k, v in c_bad[0].items()}
        c_bad[0] = Composition(c0_bad)
        with pytest.raises(CompUnNormalizedError):
            _ = comp_space.translate_format(c_bad, sc_size,
                                            from_format="comp",
                                            to_format="n")
