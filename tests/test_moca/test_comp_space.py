import pytest
import numpy as np
import numpy.testing as npt
import itertools

from smol.moca.comp_space import *
from smol.cofe.space.domain import Vacancy
from pymatgen.core import Species

from smol.moca.utils.math_utils import gcd_list, NUM_TOL

from tests.utils import assert_msonable


@pytest.fixture(scope="module")
def comp_space(ensemble):
    bits = [s.species for s in ensemble.sublattices]
    sl_sizes = np.array([len(s.sites) for s in ensemble.sublattices])
    gcd = gcd_list(sl_sizes)
    sl_sizes = sl_sizes // gcd
    return CompSpace(bits, sl_sizes)


@pytest.fixture(scope="module")
def comp_space_lmtpo():
    li = Species.from_string('Li+')
    mn = Species.from_string('Mn3+')
    ti = Species.from_string('Ti4+')
    o = Species.from_string('O2-')
    p = Species.from_string('P3-')

    bits = [[li, mn, ti],[p, o]]
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
                     charge_balanced=True,
                     optimize_basis=True,
                     table_ergodic=True)


# Generic attibutes test.
def test_generic_attributes(comp_space):
    _ = comp_space.dim_ids
    species_set = set(itertools.chain(*comp_space.bits))
    vac_count = 0
    for sp in comp_space.species:
        assert sp in species_set
        if isinstance(sp, Vacancy):
            vac_count += 1
    assert vac_count <= 1

    for sl_id, dim_ids_sl in enumerate(comp_space.dim_ids_nondisc):
        for sp_id, d in enumerate(dim_ids_sl):
            bit = comp_space.bits[sl_id][sp_id]
            sp = comp_space.species[d]
            if not isinstance(sp, Vacancy):
                assert bit == sp
            else:
                assert isinstance(bit, Vacancy)
    assert sorted(comp_space.species) == comp_space.species

    A = comp_space.A
    b = comp_space.b
    assert len(A) == 1 + len(comp_space.bits)
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
            ns = grid @ comp_space.basis + comp_space.n0
            npt.assert_array_equal(grid, comp_space._comp_grids[sc])
            # x-format
            npt.assert_array_equal(A @ ns.T - b[:, None], 0)
            assert np.all(ns >= 0)

    # Flip table not optimized.
    npt.assert_array_equal(comp_space.flip_table, comp_space.basis)


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
    for k in comp_space.keys():
        npt.assert_array_equal(comp_space._comp_grids[k],
                               comp_space_reload._comp_grids[k])


# Pre-computed data test.
def test_sc_size(comp_space_lmtpo, comp_space_lmntof):
    assert comp_space_lmtpo.min_sc_size == 6
    assert comp_space_lmntof.min_sc_size == 6


def test_n_comps(comp_space_lmtpo, comp_space_lmntof):
    assert comp_space_lmtpo.n_comps_estimate == 7776
    assert comp_space_lmntof.n_comps_estimate == 46656


# TODO: complete writing CompSpace spacial tests.
def test_basis(comp_space_lmtpo, comp_space_lmntof):
    pass