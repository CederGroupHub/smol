import numpy as np
import numpy.testing as npt
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import PeriodicSite, Structure

from smol.capp.generate.random import _gen_neutral_occu
from smol.cofe.extern import EwaldTerm
from smol.cofe.space.domain import Vacancy, get_allowed_species
from tests.utils import assert_msonable, assert_pickles


def test_get_ewald_structure(ce_processor):
    supercell = ce_processor.structure
    sm = StructureMatcher()
    ew = EwaldTerm(eta=0.15)
    ew_structure, ew_inds = ew.get_ewald_structure(supercell)

    spaces = get_allowed_species(supercell)
    nbits = np.array([len(space) - 1 for space in spaces])
    sites = []
    for space, site in zip(spaces, supercell):
        for b in space:
            if isinstance(b, Vacancy):  # skip vacancies
                continue
            sites.append(PeriodicSite(b, site.frac_coords, supercell.lattice))

    s = Structure.from_sites(sites)
    assert sm.fit(ew_structure, s)

    assert ew_inds.shape == (len(supercell), max(nbits) + 1)
    start = 0
    for space, inds in zip(spaces, ew_inds):
        if isinstance(space[-1], Vacancy):
            n_sp = len(space) - 1
        else:
            n_sp = len(space)
        npt.assert_allclose(inds[:n_sp], np.arange(start, start + n_sp))
        npt.assert_allclose(inds[n_sp:], -1)
        start += n_sp


def test_val_from_occupancy(ce_processor, rng):
    # Test 10 times at random, with charge balance.
    supercell = ce_processor.structure
    sublattices = ce_processor.get_sublattices()
    n_success = 0
    for _ in range(10):
        try:
            # We should only test neutral occupancies.
            occu = _gen_neutral_occu(sublattices, rng=rng)
            n_success += 1
        except TimeoutError:
            occu = None

        if occu is not None:
            s = ce_processor.structure_from_occupancy(occu)
            assert s.charge == 0
            ew = EwaldTerm(eta=0.15)
            np.testing.assert_almost_equal(
                ew.value_from_occupancy(occu, supercell),
                EwaldSummation(s, eta=ew.eta).total_energy,
                decimal=7,
            )
            ew = EwaldTerm(eta=0.15, use_term="real")
            np.testing.assert_almost_equal(
                ew.value_from_occupancy(occu, supercell),
                EwaldSummation(s, eta=ew.eta).real_space_energy,
                decimal=7,
            )
            ew = EwaldTerm(eta=0.15, use_term="reciprocal")
            np.testing.assert_almost_equal(
                ew.value_from_occupancy(occu, supercell),
                EwaldSummation(s, eta=ew.eta).reciprocal_space_energy,
                decimal=7,
            )
            ew = EwaldTerm(eta=0.15, use_term="point")
            np.testing.assert_almost_equal(
                ew.value_from_occupancy(occu, supercell),
                EwaldSummation(s, eta=ew.eta).point_energy,
                decimal=7,
            )
    assert n_success > 0


def test_msonable():
    ew = EwaldTerm(eta=0.15, real_space_cut=0.5, use_term="point")
    assert_msonable(ew)


def test_pickles():
    ew = EwaldTerm(eta=0.15, real_space_cut=0.5, use_term="point")
    assert_pickles(ew)
