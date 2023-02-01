import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.core import Composition, DummySpecies

from smol.cofe.space.domain import SiteSpace
from smol.moca.sublattice import Sublattice
from tests.utils import assert_msonable


@pytest.fixture
def sublattice(rng):
    composition = Composition(
        {
            DummySpecies("A"): 0.3,
            DummySpecies("X"): 0.3,
            DummySpecies("D"): 0.2,
            DummySpecies("E"): 0.2,
        }
    )
    site_space = SiteSpace(composition)
    sites = rng.choice(range(100), size=60)
    return Sublattice(site_space, sites)


def test_restrict_sites(sublattice, rng):
    npt.assert_array_equal(sublattice.sites, np.sort(sublattice.sites))
    npt.assert_array_equal(sublattice.sites, np.unique(sublattice.sites))
    npt.assert_array_equal(
        sublattice.encoding, np.arange(len(sublattice.species), dtype=int)
    )
    sites = rng.choice(sublattice.sites, size=min(10, len(sublattice.sites)))
    # test sites properly restricted
    sublattice.restrict_sites(sites)
    assert not any(s in sublattice.active_sites for s in sites)
    assert all(s in sublattice.restricted_sites for s in sites)
    assert len(sublattice.active_sites) != len(sublattice.sites)
    # test reset
    sublattice.reset_restricted_sites()
    npt.assert_array_equal(sublattice.active_sites, sublattice.sites)
    assert len(sublattice.restricted_sites) == 0


def test_split(sublattice, rng):
    sublattice.restrict_sites(
        rng.choice(sublattice.sites, size=len(sublattice.sites) // 2)
    )
    occu = np.zeros(100, dtype=int) - 1
    pool = sublattice.sites.copy()
    for code, x in zip(sublattice.encoding, sublattice.site_space.values()):
        n = int(round(x * len(sublattice.sites)))
        select = rng.choice(pool, size=min(n, len(pool)), replace=False)
        occu[select] = code
        pool = np.setdiff1d(pool, select)
    if len(pool) > 0:
        occu[pool] = sublattice.encoding[-1]
    splits = sublattice.split_by_species(
        occu, [sublattice.encoding[0:1], sublattice.encoding[1:]]
    )
    assert len(splits) == 2
    assert not splits[0].is_active
    assert set(splits[0].species) == set(sublattice.species[0:1])
    assert np.all(splits[0].encoding == sublattice.encoding[0])
    assert set(splits[1].species) == set(sublattice.species[1:])
    assert set(splits[1].encoding) == set(sublattice.encoding[1:])
    assert splits[0].site_space[splits[0].species[0]] == 1

    comp_spl1 = np.array(list(sublattice.site_space.values()))[1:]
    n1 = comp_spl1.sum()
    comp_spl1 = comp_spl1 / n1
    npt.assert_array_equal(comp_spl1, list(splits[1].site_space.values()))
    assert list(splits[0].site_space.values())[0] == 1

    sites_spl0 = sublattice.sites[
        np.any(occu[sublattice.sites, None] == sublattice.encoding[None, 0:1], axis=-1)
    ]
    sites_spl1 = sublattice.sites[
        np.any(occu[sublattice.sites, None] == sublattice.encoding[None, 1:], axis=-1)
    ]
    npt.assert_array_equal(np.sort(splits[0].sites), np.sort(sites_spl0))
    npt.assert_array_equal(np.sort(splits[1].sites), np.sort(sites_spl1))
    npt.assert_array_equal(
        np.sort(np.concatenate((sites_spl0, sites_spl1))), np.sort(sublattice.sites)
    )

    active_spl0 = np.array([], dtype=int)
    active_spl0_prev = sublattice.active_sites[
        np.any(
            occu[sublattice.active_sites, None] == sublattice.encoding[None, 0:1],
            axis=-1,
        )
    ]
    active_spl1 = sublattice.active_sites[
        np.any(
            occu[sublattice.active_sites, None] == sublattice.encoding[None, 1:],
            axis=-1,
        )
    ]
    npt.assert_array_equal(np.sort(splits[0].active_sites), np.sort(active_spl0))
    npt.assert_array_equal(np.sort(splits[1].active_sites), np.sort(active_spl1))
    npt.assert_array_equal(
        np.sort(np.concatenate((active_spl0_prev, active_spl1))),
        np.sort(sublattice.active_sites),
    )

    # Test a split when all sites are in split case 1.
    occu = np.zeros(100, dtype=int) - 1
    occu[sublattice.sites] = sublattice.encoding[1]
    splits = sublattice.split_by_species(
        occu, [sublattice.encoding[0:1], sublattice.encoding[1:]]
    )
    assert not splits[0].is_active
    assert len(splits[0].sites) == 0
    assert len(splits[0].active_sites) == 0
    assert splits[1].is_active
    npt.assert_array_equal(np.sort(splits[1].sites), np.sort(sublattice.sites))
    npt.assert_array_equal(
        np.sort(splits[1].active_sites), np.sort(sublattice.active_sites)
    )

    splits_by_species = sublattice.split_by_species(
        occu, [sublattice.species[0:1], sublattice.species[1:]]
    )
    for sublatt1, sublatt2 in zip(splits_by_species, splits):
        npt.assert_array_equal(sublatt1.sites, sublatt2.sites)
        npt.assert_array_equal(sublatt1.active_sites, sublatt2.active_sites)
        assert sublatt1.species == sublatt2.species
        npt.assert_array_equal(sublatt1.encoding, sublatt2.encoding)


def test_msonable(sublattice):
    # Test msnoable serialization
    d = sublattice.as_dict()
    slatt = Sublattice.from_dict(d)
    assert sublattice.site_space == slatt.site_space
    npt.assert_array_equal(sublattice.sites, slatt.sites)
    npt.assert_array_equal(sublattice.active_sites, slatt.active_sites)
    assert_msonable(sublattice)


def test_inactiveness(sublattice, rng):
    composition = Composition({DummySpecies("A"): 1})
    site_space = SiteSpace(composition)
    sites = rng.choice(range(100), size=60, replace=False)
    inactive_sublattice = Sublattice(site_space, sites)
    npt.assert_array_equal(inactive_sublattice.encoding, np.array([0]))
    assert len(inactive_sublattice.active_sites) == 0
    npt.assert_array_equal(
        np.sort(inactive_sublattice.restricted_sites),
        np.sort(inactive_sublattice.sites),
    )
    assert not inactive_sublattice.is_active

    sublattice.restrict_sites(sublattice.sites)
    assert not sublattice.is_active


def test_equal(sublattice, rng):
    sublatt2 = Sublattice.from_dict(sublattice.as_dict())
    assert sublattice == sublatt2
    sublatt2.restrict_sites(rng.choice(sublatt2.sites, 5, replace=False))
    assert sublattice == sublatt2

    # change the site_space
    sublatt2.site_space = SiteSpace(Composition({"A": 0.5}))
    assert sublattice != sublatt2

    # change the sites
    sublatt2.site_space = sublattice.site_space
    sublatt2.reset_restricted_sites()
    sublatt2.sites = np.arange(len(sublattice.sites) - 5)
    assert sublattice != sublatt2

    # change the encoding to a random one
    sublatt2.site_space = sublattice.site_space
    sublatt2.sites = sublattice.sites
    sublatt2.reset_restricted_sites()
    sublatt2.encoding = rng.choice(
        range(10, 20 + len(sublatt2.encoding)), replace=False
    )
    assert sublattice != sublatt2
