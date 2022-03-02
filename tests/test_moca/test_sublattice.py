import pytest
import numpy as np
import numpy.testing as npt

from pymatgen.core import Composition, DummySpecies
from smol.cofe.space.domain import SiteSpace
from smol.moca.sublattice import Sublattice, InactiveSublattice
from tests.utils import assert_msonable


@pytest.fixture
def sublattice():
    composition = Composition(
        {
            DummySpecies("A"): 0.3,
            DummySpecies("X"): 0.3,
            DummySpecies("D"): 0.2,
            DummySpecies("E"): 0.2,
        }
    )
    site_space = SiteSpace(composition)
    sites = np.random.choice(range(100), size=60)
    return Sublattice(site_space, sites)


def test_restrict_sites(sublattice):
    sites = np.random.choice(sublattice.sites, size=10)
    # test sites properly restricted
    sublattice.restrict_sites(sites)
    assert not any(s in sublattice.active_sites for s in sites)
    assert all(s in sublattice.restricted_sites for s in sites)
    assert len(sublattice.active_sites) != len(sublattice.sites)
    # test reset
    sublattice.reset_restricted_sites()
    npt.assert_array_equal(sublattice.active_sites, sublattice.sites)
    assert len(sublattice.restricted_sites) == 0


def test_msonable(sublattice):
    # Test msnoable serialization
    d = sublattice.as_dict()
    slatt = Sublattice.from_dict(d)
    assert sublattice.site_space == slatt.site_space
    npt.assert_array_equal(sublattice.sites, slatt.sites)
    npt.assert_array_equal(sublattice.active_sites, slatt.active_sites)
    assert_msonable(sublattice)


def test_inactive_sublattice():
    composition = Composition({DummySpecies("A"): 1})
    site_space = SiteSpace(composition)
    sites = np.random.choice(range(100), size=60)
    inactive_sublattice = InactiveSublattice(site_space, sites)
    assert_msonable(inactive_sublattice)
