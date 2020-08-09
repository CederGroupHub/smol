import pytest
from collections import OrderedDict
import numpy as np
import numpy.testing as npt

from smol.moca.ensemble.sublattice import Sublattice
from tests.utils import assert_msonable


@pytest.fixture
def sublattice():
    site_space = OrderedDict({'A': 0.3, 'B': 0.3, 'C': 0.2, 'D': 0.2})
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


def test_print_repr(sublattice):
    # just print and repr to check no errors raised
    print(sublattice)
    repr(sublattice)


def test_msonable(sublattice):
    # Test msnoable serialization
    assert_msonable(sublattice)
