import pytest
from random import choices
from itertools import combinations
import numpy as np
from smol.cofe.space import Cluster
from tests.utils import assert_msonable


@pytest.fixture(scope='package')
def cluster(structure):
    num_sites = np.random.randint(2, 6)
    sites = choices(structure.frac_coords, k=num_sites)
    # add random integer multiples
    for site in sites:
        site += np.random.randint(-3, 3)
    cluster = Cluster(sites, structure.lattice)
    cluster.assign_ids(1)
    return cluster


def test_from_sites(structure):
    assert Cluster(structure.frac_coords, structure.lattice) == Cluster.from_sites(structure.sites)


def test_size(cluster):
    assert len(cluster.sites) == len(cluster)


def test_diameter(cluster):
    coords = cluster.lattice.get_cartesian_coords(cluster.sites)
    diameter = max([np.sum((i - j)**2) for i, j in combinations(coords, 2)])**0.5
    assert diameter == cluster.diameter


def test_periodicity(cluster):
    new_sites = cluster.sites.copy()
    new_sites += np.random.randint(-5, 5)
    assert Cluster(new_sites, cluster.lattice) == cluster
    new_sites[0] += 0.005 + np.random.random()
    assert Cluster(new_sites, cluster.lattice) != cluster


def test_edge_case(structure):
    c1 = np.array([0.25, 0.25, 0.25])
    c2 = np.array([0, 0, 1])
    c3 = np.array([0.25, 1.25, -0.75])
    c4 = np.array([0, 1, 0])
    clust1 = Cluster([c1, c2], structure.lattice)
    clust2 = Cluster([c3, c4], structure.lattice)
    assert clust1 == clust2


def test_msonable(cluster):
    _ = repr(cluster)
    _ = str(cluster)
    assert_msonable(cluster)