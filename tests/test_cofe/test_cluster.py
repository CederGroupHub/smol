import json
import os
from itertools import combinations

import numpy as np
import pytest
from ruamel import yaml

from smol.cofe.space import Cluster
from smol.cofe.space.domain import get_site_spaces
from tests.utils import assert_msonable


@pytest.fixture()
def cluster(structure, rng):
    num_sites = rng.integers(2, 6)
    sites = rng.choice(structure, size=num_sites)
    coords = [site.frac_coords.copy() for site in sites]
    # add random integer multiples
    for coord in coords:
        coord += np.random.randint(-3, 3)
    cluster = Cluster(get_site_spaces(sites), coords, structure.lattice)
    cluster.assign_ids(1)
    return cluster


def test_size(cluster):
    assert len(cluster.sites) == len(cluster)


def test_diameter(cluster):
    coords = cluster.lattice.get_cartesian_coords(cluster.frac_coords)
    diameter = max(np.sum((i - j) ** 2) for i, j in combinations(coords, 2)) ** 0.5
    assert diameter == cluster.diameter


def test_periodicity(cluster, rng):
    new_coords = cluster.frac_coords.copy()
    new_coords += rng.integers(-5, 5)
    site_spaces = [s.species for s in cluster.sites]
    assert Cluster(site_spaces, new_coords, cluster.lattice) == cluster
    new_coords[0] += 0.005 + rng.random()
    assert Cluster(site_spaces, new_coords, cluster.lattice) != cluster


def test_edge_case(structure):
    c1 = np.array([0.25, 0.25, 0.25])
    c2 = np.array([0, 0, 1])
    c3 = np.array([0.25, 1.25, -0.75])
    c4 = np.array([0, 1, 0])
    sspaces = get_site_spaces(structure)
    clust1 = Cluster(sspaces, [c1, c2], structure.lattice)
    clust2 = Cluster(sspaces, [c3, c4], structure.lattice)
    assert clust1 == clust2


def test_msonable(cluster):
    _ = repr(cluster)
    _ = str(cluster)
    assert_msonable(cluster, skip_keys=["sites"])


def test_to_from(cluster, tmpdir):
    yml = cluster.to("yaml")
    js = cluster.to("json")

    cluster2 = Cluster.from_str(js, "json")
    assert cluster == cluster2

    cluster2 = Cluster.from_str(yml, "yaml")
    assert cluster == cluster2

    with open(os.path.join(tmpdir, "cluster.yaml"), "w") as f:
        yaml.dump(yml, f)
    with open(os.path.join(tmpdir, "cluster.json"), "w") as f:
        json.dump(js, f)

    # cluster2 = Cluster.from_file(os.path.join(tmpdir, "cluster.yaml"))
    # assert cluster == cluster2
    # cluster2 = Cluster.from_file(os.path.join(tmpdir, "cluster.json"))
    # assert cluster == cluster2

    # cluster.to("yaml", os.path.join(tmpdir, "cluster.yaml"))
    # cluster.to("json", os.path.join(tmpdir, "cluster.json"))

    # cluster2 = Cluster.from_file(os.path.join(tmpdir, "cluster.yaml"))
    # assert cluster == cluster2
    # cluster2 = Cluster.from_file(os.path.join(tmpdir, "cluster.json"))
    # assert cluster == cluster2

    with pytest.raises(ValueError):
        cluster.to("bad_format")
