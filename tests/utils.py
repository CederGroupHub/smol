"""
A few testing utilities that may be useful to just import and run.
Some of these are borrowed from pymatgen test scripts.
"""

import json
import pickle

import numpy as np
import numpy.testing as npt
from monty.json import MontyDecoder, MSONable
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedStructureEntry


def assert_table_set_equal(a1, a2):
    a1 = np.array(a1)
    a2 = np.array(a2)
    assert len(a1.shape) == 2
    assert a1.shape == a2.shape
    a1_set = np.concatenate((a1, -a1), axis=0)
    a2_set = np.concatenate((a2, -a2), axis=0)
    a1_set = np.array(sorted(tuple(r) for r in a1_set))
    a2_set = np.array(sorted(tuple(r) for r in a2_set))
    npt.assert_array_equal(a1_set, a2_set)


def assert_msonable(obj, skip_keys=None, test_if_subclass=True):
    """
    Tests if obj is MSONable and tries to verify whether the contract is
    fulfilled.
    By default, the method tests whether obj is an instance of MSONable.
    This check can be deactivated by setting test_if_subclass to False.
    """
    if test_if_subclass:
        assert isinstance(obj, MSONable)

    skip_keys = [] if skip_keys is None else skip_keys
    d1 = obj.as_dict()
    d2 = obj.__class__.from_dict(obj.as_dict()).as_dict()
    for key in d1.keys():
        if key in skip_keys:
            continue
        assert d1[key] == d2[key]

    try:
        _ = json.loads(obj.to_json(), cls=MontyDecoder)
    except Exception as e:
        raise AssertionError(e)


def assert_pickles(obj):
    """Test if obj is picklable."""
    try:
        p = pickle.dumps(obj)
        obj_copy = pickle.loads(p)
    except Exception as e:
        raise AssertionError(e)

    assert isinstance(obj_copy, obj.__class__)

    if isinstance(obj, MSONable):
        d1 = obj.as_dict()
        d2 = obj_copy.as_dict()
        for key in d1.keys():
            assert d1[key] == d2[key]
    else:
        # fallback for objects that are not MSONable
        # not a complete test, since we are only checking that attribute names match
        d1 = obj.__dict__
        d2 = obj_copy.__dict__
        assert d1.keys() == d2.keys()


def gen_fake_training_data(prim_structure, n=10, rng=None):
    """Generate a fake structure, energy training set."""
    rng = np.random.default_rng(rng)
    training_data = []
    for energy in rng.random(n):
        struct = gen_random_ordered_structure(
            prim_structure, size=rng.integers(2, 6), rng=rng
        )
        energy *= -len(struct)
        training_data.append(ComputedStructureEntry(struct, energy))
    return training_data


def gen_random_ordered_structure(prim, size=3, rng=None):
    """Generate an random ordered structure from a disordered prim.

    Args:
        prim (pymatgen.Structure):
            disordered primitive structure:
        size (optional):
            size argument to structure.make_supercell
        rng (optional): {None, int, array_like[ints], SeedSequence, BitGenerator, Generator},
            A RNG, seed or otherwise to initialize defauly_rng

    Returns:
        ordered structure
    """
    rng = np.random.default_rng(rng)
    structure = prim.copy()
    structure.make_supercell(size)
    for site in structure:
        site.species = Composition({rng.choice(list(site.species.keys())): 1})
    return structure


def compute_cluster_interactions(expansion, structure, normalized=True, scmatrix=None):
    # compute cluster interactions from definitionas in pure python
    corrs = expansion.cluster_subspace.corr_from_structure(
        structure, normalized=normalized, scmatrix=scmatrix
    )
    vals = expansion.eci * corrs
    interactions = np.array(
        [
            np.sum(
                vals[expansion.eci_orbit_ids == i]
                * expansion.cluster_subspace.function_ordering_multiplicities[
                    expansion.eci_orbit_ids == i
                ]
            )
            for i in range(len(expansion.cluster_subspace.orbits) + 1)
        ]
    )
    return interactions
