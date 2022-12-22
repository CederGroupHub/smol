"""
A few testing utilities that may be useful to just import and run.
Some of these are borrowed from pymatgen test scripts.
"""

import json

import numpy as np
import numpy.testing as npt
from monty.json import MontyDecoder, MSONable
from pymatgen.entries.computed_entries import ComputedStructureEntry

from smol.capp.generate.random import gen_random_structure


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
    _ = json.loads(obj.to_json(), cls=MontyDecoder)


def gen_fake_training_data(prim_structure, n=10, rng=None):
    """Generate a fake structure, energy training set."""
    rng = np.random.default_rng(rng)
    training_data = []
    for energy in rng.random(n):
        struct = gen_random_structure(prim_structure, size=rng.integers(2, 6), rng=rng)
        energy *= -len(struct)
        training_data.append(ComputedStructureEntry(struct, energy))
    return training_data
