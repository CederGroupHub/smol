"""Input and output functions.

Some convenience functions to save and load CE and MC workflows in a "standard"
way.

Can also at some point add input and output for other CE code packages to make
it easier to translate work.
"""

__author__ = "Luis Barroso-Luque"

import json

from monty.json import MontyDecoder, MontyEncoder, MSONable


def save_work(file_path, *msonables):
    """Save msonable classes used in a CE and/or MC workflow.

    Save a set of unique MSONable objects used in a workflow for a cluster
    expansion and/or monte carlo run as a json dictionary.

    Only one of each class type can be saved per file.

    Args:
        file_path (str):
            file path
        *msonables (monty.MSONable):
            MSONable child classes.
    """
    work_d = {}
    for msonable in msonables:
        if not isinstance(msonable, MSONable):
            raise AttributeError(
                "Attempting to save an object which is not " f"MSONable: {msonable}"
            )
        work_d[msonable.__class__.__name__] = msonable

    with open(file_path, "w", encoding="utf-8") as fpath:
        json.dump(work_d, fpath, cls=MontyEncoder)


def load_work(file_path):
    """Load a dictionary and instantiate the MSONable objects.

    Args:
        file_path (str):

    Returns: Dictionary with smol objects
        dict
    """
    with open(file_path, encoding="utf-8") as fpath:
        work_d = json.load(fpath, cls=MontyDecoder)

    return work_d
