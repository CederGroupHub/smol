"""
Input and output functions. Can also at some point add input and output for
other CE code packages to make it easier to translate work.
"""

__author__ = "Luis Barroso-Luque"

import json
from monty.json import MSONable, MontyEncoder, MontyDecoder


def save_work(file_path, *msonables):
    """
    Save a set of unique MSONable objects used in a workflow for a cluster
    expansion and/or monte carlo run as a json dictionary.

    Args:
        file_path (str):
            file path
        *msonables (monty.MSONable):
            MSONable child classes.
    """
    work_dict = {}
    for msonable in msonables:
        if not isinstance(msonable, MSONable):
            raise AttributeError('Attempting to save an object which is not '
                                 f'MSONable: {msonable}')
        work_dict[msonable.__class__.__name__] = msonable

    with open(file_path, 'w') as fp:
        json.dump(work_dict, fp, cls=MontyEncoder)


def load_work(file_path):
    """
    Load a dictionary of with the save_work function and instantiate the
    MSONable objects.

    Args:
        file_path (str):

    Returns: Dictionary with smol objects
        dict
    """
    with open(file_path, 'r') as fp:
        work_dict = json.load(fp, cls=MontyDecoder)

    return work_dict
