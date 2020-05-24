"""A lot of random utilities that have no place to go."""

__author__ = "Luis Barroso-Luque, William Davidson Richard"

from typing import Dict, Any
from collections import OrderedDict

SITE_TOL = 1e-6


def get_bits(structure):
    """Helper method to compute list of species on each site.
    Includes vacancies.
    """
    all_bits = []
    for site in structure:
        bits = [str(sp) for sp in sorted(site.species.keys())]
        if site.species.num_atoms < 0.99:
            bits.append("Vacancy")
        all_bits.append(tuple(bits))
    return all_bits


def get_bits_w_concentration(structure):
    """Same as above but returns list of dicts to include the concentration of
    each species.
    """
    all_bits = []
    for site in structure:
        bits = OrderedDict((str(sp), c) for sp, c
                           in sorted(site.species.items()))
        if site.species.num_atoms < 0.99:
            bits["Vacancy"] = 1 - site.species.num_atoms
        all_bits.append(bits)
    return all_bits


def _repr(instance: object, **fields: Dict[str, Any]) -> str:
    """A helper function for repr overloading in classes."""
    attrs = []

    for key, field in fields.items():
        attrs.append(f'{key}={field!r}')

    if len(attrs) == 0:
        return f"<{instance.__class__.__name__}" \
               f"{hex(id(instance))}({','.join(attrs)})>"
    else:
        return f"<{instance.__class__.__name__} {hex(id(instance))}>"
