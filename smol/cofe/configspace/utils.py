"""
A lot of random utilies that have no place to go :(
"""

from typing import Dict, Any

SITE_TOL = 1e-6


def get_bits(structure):
    """
    Helper method to compute list of species on each site.
    Includes vacancies
    """
    all_bits = []
    for site in structure:
        bits = []
        for sp in sorted(site.species.keys()):
            bits.append(str(sp))
        if site.species.num_atoms < 0.99:
            bits.append("Vacancy")
        all_bits.append(bits)
    return all_bits


def _repr(object: object, **fields: Dict[str, Any]) -> str:
    """
    A helper function for repr overloading in classes
    """
    attrs = []

    for key, field in fields.items():
        attrs.append(f'{key}={field!r}')

    if len(attrs) == 0:
        return f"<{object.__class__.__name__}" \
               f"{hex(id(object))}({','.join(attrs)})>"
    else:
        return f"<{object.__class__.__name__} {hex(id(object))}>"
