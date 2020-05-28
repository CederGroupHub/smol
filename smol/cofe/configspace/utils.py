"""A few random utilities that have no place to go?"""

__author__ = "Luis Barroso-Luque"

from typing import Dict, Any
from collections import OrderedDict

SITE_TOL = 1e-6


def get_site_domains(structure, include_measure=False):
    """
    Helper method to obtain the single site domains for the sites in a
    structure. The single site domains are represented by the allowed species
    for each site (with an optional measure/concentration for disordered sites)
    Vacancies are included in sites where the site element composition is not
    1 (i.e. the total occupation is not 1)

    Args:
        structure (Structure):
            Structure to determine site domains from at least some sites should
            be disordered, otherwise there is no point in using this.
        include_measure (bool): (optional)
             To include the site element compositions as the domain measure.

    Returns:
        list: Of allowed species for each site if include_measure is False
        Ordereddict: Of allowed species and their measure for each site if
            include_measure is True
    """
    all_site_domains = []
    for site in structure:
        # sorting is crucial to ensure consistency!
        if include_measure:
            site_domain = OrderedDict((str(sp), c) for sp, c
                                      in sorted(site.species.items()))
            if site.species.num_atoms < 0.99:
                site_domain["Vacancy"] = 1 - site.species.num_atoms
        else:
            site_domain = [str(sp) for sp in sorted(site.species.keys())]
            if site.species.num_atoms < 0.99:
                site_domain.append("Vacancy")
        all_site_domains.append(site_domain)
    return all_site_domains


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
