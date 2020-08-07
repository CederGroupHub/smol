"""
Implementation functions to generate lists of site spaces and allowed species
for a given disordered strucuture.

A site space is simply made up of the allowed species and their concentrations
including dummy species such as vacancies. A site spaces makes up the domain
of a site function and many site spaces make up the space of configurations.
"""

__author__ = "Luis Barroso-Luque, Fengyu Xie"

from collections import OrderedDict
from pymatgen.core.periodic_table import DummySpecie


# class CESpecie(Specie)
# Will be defined if we need more specie dimensions in the future.


def get_allowed_species(structure):
    """Get the allowed species for each site in a disoredered structure.

    This will get all the allowed species for each site in a pymatgen structure
    If the site composition does not add to 1, a vacancy DummySpecie will
    be appended to the allowed species at that site.

    Args:
        structure (Structure):
            Structure to determine site spaces from at least some sites should
            be disordered, otherwise there is no point in using this.

    Returns:
        list of Specie: of allowed species for each site
    """
    all_site_spaces = get_site_spaces(structure)
    return [list(site_space.keys()) for site_space in all_site_spaces]


def get_site_spaces(structure, include_measure=False):
    """Get site spaces for each site in a disordered structure.

    Method to obtain the single site spaces for the sites in a structure.
    The single site spaces are represented by the allowed species for each site
    and the measure/concentration for disordered sites.

    Vacancies are included in sites where the site element composition does not
    sum to 1 (i.e. the total occupation is not 1)

    Args:
        structure (Structure):
            Structure to determine site spaces from at least some sites should
            be disordered, otherwise there is no point in using this.
        include_measure (bool): optional
            If true will take the site compositions as the site space measure,
            otherwise a uniform measure is assumed.

    Returns:
        Ordereddict: Of allowed species and their corresponding measure.
    """
    all_site_spaces = []
    for site in structure:
        if site.species.num_atoms > 1:
            raise ValueError(f"The composition in site {site} is greater than "
                             "1. Makes sure compositions are fractional.")
        # sorting is crucial to ensure consistency!
        site_space = OrderedDict((spec, comp) for spec, comp
                                 in sorted(site.species.items()))
        if site.species.num_atoms < 0.99:
            site_space[DummySpecie("_vacancy")] = 1 - site.species.num_atoms
        if not include_measure:  # make uniform if measure not included
            for spec in site_space.keys():
                site_space[spec] = 1.0 / len(site_space)
        all_site_spaces.append(site_space)
    return all_site_spaces
