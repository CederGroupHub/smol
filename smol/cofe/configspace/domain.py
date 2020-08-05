"""
Implementation of the SiteSpace and functions to generate site spaces for a
given disordered strucuture.

A site space is simply made up of the allowed species and their concentrations
including dummy species such as vacancies. A site spaces makes up the domain
of a site function and many site spaces make up the space of configurations.
"""

from collections import OrderedDict


def get_allowed_species(structure):
    """Get the allowed species for each site in a disoredered structure.

    Method to obtain the single site spaces for the sites in a structure.
    The single site spaces are represented by the allowed species for each site
    and the measure/concentration for disordered sites.

    Vacancies are included in sites where the site element composition does not
    sum to 1 (i.e. the total occupation is not 1)

    Args:
        structure (Structure):
            Structure to determine site spaces from at least some sites should
            be disordered, otherwise there is no point in using this.

    Returns:
        list: Of allowed species for each site
    """
    all_allowed_species = []
    for site in structure:
        # sorting is crucial to ensure consistency!
        allowed_species = [str(sp) for sp in sorted(site.species.keys())]
        if site.species.num_atoms < 0.99:
            allowed_species.append("Vacancy")
        all_allowed_species.append(allowed_species)
    return all_allowed_species


def get_site_spaces(structure):
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

    Returns:
        Ordereddict: Of allowed species and their corresponding measure.
    """
    all_site_spaces = []
    for site in structure:
        # sorting is crucial to ensure consistency!
        site_space = OrderedDict((str(sp), c) for sp, c
                                 in sorted(site.species.items()))
        if site.species.num_atoms < 0.99:
            site_space["Vacancy"] = 1 - site.species.num_atoms
        all_site_spaces.append(site_space)
    return all_site_spaces
