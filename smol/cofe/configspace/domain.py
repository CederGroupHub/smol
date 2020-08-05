"""
Implementation of the SiteSpace and functions to generate site spaces for a
given disordered strucuture.

A site space is simply made up of the allowed species and their concentrations
including dummy species such as vacancies. A site spaces makes up the domain
of a site function and many site spaces make up the space of configurations.
"""
__author__ = "Luis Barroso-Luque, Fengyu Xie"

from collections import OrderedDict
from pymatgen.core.periodic_table import Specie, DummySpecie


# class CESpecie(Specie)
# Will be defined if we need more specie dimensions in the future.


def get_site_space(composition, include_measure=False):
    """
    Inputs:
        composition: pymatgen.core.composition, must be
                     fractional composition.
        include_measure: Boolean. If ture, will keep
                     concentration values as a measure
                     in concentration dependent basis;
                     otherwise, all species will be given
                     equal weights.
    Output:
        site_space:
            OrderedDict: {Specie: measure}
    """
    if composition._natoms > 1:
        raise ValueError("Given measures does not sum to 1! Are you sure you are \
                          using a fractional composition?")

    site_space = OrderedDict(sorted(composition.items()))
    if composition._natoms < 0.99:
        vac = DummySpecie('_Vacancy')
        site_space[vac] = 1 - composition._natoms

    if not include_measure:
        d = len(site_space)
        for sp in site_space:
            site_space[sp] = float(1) / d

    return site_space


def get_site_spaces_from_structure(structure, include_measure=False):
    """Get site domains for sites in a disordered structure, from a pymatgen.structure
    object.

    Not recommended, because this type of domains will only contain elemental and charge
    information, and no additional information such as magentization can be included.
    We recommend users to initialize a cluster expansion from lattice, site_coords, custom
    site_spaces, and sublattice list.

    Vacancies are included in sites where the site element composition does not
    sum to 1 (i.e. the total occupation is not 1)

    Args:
        structure (Structure):
            Structure to determine site domains from at least some sites should
            be disordered, otherwise there is no point in using this.
        include_measure (bool): (optional)
             To include the site element compositions as the domain measure.

    Returns:
        List of site_space, each site_space is an instance of SiteSpace class.
    """
    return [get_site_space(st.species, include_measure=include_measure) \
            for st in structure]



