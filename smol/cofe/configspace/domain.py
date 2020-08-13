"""Functions to obtain site spaces and allowed species.

Implementation functions to generate lists of site spaces and allowed species
for a given disordered strucuture.

A site space is simply made up of the allowed species and their concentrations
including dummy species such as vacancies. A site spaces makes up the domain
of a site function and many site spaces make up the space of configurations.
"""

__author__ = "Luis Barroso-Luque, Fengyu Xie"

from collections import OrderedDict
from pymatgen.core.periodic_table import DummySpecie, get_el_sp


def get_allowed_species(structure):
    """Get the allowed species for each site in a disordered structure.

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
            site_space[Vacancy()] = 1 - site.species.num_atoms
        if not include_measure:  # make uniform if measure not included
            for spec in site_space.keys():
                site_space[spec] = 1.0 / len(site_space)
        all_site_spaces.append(site_space)
    return all_site_spaces


def get_specie(obj):
    """Method wrapping pymatgen.core.periodic_table.get_el_sp to catch and
    return Vacancies when desired.

    Args:
        obj (Element/Specie/str/int):
            An arbitrary object.  Supported objects
            are actual Element/Specie objects, integers (representing atomic
            numbers) or strings (element symbols or species strings).

    Returns:
        Specie, Element or Vacancy, with a bias for the maximum number of
        properties that can be determined.
    Raises:
        ValueError if obj cannot be converted into an Element or Specie.
    """
    if isinstance(obj, Vacancy):
        return obj

    if isinstance(obj, (list, tuple)):
        return [get_specie(o) for o in obj]

    if isinstance(obj, str) and 'Vac' in obj.capitalize():
        return Vacancy()

    return get_el_sp(obj)


class Vacancy(DummySpecie):
    """Wrapper class around DummySpecie to treat vacancies as their own."""

    def __init__(self, symbol: str = "A", oxidation_state: float = 0,
                 properties: dict = None):
        """
        Args:
            symbol (str): An assigned symbol for the vacancy. Strict
                rules are applied to the choice of the symbol. The vacancy
                symbol cannot have any part of first two letters that will
                constitute an Element symbol. Otherwise, a composition may
                be parsed wrongly. E.g., "X" is fine, but "Vac" is not
                because Vac contains V, a valid Element.
            oxidation_state (float): Oxidation state for Vacancy. More like
                the charge of a point defect. Defaults to zero.
        """
        super().__init__(symbol=symbol, oxidation_state=oxidation_state,
                         properties=properties)

    def __eq__(self, other):
        """Test equality, only equal if isinstance"""
        if not isinstance(other, Vacancy):
            return False
        else:
            return super().__eq__(other)

    def __hash__(self):
        """Unique hash, append an underscore to avoid clash with Dummy."""
        # this should prevent any hash collisions with other Specie instances
        # since the symbol for a DummySpecie can not start with "v"
        return hash("v" + self.symbol)

    def __str__(self):
        """Get string representation, add a v to differentiate."""
        return "v" + super().__str__()

    def __repr__(self):
        """Get an explicit representation."""
        return "Vacancy " + self.__str__()
