"""Implementation of SiteSpace and functions to get spaces and allowed species.

Implementation functions to generate lists of site spaces and allowed species
for a given disordered structure.

A site space is made up of the allowed species and their concentrations
including special species such as vacancies. A site spaces makes up the domain
of a site function and many site spaces make up the space of configurations.
"""

from __future__ import annotations

__author__ = "Luis Barroso-Luque, Fengyu Xie"

from collections import OrderedDict
from collections.abc import Hashable, Mapping

from monty.json import MSONable
from pymatgen.core import Composition
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.util.string import formula_double_format


def get_allowed_species(structure) -> list[Species]:
    """Get the allowed species for each site in a disordered structure.

    This will get all the allowed species for each site in a pymatgen structure.
    If the site composition does not add to 1, a Vacancy will be appended to
    the allowed species at that site.

    Same as in a site space, the order of the allowed species at each site is
    important because that implicitly represents the code that will be used
    when evaluating site basis functions.

    Args:
        structure (Structure):
            Structure to determine site spaces from. At least some sites should
            be disordered, otherwise there is no point in using this.

    Returns:
        list of Specie: list of allowed Species for each site
    """
    all_site_spaces = get_site_spaces(structure)
    return [list(site_space.keys()) for site_space in all_site_spaces]


def get_site_spaces(structure, include_measure=False) -> list[SiteSpace]:
    """Get site spaces for each site in a disordered structure.

    Method to obtain the single site spaces for the sites in a structure.
    The single site spaces are represented by the allowed species for each site
    and the measure/concentration for disordered sites. The order is important so
    the species need to be sorted, since that implicitly sets the code/index
    used to evaluate basis functions.

    Vacancies are included in sites where the site element composition does not
    sum to 1 (i.e. the total occupation is not 1).

    Args:
        structure (Structure):
            Structure to determine site spaces from. At least some sites should
            be disordered, otherwise there is no point in using this.
        include_measure (bool): optional
            if True, will take the site compositions as the site space measure,
            otherwise a uniform measure is assumed.

    Returns:
        SiteSpace: Allowed species and their corresponding measure.
    """
    all_site_spaces = []
    for site in structure:
        if include_measure:
            site_space = SiteSpace(site.species)
        else:  # make comp uniform if measure not included
            num_species = len(site.species)
            if site.species.num_atoms < 0.99:
                num_species += 1
            site_space = SiteSpace(
                Composition({sp: 1.0 / num_species for sp in site.species.keys()})
            )
        all_site_spaces.append(site_space)
    return all_site_spaces


def get_species(obj) -> Species | Element | Vacancy | DummySpecies:
    """Get specie object.

    Wraps pymatgen.core.periodic_table.get_el_sp to be able to catch and
    return Vacancies when needed.

    Args:
        obj (Element/Specie/str/int):
            an arbitrary object.  Supported objects
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
        return [get_species(o) for o in obj]

    if isinstance(obj, str) and "vac" in obj.lower():
        return Vacancy()

    return get_el_sp(obj)


class SiteSpace(Mapping, Hashable, MSONable):
    """Represents a site space.

    A site spaces is a set of allowed species at a site and their corresponding
    measure (composition in the disordered state).

    This class is somewhat similar to pymatgen Composition. However a SiteSpace
    must always have a total composition = 1 since vacancies are explicitly
    included as species. A sorted order of the included species is also kept to
    ensure a consistent encoding when creating basis functions. Vacancies are
    always the last species. Key errors are raised for species not in the
    site space.
    """

    def __init__(self, composition):
        """Initialize a site space.

        Args:
            composition (Composition):
                Composition object that specifies the site space.
        """
        # num_atoms = 0 identified as vacancy-only site-space.
        if composition.num_atoms < 0 or composition.num_atoms > 1:
            raise ValueError(
                "Number of atoms in Composition must be in [0, 1]"
                f" got {composition.num_atoms}!"
            )

        if any(isinstance(sp, Vacancy) for sp in composition):
            if composition.num_atoms != 1:
                raise ValueError(
                    f"Composition {composition} has a Vacancy but"
                    " number of atoms is not 1."
                )

            if sum(isinstance(sp, Vacancy) for sp in composition) > 1:
                raise ValueError(
                    "There are multiple vacancies in this composition "
                    f"{composition}."
                )

        self._composition = composition
        self._data = OrderedDict(
            (spec, comp) for spec, comp in sorted(composition.items())
        )
        if composition.num_atoms < 0.99:
            self._data[Vacancy()] = 1 - composition.num_atoms

    @property
    def composition(self):
        """Return the underlying composition."""
        return self._composition

    @property
    def codes(self):
        """Return range of species (not necessarily sub-lattice encodings)."""
        return tuple(range(len(self)))

    def __getitem__(self, item):
        """Get a specie in the sitespace."""
        try:
            specie = get_species(item)
            return self._data[specie]
        except ValueError as exp:
            raise TypeError(
                f"Invalid key {item}, {type(item)} for Composition"
                f".\nValueError exception:\n{exp}"
            ) from exp

    def __len__(self):
        """Get number os species."""
        return len(self._data)

    def __iter__(self):
        """Return OrderedDict iter."""
        return self._data.keys().__iter__()

    def __eq__(self, other):
        """Compare equality, requires same order."""
        if not isinstance(other, SiteSpace):
            return False

        return all(
            item1 == item2 for item1, item2 in zip(self._data.items(), other.items())
        )

    def __lt__(self, other):
        """Compare order of two SiteSpaces."""
        return list(self._data.keys()) < list(other._data.keys())

    def __hash__(self):
        """Take the hash of the composition for now."""
        return hash(self.composition)

    def __str__(self):
        """Get pretty string."""
        return "\n".join(["Site Space", repr(self)])

    def __repr__(self):
        """Get repr."""
        out = str(self._composition) + " "
        if len(self._data) > len(self._composition):
            items = list(self._data.items())[len(self._composition) :]
            out += " ".join(
                [
                    f"{str(k)}{formula_double_format(v, ignore_ones=False)}"
                    for k, v in items
                ]
            )
        return out

    def as_dict(self) -> dict:
        """Get MSONable dict representation."""
        site_space_d = super().as_dict()
        site_space_d["composition"] = self._composition.as_dict()
        return site_space_d

    @classmethod
    def from_dict(cls, d) -> SiteSpace:
        """Create a SiteSpace from dict representation."""
        return cls(Composition.from_dict(d["composition"]))


class Vacancy(DummySpecies):
    """Wrapper class around DummySpecie to treat vacancies as their own species."""

    def __init__(
        self,
        symbol: str = "A",
        oxidation_state: float = 0,
        properties: dict | None = None,
    ):
        """Initialize a Vacancy.

        Args:
            symbol (str): an assigned symbol for the vacancy. Strict
                rules are applied to the choice of the symbol. The vacancy
                symbol cannot have any part of first two letters that will
                constitute an Element symbol. Otherwise, a composition may
                be parsed wrongly. E.g., "X" is fine, but "Vac" is not
                because Vac contains V, a valid Element.
            oxidation_state (float): oxidation state for Vacancy. More like
                the charge of a point defect. Defaults to zero.
            properties (dict): Deprecated. See pymatgen 2023.6.28 release notes.
        """
        super().__init__(
            symbol=symbol, oxidation_state=oxidation_state, properties=properties
        )

    def __eq__(self, other):
        """Test equality, only equal if isinstance."""
        return False if not isinstance(other, Vacancy) else super().__eq__(other)

    def __hash__(self):
        """Get hash, append an underscore to avoid clash with Dummy."""
        # this should prevent any hash collisions with other Specie instances
        # since the symbol for a DummySpecie can not start with "v"
        return hash("v" + self.symbol)

    def __str__(self):
        """Get string representation, add a v to differentiate."""
        return "vac" + super().__str__()

    def __repr__(self):
        """Get an explicit representation."""
        return "Vacancy " + self.__str__()

    def __copy__(self):
        """Copy the vacancy object."""
        return Vacancy(self.symbol, self._oxi_state)

    def __deepcopy__(self, memo):
        """Deepcopy the vacancy object."""
        return Vacancy(self.symbol, self._oxi_state)
