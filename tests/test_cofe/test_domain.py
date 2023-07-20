from copy import copy, deepcopy
from random import choice

import pytest
from pymatgen.core import Composition, DummySpecies, Element, Species

from smol.cofe.space.domain import (
    SiteSpace,
    Vacancy,
    get_allowed_species,
    get_site_spaces,
    get_species,
)
from tests.utils import assert_msonable


@pytest.mark.parametrize("measure", [True, False])
def test_get_site_spaces(structure, measure):
    for space, site in zip(
        get_site_spaces(structure, include_measure=measure), structure
    ):
        if measure:
            assert space.composition == site.species
        else:
            nspec = len(space)
            assert space.composition == Composition(
                {s: 1.0 / nspec for s in site.species}
            )
        if site.species.num_atoms < 0.99:
            assert "Vacancy" in space
            assert list(space.keys())[:-1] == sorted(site.species)
        else:
            assert list(space.keys()) == sorted(site.species)


def test_get_allowed_species(structure):
    for allowed_sps, site in zip(get_allowed_species(structure), structure):
        assert all(s in allowed_sps for s in site.species)
        if site.species.num_atoms < 0.99:  # check there is a vacancy
            assert any(isinstance(s, Vacancy) for s in allowed_sps)


@pytest.mark.parametrize(
    "vacancy", ["vacancy", "Vacancy", "Vac6", "vacaret", Vacancy(), Vacancy("X")]
)
def test_get_specie_vacancy(vacancy):  # smol part
    assert isinstance(get_species(vacancy), Vacancy)


@pytest.mark.parametrize(
    "species",
    [
        "Li",
        "Li+",
        Element("Li"),
        Species("Li", 1),
        DummySpecies(),
        "X",
        1,
        ("Li+", "Mn2+"),
    ],
)
def test_get_specie_others(species):  # pymatgen part
    sp = get_species(species)
    if isinstance(species, (list, tuple)):
        assert all(isinstance(s, Species) or isinstance(s, Element) for s in sp)
    else:
        assert isinstance(sp, Species) or isinstance(sp, Element)


def test_vacancy():
    dummy = DummySpecies("X")
    vacancy = Vacancy("X")
    assert hash(dummy) != hash(vacancy)
    assert dummy.symbol == vacancy.symbol
    assert vacancy == Vacancy("X")
    assert vacancy != Vacancy("A")
    assert vacancy != dummy
    assert vacancy != Species("Li")
    assert vacancy != 5.67
    _ = str(vacancy)
    _ = repr(vacancy)
    assert deepcopy(vacancy).as_dict() == vacancy.as_dict()
    assert copy(vacancy).as_dict() == vacancy.as_dict()


def test_site_space(structure):
    spaces = get_site_spaces(structure, include_measure=True)
    # Check creating a set works
    unique_spaces = []
    for space in spaces:
        if space not in unique_spaces:
            unique_spaces.append(space)
    assert all(s in spaces for s in unique_spaces) and all(
        s in unique_spaces for s in spaces
    )
    assert all(tuple(range(len(space))) == space.codes for space in spaces)
    i = choice(range(len(spaces)))
    _ = str(spaces[i])
    _ = repr(spaces[i])
    for space in spaces:
        assert_msonable(space)
        for specie in space:
            assert space[specie]
            assert space[str(specie)]

    # test a few invalid types
    with pytest.raises(TypeError):
        _ = space[4.56], space[True]


def test_bad_site_space():
    with pytest.raises(ValueError):
        SiteSpace(Composition({"A": 0.5, Vacancy(): 0.1}))
    with pytest.raises(ValueError):
        SiteSpace(Composition({"A": 0.5, "B": 0.6}))
    with pytest.raises(ValueError):
        SiteSpace(Composition({Vacancy(): 0.5, Vacancy(): 0.5}))
