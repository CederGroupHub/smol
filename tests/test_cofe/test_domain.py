import pytest
import numpy as np
import numpy.testing as npt
from tests.utils import assert_msonable

from smol.cofe.configspace.domain import (get_specie, get_allowed_species,
                                          get_site_spaces, SiteSpace, Vacancy)

