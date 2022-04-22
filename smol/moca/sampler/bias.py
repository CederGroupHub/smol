"""Bias term definitions for biased sampling techniques.

Bias terms can be added to an MCKernel in order to generate samples that are
biased accordingly.
"""

__author__ = "Fengyu Xie, Luis Barroso-Luque"

from abc import ABC, abstractmethod
from collections import Counter
from math import log

import numpy as np

from smol.cofe.space.domain import get_species
from smol.utils import class_name_from_str, derived_class_factory


class MCBias(ABC):
    """Base bias term class.

    Attributes:
        sublattices (List[Sublattice]):
            list of sublattices with active sites.
    """

    def __init__(self, sublattices, *args, **kwargs):
        """Initialize MCBias.

        Args:
            sublattices (List[Sublattice]):
                list of active sublattices, containing species information and
                site indices in sublattice.
            args:
                Additional arguments buffer.
            kwargs:
                Additional keyword arguments buffer.
        """
        self.sublattices = sublattices
        self.active_sublattices = [
            sublatt for sublatt in self.sublattices if sublatt.is_active
        ]

    @abstractmethod
    def compute_bias(self, occupancy):
        """Compute bias from occupancy.

        Args:
            occupancy(np.ndarray):
                encoded occupancy string.
        Returns:
            Float, bias value.
        """
        return

    @abstractmethod
    def compute_bias_change(self, occupancy, step):
        """Compute bias change from step.

        The returned value needs to be the difference of bias logs,
        log(bias_f) - log(bias_i), when the bias terms would directly multiply
        the ensemble probability (i.e. exp(-beta * E) * bias).

        Args:
            occupancy: (ndarray):
                encoded occupancy array.
            step: (List[tuple(int,int)]):
                step returned by MCUsher.
        Return:
            Float, change of bias value after step.
        """
        return


class FugacityBias(MCBias):
    """Fugacity fraction bias.

    This bias corresponds directly to using a composition bias. Using this
    with a CanonicalEnsemble keeps fugacity fractions constant, which
    implicitly sets the chemical potentials, albeit for a specific temperature.
    Since one species per sublattice is the reference species, to calculate
    actual fugacities, the reference fugacity must be computed as an ensemble
    average and all other fugacities can then be calculated.
    From the fugacities and the set temperature, the corresponding chemical
    potentials can then be calculated.
    """

    def __init__(self, sublattices, fugacity_fractions=None):
        """Initialize fugacity ratio bias.

        Args:
            sublattices (List[Sublattice]):
                list of active sublattices, containing species information and
                site indices in sublattice.
            fugacity_fractions (sequence of dicts): optional
                Dictionary of species name and fugacity fraction for each
                sublattice (.i.e think of it as the sublattice concentrations
                for random structure). If not given this will be taken from the
                prim structure used in the cluster subspace. Needs to be in
                the same order as the corresponding sublattice.
        """
        super().__init__(sublattices)
        self._fus = None
        self._fu_table = None
        # Consider only species on active sub-lattices
        self._species = [
            set(sublatt.site_space.keys()) for sublatt in self.active_sublattices
        ]

        if fugacity_fractions is not None:
            # check that species are valid
            fugacity_fractions = [
                {get_species(k): v for k, v in sub.items()}
                for sub in fugacity_fractions
            ]
        else:
            fugacity_fractions = [
                dict(sublatt.site_space) for sublatt in self.active_sublattices
            ]
        self.fugacity_fractions = fugacity_fractions

    @property
    def fugacity_fractions(self):
        """Get the fugacity fractions for species on active sublatts."""
        return self._fus

    @fugacity_fractions.setter
    def fugacity_fractions(self, value):
        """Set the fugacity fractions on active sublatts and update table."""
        for sub in value:
            for spec, count in Counter(map(get_species, sub.keys())).items():
                if count > 1:
                    raise ValueError(
                        f"{count} values of the fugacity for the same "
                        f"species {spec} were provided.\n Make sure the "
                        f"dictionaries you are using have only "
                        f"string keys or only Species objects as keys."
                    )

        value = [{get_species(k): v for k, v in sub.items()} for sub in value]
        if not all(sum(fus.values()) == 1 for fus in value):
            raise ValueError("Fugacity ratios must add to one.")
        for (spec, vals) in zip(self._species, value):
            if spec != set(vals.keys()):
                raise ValueError(
                    f"Fugacity fractions given are missing or not valid "
                    f"species.\n"
                    f"Values must be given for each  of the following: "
                    f"{self._species}"
                )
        self._fus = value
        self._fu_table = self._build_fu_table(value)

    def compute_bias(self, occupancy):
        """Compute bias from occupancy.

        Args:
            occupancy(np.ndarray):
                Encoded occupancy string.
        Returns:
            Float, bias value.
        """
        return sum(
            log(self._fu_table[site][species]) for site, species in enumerate(occupancy)
        )

    def compute_bias_change(self, occupancy, step):
        """Compute bias change from step.

        The returned value needs to be the difference of bias logs,
        log(bias_f) - log(bias_i), when the bias terms would directly multiply
        the ensemble probability (i.e. exp(-beta * E) * bias).

        Args:
            occupancy: (ndarray):
                encoded occupancy array.
            step: (List[tuple(int,int)]):
                step returned by MCUsher.
        Return:
            float, change of bias value after step.
        """
        delta_log_fu = sum(
            log(self._fu_table[f[0]][f[1]] / self._fu_table[f[0]][occupancy[f[0]]])
            for f in step
        )
        return delta_log_fu

    def _build_fu_table(self, fugacity_fractions):
        """Build an array for fugacity fractions for all sites in system.

        Rows represent sites and columns species. This allows quick evaluation
        of fugacity fraction changes from flips. Not that the total number
        of columns will be the number of species in the largest site space. For
        smaller site spaces the values at those rows are meaningless and will
        be given values of 1. Also rows representing sites with not partial
        occupancy will have all 1 values and should never be used.
        """
        num_cols = max(max(sublatt.encoding) for sublatt in self.sublattices) + 1
        # Sublattice can only be initialized as default, or splitted from default.
        num_rows = sum(len(sl.sites) for sl in self.sublattices)
        table = np.ones((num_rows, num_cols))
        for fus, sublatt in zip(fugacity_fractions, self.active_sublattices):
            ordered_fus = np.array([fus[sp] for sp in sublatt.site_space])
            table[sublatt.sites[:, None], sublatt.encoding] = ordered_fus[None, :]
        return table


def mcbias_factory(bias_type, sublattices, *args, **kwargs):
    """Get a MCMC bias from string name.

    Args:
        bias_type (str):
            string specifying bias name to instantiate.
        sublattices (List[Sublattice]):
            list of active sublattices, containing species information and
            site indices in sublattice.
        *args:
            positional args to instatiate a bias term.
        *kwargs:
            keyword argument to instantiate a bias term.
    """
    if "bias" not in bias_type and "Bias" not in bias_type:
        bias_type += "-bias"
    bias_name = class_name_from_str(bias_type)
    return derived_class_factory(bias_name, MCBias, sublattices, *args, **kwargs)
