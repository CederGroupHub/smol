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
from smol.moca.composition import get_oxi_state
from smol.moca.metadata import Metadata
from smol.moca.occu_utils import get_dim_ids_table, occu_to_counts
from smol.utils.class_utils import class_name_from_str, derived_class_factory


class MCBias(ABC):
    """Base bias term class.

    Note: Any MCBias should be implemented as beta * E - bias
    will be minimized in thermodynamics kernel.

    Attributes:
        sublattices (List[Sublattice]):
            list of sublattices with active sites.
    """

    def __init__(self, sublattices, rng=None, *args, **kwargs):
        """Initialize MCBias.

        Args:
            sublattices (List[Sublattice]):
                list of active sublattices, containing species information and
                site indices in sublattice.
            rng (np.Generator): optional
                The given PRNG must be the same instance as that used by the kernel and
                any bias terms, otherwise reproducibility will be compromised.
            args:
                Additional arguments buffer.
            kwargs:
                Additional keyword arguments buffer.
        """
        self.sublattices = sublattices
        self.active_sublattices = [
            sublatt for sublatt in self.sublattices if sublatt.is_active
        ]
        self._rng = np.random.default_rng(rng)

        self.spec = Metadata(
            type=self.__class__.__name__,
            sublattices=[
                sublatt.site_space.as_dict()["composition"]
                for sublatt in self.sublattices
            ],
        )

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

    def compute_bias_change(self, occupancy, step):
        """Compute bias change from step.

        Args:
            occupancy: (ndarray):
                encoded occupancy array.
            step: (List[tuple(int,int)]):
                step returned by MCUsher.
        Return:
            Float, change of bias value after step.
        """
        occu_next = occupancy.copy()
        for site, code in step:
            occu_next[site] = code
        return self.compute_bias(occu_next) - self.compute_bias(occupancy)


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

    def __init__(self, sublattices, fugacity_fractions=None, **kwargs):
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
        kwargs:
            Keyword arguments for initializing MCUsher.
        """
        super().__init__(sublattices, **kwargs)
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

        # update spec
        self.spec.fugacity_fractions = fugacity_fractions

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
        for spec, vals in zip(self._species, value):
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
            log(self._fu_table[site, species]) for site, species in enumerate(occupancy)
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
        # if a site is flipped twice in a step only use the last flip
        steps = {site: code for site, code in step}
        delta_log_fu = sum(
            log(self._fu_table[site, code] / self._fu_table[site, occupancy[site]])
            for site, code in steps.items()
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
        # Sublattice can only be initialized as default, or split from default.
        num_rows = sum(len(sl.sites) for sl in self.sublattices)
        table = np.ones((num_rows, num_cols))
        for fus, sublatt in zip(fugacity_fractions, self.active_sublattices):
            ordered_fus = np.array([fus[sp] for sp in sublatt.site_space])
            table[sublatt.sites[:, None], sublatt.encoding] = ordered_fus[None, :]
        return table


class SquareChargeBias(MCBias):
    """Square charge bias.

    This bias penalizes energy on square of the system net charge.
    """

    def __init__(self, sublattices, penalty=0.5, **kwargs):
        """Square charge bias.

        Args:
            sublattices (List[Sublattice]):
                List of active sublattices, containing species information and
                site indices in sublattice. Must include all sites!
            penalty (float): optional
                Penalty factor. energy/kT will be penalized by adding penalty
                * charge**2.
                Must be positive. Default to 0.5, which works
                for most of the cases.
        kwargs:
            Keyword arguments for initializing MCUsher.
        """
        super().__init__(sublattices, **kwargs)
        charges = [
            [get_oxi_state(sp) for sp in sublatt.species]
            for sublatt in self.sublattices
        ]
        if penalty <= 0:
            raise ValueError("Penalty factor should be > 0!")
        self.penalty = penalty
        num_cols = max(max(sl.encoding) for sl in self.sublattices) + 1
        num_rows = sum(len(sl.sites) for sl in self.sublattices)
        table = np.zeros((num_rows, num_cols))
        for cs, sublatt in zip(charges, self.sublattices):
            cs = np.array(cs)
            table[sublatt.sites[:, None], sublatt.encoding] = cs[None, :]
        self._c_table = table

        # record specifications
        self.spec.penalty = penalty

    def compute_bias(self, occupancy):
        """Compute bias from occupancy.

        Args:
            occupancy(np.ndarray):
                Encoded occupancy string.
        Returns:
            Float, bias value.
        """
        c = np.sum(self._c_table[np.arange(len(occupancy), dtype=int), occupancy])
        # Returns a negative value because of the implementation in mckernels.
        return -self.penalty * c**2


class SquareHyperplaneBias(MCBias):
    """Square hyperplane bias.

    This bias penalizes energy on sum square of distance from a composition n
    ("counts" format) to hyperplanes A n = b (in the unconstrained composition space,
    see CompositionSpace document). In other words, this bias penalizes the
    composition's deviation from the constraints in A n = b.
    """

    def __init__(
        self,
        sublattices,
        hyperplane_normals,
        hyperplane_intercepts,
        penalty=0.5,
        **kwargs,
    ):
        """Square composition bias.

        Use this when you have other constraints to the composition
        than the charge constraint.
        Args:
           sublattices (List[Sublattice]):
                List of active sublattices, containing species information and
                site indices in sublattice. Must include all sites!
           hyperplane_normals (2D ArrayLike):
                Normal vectors of hyperplanes, each in a row. (The matrix A.)
           hyperplane_intercepts (1D ArrayLike):
                Intercepts of each hyperplane. (The vector b.)
           The matrix A and the vector b together forms a set of
           constraint hyperplanes: A n = b. (Per-super-cell, not per-primitive cell).
           penalty (float): optional
                Penalty factor. energy/kT will be penalized by adding penalty
                * ||A n - b||**2. Must be positive.
                Default to 0.5, which works for most of the cases.
        kwargs:
            Keyword arguments for initializing MCUsher.
        """
        super().__init__(sublattices, **kwargs)
        if penalty <= 0:
            raise ValueError("Penalty factor should be > 0!")
        self.penalty = penalty
        self._A = np.array(hyperplane_normals, dtype=int)
        self._b = np.array(hyperplane_intercepts, dtype=int)
        self._dim_ids_table = get_dim_ids_table(self.sublattices)
        self.d = sum(len(sublatt.species) for sublatt in sublattices)

        # record specifications
        self.spec.penalty = penalty
        self.spec.hyperplane_normals = self._A.tolist()
        self.spec.hyperplane_intercepts = self._b.tolist()

    def compute_bias(self, occupancy):
        """Compute bias from occupancy.

        Args:
            occupancy(np.ndarray):
                Encoded occupancy string.
        Returns:
            Float, bias value.
        """
        n = occu_to_counts(occupancy, self.d, self._dim_ids_table)
        return -self.penalty * np.sum((self._A @ n - self._b) ** 2)


def mcbias_factory(bias_type, sublattices, *args, **kwargs):
    """Get a MCMC bias from string name.

    Args:
        bias_type (str):
            string specifying bias name to instantiate.
        sublattices (List[Sublattice]):
            list of active sublattices, containing species information and
            site indices in sublattice.
        *args:
            positional args to instantiate a bias term.
        *kwargs:
            keyword argument to instantiate a bias term.
    """
    if "bias" not in bias_type and "Bias" not in bias_type:
        bias_type += "-bias"
    bias_name = class_name_from_str(bias_type)
    return derived_class_factory(bias_name, MCBias, sublattices, *args, **kwargs)
