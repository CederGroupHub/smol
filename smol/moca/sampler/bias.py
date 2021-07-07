"""Bias term definitions for biased sampling techniques."""
from abc import ABC, abstractmethod
import numpy as np

from ..comp_space import get_oxi_state
from smol.utils import derived_class_factory


class MCMCBias(ABC):
    """Base bias term class.

    Attributes:
        sublattices(List[Sublattice]):
            List of sublattices, including all active
            and inactive sites.
    """

    def __init__(self, all_sublattices, *args, **kwargs):
        """Initialize Basebias.

        Args:
            all_sublattices(List[smol.moca.sublattice]):
                List of sublattices, containing species information and site
                indices in sublattice.
                Must be all sublattices, regardless of active or not,
                otherwise charge may not be balanced!!
            args:
                Additional arguments buffer.
            kwargs:
                Additional keyword arguments buffer.
        """
        self.sublattices = all_sublattices

    @abstractmethod
    def compute_bias(self, occupancy):
        """Compute bias from occupancy.

        Args:
            occupancy(np.ndarray):
                Encoded occupancy string.
        Returns:
            Float, bias value.
        """
        return

    @abstractmethod
    def compute_bias_change(self, occupancy, step):
        """Compute bias change from step.

        Args:
            occupancy(np.array):
                Encoded occupancy array.
            step(List[tuple(int,int)]):
                Step returned by MCUsher.
        Return:
            Float, change of bias value after step.
        """
        return


class Nullbias(MCMCBias):
    """Null bias, always 0."""

    def __init__(self, all_sublattices, *args, **kwargs):
        """Initialize Nullbias.

        Args:
            all_sublattices(List[smol.moca.sublattice]):
                List of sublattices, containing species information and site
                indices in sublattice.
                Must be all sublattices, regardless of active or not,
                otherwise charge may not be balanced!!
            args:
                Additional arguments buffer.
            kwargs:
                Additional keyword arguments buffer.
        """
        super().__init__(all_sublattices, *args, **kwargs)

    def compute_bias(self, occupancy):
        """Compute bias from occupancy.

        Args:
            occupancy(np.ndarray):
                Encoded occupancy string.
        Returns:
            Float, bias value.
        """
        return 0

    def compute_bias_change(self, occupancy, step):
        """Compute bias change from step.

        Args:
            occupancy(np.array):
                Encoded occupancy array.
            step(List[tuple(int,int)]):
                Step returned by MCUsher.
        Return:
            Float, change of bias value after step.
        """
        return 0


class Squarechargebias(MCMCBias):
    """Square charge bias term class, lam * C^2."""

    def __init__(self, all_sublattices, lam=0.5, *args, **kwargs):
        """Initialize Squarechargebias.

        Args:
            all_sublattices(List[smol.moca.sublattice]):
                List of sublattices, containing species information and site
                indices in sublattice.
                Must be all sublattices, regardless of active or not,
                otherwise charge may not be balanced!!
            lam(Float, optional):
                Lam value in bias term. Should be positive. Default to 0.5.
        """
        super().__init__(all_sublattices, *args, **kwargs)
        self.lam = lam

        self._charge_table = self._build_charge_table()

    def _build_charge_table(self):
        """Build array containing charge of species on each site.

        Rows reperesent sites and columns represent species. Allows
        quick evaluation of charge and charge change from steps.
        """
        num_cols = max(len(s.site_space) for s in self.sublattices)
        num_rows = sum(len(s.sites) for s in self.sublattices)

        table = np.zeros((num_rows, num_cols))
        for s in self.sublattices:
            ordered_cs = [get_oxi_state(sp) for sp in s.site_space]
            table[s.sites, :len(ordered_cs)] = ordered_cs
        return table

    def _get_charge(self, occupancy):
        """Compute charge from occupancy."""
        occu = np.array(occupancy, dtype=int)
        ids = np.arange(len(occupancy), dtype=int)
        return np.sum(self._charge_table[ids, occu])

    def compute_bias(self, occupancy):
        """Compute bias from occupancy.

        Args:
            occupancy(np.ndarray):
                Encoded occupancy string.
        Returns:
            Float, bias value.
        """
        return (self.lam * self._get_charge(occupancy) ** 2)

    def compute_bias_change(self, occupancy, step):
        """Compute bias change from step.

        Args:
            occupancy(np.array):
                Encoded occupancy array.
            step(List[tuple(int,int)]):
                Step returned by MCUsher.
        Return:
            Float, change of bias value after step.
        """
        if len(step) == 0:
            return 0

        step_arr = np.array(step, dtype=int)
        occu_before = occupancy.copy()
        occu_after = occupancy.copy()
        occu_after[step_arr[:, 0]] = step_arr[:, 1]

        return self.lam * (self._get_charge(occu_after) -
                           self._get_charge(occu_before)) ** 2


class Squarecompconstraintbias(MCMCBias):
    """Square composition deviation bias. lam * sum(||Cx-b||^2)."""

    def __init__(self, all_sublattices, C, b, lam=0.5,
                 *args, **kwargs):
        """Initialize Squarecompconstraintbias.

        Note: This class can also be used to represent the charge balance
              constraint, but will be less interpreble than using
              Squarechargebias. Use this only when you have multiple bias
              to enforce together.

        Args:
            all_sublattices(List[Sublattice]):
                List of sublattices, containing species information and site
                indices in sublattice.
            C(np.ndarray, 2 dimensional), b(np.ndarray):
                In unconstrained composition space, Cx == b defines any
                composition constraint. We will measure |Cx-b|^2 as penalty.
                Note: the x here is unconstrained coordinates, and they are
                      NOT normalized by supercell size!
            lam(float or np.ndarray), optional:
                Penalization factor(s). When using an array, should be the
                same length as b. Must all be positive numbers.
                Default to 0.5 for all bias terms.
        """
        super().__init__(all_sublattices, *args, **kwargs)
        self.C = np.array(C)
        self.b = np.array(b)
        self.bits = [s.species for s in all_sublattices]
        self.sl_list = [s.sites for s in all_sublattices]
        if isinstance(lam, (int, float)):
            self.lams = np.repeat(lam, len(self.b))
        elif len(lam) == len(self.b):
            self.lams = np.array(lam)
        else:
            raise ValueError("Array lambdas provided, but length does not " +
                             "match number of composition constraints.")

    def _compute_x(self, occupancy):
        """Compute unconstrained coordinates from occupancy."""
        occu = np.array(occupancy)
        compstat = [[(occu[self.sl_list[sl_id]] == sp_id).sum()
                    for sp_id, sp in enumerate(sl)]
                    for sl_id, sl in enumerate(self.bits)]
        ucoords = []
        for sl in compstat:
            ucoords.extend(sl[:-1])
        return np.array(ucoords)

    def compute_bias(self, occupancy):
        """Compute composition constraint bias from occupancy.

        Args:
            occupancy(np.ndarray):
                Encoded occupancy string.
        """
        x = self._compute_x(occupancy)
        return np.sum(self.lams * (self.C @ x - self.b) ** 2)

    def compute_bias_change(self, occupancy, step):
        """Compute bias change from step.

        Args:
            occupancy(np.ndarray):
                Encoded occupancy string.
            step(List[tuple(int,int)]):
                Step returned by MCUsher.
        Return:
            Float, change of bias value after step.
        """
        if len(step) == 0:
            return 0

        step = np.array(step, dtype=int)
        occu_now = np.array(occupancy)
        occu_next = np.array(occupancy)
        occu_next[step[:, 0]] = step[:, 1]
        return self.compute_bias(occu_next) - self.compute_bias(occu_now)


def mcbias_factory(bias_type, all_sublattices, *args, **kwargs):
    """Get a MCMC bias from string name.

    Args:
        bias_type (str):
            string specyting bias name to instantiate.
        all_sublattices (list of Sublattice):
            list of Sublattices to calculate bias for.
            Must contain all sublattices, including active
            and inactive, otherwise might be unable to
            calculate some type of bias, for example,
            charge bias.
        *args:
            positional args to instatiate a bias term.
        *kwargs:
            Keyword argument to instantiate a bias term.
    """
    return derived_class_factory(bias_type.capitalize(), MCMCBias,
                                 all_sublattices, *args, **kwargs)
