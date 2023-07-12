"""
A Wang-Landau kernel for multicanonical estimation of the density of states.

Based on: https://link.aps.org/doi/10.1103/PhysRevLett.86.2050
"""

__author__ = "Luis Barroso-Luque, Fengyu Xie"

from functools import partial
from math import log

import numpy as np

from smol.moca.kernel.base import ALL_MCUSHERS, MCKernel


class WangLandau(MCKernel):
    """A Wang-Landau sampling kernel.

    Allows wang-landau sampling method. Stores histogram and DOS in its instance.
    """

    valid_mcushers = ALL_MCUSHERS
    valid_bias = None  # Wang-Landau does not need bias.

    def __init__(
        self,
        ensemble,
        step_type,
        min_enthalpy,
        max_enthalpy,
        bin_size,
        *args,
        flatness=0.8,
        mod_factor=1.0,
        check_period=1000,
        update_period=1,
        mod_update=None,
        seed=None,
        **kwargs,
    ):
        """Initialize a WangLandau Kernel.

        Args:
            ensemble (Ensemble):
                The ensemble object to use to generate samples
            step_type (str):
                An MC step type corresponding to an MCUsher. See valid_mcushers
            min_enthalpy (float):
                The minimum enthalpy to sample. Enthalpy means the dot product
                of natural parameters and ensemble features. In canonical ensemble,
                it shall be the energy. In semigrand ensemble, it shall be E- mu N.
                Unit is eV scaled at the super-cell.
            max_enthalpy (float):
                The maximum enthalpy to sample.
            bin_size (float):
                The enthalpy bin size to determine different states.
            flatness (float): optional
                The flatness factor used when checking histogram flatness.
                Must be between 0 and 1.
            mod_factor (float):
                log of the initial modification factor used to update the
                DOS/entropy.
                Default is 1, which means the DOS shall be multiplied by e^1.
            check_period (int): optional
                The period in number of steps for the histogram flatness to be
                checked.
            update_period (int): optional
                The period in number of steps to update the histogram and entropy.
                In some cases periods slightly larger than each step, may allow
                more efficient exploration of energy space.
            mod_update (float|Callable): optional
                A function used to update the fill factor when the histogram
                satisfies the flatness criteria. It must monotonically decrease
                to 0.
                When a number is provided, at each update we will divide the
                mod factor by this number.
            seed (int): optional
                non-negative integer to seed the PRNG
            args:
                Positional arguments to instantiate MCUsher.
            kwargs:
                Keyword arguments to instantiate MCUsher.
        """
        if min_enthalpy > max_enthalpy:
            raise ValueError("min_enthalpy can not be larger than max_enthalpy.")
        if (max_enthalpy - min_enthalpy) / bin_size <= 1:
            raise ValueError(
                "The values provided for min and max enthalpy and bin sizer result in a single bin!"
            )
        if mod_factor <= 0:
            raise ValueError("mod_factor must be greater than 0.")

        self.flatness = flatness
        self.check_period = check_period
        self.update_period = update_period
        self._m = mod_factor
        self._window = (min_enthalpy, max_enthalpy, bin_size)

        if callable(mod_update):
            self._mod_update = mod_update
        elif mod_update is not None:
            self._mod_update = partial(_divide, m=mod_update)
        else:
            self._mod_update = partial(_divide, m=2.0)

        self._levels = np.arange(min_enthalpy, max_enthalpy, bin_size)

        # The correct initialization will be handled in set_aux_state.
        self._current_enthalpy = np.inf
        self._current_features = np.zeros(len(ensemble.natural_parameters))
        self._entropy = np.zeros(len(self._levels))
        self._histogram = np.zeros(len(self._levels), dtype=int)
        # Used to compute average features only.
        self._occurrences = np.zeros(len(self._levels), dtype=int)
        # Mean feature vector per level.
        self._mean_features = np.zeros(
            (len(self._levels), len(ensemble.natural_parameters))
        )
        self._steps_counter = 0  # Save number of valid states elapsed.

        # Population of initial trace included here.
        super().__init__(
            ensemble=ensemble, step_type=step_type, *args, seed=seed, **kwargs
        )

        if self.bias is not None:
            raise ValueError("Cannot apply bias to Wang-Landau simulation!")

        # add inputs to specification
        self.spec.min_enthalpy = min_enthalpy
        self.spec.max_enthalpy = max_enthalpy
        self.spec.bin_size = bin_size
        self.spec.flatness = flatness
        self.spec.check_period = check_period
        self.spec.update_period = update_period
        self.spec.levels = self._levels.tolist()

        # Additional clean-ups.
        self._histogram[:] = 0
        self._occurrences[:] = 0
        self._entropy[:] = 0
        self._steps_counter = 0  # Log n_steps elapsed.

    @property
    def bin_size(self):
        """Enthalpy bin size."""
        return self._window[2]

    @property
    def levels(self):
        """Visited enthalpy levels."""
        return self._levels[self._entropy > 0]

    @property
    def entropy(self):
        """log(dos) on visited levels."""
        return self._entropy[self._entropy > 0]

    @property
    def dos(self):
        """Density of states on visited levels."""
        return np.exp(self.entropy - self.entropy.min())

    @property
    def histogram(self):
        """Histogram on visited levels."""
        return self._histogram[self._entropy > 0]

    @property
    def mod_factor(self):
        """Mod factor."""
        return self._m

    def _get_bin_id(self, e):
        """Get bin index of an enthalpy."""
        # Does not check if enthalpy is in window.
        if e == np.inf:  # This happens at init.
            return np.inf
        return int((e - self._window[0]) // self._window[2])

    def _get_bin_enthalpy(self, bin_id):
        """Get enthalpy form bin index."""
        return bin_id * self._window[2] + self._window[0]

    def _accept_step(self, occupancy, step):
        bin_id = self._get_bin_id(self._current_enthalpy)
        new_enthalpy = self._current_enthalpy + self.trace.delta_trace.enthalpy

        # reject if outside of window
        if new_enthalpy < self._window[0] or new_enthalpy >= self._window[1]:
            self.trace.accepted = np.array(False)
        else:
            new_bin_id = self._get_bin_id(new_enthalpy)
            entropy = self._entropy[bin_id]
            new_entropy = self._entropy[new_bin_id]
            log_factor = self.mcusher.compute_log_priori_factor(occupancy, step)
            exponent = entropy - new_entropy + log_factor
            self.trace.accepted = np.array(
                True if exponent >= 0 else exponent > log(self._rng.random())
            )
        return self.trace.accepted

    def _do_accept_step(self, occupancy, step):
        """Accept/reject a given step and populate trace and aux states accordingly.

        Args:
            occupancy (ndarray):
                Current occupancy
            step (Sequence of tuple):
                Sequence of tuples of ints representing an MC step

        Returns:
            ndarray: new occupancy
        """
        occupancy = super()._do_accept_step(occupancy, step)
        self._current_features += self.trace.delta_trace.features
        self._current_enthalpy += self.trace.delta_trace.enthalpy

        return occupancy

    def _do_post_step(self):
        """Populate trace and aux states for a step regardless of acceptance.

        Populate histogram and entropy, and update counters accordingly.
        """
        bin_id = self._get_bin_id(self._current_enthalpy)

        # Only if bin_id is valid, because bin_id is not checked.
        if 0 <= bin_id < len(self._levels):
            # compute the cumulative statistics
            self._steps_counter += 1
            total = self._occurrences[bin_id]
            self._mean_features[bin_id, :] = (
                1
                / (total + 1)
                * (self._current_features + total * self._mean_features[bin_id, :])
            )
            # update histogram, entropy and occurrences each update_period steps
            if self._steps_counter % self.update_period == 0:
                # Add to DOS and histogram
                self._entropy[bin_id] += self._m
                self._histogram[bin_id] += 1
                self._occurrences[bin_id] += 1

        # fill up values in the trace
        self.trace.histogram = self._histogram
        self.trace.occurrences = self._occurrences
        self.trace.entropy = self._entropy
        self.trace.cumulative_mean_features = self._mean_features
        self.trace.mod_factor = np.array([self._m])

        if self._steps_counter % self.check_period == 0:
            # remove zero entropy entries (use entropy instead of histogram to mask so
            # that once a state is visited it is always considered there after)
            histogram = self._histogram[self._entropy > 0]
            # Check that at least two distinct bins have been visited, then check flatness
            if (
                len(histogram) >= 2
                and (histogram > self.flatness * histogram.mean()).all()
            ):
                # reset histogram and decrease modification factor
                self._histogram[:] = 0
                self._m = self._mod_update(self._m)

        return self.trace

    def compute_initial_trace(self, occupancy):
        """Compute initial values for sample trace given an occupancy.

        Add the micro-canonical entropy and histograms to the trace.
        Args:
            occupancy (ndarray):
                Initial occupancy
        Returns:
            Trace
        """
        trace = super().compute_initial_trace(occupancy)

        # This will not be counting the state corresponding to the given occupancy.
        # but that is ok since we are not recording the initial trace when sampling.
        trace.histogram = self._histogram
        trace.occurrences = self._occurrences
        trace.entropy = self._entropy
        trace.cumulative_mean_features = self._mean_features
        trace.mod_factor = self._m

        return trace

    def set_aux_state(self, occupancy, *args, **kwargs):
        """Set the auxiliary occupancies based on an occupancy.

        This is necessary for WangLandau to work properly because
        it needs to store the current enthalpy and features.
        """
        features = np.array(self.ensemble.compute_feature_vector(occupancy))
        enthalpy = np.dot(features, self.natural_params)
        self._current_features = features
        self._current_enthalpy = enthalpy
        self.mcusher.set_aux_state(occupancy)


def _divide(x, m):
    """Use to allow WangLandau to be pickled."""
    return x / m
