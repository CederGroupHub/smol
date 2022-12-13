"""Implementation of MCMC transition kernel classes.

A kernel essentially is an implementation of the MCMC algorithm that is used
to generate states for sampling an MCMC chain.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod
from math import log

import numpy as np

from smol.constants import kB
from smol.moca.sampler.bias import MCBias, mcbias_factory
from smol.moca.sampler.mcusher import MCUsher, mcusher_factory
from smol.moca.sampler.namespace import Metadata, StepTrace, Trace
from smol.utils import class_name_from_str, derived_class_factory, get_subclasses

ALL_MCUSHERS = list(get_subclasses(MCUsher).keys())
ALL_BIAS = list(get_subclasses(MCBias).keys())


class MCKernel(ABC):
    """Abstract base class for transition kernels.

    A kernel is used to implement a specific MC algorithm used to sample
    the ensemble classes. For an illustrative example of how to derive from this
    and write a specific kernel see the Metropolis kernel.
    """

    # Lists of valid helper classes, set these in derived kernels
    valid_mcushers = None
    valid_bias = None

    def __init__(
        self,
        ensemble,
        step_type,
        *args,
        seed=None,
        bias_type=None,
        bias_kwargs=None,
        **kwargs,
    ):
        """Initialize MCKernel.

        Args:
            ensemble (Ensemble):
                an Ensemble instance to obtain the features and parameters
                used in computing log probabilities.
            step_type (str):
                string specifying the MCMC step type.
            seed (int): optional
                non-negative integer to seed the PRNG
            bias_type (str): optional
                name for bias type instance.
            bias_kwargs (dict): optional
                dictionary of keyword arguments to pass to the bias
                constructor.
            args:
                positional arguments to instantiate the MCUsher for the
                corresponding step size.
            kwargs:
                keyword arguments to instantiate the MCUsher for the
                corresponding step size.
        """
        self.natural_params = ensemble.natural_parameters
        self._seed = seed if seed is not None else np.random.SeedSequence().entropy
        self._rng = np.random.default_rng(self._seed)
        self._compute_features = ensemble.compute_feature_vector
        self._feature_change = ensemble.compute_feature_vector_change
        self.trace = StepTrace(accepted=np.array([True]))
        self._usher, self._bias = None, None

        mcusher_name = class_name_from_str(step_type)
        self.mcusher = mcusher_factory(
            mcusher_name, ensemble.sublattices, *args, rng=self._rng, **kwargs
        )

        self.spec = Metadata(
            self.__class__.__name__, seed=self._seed, step=self.mcusher.spec
        )

        if bias_type is not None:
            bias_name = class_name_from_str(bias_type)
            bias_kwargs = {} if bias_kwargs is None else bias_kwargs
            self.bias = mcbias_factory(
                bias_name, ensemble.sublattices, rng=self._rng, **bias_kwargs
            )
            self.spec.bias = self.bias.spec

        # run a initial step to populate trace values
        _ = self.single_step(np.zeros(ensemble.num_sites, dtype=int))

    @property
    def mcusher(self):
        """Get the MCUsher."""
        return self._usher

    @mcusher.setter
    def mcusher(self, usher):
        """Set the MCUsher."""
        if usher.__class__.__name__ not in self.valid_mcushers:
            raise ValueError(f"{type(usher)} is not a valid MCUsher for this kernel.")
        self._usher = usher

    @property
    def seed(self):
        """Get seed for PRNG."""
        return self._seed

    @property
    def bias(self):
        """Get the bias."""
        return self._bias

    @bias.setter
    def bias(self, bias):
        """Set the bias."""
        if bias.__class__.__name__ not in self.valid_bias:
            raise ValueError(f"{type(bias)} is not a valid MCBias for this kernel.")
        if "bias" not in self.trace.delta_trace.names:
            self.trace.delta_trace.bias = np.zeros(1)
        self._bias = bias

    def set_aux_state(self, occupancies, *args, **kwargs):
        """Set the auxiliary occupancies from initial or checkpoint values."""
        self._usher.set_aux_state(occupancies, *args, **kwargs)

    @abstractmethod
    def single_step(self, occupancy):
        """Attempt an MCMC step.

        Returns the next state in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            StepTrace: a step trace for states and traced values for a single
                       step
        """
        return self.trace

    def compute_initial_trace(self, occupancy):
        """Compute initial values for sample trace given an occupancy.

        Args:
            occupancy (ndarray):
                Initial occupancy

        Returns:
            Trace
        """
        trace = Trace()
        trace.occupancy = occupancy
        trace.features = self._compute_features(occupancy)
        # set scalar values into shape (1,) array for sampling consistency.
        trace.enthalpy = np.array([np.dot(self.natural_params, trace.features)])
        if self.bias is not None:
            trace.bias = np.array([self.bias.compute_bias(occupancy)])
        trace.accepted = np.array([True])
        return trace


class ThermalKernel(MCKernel):
    """Abstract base class for transition kernels with a set temperature.

    Basically all kernels should derive from this with the exception of those
    for multicanonical sampling and related methods
    """

    def __init__(self, ensemble, step_type, temperature, *args, **kwargs):
        """Initialize ThermalKernel.

        Args:
            ensemble (Ensemble):
                an Ensemble instance to obtain the features and parameters
                used in computing log probabilities.
            step_type (str):
                string specifying the MCMC step type.
            temperature (float):
                temperature at which the MCMC sampling will be carried out.
            args:
                positional arguments to instantiate the MCUsher for the
                corresponding step size.
            kwargs:
                keyword arguments to instantiate the MCUsher for the
                corresponding step size.
        """
        # hacky for initialization single_step to run
        self.beta = 1.0 / (kB * temperature)
        super().__init__(ensemble, step_type, *args, **kwargs)
        self.temperature = temperature

    @property
    def temperature(self):
        """Get the temperature of kernel."""
        return self.trace.temperature

    @temperature.setter
    def temperature(self, temperature):
        """Set the temperature and beta accordingly."""
        self.trace.temperature = np.array(temperature)
        self.beta = 1.0 / (kB * temperature)

    def compute_initial_trace(self, occupancy):
        """Compute initial values for sample trace given occupancy.

        Args:
            occupancy (ndarray):
                Initial occupancy

        Returns:
            Trace
        """
        trace = super().compute_initial_trace(occupancy)
        trace.temperature = np.array([self.trace.temperature])
        return trace

    def set_aux_state(self, occupancies, *args, **kwargs):
        """Set the auxiliary occupancies from initial or checkpoint values."""
        self._usher.set_aux_state(occupancies, *args, **kwargs)


class UniformlyRandom(MCKernel):
    """A Kernel that accepts all proposed steps.

    This kernel samples the random limit distribution where all states have the
    same probability (corresponding to an infinite temperature). If a
    bias is added then the corresponding distribution will be biased
    accordingly.
    """

    valid_mcushers = ALL_MCUSHERS
    valid_bias = ALL_BIAS

    def single_step(self, occupancy):
        """Attempt an MCMC step.

        Returns the next occupancies in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            StepTrace
        """
        step = self._usher.propose_step(occupancy)
        log_factor = self._usher.compute_log_priori_factor(occupancy, step)
        self.trace.delta_trace.features = self._feature_change(occupancy, step)
        self.trace.delta_trace.enthalpy = np.array(
            np.dot(self.natural_params, self.trace.delta_trace.features)
        )

        if self._bias is not None:
            self.trace.delta_trace.bias = np.array(
                self._bias.compute_bias_change(occupancy, step)
            )
            exponent = self.trace.delta_trace.bias + log_factor
        else:
            exponent = log_factor

        self.trace.accepted = np.array(
            True if exponent >= 0 else exponent > log(self._rng.random())
        )

        if self.trace.accepted:
            for tup in step:
                occupancy[tup[0]] = tup[1]
            self._usher.update_aux_state(step)

        self.trace.occupancy = occupancy
        return self.trace


class Metropolis(ThermalKernel):
    """A Metropolis-Hastings kernel.

    The classic and nothing but the classic.
    """

    valid_mcushers = ALL_MCUSHERS
    valid_bias = ALL_BIAS

    def single_step(self, occupancy):
        """Attempt an MC step.

        Returns the next occupancies in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            StepTrace
        """
        step = self._usher.propose_step(occupancy)
        log_factor = self._usher.compute_log_priori_factor(occupancy, step)
        self.trace.delta_trace.features = self._feature_change(occupancy, step)
        self.trace.delta_trace.enthalpy = np.array(
            np.dot(self.natural_params, self.trace.delta_trace.features)
        )

        if self._bias is not None:
            self.trace.delta_trace.bias = np.array(
                self._bias.compute_bias_change(occupancy, step)
            )
            exponent = (
                -self.beta * self.trace.delta_trace.enthalpy
                + self.trace.delta_trace.bias
                + log_factor
            )

        else:
            exponent = -self.beta * self.trace.delta_trace.enthalpy + log_factor

        self.trace.accepted = np.array(
            True if exponent >= 0 else exponent > log(self._rng.random())
        )

        if self.trace.accepted:
            for tup in step:
                occupancy[tup[0]] = tup[1]
            self._usher.update_aux_state(step)
        self.trace.occupancy = occupancy

        return self.trace


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
            self._mod_update = lambda x: x / mod_update
        else:
            self._mod_update = lambda x: x / 2.0

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

    def single_step(self, occupancy):
        """Attempt an MC step.

        Returns the next steptrace in the chain.
        Args:
            occupancy (ndarray):
                encoded occupancy.
        Returns:
            StepTrace
        """
        bin_id = self._get_bin_id(self._current_enthalpy)
        step = self._usher.propose_step(occupancy)
        self.trace.delta_trace.features = self._feature_change(occupancy, step)
        self.trace.delta_trace.enthalpy = np.array(
            np.dot(self.natural_params, self.trace.delta_trace.features)
        )

        new_enthalpy = self._current_enthalpy + self.trace.delta_trace.enthalpy

        # reject if outside of window
        if new_enthalpy < self._window[0] or new_enthalpy >= self._window[1]:
            self.trace.accepted = np.array(False)
            new_bin_id = bin_id
        else:
            new_bin_id = self._get_bin_id(new_enthalpy)
            entropy = self._entropy[bin_id]
            new_entropy = self._entropy[new_bin_id]
            log_factor = self._usher.compute_log_priori_factor(occupancy, step)
            exponent = entropy - new_entropy + log_factor
            self.trace.accepted = np.array(
                True if exponent >= 0 else exponent > log(self._rng.random())
            )

        if self.trace.accepted:
            for f in step:
                occupancy[f[0]] = f[1]

            bin_id = new_bin_id
            self._current_features += self.trace.delta_trace.features
            self._current_enthalpy = new_enthalpy
            self._usher.update_aux_state(step)
        self.trace.occupancy = occupancy

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
        features = np.array(self._compute_features(occupancy))
        enthalpy = np.dot(features, self.natural_params)
        self._current_features = features
        self._current_enthalpy = enthalpy
        self._usher.set_aux_state(occupancy)


def mckernel_factory(kernel_type, ensemble, step_type, *args, **kwargs):
    """Get a MCMC Kernel from string name.

    Args:
        kernel_type (str):
            string specifying kernel type to instantiate.
        ensemble (Ensemble)
            an Ensemble object to create the MCMC kernel from.
        step_type (str):
            string specifying the proposal type (i.e. key for MCUsher type)
        *args:
            positional arguments passed to class constructor
        **kwargs:
            keyword arguments passed to class constructor

    Returns:
        MCKernel: instance of derived class.
    """
    kernel_name = class_name_from_str(kernel_type)
    return derived_class_factory(
        kernel_name, MCKernel, ensemble, step_type, *args, **kwargs
    )
