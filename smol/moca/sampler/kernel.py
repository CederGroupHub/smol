"""Implementation of MCMC transition kernel classes.

A kernel essentially is an implementation of the MCMC algorithm that is used
to generate states for sampling an MCMC chain.
"""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod
from math import log
from types import SimpleNamespace

import numpy as np

from smol.constants import kB
from smol.moca.sampler.bias import MCBias, mcbias_factory
from smol.moca.sampler.mcusher import MCUsher, mcusher_factory
from smol.utils import class_name_from_str, derived_class_factory, get_subclasses

ALL_MCUSHERS = list(get_subclasses(MCUsher).keys())
ALL_BIAS = list(get_subclasses(MCBias).keys())


class Trace(SimpleNamespace):
    """Simple Trace class.

    A Trace is a simple namespace to hold states and values to be recorded
    during MC sampling.
    """

    def __init__(self, /, **kwargs):  # noqa
        if not all(isinstance(val, np.ndarray) for val in kwargs.values()):
            raise TypeError("Trace only supports attributes of type ndarray.")
        super().__init__(**kwargs)

    @property
    def names(self):
        """Get all attribute names."""
        return tuple(self.__dict__.keys())

    def items(self):
        """Return generator for (name, attribute)."""
        yield from self.__dict__.items()

    def __setattr__(self, name, value):
        """Set only ndarrays as attributes."""
        if isinstance(value, (float, int)):
            value = np.array([value])

        if not isinstance(value, np.ndarray):
            raise TypeError("Trace only supports attributes of type ndarray.")
        self.__dict__[name] = value

    def as_dict(self):
        """Return copy of underlying dictionary."""
        return self.__dict__.copy()


class StepTrace(Trace):
    """StepTrace class.

    Same as Trace above but holds a default "delta_trace" inner trace to hold
    trace values that represent changes from previous values, to be handled
    similarly to delta_features and delta_energy.

    A StepTrace object is set as an MCKernel's attribute to record
    kernel specific values during sampling.
    """

    def __init__(self, /, **kwargs):  # noqa
        super().__init__(**kwargs)
        super(Trace, self).__setattr__("delta_trace", Trace())

    @property
    def names(self):
        """Get all field names. Removes delta_trace from field names."""
        return tuple(name for name in super().names if name != "delta_trace")

    def items(self):
        """Return generator for (name, attribute). Skips delta_trace."""
        for name, value in self.__dict__.items():
            if name == "delta_trace":
                continue
            yield name, value

    def __setattr__(self, name, value):
        """Set only ndarrays as attributes."""
        if name == "delta_trace":
            raise ValueError("Attribute name 'delta_trace' is reserved.")
        if not isinstance(value, np.ndarray):
            raise TypeError("Trace only supports attributes of type ndarray.")
        self.__dict__[name] = value

    def as_dict(self):
        """Return copy of serializable dictionary."""
        step_trace_d = self.__dict__.copy()
        step_trace_d["delta_trace"] = step_trace_d["delta_trace"].as_dict()
        return step_trace_d


# TODO make it easier to have multiple walkers, either have the sampler have
#  a list of kernel copies or make a multi-kernel class that simply holds
#  the copies but ow behaves the same, that will really simplify writing kernels!
#  TODO it is so tedious and unessecary to have multiple walkers for the WL Kernel !


class MCKernel(ABC):
    """Abtract base class for transition kernels.

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
        nwalkers=1,
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
            nwalkers (int): optional
                Number of walkers/chains to sampler.
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
        self._nwalkers = nwalkers
        self.trace = StepTrace(accepted=np.array([True]))
        self._usher, self._bias = None, None

        mcusher_name = class_name_from_str(step_type)
        self.mcusher = mcusher_factory(
            mcusher_name,
            ensemble.sublattices,
            *args,
            rng=self._rng,
            **kwargs,
        )

        if bias_type is not None:
            bias_name = class_name_from_str(bias_type)
            bias_kwargs = {} if bias_kwargs is None else bias_kwargs
            self.bias = mcbias_factory(
                bias_name,
                ensemble.sublattices,
                rng=self._rng,
                **bias_kwargs,
            )

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
        """Compute inital values for sample trace given an occupancy.

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

    def iter_steps(self, occupancies):
        """Iterate steps for each walker over an array of occupancies."""
        for occupancy in occupancies:
            yield self.single_step(occupancy)


class ThermalKernel(MCKernel):
    """Abtract base class for transition kernels with a set temperature.

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
        """Compute inital values for sample trace given occupancy.

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
        self.trace.delta_trace.features = self._feature_change(occupancy, step)
        self.trace.delta_trace.enthalpy = np.array(
            np.dot(self.natural_params, self.trace.delta_trace.features)
        )

        if self._bias is not None:
            self.trace.delta_trace.bias = np.array(
                self._bias.compute_bias_change(occupancy, step)
            )
            self.trace.accepted = np.array(
                True
                if self.trace.delta_trace.bias >= 0
                else self.trace.delta_trace.bias > log(self._rng.random())
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
            )
            self.trace.accepted = np.array(
                True if exponent >= 0 else exponent > log(self._rng.random())
            )
        else:
            self.trace.accepted = np.array(
                True
                if self.trace.delta_trace.enthalpy <= 0
                else -self.beta * self.trace.delta_trace.enthalpy
                > log(self._rng.random())  # noqa
            )

        if self.trace.accepted:
            for tup in step:
                occupancy[tup[0]] = tup[1]
            self._usher.update_aux_state(step)
        self.trace.occupancy = occupancy

        return self.trace


class WangLandau(MCKernel):
    """
    A Kernel for Wang Landau Sampling.

    Inheritance naming is probably a misnomer, since WL is non-Markovian. But
    alas we can be code descriptivists.
    """

    valid_mcushers = ALL_MCUSHERS
    valid_bias = None

    def __init__(
        self,
        ensemble,
        step_type,
        bin_size,
        min_energy,
        max_energy,
        flatness=0.8,
        mod_factor=1.0,
        check_period=1000,
        update_period=1,
        mod_update=None,
        seed=None,
        nwalkers=1,
        **kwargs,
    ):
        """Initialize a WangLandau Kernel.

        Args:
            ensemble (Ensemble):
                The ensemble object to use to generate samples
            step_type (str):
                An MC step type corresponding to an MCUsher. See valid_mcushers
            bin_size (float):
                The energy bin size to determine different states.
            min_energy (float):
                The minimum energy to sample. Energy value should be given per
                supercell (i.e. same order of magnitude as what will be sampled).
            max_energy (float):
                The maximum energy to sample. Same as above.
            flatness (float): optional
                The flatness factor used when checking histogram flatness.
                Must be between 0 and 1.
            mod_factor (float):
                The modification factor used to update the DOS/entropy.
                Default is log(e^1).
            check_period (int): optional
                The period in number of steps for the histogram flatness to be
                checked.
            update_period (int): optional
                The period number in steps to update the histogram and entropy.
                In some cases periods slightly larger than each step, may allow
                more efficient exploration of energy space.
            mod_update (Callable): optional
                A function used to update the fill factor when the histogram
                satisfies the flatness criteria. The function is used to update
                the entropy value for an energy level and so must monotonically
                decrease to 0.
            seed (int): optional
                non-negative integer to seed the PRNG
            nwalkers (int): optional
                Number of walkers/chains to sampler. Default is 1.
        """
        if min_energy > max_energy:
            raise ValueError("min_energy can not be larger than max_energy.")
        elif mod_factor <= 0:
            raise ValueError("mod_factor must be greater than 0.")

        self.flatness = flatness
        self.check_period = check_period
        self.update_period = update_period
        self._mfactors = np.array(
            nwalkers
            * [
                mod_factor,
            ]
        )
        self._window = (min_energy, max_energy)
        self._bin_size = bin_size
        self._mod_update = mod_update if mod_update is not None else lambda f: f / 2.0

        self._energy_levels = np.arange(min_energy, max_energy, bin_size)
        self._aux_states = {
            "histogram": np.zeros((nwalkers, len(self._energy_levels)), dtype=int),
            "entropy": np.zeros((nwalkers, len(self._energy_levels))),
            "current-energy": np.zeros(nwalkers),
            "check-counter": 0,  # count steps to check flatness at given intervals
            "update-counter": 0,  # count steps to update histogram and entropy
            # keep a total-histogram for online means (and variances eventually?)
            "total-histogram": np.zeros(
                (nwalkers, len(self._energy_levels)), dtype=int
            ),
            "current-features": np.zeros((nwalkers, len(ensemble.natural_parameters))),
            # for cumulative mean features per energy level
            "mean-features": np.zeros(
                (nwalkers, len(self.energy_levels), len(ensemble.natural_parameters))
            ),
        }

        super().__init__(
            ensemble=ensemble,
            step_type=step_type,
            nwalkers=nwalkers,
            seed=seed,
            **kwargs,
        )
        # clear aux_states from single_step call in super.__init__ (not v clean though)
        self._aux_states["histogram"][:] = 0
        self._aux_states["entropy"][:] = 0

    @property
    def bin_size(self):
        """Get the size of energy bins."""
        return self._bin_size

    @property
    def energy_levels(self):
        """Get energies that have been visited or are inside initial window."""
        return self._energy_levels

    @property
    def entropy(self):
        """Return entropy values for each walker."""
        return self._aux_states["entropy"]

    @property
    def dos(self):
        """Get current DOS for each walker."""
        # TODO look into handling this to avoid overflow issues...
        return np.exp(self.entropy)

    @property
    def histogram(self):
        """Get current histograms for each walker."""
        return self._aux_states["histogram"]

    @property
    def mod_factors(self):
        """Get the entropy modification factors."""
        return self._mfactors

    def _get_bin(self, energy):
        """Get the histogram bin for _aux_states dict from given energy."""
        return int((energy - self._window[0]) / self._bin_size)

    def _get_bin_energy(self, bin_number):
        """Get the bin energy for _aux_states dict from given energy."""
        return bin_number * self._bin_size + self._window[0]

    def iter_steps(self, occupancies):
        """Iterate steps for each walker over an array of occupancies."""
        for walker, occupancy in enumerate(occupancies):
            yield self.single_step(occupancy, walker)

        self._aux_states["check-counter"] += 1
        # check if histograms are flat and reset accordingly
        if self._aux_states["check-counter"] % self.check_period == 0:
            for i, histogram in enumerate(self._aux_states["histogram"]):
                histogram = histogram[histogram > 0]  # remove zero entries
                if (histogram > self.flatness * histogram.mean()).all():
                    self._mfactors[i] = self._mod_update(self._mfactors[i])
                    self._aux_states["histogram"][i, :] = 0  # reset histogram

    def single_step(self, occupancy, walker=0):
        """Attempt an MC step.

        Returns the next occupancies in the chain and if the attempted step was
        successful based on the WL algorithm.

        Args:
            occupancy (ndarray):
                encoded occupancy.
            walker (int):
                index of walker to take step.

        Returns:
            tuple: (acceptance, occupancy, features change, enthalpy change)
        """
        energy = self._aux_states["current-energy"][walker]
        features = self._aux_states["current-features"][walker]
        bin_num = self._get_bin(energy)

        step = self._usher.propose_step(occupancy)
        self.trace.delta_trace.features = self._feature_change(occupancy, step)
        self.trace.delta_trace.enthalpy = np.array(
            np.dot(self.natural_params, self.trace.delta_trace.features)
        )

        new_energy = energy + self.trace.delta_trace.enthalpy  # energy really

        if new_energy < self._window[0]:  # reject
            self.trace.accepted = np.array(False)
        elif new_energy > self._window[1]:  # reject
            self.trace.accepted = np.array(False)
        else:
            state = self._aux_states["entropy"][walker, bin_num]
            new_bin_num = self._get_bin(new_energy)
            new_state = self._aux_states["entropy"][walker, new_bin_num]
            self.trace.accepted = np.array(state - new_state >= log(self._rng.random()))

        if self.trace.accepted:
            for f in step:
                occupancy[f[0]] = f[1]
            bin_num = new_bin_num
            features += self.trace.delta_trace.features
            self._aux_states["current-energy"][walker] = new_energy
            self._aux_states["current-features"][walker] = features
            self._usher.update_aux_state(step)

        # only if bin_num is valid
        if 0 < bin_num < len(self._energy_levels):
            # compute the cumulative statistics
            total = self._aux_states["total-histogram"][walker, bin_num]
            curr_mean = self._aux_states["mean-features"][walker, bin_num]
            self._aux_states["mean-features"][walker, bin_num] = (1 / (total + 1)) * (
                features + total * curr_mean
            )
            self._aux_states["update-counter"] += 1
            # check if histograms are flat and reset accordingly
            if self._aux_states["update-counter"] % self.update_period == 0:
                # update DOS and histogram
                self._aux_states["entropy"][walker, bin_num] += self._mfactors[walker]
                self._aux_states["histogram"][walker, bin_num] += 1
                self._aux_states["total-histogram"][walker, bin_num] += 1

        # fill up values in the trace
        self.trace.histogram = self._aux_states["histogram"][walker]
        self.trace.entropy = self._aux_states["entropy"][walker]
        self.trace.cumulative_mean_features = self._aux_states["mean-features"][walker]
        # this multiple walker thing is so unessecary!!!
        self.trace.mod_factor = np.array([self._mfactors[walker]])

        return self.trace

    def compute_initial_trace(self, occupancy):
        """Compute inital values for sample trace given an occupancy.

        Add the microcanonical entropy and histograms to the trace

        Args:
            occupancy (ndarray):
                Initial occupancy

        Returns:
            Trace
        """
        trace = super().compute_initial_trace(occupancy)
        # This will not be counting the sate correspondint to the given occupancy.
        # but that is ok since we are not recording the initial trace when sampling.
        trace.histogram = self._aux_states["histogram"][0]
        trace.entropy = self._aux_states["entropy"][0]
        trace.cumulative_mean_features = self._aux_states["mean-features"][0]
        trace.mod_factor = self._mfactors[0]

        return trace

    def set_aux_state(self, occupancies, *args, **kwargs):
        """Set the auxiliary occupancies based on an occupancy."""
        features = np.array(list(map(self._compute_features, occupancies)))
        energies = np.dot(features, self.natural_params)
        self._aux_states["current-features"][:] = features
        self._aux_states["current-energy"][:] = energies
        self._usher.set_aux_state(occupancies)


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
