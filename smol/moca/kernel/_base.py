"""Implementation of Base MCKernel and Mixin classes."""

__author__ = "Luis Barroso-Luque"

from abc import ABC, abstractmethod

import numpy as np

from smol._utils import class_name_from_str, get_subclasses
from smol.constants import kB
from smol.moca.kernel._trace import StepTrace, Trace
from smol.moca.kernel.bias import MCBias, mcbias_factory
from smol.moca.kernel.mcusher import MCUsher, mcusher_factory
from smol.moca.metadata import Metadata

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
        self._ensemble = ensemble
        self.trace = StepTrace(accepted=np.array(True))
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
    def ensemble(self):
        """Return the ensemble instance."""
        return self._ensemble

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
        self.mcusher.set_aux_state(occupancies, *args, **kwargs)

    def _compute_step_trace(self, occupancy, step):
        """Compute the trace for a single step.

        Args:
            occupancy (ndarray):
                Current occupancy

        """
        self.trace.delta_trace.features = self.ensemble.compute_feature_vector_change(
            occupancy, step
        )
        self.trace.delta_trace.enthalpy = np.array(
            np.dot(self.natural_params, self.trace.delta_trace.features),
            dtype=np.float64,
        )
        if self._bias is not None:
            self.trace.delta_trace.bias = np.array(
                self._bias.compute_bias_change(occupancy, step),
                dtype=np.float64,
            )
        # TODO consider adding bias = 0 when no bias is set to simplify code in
        #  single_step of kernels. may be faster maybe not...

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
        trace.features = self.ensemble.compute_feature_vector(occupancy)
        # set scalar values into shape (1,) array for sampling consistency.
        trace.enthalpy = np.array(
            [np.dot(self.natural_params, trace.features)], dtype=np.float64
        )
        if self.bias is not None:
            trace.bias = np.array([self.bias.compute_bias(occupancy)], dtype=np.float64)
        trace.accepted = np.array([True], dtype=np.bool)
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
        self.trace.temperature = np.array(temperature, dtype=np.float64)
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
        trace.temperature = np.array([self.trace.temperature], dtype=np.float64)
        return trace

    def set_aux_state(self, occupancies, *args, **kwargs):
        """Set the auxiliary occupancies from initial or checkpoint values."""
        self._usher.set_aux_state(occupancies, *args, **kwargs)


class MulticellKernelMixin(MCKernel):
    """MulticellKernelMixin class.

    This class should not be instantiated, but be used as a Mixin class in other Kernel
    class definitions. See MulticellMetropolis for an example.

    Use this to create kernels that apart from proposing new occupancy states, jumping
    among different kernels is also allowed.

    This is useful for example when one wants to sample over different supercell shapes,
    i.e. for generating special quasi-random structures.

    The ensembles in all kernels must have supercells of the same size, only the shape
    can change. All ensembles must be created from the same Hamiltonian. And all kernels
    must be of the same kind.

    There is no requirement for the individual kernels to have the same step type, bias,
    or any other setup parameter, which allows very flexible sampling strategies.
    However, users are responsible for ensuring that the overall sampling strategy is
    correct! (i.e. satisfies balance or detailed balance).

    Sampling is slower since the property difference of a kernel to kernel jump requires
    complete re-computation of the feature vector.
    """

    valid_bias = None  # currently no bias allowed for kernel hops

    def __init__(
        self,
        mckernels,
        kernel_probabilities=None,
        kernel_hop_periods=1,
        kernel_hop_probabilities=None,
        seed=None,
    ):
        """Initialize MCKernel.

        Args:
            mckernels (list of MCKernels):
                a list of MCKernels instances to obtain the features and parameters
                used in computing log probabilities.
            seed (int): optional
                non-negative integer to seed the PRNG
            kernel_probabilities (Sequence of floats): optional
                Probability for choosing each of the supplied kernels
            kernel_hop_periods (int or Sequence of ints): optional
                number of steps between kernel hop attempts.
            kernel_hop_probabilities (Sequence of floats): optional
                probabilities for choosing each of the specified hop periods.
        """
        if any(not isinstance(kernel, type(mckernels[0])) for kernel in mckernels):
            raise ValueError("All kernels must be of the same type.")

        if any(
            not kernel.ensemble.num_sites != mckernels[0].ensemble.num_sites
            for kernel in mckernels
        ):
            raise ValueError("All ensembles must have the same number of sites.")

        if any(
            not np.allclose(kernel.natural_params, mckernels[0].natural_params)
            for kernel in mckernels
        ):
            raise ValueError("All ensembles must have the same natural parameters.")

        if kernel_probabilities is not None:
            if sum(kernel_probabilities) != 1.0:
                raise ValueError("The kernel_probabilities do not sum to 1.")
            if len(kernel_probabilities) != len(mckernels):
                raise ValueError(
                    "The length of kernel_probabilities must be equal to the number of mckernels."
                )
            self.kernel_p = np.array(kernel_probabilities)
        else:
            self._kernel_p = np.array(
                [1.0 / len(mckernels) for _ in range(len(mckernels))]
            )

        if isinstance(kernel_hop_periods, int):
            self._hop_periods = np.array([kernel_hop_periods], dtype=int)
        else:
            self._hop_periods = np.array(kernel_hop_periods, dtype=int)

        if kernel_hop_probabilities is not None:
            if sum(kernel_hop_probabilities) != 1.0:
                raise ValueError("The kernel_hop_probabilities do not sum to 1.")
            if len(kernel_hop_probabilities) != len(kernel_hop_periods):
                raise ValueError(
                    "The length of kernel_hop_periods and kernel_hop_probabilities does not match."
                )
            self._hop_p = np.array(kernel_hop_probabilities)
        else:
            self._hop_p = np.array(
                [1.0 / len(self._hop_periods) for _ in range(len(self._hop_periods))]
            )

        self._seed = seed if seed is not None else np.random.SeedSequence().entropy
        self._rng = np.random.default_rng(self._seed)

        self._kernels = mckernels
        self._current_hop_period = self._rng.choice(self._hop_periods, p=self._hop_p)
        self._kernel_hop_counter = 0
        self._current_enthalpy = np.inf

        self.trace = StepTrace(
            accepted=np.array(True), kernel_index=np.array(0, dtype=int)
        )

        # run a initial step to populate trace values
        _ = self.single_step(
            np.zeros(self.current_kernel.ensemble.num_sites, dtype=int)
        )

    @property
    def mckernels(self):
        """Get the MCKernels."""
        return self._kernels

    @property
    def current_kernel(self):
        """Get the current kernel."""
        return self._kernels[self.trace.kernel_index]

    @property
    def ensemble(self):
        """Return the ensemble instance."""
        return self.current_kernel.ensemble

    @property
    def mcusher(self):
        """Get the MCUsher."""
        return self.current_kernel.mcusher

    @mcusher.setter
    def mcusher(self, usher):
        """Set the MCUsher."""
        raise RuntimeError(
            "Cannot set the MCUsher for a CompositeKernel.\n"
            "You must set it for an individual kernel."
        )

    @property
    def bias(self):
        """Get the bias."""
        return self.current_kernel.bias

    @bias.setter
    def bias(self, bias):
        """Set the bias."""
        raise RuntimeError(
            "Cannot set the bias for a CompositeKernel.\n"
            "You must set it for an individual kernel."
        )

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
        # check if we should attempt to hop kernels
        if self._kernel_hop_counter % self._current_hop_period == 0:
            # new_kernel_index = self._rng.choice(
            #    range(len(self._kernels)), p=self._kernel_p
            # )  # uncomment this line
            # TODO do the acceptance stuff and set new trace values based on the
            # mixin class!
            # this will require to separate the single_step method into computing
            # necessary values, then doing the acceptance, then setting the trace

            # choose new hop period and reset counter
            self._current_hop_period = self._rng.choice(
                self._hop_periods, p=self._hop_p
            )
            self._kernel_hop_counter = 0
        # if not do a single step with current kernel
        else:
            self.trace = self.current_kernel.single_step(occupancy)
            self._kernel_hop_counter += 1

        # for this we need to save the latest property...
        self._current_enthalpy += self.trace.delta_trace.enthalpy

        return self.trace

    def set_aux_state(self, occupancy, *args, **kwargs):
        """Set the auxiliary occupancies based on an occupancy.

        This is necessary for WangLandau to work properly because
        it needs to store the current enthalpy and features.
        """
        self._current_enthalpy = np.dot(
            np.array(self.ensemble.compute_feature_vector(occupancy)),
            self.natural_params,
        )
        self._usher.set_aux_state(occupancy)

    def compute_initial_trace(self, occupancy):
        """Compute the initial trace for a given occupancy."""
        # TODO finish this... set the current kernel index
        return self.trace
