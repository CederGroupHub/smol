"""Implementation of Base MCKernel and Mixin classes."""

__author__ = "Luis Barroso-Luque"

from abc import ABC, ABCMeta, abstractmethod

import numpy as np

from smol.constants import kB
from smol.moca.kernel.bias import MCBias, mcbias_factory
from smol.moca.kernel.mcusher import MCUsher, mcusher_factory
from smol.moca.metadata import Metadata
from smol.moca.trace import StepTrace, Trace
from smol.utils.class_utils import class_name_from_str, get_subclasses

ALL_MCUSHERS = list(get_subclasses(MCUsher).keys())
ALL_BIAS = list(get_subclasses(MCBias).keys())


class MCKernelInterface(metaclass=ABCMeta):
    """MCKernelInterface for defining Monte Carlo Kernels.

    This interface defines the bare minimum methods and properties that must be
    implemented in MCKernel.
    """

    @property
    @abstractmethod
    def trace(self) -> StepTrace:
        """Return the StepTrace object used to record the values for the proposed step.

        The trace most often holds the occupancy, an acceptance value, the change in
        features and change in generalized enthalpy from the previous step.

        Basically, any value that is needed to compute thermodynamic properties
        from a MC sample should be recorded here.
        """

    @property
    @abstractmethod
    def spec(self) -> Metadata:
        """Return the Metadata object recording sampling specifications.

        Specifications should record all the necessary information to be able to
        reproduce the same results.
        """

    @abstractmethod
    def single_step(self, occupancy) -> StepTrace:
        """Carry out a single MC step.

        It is recommended to use a breakup of single step into methods as is done in
        the MCKernel base class, by using StandardSingleStepMixin:
        1) _compute_step_trace(occupancy, step)
        2) _accept_step(occupancy, step)
        3) _do_post_step()

        The above is not a hard requirement, as long as the StepTrace recording
        necessary values is returned, and the occupancy is modified in place
        according to the step. (not sure if that is the best design decision, if
        we decide it is not then we need to adjust the run method in Sampler)
        """

    @abstractmethod
    def compute_initial_trace(self, occupancy) -> Trace:
        """Compute an initial Trace.

        This function myst compute an initial Trace recording all the necessary absolute
        values that would be recorded during sampling using single_step. That is, this
        should not include "delta" or changes in values, but the absolute values of
        properties corresponding to the given occupancy.
        """

    @abstractmethod
    def set_aux_state(self, occupancy, *args, **kwargs):
        """Set the auxiliary occupancies from initial or checkpoint values.

        This class should be used to setup any auxiliary values that are not recorded
        in the traced values.
        """


class StandardSingleStepMixin(ABC):
    """A mixin interface class that breaks up the single_step method of a Kernel.

    The breakup is done using the following methods:

    1) _compute_step_trace(occupancy, step)
    2) _accept_step(occupancy, step)
    3) _do_post_step()

    The above methods must be implemented in derived classes.
    """

    @abstractmethod
    def _compute_step_trace(self, occupancy, step) -> None:
        """Compute the trace for a single step.

        Every quantity that is required to accept/reject should be computed here.

        Args:
            occupancy (ndarray):
                Current occupancy
            step (Sequence of tuple):
                Sequence of tuples of ints representing an MC step
        """

    @abstractmethod
    def _accept_step(self, occupancy, step) -> bool:
        """Accept/reject a given step.

        Args:
            occupancy (ndarray):
                Current occupancy
            step (Sequence of tuple):
                Sequence of tuples of ints representing an MC step
        """

    @abstractmethod
    def _do_accept_step(self, occupancy, step) -> np.ndarray:
        """Populate trace and aux states for an accepted step.

        All updates that should be done if a step has been accepted should be done here.

        The updated occupancy is returned.

        Args:
            occupancy (ndarray):
                Current occupancy
            step (Sequence of tuple):
                Sequence of tuples of ints representing an MC step

        Returns:
            ndarray: new occupancy
        """

    def _do_post_step(self) -> None:
        """Do any post step updates.

        Any updates that should be done after a step has been attempted regardless of
        acceptance should be done here.
        """
        return

    def single_step(self, occupancy):
        """Attempt an MCMC step.

        Returns the next state in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            StepTrace:
                A trace for states, traced values, and differences in traced values
                for the MC step
        """
        step = self.mcusher.propose_step(occupancy)
        self._compute_step_trace(occupancy, step)
        if self._accept_step(occupancy, step):
            occupancy = self._do_accept_step(occupancy, step)
        self._trace.occupancy = occupancy
        self._do_post_step()
        return self.trace


class MCKernel(StandardSingleStepMixin, MCKernelInterface, ABC):
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
        self._trace = StepTrace(accepted=np.array(True))
        self._usher, self._bias = None, None

        mcusher_name = class_name_from_str(step_type)
        self.mcusher = mcusher_factory(
            mcusher_name, ensemble.sublattices, *args, rng=self._rng, **kwargs
        )

        self._spec = Metadata(
            self.__class__.__name__, seed=self._seed, step=self.mcusher.spec
        )

        if bias_type is not None:
            bias_name = class_name_from_str(bias_type)
            bias_kwargs = {} if bias_kwargs is None else bias_kwargs
            self._bias = mcbias_factory(
                bias_name, ensemble.sublattices, rng=self._rng, **bias_kwargs
            )
            self.spec.bias = self.bias.spec

        # run a initial step to populate trace values
        _ = self.single_step(np.zeros(ensemble.num_sites, dtype=int))

    @property
    def trace(self):
        """Return step trace."""
        return self._trace

    @property
    def spec(self):
        """Return sampling specifications."""
        return self._spec

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

    def set_aux_state(self, occupancy, *args, **kwargs):
        """Set the auxiliary occupancies from initial or checkpoint values."""
        self.mcusher.set_aux_state(occupancy, *args, **kwargs)

    def _compute_step_trace(self, occupancy, step):
        """Compute the trace for a single step.

        Args:
            occupancy (ndarray):
                Current occupancy
            step (Sequence of tuple):
                Sequence of tuples of ints representing an MC step
        """
        self.trace.delta_trace.features = self.ensemble.compute_feature_vector_change(
            occupancy, step
        )
        self.trace.delta_trace.enthalpy = np.array(
            np.dot(self.natural_params, self.trace.delta_trace.features),
            dtype=np.float64,
        )
        if self.bias is not None:
            self.trace.delta_trace.bias = np.array(
                self.bias.compute_bias_change(occupancy, step),
                dtype=np.float64,
            )
        # TODO consider adding bias = 0 when no bias is set to simplify code in
        #  single_step of kernels. may be faster maybe not...

    @abstractmethod
    def _accept_step(self, occupancy, step):
        """Accept/reject a given step.

        Args:
            occupancy (ndarray):
                Current occupancy
            step (Sequence of tuple):
                Sequence of tuples of ints representing an MC step
        """
        return self.trace.accepted

    def _do_accept_step(self, occupancy, step):
        """Populate trace and aux states for an accepted step.

        Args:
            occupancy (ndarray):
                Current occupancy
            step (Sequence of tuple):
                Sequence of tuples of ints representing an MC step

        Returns:
            ndarray: new occupancy
        """
        for site, species in step:
            occupancy[site] = species

        self.mcusher.update_aux_state(step)
        return occupancy

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
        trace.accepted = np.array([True], dtype=bool)
        return trace


class ThermalKernelMixin:
    """Mixin class for transition kernels with a set temperature.

    Basically all kernels should use this Mixin class with the exception of those
    for multicanonical sampling and related methods (i.e. see Wang-Landau)

    Never really found a final say regarding Mixins with __init__ and attributes....
    ...oh well, here's one...
    """

    _kB: float = kB  # Boltzmann constant in eV/K

    def __init__(self, temperature, *args, **kwargs):
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
        self.beta = 1.0 / (self.kB * temperature)
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    @property
    def kB(self):
        """Get the temperature scale factor."""
        return self._kB

    @kB.setter
    def kB(self, kB):
        """Set the temperature scale factor."""
        self._kB = kB
        self.beta = 1.0 / (self.kB * self.temperature)

    @property
    def temperature(self):
        """Get the temperature of kernel."""
        return self.trace.temperature

    @temperature.setter
    def temperature(self, temperature):
        """Set the temperature and beta accordingly."""
        self.trace.temperature = np.array(temperature, dtype=np.float64)
        self.beta = 1.0 / (self.kB * temperature)

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


class MulticellKernel(StandardSingleStepMixin, MCKernelInterface, ABC):
    """Abstract MulticellKernel class.

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
    """

    def __init__(
        self,
        mckernels,
        kernel_probabilities=None,
        kernel_hop_periods=5,
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
            kernel.ensemble.num_sites != mckernels[0].ensemble.num_sites
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
                    "The length of kernel_probabilities must be equal to the number of"
                    " mckernels."
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
                    "The length of kernel_hop_periods and kernel_hop_probabilities does"
                    "not match."
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
        self._kernel_hop_counter = 1
        self._current_kernel_index = np.array(0, dtype=int)
        # keep values of proposed features and occupancy for cell hops
        self._new_features = None
        self._features = np.zeros(
            (len(mckernels), len(self._kernels[0].natural_params))
        )
        self._spec = Metadata(
            self.__class__.__name__,
            seed=self._seed,
            kernel_probabilities=self._kernel_p,
            kernel_hop_periods=self._hop_periods,
            kernel_hop_probabilities=self._hop_p,
            mckernels=[kernel.spec for kernel in mckernels],
        )
        # set kernel indices
        for i, kernel in enumerate(self._kernels):
            kernel.trace.kernel_index = np.array(i, dtype=int)
        self._trace = self._kernels[self._current_kernel_index].trace
        self._kernel_hop_counter = 1  # reset the counter

    @property
    def trace(self):
        """Return step trace."""
        return self._trace

    @property
    def spec(self):
        """Return sampling specifications."""
        return self._spec

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

    @property
    def seed(self):
        """Get seed for PRNG."""
        return self._seed

    @property
    def bias(self):
        """Get the bias."""
        return self.current_kernel.bias

    @bias.setter
    def bias(self, bias):
        """Set the bias."""
        raise RuntimeError(
            "Cannot set the bias for a MultiCellKernel.\n"
            "You must set it for an individual kernel."
        )

    def _compute_step_trace(self, occupancy, step):
        """Compute the step trace.

        Args:
            occupancy (np.ndarray): the current occupancy
            step (np.ndarray): the proposed step

        Returns:
            StepTrace: the step trace
        """
        occupancy = occupancy.copy()  # make a copy to avoid modifying the original
        for site, species in step:
            occupancy[site] = species

        self._new_features = self.ensemble.compute_feature_vector(occupancy)
        # from kernel currently at
        prev_features = self._features[self._current_kernel_index]
        self.trace.delta_trace.features = self._new_features - prev_features
        delta_enthalpy = np.dot(
            self.trace.delta_trace.features, self.ensemble.natural_parameters
        )
        self.trace.delta_trace.enthalpy = delta_enthalpy

    def _do_accept_step(self, occupancy, step):
        """Populate trace and aux states for an accepted step.

        Args:
            occupancy (ndarray):
                Current occupancy
            step (Sequence of tuple):
                Sequence of tuples of ints representing an MC step

        Returns:
            ndarray: new occupancy
        """
        for site, species in step:
            occupancy[site] = species

        # update the features
        self._features[self.trace.kernel_index, :] = self._new_features
        return occupancy

    def single_step(self, occupancy):
        """Attempt an MCMC step.

        Returns the next state in the chain and if the attempted step was
        successful.

        Args:
            occupancy (ndarray):
                encoded occupancy.

        Returns:
            StepTrace: a step trace for states and traced values for a single step
        """
        # check if we should attempt to hop kernels
        if self._kernel_hop_counter % self._current_hop_period == 0:
            # this is the trace for the new kernel (the one attempting to hop to)
            # in some cases it may hop to the same kernel, but that is ok
            self._trace = self._rng.choice(self._kernels, p=self._kernel_p).trace
            # set the occupancy to the one currently in the trace of the kernel
            # attempting to hop to, not the one passed from the previous step
            self._trace = super().single_step(self.trace.occupancy)
            if self.trace.accepted:
                # update the current kernel index
                self._current_kernel_index = self.trace.kernel_index
                # modify the occupancy in place, to follow MCKernelInterface
                occupancy[:] = self.trace.occupancy
            else:
                # reset the trace to the one of the previous kernel if step not accepted
                self._trace = self._kernels[self._current_kernel_index].trace
                self._trace.occupancy = occupancy
                self._trace.accepted = np.array(False)

            # reset the hop period and counter
            self._current_hop_period = self._rng.choice(
                self._hop_periods, p=self._hop_p
            )
            self._kernel_hop_counter = 1
        else:  # if not do a single step with current kernel
            self._trace = self.current_kernel.single_step(occupancy)
            self._kernel_hop_counter += 1
            # We need to save the latest property...
            if self.trace.accepted:
                self._features[
                    self._current_kernel_index
                ] += self.trace.delta_trace.features

        # assert self.trace.kernel_index == self._current_kernel_index
        return self.trace

    def set_aux_state(self, occupancy, *args, **kwargs):
        """Set the occupancies, features and enthalpy for each supercell kernel.

        Since multiple underlying kernels exist, occupancy can be a 2D array
        of occupancies for each kernel, ie dim = (n_kernels, n_sites).
        """
        # set the current and previous values based on given occupancy
        new_features = []
        if occupancy.ndim == 2 and occupancy.shape[0] == len(self._kernels):
            for kernel, occupancy in zip(self._kernels, occupancy):
                occupancy = np.ascontiguousarray(occupancy, dtype=int)
                kernel.trace.occupancy = occupancy
                kernel.set_aux_state(occupancy, *args, **kwargs)
                new_features.append(kernel.ensemble.compute_feature_vector(occupancy))
            self._features = np.vstack(new_features)
        else:  # if only one assume it is continuing from a run...
            self._features[
                self._current_kernel_index, :
            ] = self.current_kernel.ensemble.compute_feature_vector(occupancy)
            self.current_kernel.set_aux_state(occupancy, *args, **kwargs)

    def compute_initial_trace(self, occupancy):
        """Compute the initial trace for a given occupancy."""
        trace = self.current_kernel.compute_initial_trace(occupancy)
        trace.kernel_index = self._current_kernel_index
        return trace
