"""Implementation of base sampler classes.

A sampler essentially is an implementation of the MCMC algorithm that is used
by the corresponding ensemble to generate Monte Carlo samples.
"""

__author__ = "Luis Barroso-Luque"

import os
from datetime import datetime
from warnings import warn

import numpy as np

from smol.moca.kernel import mckernel_factory
from smol.moca.kernel.base import MulticellKernel
from smol.moca.sampler.container import SampleContainer
from smol.moca.trace import Trace
from smol.utils.progressbar import progress_bar


class Sampler:
    """
    A sampler is used to run MCMC sampling simulations.

    The specific MCMC algorithm is defined by the given MCKernel.
    The default will use a simple Metropolis random walk kernel.
    """

    def __init__(self, kernels, container):
        """Initialize BaseSampler.

        It is recommended to initialize a sampler with the from_ensemble
        method, unless you need more control over the sampling kernel,
        mcusher, etc.

        Args:
            kernels ([MCKernel]):
                a list of MCKernel instances for each walker.
                Note: we only support the same type of kernel and the
                 same ensemble for all walkers.
            container (SampleContainer):
                a sampler container to store samples.
        """
        self._kernels = kernels
        self._container = container

        #  Save kernel specifications  #
        self._container.metadata.kernels = [kernel.spec for kernel in kernels]

    @classmethod
    def from_ensemble(
        cls,
        ensemble,
        *args,
        step_type=None,
        kernel_type=None,
        seeds=None,
        nwalkers=1,
        **kwargs,
    ):
        """
        Create a sampler based on an Ensemble instance.

        This is the easier way to set up a Sampler. This will
        automatically populate and create an appropriate SampleContainer.

        Args:
            ensemble (Ensemble):
                an Ensemble class to obtain sample probabilities from.
            step_type (str): optional
                type of step to run MCMC with. If not given the default depends on
                whether chemical potentials are defined.
            *args:
                positional arguments to pass to the MCKernel constructor.
                More often than not you want to specify the temperature!
            kernel_type (str): optional
                string specifying the specific MCMC transition kernel. This
                represents the underlying MC algorithm. Currently only
                Metropolis is supported.
            seeds ([int]): optional
                seed for the PRNG in each kernel.
            nwalkers (int): optional
                number of walkers/chains to sampler. Default is 1. More than 1
                is still experimental...
            **kwargs:
                keyword arguments to pass to the MCKernel constructor.
                More often than not you want to specify the temperature!

        Returns:
            Sampler
        """
        if step_type is None:
            if (
                hasattr(ensemble, "chemical_potentials")
                and ensemble.chemical_potentials is not None
            ):
                step_type = "flip"
            else:
                step_type = "swap"

        if kernel_type is None:
            kernel_type = "Metropolis"
        if seeds is not None:
            if len(seeds) != nwalkers:
                raise ValueError("Number of seeds does not match number of kernels!")
        else:
            seeds = [None for _ in range(nwalkers)]

        mckernels = [
            mckernel_factory(
                kernel_type, ensemble, step_type, seed=seed, *args, **kwargs
            )
            for seed in seeds
        ]

        # get a trial trace to initialize sample container trace
        _trace = mckernels[0].compute_initial_trace(
            np.zeros(ensemble.num_sites, dtype=int)
        )
        sample_trace = Trace(
            **{
                name: np.empty((0, nwalkers, *value.shape), dtype=value.dtype)
                for name, value in _trace.items()
            }
        )

        sampling_metadata = ensemble.thermo_boundaries

        # Container will be initialized to read all sub-lattices,
        # active or not.
        container = SampleContainer(
            ensemble,
            sample_trace,
            sampling_metadata,
        )
        return cls(mckernels, container)

    @property
    def mckernels(self):
        """Get the underlying ensemble."""
        return self._kernels

    @property
    def seeds(self):
        """Seed for the random number generator."""
        return [kernel.seed for kernel in self._kernels]

    @property
    def samples(self):
        """Get the SampleContainer."""
        return self._container

    def efficiency(self, discard=0, flat=True):
        """Get the efficiency of MCMC sampling (accepted/total)."""
        return self.samples.sampling_efficiency(discard=discard, flat=flat)

    def clear_samples(self):
        """Clear samples from sampler container."""
        self.samples.clear()

    def sample(self, nsteps, initial_occupancies, thin_by=1, progress=False):
        """Generate MCMC samples.

        Yield a sampler state every `thin_by` iterations. A state is give by
        a tuple of (occupancies, features, enthalpy)

        Args:
            nsteps (int):
                number of iterations to run.
            initial_occupancies (ndarray):
                array of occupancies
            thin_by (int): optional
                number to thin iterations by and provide samples.
            progress (bool):
                If true will show a progress bar.

        Yields:
            tuple: accepted, occupancies, features change, enthalpies change
        """
        if nsteps % thin_by != 0:
            warn(
                f"The number of steps {nsteps} is not a multiple of thin_by "
                f" {thin_by}. The last {nsteps % thin_by} will be ignored.",
                category=RuntimeWarning,
            )

        occupancies, trace = self.setup_sample(initial_occupancies)
        # Initialise progress bar
        chains, nsites = self.samples.shape
        desc = f"Sampling {chains} chain(s) from a cell with {nsites} sites"
        with progress_bar(progress, total=nsteps, description=desc) as p_bar:
            for _ in range(nsteps // thin_by):
                for _ in range(thin_by):
                    # the occupancies are modified in place at each step
                    for i, strace in enumerate(self._single_step(occupancies)):
                        for name, value in strace.items():
                            val = getattr(trace, name)
                            val[i] = value
                            # this will mess up recording values for > 1 walkers
                            # setattr(trace, name, value)
                        if strace.accepted:
                            for name, delta_val in strace.delta_trace.items():
                                val = getattr(trace, name)
                                val[i] += delta_val
                    p_bar.update()
                # yield copies
                yield trace

    def run(
        self,
        nsteps,
        initial_occupancies=None,
        thin_by=1,
        progress=False,
        stream_chunk=0,
        stream_file=None,
        keep_last_chunk=False,
        swmr_mode=False,
    ):
        """Run an MCMC sampling simulation.

        This will run and save the samples every thin_by into a
        SampleContainer.

        Args:
            nsteps (int):
                number of total MC steps.
            initial_occupancies (ndarray):
                array of occupancies. If None, the last sampler will be taken.
                You should only provide this the first time you call run. If
                you want to reset then you should call reset before to start
                a fresh run.
            thin_by (int): optional
                the amount to thin by for saving samples.
            progress (bool): optional
                if true will show a progress bar.
            stream_chunk (int): optional
                chunk of samples to stream into a file. If > 0 samples will
                be flushed to backend file in stream_chucks
            stream_file (str): optional
                file name to use as backend. If file already exists will try
                to append to datasets. If not given will create a new file.
            keep_last_chunk (bool): optional
                if True will keep the last chunk of samples in memory, but will reset
                the sampler otherwise. This is useful if the last occupancies are
                needed to start another run (ie for simulated annealing with streaming)
            swmr_mode (bool): optional
                if true allows to read file from other processes. Single Writer
                Multiple Readers.
        """
        if initial_occupancies is None:
            try:
                initial_occupancies = self.samples.get_occupancies(flat=False)[-1]
                # auxiliary states from kernels should be set here
            except IndexError as index_error:
                raise RuntimeError(
                    "There are no saved samples to obtain the initial occupancies."
                    "These must be provided."
                ) from index_error
        elif self.samples.num_samples > 0:
            warn(
                "Initial occupancies where provided with a pre-existing set of samples."
                "\n Make real sure that is what you want. "
                "If not, reset the samples in the sampler.",
                RuntimeWarning,
            )

        if stream_chunk > 0:
            if stream_file is None:
                now = datetime.now()
                file_name = "moca-samples-" + now.strftime("%Y-%m-%d-%H%M%S%f")
                stream_file = os.path.join(os.getcwd(), file_name + ".h5")
            backend = self.samples.get_backend(
                stream_file, nsteps // thin_by, swmr_mode=swmr_mode
            )
            # allocate memory only if there is None available
            if len(self.samples.get_occupancies()) == 0:
                self.samples.allocate(stream_chunk)
        else:
            backend = None
            self.samples.allocate(nsteps // thin_by)

        for i, trace in enumerate(
            self.sample(nsteps, initial_occupancies, thin_by=thin_by, progress=progress)
        ):
            self.samples.save_sampled_trace(trace, thinned_by=thin_by)
            if backend is not None and (i + 1) % stream_chunk == 0:
                self.samples.flush_to_backend(backend)

        if backend is not None:
            backend.close()
            # Only clear samples if requested.
            if keep_last_chunk is False:
                self.clear_samples()

        # A checkpoint of aux states should be saved to container here.
        # Note that to save any general "state" we will need to make sure it is
        # properly serializable to save as json and also to save in h5py

    def anneal(
        self,
        temperatures,
        mcmc_steps,
        initial_occupancies=None,
        thin_by=1,
        progress=False,
        stream_chunk=0,
        stream_file=None,
        swmr_mode=True,
    ):
        """Carry out a simulated annealing procedure.

        Will MC sampling for each temperature for the specified number of steps,
        taking the last sampled configuration at each temperature as the starting
        configuration for the next temperature.

        Everything is saved to the same SampleContainer, or if streaming, saved to the
        same file. To save each temperature in a different container, simply run
        a similar "for loop" creating a new sampler at each temperature.

        Args:
            temperatures (Sequence):
               sequence of temperatures to anneal, should be strictly
               decreasing.
               Note: only available for ThermalKernel.
            mcmc_steps (int):
               number of Monte Carlo steps to run at each temperature.
            initial_occupancies (ndarray):
                array of occupancies. If None, the last sample will be taken.
                You should only provide this the first time you call run.
            thin_by (int): optional
                amount to thin by for saving samples.
            progress (bool):
                if true will show a progress bar.
            stream_chunk (int): optional
                chunk of samples to stream into a file. If > 0 samples will
                be flushed to backend file in stream_chucks
            stream_file (str): optional
                file name to use as backend. If file already exists will try
                to append to datasets. If not given will create a new file.
            swmr_mode (bool): optional
                if true allows to read file from other processes. Single Writer
                Multiple Readers.
        """
        if temperatures[0] < temperatures[-1]:
            raise ValueError(
                "End temperature is greater than start "
                f"temperature {temperatures[-1]:.2f} > {temperatures[0]:.2f}."
            )

        # initialize for first temperature.
        for kernel in self._kernels:
            kernel.temperature = temperatures[0]
        self.run(
            mcmc_steps,
            initial_occupancies=initial_occupancies,
            thin_by=thin_by,
            progress=progress,
            stream_chunk=stream_chunk,
            stream_file=stream_file,
            swmr_mode=swmr_mode,
            keep_last_chunk=True,
        )
        for temperature in temperatures[1:]:
            for kernel in self._kernels:
                kernel.temperature = temperature
            self.run(
                mcmc_steps,
                thin_by=thin_by,
                progress=progress,
                stream_chunk=stream_chunk,
                stream_file=stream_file,
                swmr_mode=swmr_mode,
                keep_last_chunk=True,
            )

        # If streaming to file was done then clear samplers now.
        if stream_chunk > 0:
            self.clear_samples()

    def setup_sample(self, initial_occupancies):
        """Copy and reshape occupancies and compute initial trace for sampling.

        Args:
            initial_occupancies (ndarray):
                initial occupancies for sampling.

        Returns: (occupancies, trace)
            occupancies (ndarray):
                initial occupancies for sampling.
            trace (Trace):
                initial trace for sampling.
        """
        # TODO check that initial states are independent if num_walkers > 1
        # TODO make samplers with single chain, multiple and multiprocess
        occupancies = initial_occupancies.copy()
        if occupancies.shape != self.samples.shape:
            occupancies = self._reshape_occu(occupancies)

        # set up any auxiliary states from initial occupancies
        selected_occus = []
        for kernel, occupancy in zip(self._kernels, occupancies):
            kernel.set_aux_state(occupancy)
            # select the current kernel occu
            if isinstance(kernel, MulticellKernel):
                if occupancies.ndim == 3 and occupancies.shape[1] == len(
                    kernel.mckernels
                ):
                    selected_occus.append(occupancy[kernel.trace.kernel_index])
        if len(selected_occus) > 0:
            occupancies = np.vstack(selected_occus)

        # get initial traces and stack them
        trace = Trace()
        traces = list(
            map(
                lambda kernel, occu: kernel.compute_initial_trace(occu),
                self._kernels,
                occupancies,
            )
        )
        for name in traces[0].names:
            stack = np.stack([getattr(tr, name) for tr in traces], axis=0)
            setattr(trace, name, stack)

        return occupancies, trace

    def _single_step(self, occupancies):
        """Do a single step for all kernels."""
        for kernel, occupancy in zip(self._kernels, occupancies):
            yield kernel.single_step(occupancy)

    def _reshape_occu(self, occupancies):
        """Reshape occupancies for the single walker case."""
        # check if this is only a single walker.
        if len(occupancies.shape) == 1 and self.samples.shape[0] == 1:
            occupancies = np.reshape(occupancies, (1, len(occupancies)))
        elif any(isinstance(kernel, MulticellKernel) for kernel in self._kernels):
            # reshape for multicell kernels
            occupancies = np.reshape(occupancies, (1, *occupancies.shape))
        else:
            raise AttributeError(
                "The given initial occcupancies have incompompatible dimensions. Shape should"
                f" be {self.samples.shape}."
            )
        return occupancies


# TODO potential define two _sample functions serial/parallel that get called
#  in one sampler function.
