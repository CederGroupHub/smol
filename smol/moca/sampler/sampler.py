"""Implementation of base sampler classes.

A sampler essentially is an implementation of the MCMC algorithm that is used
by the corresponding ensemble to generate Monte Carlo samples.
"""

__author__ = "Luis Barroso-Luque"

import os
from datetime import datetime
from warnings import warn

import numpy as np

from smol.moca.sampler.container import SampleContainer
from smol.moca.sampler.kernel import Trace, mckernel_factory
from smol.utils import progress_bar


class Sampler:
    """
    A sampler is used to run MCMC sampling simulations.

    The specific MCMC algorithm is defined by the given MCKernel.
    The default will use a a simple Metropolis random walk kernel.
    """

    def __init__(self, kernel, container, seed=None):
        """Initialize BaseSampler.

        It is recommended to initialize a sampler with the from_ensemble
        method, unless you need more control.

        Args:
            kernel (MCKernel):
                An MCKernel instance.
            container (SampleContainer):
                A sampler containter to store samples. If given num_walkers is
                taken from the container.
            seed (int): optional
                seed for random number generator.
        """
        self._kernel = kernel
        self._container = container
        # Set and save the seed for random. This allows reproducible results.
        if seed is None:
            seed = np.random.seed()
        #  Save the seed for reproducibility
        self._container.metadata["seed"] = seed
        self._seed = seed

    @classmethod
    def from_ensemble(
        cls,
        ensemble,
        *args,
        step_type=None,
        kernel_type=None,
        seed=None,
        nwalkers=1,
        **kwargs,
    ):
        """
        Create a sampler based on an Ensemble instances.

        This is the easier way to spin up a Sampler. This will
        automatically populate and create an appropriate SampleContainer.

        Args:
            ensemble (Ensemble):
                An Ensemble class to obtain sample probabilities from.
            step_type (str): optional
                type of step to run MCMC with. If not given the default is the
                first entry in the Ensemble.valid_mcmc_steps.
            *args:
                Positional arguments to pass to the MCKernel constructor.
                More often than not you want to specify the temperature!
            kernel_type (str): optional
                string specifying the specific MCMC transition kernel. This
                represents the underlying MC algorithm. Currently only
                Metropolis is supported.
            seed (int): optional
                Seed for the PRNG.
            nwalkers (int): optional
                Number of walkers/chains to sampler. Default is 1. More than 1
                is still experimental...
            **kwargs:
                Keyword arguments to pass to the MCKernel constructor.
                More often than not you want to specify the temperature!

        Returns:
            Sampler
        """
        if step_type is None:
            step_type = ensemble.valid_mcmc_steps[0]
        elif step_type not in ensemble.valid_mcmc_steps:
            raise ValueError(
                f"Step type {step_type} can not be used for sampling a "
                f"{type(ensemble)}."
            )
        if kernel_type is None:
            kernel_type = "Metropolis"

        mckernel = mckernel_factory(kernel_type, ensemble, step_type, *args, **kwargs)
        # get a trial trace to initialize sample container trace
        _trace = mckernel.compute_initial_trace(np.zeros(ensemble.num_sites, dtype=int))
        sample_trace = Trace(
            **{
                name: np.empty((0, nwalkers, *value.shape), dtype=value.dtype)
                for name, value in _trace.items()
            }
        )

        sampling_metadata = {"name": type(ensemble).__name__}
        sampling_metadata.update(ensemble.thermo_boundaries)
        sampling_metadata.update({"kernel": kernel_type, "step": step_type})
        container = SampleContainer(
            ensemble.sublattices,
            ensemble.natural_parameters,
            ensemble.num_energy_coefs,
            sample_trace,
            sampling_metadata,
        )
        return cls(mckernel, container, seed=seed)

    @property
    def mckernel(self):
        """Get the underlying ensemble."""
        return self._kernel

    @property
    def seed(self):
        """Seed for the random number generator."""
        return self._seed

    @seed.setter
    def seed(self, seed):
        """Set the seed for the PRNG."""
        self._seed = seed

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
                Number of iterations to run.
            initial_occupancies (ndarray):
                array of occupancies
            thin_by (int): optional
                Number to thin iterations by and provide samples.
            progress (bool):
                If true will show a progress bar.

        Yields:
            tuple: accepted, occupancies, features change, enthalpies change
        """
        occupancies = initial_occupancies.copy()
        if occupancies.shape != self.samples.shape:
            occupancies = self._reshape_occu(occupancies)
        if nsteps % thin_by != 0:
            warn(
                f"The number of steps {nsteps} is not a multiple of thin_by "
                f" {thin_by}. The last {nsteps % thin_by} will be ignored.",
                category=RuntimeWarning,
            )
        # TODO check that initial states are independent if num_walkers > 1
        # TODO make samplers with single chain, multiple and multiprocess
        # TODO kernel should take only 1 occupancy
        # set up any auxiliary states from inital occupancies
        self._kernel.set_aux_state(occupancies)

        # get initial traces and stack them
        trace = Trace()
        traces = list(map(self._kernel.compute_initial_trace, occupancies))
        for name in traces[0].names:
            stack = np.vstack([getattr(tr, name) for tr in traces])
            setattr(trace, name, stack)

        # Initialise progress bar
        chains, nsites = self.samples.shape
        desc = f"Sampling {chains} chain(s) from a cell with {nsites} sites..."
        with progress_bar(progress, total=nsteps, description=desc) as p_bar:
            for _ in range(nsteps // thin_by):
                for _ in range(thin_by):
                    for i, strace in enumerate(
                        map(self._kernel.single_step, occupancies)
                    ):
                        for name, value in strace.items():
                            setattr(trace, name, value)
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
                If true will show a progress bar.
            stream_chunk (int): optional
                Chunk of samples to stream into a file. If > 0 samples will
                be flushed to backend file in stream_chucks
            stream_file (str): optional
                file name to use as backend. If file already exists will try
                to append to datasets. If not given will create a new file.
            swmr_mode (bool): optional
                If true allows to read file from other processes. Single Writer
                Multiple Readers.
        """
        if initial_occupancies is None:
            try:
                initial_occupancies = self.samples.get_occupancies(flat=False)[
                    -1
                ]  # noqa
                # auxiliary states from kernels should be set here
            except IndexError as ind_error:
                raise RuntimeError(
                    "There are no saved samples to obtain the "
                    "initial occupancies. These must be "
                    "provided."
                ) from ind_error
        elif self.samples.num_samples > 0:
            warn(
                "Initial occupancies where provided with a pre-existing "
                "set of samples.\n Make real sure that is what you want. "
                "If not, reset the samples in the sampler.",
                RuntimeWarning,
            )
        else:
            if initial_occupancies.shape != self.samples.shape:
                initial_occupancies = self._reshape_occu(initial_occupancies)

        if stream_chunk > 0:
            if stream_file is None:
                now = datetime.now()
                file_name = "moca-samples-" + now.strftime("%Y-%m-%d-%H%M%S%f")
                stream_file = os.path.join(os.getcwd(), file_name + ".h5")
            backend = self.samples.get_backend(
                stream_file, nsteps // thin_by, swmr_mode=swmr_mode
            )
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
            self.clear_samples()
            backend.close()

        # A checkpoing of aux states should be saved to container here.
        # Note that to save any general "state" we will need to make sure it is
        # properly serializable to save as json and also to save in h5py

    def anneal(
        self,
        temperatures,
        mcmc_steps,
        initial_occupancies=None,
        thin_by=1,
        progress=False,
    ):
        """Carry out a simulated annealing procedure.

        Uses the total number of temperatures given by "steps" interpolating
        between the start and end temperature according to a cooling function.
        The start temperature is the temperature set for the ensemble.

        Args:
            temperatures (Sequence):
               Sequence of temperatures to anneal, should be strictly
               decreasing.
            mcmc_steps (int):
               number of Monte Carlo steps to run at each temperature.
            initial_occupancies (ndarray):
                array of occupancies. If None, the last sample will be taken.
                You should only provide this the first time you call run.
            thin_by (int): optional
                the amount to thin by for saving samples.
            progress (bool):
                If true will show a progress bar.
        """
        if temperatures[0] < temperatures[-1]:
            raise ValueError(
                "End temperature is greater than start "
                f"temperature {temperatures[-1]:.2f} > "
                f"{temperatures[0]:.2f}."
            )
        # initialize for first temperature.
        self._kernel.temperature = temperatures[0]
        self.run(
            mcmc_steps,
            initial_occupancies=initial_occupancies,
            thin_by=thin_by,
            progress=progress,
        )
        for temperature in temperatures[1:]:
            self._kernel.temperature = temperature
            self.run(mcmc_steps, thin_by=thin_by, progress=progress)

    def _reshape_occu(self, occupancies):
        """Reshape occupancies for the single walker case."""
        # check if this is only a single walker.
        if len(occupancies.shape) == 1 and self.samples.shape[0] == 1:
            occupancies = np.reshape(occupancies, (1, len(occupancies)))
        else:
            raise AttributeError(
                "The given initial occcupancies have "
                "incompompatible dimensions. Shape should"
                f" be {self.samples.shape}."
            )
        return occupancies


# TODO potential define two _sample functions serial/parallel that get called
#  in one sampler function.
