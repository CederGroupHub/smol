"""Implementation of SampleContainer class.

A SampleContainer is used to store data from Monte Carlo sampling simulation.
It has some minimimal methods and properties useful to start analyzing the
samples.
"""

__author__ = "Luis Barroso-Luque"

import json
import os
import warnings
from collections import defaultdict

import numpy as np
from monty.dev import requires
from monty.json import MontyDecoder, MontyEncoder, MSONable, jsanitize

from smol.moca.sampler.namespace import Metadata, Trace
from smol.moca.sublattice import Sublattice

try:
    import h5py
except ImportError:
    h5py = None


class SampleContainer(MSONable):
    """A SampleContainer class stores Monte Carlo simulation samples.

    A SampleContainer holds samples and sampling information from an MCMC
    sampling run. It is useful to obtain the raw data and minimal empirical
    properties of the underlying distribution in order to carry out further
    analysis of the MCMC sampling results.

    When getting any value from the provided attributes, the highly repeated
    args are:

        discard (int): optional
            number of samples to discard to obtain the value requested.
        thin_by (int): optional
            use every thin_by sample to obtain the value requested.
        flat (bool): optional
            if more than 1 walker is used, flattening will flatten all
            chains into one. Defaults to True.

    Attributes:
        num_sites (int):
            Size of system (usually in number of prims in supercell, but
            can be anything representative, i.e. number of sites)
        sublattices (list of Sublattice)
            Sublattices of the ensemble sampled.
        natural_parameters (ndarray):
                Array of natural parameters used in the ensemble.
        metadata (dict):
            Dictionary of metadata from the MC run that generated the samples.
    """

    def __init__(
        self,
        sublattices,
        natural_parameters,
        num_energy_coefs,
        sample_trace,
        sampling_metadata=None,
    ):
        """Initialize a sample container.

        Args:
            sublattices (list of Sublattice):
                Sublattices of the ensemble sampled.
            natural_parameters (ndarray):
                array of natural parameters used in the ensemble.
            num_energy_coefs (int):
                the number of coefficients in the natural parameters that
                correspond to the energy only.
            sample_trace (Trace):
                a trace object for the traced values during MC sampling
            sampling_metadata (Ensemble):
                sampling metadata (i.e. ensemble name, mckernel type, etc)
        """
        self.sublattices = sublattices
        self.natural_parameters = natural_parameters
        self._total_steps = 0

        if sampling_metadata is not None:
            if isinstance(sampling_metadata, Metadata):
                self.metadata = sampling_metadata
            else:
                self.metadata = Metadata(SampleContainer.__name__, **sampling_metadata)
        else:
            self.metadata = Metadata(SampleContainer.__name__)

        self._num_energy_coefs = num_energy_coefs
        # need counter because can't use shape of arraus when allocating space
        self._nsamples = 0
        self._trace = sample_trace
        self._aux_checkpoint = None
        self._backend = None  # for streaming

    @property
    def num_samples(self):
        """Get the total number of samples."""
        return self._nsamples

    @property
    def total_mc_steps(self):
        """Return the total number of MC steps taken during sampling."""
        return self._total_steps

    @property
    def shape(self):
        """Get the shape of the samples in chain."""
        return self._trace.occupancy.shape[1:]

    @property
    def traced_values(self):
        """Get the names of traced values being sampled."""
        return self._trace.names

    def sampling_efficiency(self, discard=0, flat=True):
        """Return the sampling efficiency for chains."""
        total_accepted = self._trace.accepted[discard:].sum(axis=0)
        efficiency = total_accepted / (self._total_steps - discard)
        if flat:
            efficiency = efficiency.mean()
        return efficiency

    def get_trace_value(self, name, discard=0, thin_by=1, flat=True):
        """Get sampled values of a traced value given by name."""
        value = getattr(self._trace, name)[discard + thin_by - 1 :: thin_by]
        if flat:
            value = self._flatten(value)
        return value

    def mean_trace_value(self, name, discard=0, thin_by=1, flat=True):
        """Get mean of a traced value given by name."""
        return self.get_trace_value(name, discard, thin_by, flat).mean(axis=0)

    def trace_value_variance(self, name, discard=0, thin_by=1, flat=True):
        """Get variance of a traced value given by name."""
        return self.get_trace_value(name, discard, thin_by, flat).var(axis=0)

    def get_occupancies(self, discard=0, thin_by=1, flat=True):
        """Get the occupancy chain of the samples."""
        return self.get_trace_value("occupancy", discard, thin_by, flat)

    def get_enthalpies(self, discard=0, thin_by=1, flat=True):
        """Get the generalized enthalpy changes from samples in chain."""
        return self.get_trace_value("enthalpy", discard, thin_by, flat)

    def get_feature_vectors(self, discard=0, thin_by=1, flat=True):
        """Get the feature vector changes from samples in chain."""
        return self.get_trace_value("features", discard, thin_by, flat)

    def get_energies(self, discard=0, thin_by=1, flat=True):
        """Get the energies from samples in chain."""
        if len(self.natural_parameters) == self._num_energy_coefs:
            return self.get_enthalpies(discard, thin_by, flat)
        # otherwise we have to calculate without the additional terms...
        feature_trace = self.get_feature_vectors(discard, thin_by, flat=False)
        energies = np.expand_dims(
            np.vstack(
                [
                    np.tensordot(
                        features[:, : self._num_energy_coefs],
                        self.natural_parameters[: self._num_energy_coefs],
                        axes=([1], [0]),
                    )
                    for features in feature_trace
                ]
            ),
            axis=-1,
        )
        if flat:
            energies = self._flatten(energies)
        return energies

    def get_sublattice_compositions(self, sublattice, discard=0, thin_by=1, flat=True):
        """Get the compositions of a specific sublattice."""
        counts = self.get_sublattice_species_counts(sublattice, discard, thin_by, flat)
        return counts / len(sublattice.sites)

    def get_compositions(self, discard=0, thin_by=1, flat=True):
        """Get the compositions for each occupancy in the chain."""
        counts = self.get_species_counts(discard, thin_by, flat)
        return {spec: count / self.shape[1] for spec, count in counts.items()}

    def mean_enthalpy(self, discard=0, thin_by=1, flat=True):
        """Get the mean generalized enthalpy."""
        return self.get_enthalpies(discard, thin_by, flat).mean(axis=0)

    def enthalpy_variance(self, discard=0, thin_by=1, flat=True):
        """Get the variance in enthalpy."""
        return self.get_enthalpies(discard, thin_by, flat).var(axis=0)

    def mean_energy(self, discard=0, thin_by=1, flat=True):
        """Calculate the mean energy from samples."""
        return self.get_energies(discard, thin_by, flat).mean(axis=0)

    def energy_variance(self, discard=0, thin_by=1, flat=True):
        """Calculate the variance of sampled energies."""
        return self.get_energies(discard, thin_by, flat).var(axis=0)

    def mean_feature_vector(self, discard=0, thin_by=1, flat=True):
        """Get the mean feature vector from samples."""
        return self.get_feature_vectors(discard, thin_by, flat).mean(axis=0)

    def feature_vector_variance(self, discard=0, thin_by=1, flat=True):
        """Get the variance of feature vector elements."""
        return self.get_feature_vectors(discard, thin_by, flat).var(axis=0)

    def mean_composition(self, discard=0, thin_by=1, flat=True):
        """Get mean composition for all species regardless of sublattice."""
        comps = self.get_compositions(discard, thin_by, flat)
        return {spec: comp.mean(axis=0) for spec, comp in comps.items()}

    def composition_variance(self, discard=0, thin_by=1, flat=True):
        """Get the variance in composition of all species."""
        comps = self.get_compositions(discard, thin_by, flat)
        return {spec: comp.var(axis=0) for spec, comp in comps.items()}

    def mean_sublattice_composition(self, sublattice, discard=0, thin_by=1, flat=True):
        """Get the mean composition of a specific sublattice."""
        return self.get_sublattice_compositions(
            sublattice, discard, thin_by, flat
        ).mean(axis=0)

    def sublattice_composition_variance(
        self, sublattice, discard=0, thin_by=1, flat=True
    ):
        """Get the variance in composition of a specific sublattice."""
        return self.get_sublattice_compositions(sublattice, discard, thin_by, flat).var(
            axis=0
        )

    def get_minimum_enthalpy(self, discard=0, thin_by=1, flat=True):
        """Get the minimum energy from samples."""
        return self.get_enthalpies(discard, thin_by, flat).min(axis=0)

    def get_minimum_enthalpy_occupancy(self, discard=0, thin_by=1, flat=True):
        """Find the occupancy with minimum energy from samples."""
        inds = self.get_enthalpies(discard, thin_by, flat).argmin(axis=0)
        if flat:
            occus = self.get_occupancies(discard, thin_by, flat)[inds]
        else:
            occus = self.get_occupancies(discard, thin_by, flat)[
                inds, np.arange(self.shape[0])
            ][0]
        return occus

    def get_minimum_energy(self, discard=0, thin_by=1, flat=True):
        """Get the minimum energy from samples."""
        return self.get_energies(discard, thin_by, flat).min(axis=0)

    def get_minimum_energy_occupancy(self, discard=0, thin_by=1, flat=True):
        """Find the occupancy with minimum energy from samples."""
        inds = self.get_energies(discard, thin_by, flat).argmin(axis=0)
        if flat:
            occus = self.get_occupancies(discard, thin_by, flat)[inds]
        else:
            occus = self.get_occupancies(discard, thin_by, flat)[
                inds, np.arange(self.shape[0])
            ][0]
        return occus

    def get_species_counts(self, discard=0, thin_by=1, flat=True):
        """Get the species counts for each occupancy in the chain."""
        samples = (self.num_samples - discard) // thin_by
        shape = self.shape[0] * samples if flat else (self.shape[0], samples)
        counts = defaultdict(lambda: np.zeros(shape=shape))
        for sublattice in self.sublattices:
            subcounts = self.get_sublattice_species_counts(
                sublattice, discard, thin_by, flat
            )
            for species, count in zip(sublattice.species, subcounts.T):
                counts[species] += count
        return counts

    def get_sublattice_species_counts(
        self, sublattice, discard=0, thin_by=1, flat=True
    ):
        """Get the counts of each species in a sublattices.

        Returns:
            ndarray: where last axis is the count for each species in the
            same order as the underlying site space.
        """
        if sublattice not in self.sublattices:
            raise ValueError(
                "Sublattice provided is not recognized.\n Provide one included"
                " in the sublattices attribute of this SampleContainer."
            )
        occus = self.get_occupancies(discard, thin_by, flat=False)
        counts = np.zeros((*occus.shape[:-1], len(sublattice.site_space)))
        #  This can probably be re-written in a clean/faster way
        for i, occupancies in enumerate(occus):
            for j, occupancy in enumerate(occupancies):
                codes, count = np.unique(
                    occupancy[sublattice.sites], return_counts=True
                )
                # check for zero counts
                if len(codes) != len(sublattice.site_space):
                    missed = list(set(sublattice.encoding) - set(codes))
                    codes = np.append(codes, missed)
                    count = np.append(count, len(missed) * [0])

                original_codes = sublattice.encoding.tolist()
                order = [codes.tolist().index(code) for code in original_codes]
                counts[i][j] = count[order]  # order them accordingly
        if flat:
            counts = self._flatten(counts)
        return counts

    def save_sampled_trace(self, trace, thinned_by):
        """Save a sampled trace.

        Args:
            trace (Trace)
                Trace of sampled values
            thinned_by (int):
                the amount that the sampling was thinned by. Used to update
                the total mc iterations.
        """
        for name, value in trace.items():
            getattr(self._trace, name)[self._nsamples] = value
        self._nsamples += 1
        self._total_steps += thinned_by

    def clear(self):
        """Clear all samples from container."""
        self._total_steps = 0
        self._total_steps = 0
        self._nsamples = 0
        for name, value in self._trace.items():
            setattr(
                self._trace, name, np.empty((0, *value.shape[1:]), dtype=value.dtype)
            )

    def allocate(self, nsamples):
        """Allocate more space in arrays for more samples."""
        for name, value in self._trace.items():
            arr = np.empty((nsamples, *value.shape[1:]), dtype=value.dtype)
            setattr(self._trace, name, np.append(value, arr, axis=0))

    def flush_to_backend(self, backend):
        """Flush current samples and trace to backend file.

        Args:
            backend (object):
                backend file object, currently only hdf5 supported.
        """
        start = backend["trace"].attrs["nsamples"]
        end = self._nsamples + start
        for name, value in self._trace.items():
            backend["trace"][name][start:end] = value

        backend["trace"].attrs["total_mc_steps"] += self._total_steps
        backend["trace"].attrs["nsamples"] += self._nsamples
        backend.flush()
        self._total_steps = 0
        self._nsamples = 0

    @requires(h5py is not None, "'h5py' not found. Please install it.")
    def get_backend(self, file_path, alloc_nsamples=0, swmr_mode=False):
        """Get a backend file object.

        Currently only hdf5 files supported

        Args:
            file_path (str):
                path to backend file.
            alloc_nsamples (int): optional
                number of new samples to allocate. Will only extend datasets
                if number given is larger than space left to write samples
                into.
            swmr_mode (bool): optional
                If true allows to read file from other processes. Single Writer
                Multiple Readers.

        Returns:
            h5.File object
        """
        if os.path.isfile(file_path):
            backend = self._check_backend(file_path)
            trace_grp = backend["trace"]
            available = len(trace_grp["occupancy"]) - trace_grp.attrs["nsamples"]
            # this probably fails since maxshape is not set.
            if available < alloc_nsamples:
                SampleContainer._grow_backend(backend, alloc_nsamples - available)
        else:
            backend = h5py.File(file_path, "w-", libver="latest")
            self._init_backend(backend, alloc_nsamples)

        if swmr_mode:
            backend.swmr_mode = swmr_mode
        return backend

    def _check_backend(self, file_path):
        """Check if existing backend file is populated correctly."""
        backend = h5py.File(file_path, mode="r+", libver="latest")
        if self.shape != backend["trace"]["occupancy"].shape[1:]:
            shape = backend["trace"]["occupancy"].shape[1:]
            backend.close()
            raise RuntimeError(
                f"Backend file {file_path} has incompatible dimensions "
                f"{self.shape}, {shape}."
            )
        return backend

    def _init_backend(self, backend, nsamples):
        """Initialize a backend file."""
        sublattices = [sublatt.as_dict() for sublatt in self.sublattices]
        backend.create_dataset("sublattices", data=json.dumps(sublattices))
        backend.create_dataset("natural_parameters", data=self.natural_parameters)
        backend["natural_parameters"].attrs[
            "num_energy_coefs"
        ] = self._num_energy_coefs  # noqa
        backend.create_dataset(
            "sampling_metadata", data=json.dumps(self.metadata, cls=MontyEncoder)
        )
        trace_grp = backend.create_group("trace")
        for name, value in self._trace.items():
            trace_grp.create_dataset(
                name,
                shape=(nsamples, *value.shape[1:]),
                dtype=value.dtype,
                maxshape=(None, *value.shape[1:]),
            )
        trace_grp.attrs["nsamples"] = 0
        trace_grp.attrs["total_mc_steps"] = 0

    @staticmethod
    def _grow_backend(backend, nsamples):
        """Extend space available in a backend file."""
        for name in backend["trace"]:
            backend["trace"][name].resize(
                len(backend["trace"][name]) + nsamples, axis=0
            )

    @staticmethod
    def _flatten(traced_values):
        """Flatten values in trace values with multiple walkers."""
        shape_l = list(traced_values.shape[1:])
        shape_l[0] = np.prod(traced_values.shape[:2])
        return np.squeeze(traced_values.reshape(shape_l))

    def __len__(self):
        """Return the number of samples."""
        return self._nsamples

    def as_dict(self):
        """Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        container_d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "sublattices": [s.as_dict() for s in self.sublattices],
            "natural_parameters": jsanitize(self.natural_parameters),
            "metadata": self.metadata.as_dict(),
            "num_energy_coefs": self._num_energy_coefs,
            "total_mc_steps": self._total_steps,
            "nsamples": self._nsamples,
            "trace": jsanitize(self._trace.as_dict()),
            "aux_checkpoint": self._aux_checkpoint,
        }
        # TODO need to think how to generally serialize the aux checkpoint
        return container_d

    @classmethod
    def from_dict(cls, d):
        """Instantiate a SampleContainer from dict representation.

        Args:
            d (dict):
                dictionary representation.
        Returns:
            Sublattice
        """
        sublattices = [Sublattice.from_dict(s) for s in d["sublattices"]]
        trace = Trace(**{key: np.array(val) for key, val in d["trace"].items()})
        d["metadata"] = MontyDecoder().process_decoded(d["metadata"])

        if not isinstance(d["metadata"], Metadata):  # backwards compatibility
            d["metadata"] = Metadata(cls.__name__, **d["metadata"])

        container = cls(
            sublattices,
            d["natural_parameters"],
            d["num_energy_coefs"],
            trace,
            d["metadata"],
        )
        # set the container internals
        container._total_steps = d["total_mc_steps"]
        container._nsamples = trace.occupancy.shape[0]
        container._aux_checkpoint = d["aux_checkpoint"]
        return container

    def to_hdf5(self, file_path):
        """Save SampleContainer as an HDF5 file.

        Args:
            file_path (str):
                path to file save location. If file exists and dimensions
                match samples will be appended.
        """
        # keep these to not reset writing
        total_mc_steps, nsamples = self._total_steps, self._nsamples
        backend = self.get_backend(file_path, self.num_samples)
        self.flush_to_backend(backend)
        self._total_steps, self._nsamples = total_mc_steps, nsamples
        backend.close()

    @classmethod
    @requires(h5py is not None, "'h5py' not found. Please install it.")
    def from_hdf5(cls, file_path, swmr_mode=True):
        """Instantiate a SampleContainer from an hdf5 file.

        Args:
            file_path (str):
                path to file
            swmr_mode (bool): optional
                If true allows to read file from other processes. Single Writer
                Multiple Readers.

        Returns:
            SampleContainer
        """
        with h5py.File(file_path, "r", swmr=swmr_mode) as f:
            # Check if written states matches the size of datasets
            nsamples = f["trace"].attrs["nsamples"]
            if len(f["trace"]["occupancy"]) > nsamples:
                warnings.warn(
                    f"The hdf5 file provided appears to be from an unifinished"
                    f" MC run.\n Only {nsamples} of "
                    f" {len(f['trace']['occupancy'])} samples "
                    f"have been written and will be loaded.",
                    UserWarning,
                )

            sublattices = [
                Sublattice.from_dict(s) for s in json.loads(f["sublattices"][()])
            ]
            trace = Trace(
                **{name: value[:nsamples] for name, value in f["trace"].items()}
            )
            container = cls(
                sublattices=sublattices,
                natural_parameters=f["natural_parameters"][()],
                num_energy_coefs=f["natural_parameters"].attrs[
                    "num_energy_coefs"
                ],  # noqa
                sample_trace=trace,
                sampling_metadata=json.loads(
                    f["sampling_metadata"][()], cls=MontyDecoder
                ),
            )
            container._nsamples = nsamples
            container._total_steps = f["trace"].attrs["total_mc_steps"]
        return container
