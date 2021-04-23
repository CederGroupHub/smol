"""Implementation of SampleContainer class.

A SampleContainer is used to store data from Monte Carlo sampling simulation.
It has some minimimal methods and properties useful to start analyzing the
samples.
"""

__author__ = "Luis Barroso-Luque"

import os
import warnings
from collections import defaultdict
import json
import numpy as np

from monty.json import MSONable
from smol.moca.ensemble.sublattice import Sublattice

try:
    import h5py
except ImportError:
    h5py = None
    h5err = ImportError("'h5py' not found. Please install it.")


class SampleContainer(MSONable):
    """A SampleContainter class stores Monte Carlo simulation samples.

    A SampleContainer holds samples and sampling information from an MCMC
    sampling run. It is useful to obtain the raw data and minimal empirical
    properties of the underlying distribution in order to carry out further
    analysis of the MCMC sampling results.

    When getting any value from the provided attributes, the highly repeated
    args are:
        discard (int): optional
            Number of samples to discard to obtain the value requested.
        thin_by (int): optional
            Use every thin by sample to obtain the value requested.
        flat (bool): optional
            If more than 1 walkers are used flattening will flatten all
            chains into one. Defaults to True.

    Attributes:
        num_sites (int):
            Size of system (usually in number of prims in supercell, but
            can be anything representative i.e. number of sites)
        sublattices (list of Sublattice)
            Sublattices of the ensemble sampled.
        natural_parameters (ndarray):
                array of natural parameters used in the ensemble.
        total_mc_steps (int)
            Number of iterations used in sampling
        metadata (dict):
            Dictionary of metadata from the MC run that generated the samples.
        aux_checkpoint (dict):
            Checkpoint dictionary of auxiliary states and variables to continue
            sampling from the last state of a previous MCMC run.
            (not implemented yet)
    """

    def __init__(self, num_sites, sublattices, natural_parameters,
                 num_energy_coefs, sampling_metadata=None, nwalkers=1):
        """Initialize a sample container.

        Args:
            num_sites (int):
                Total number of sites in supercell of the ensemble.
            sublattices (list of Sublattice):
                Sublattices of the ensemble sampled.
            natural_parameters (ndarray):
                array of natural parameters used in the ensemble.
            num_energy_coefs (int):
                the number of coeficients in the natural parameters that
                correspond to the energy only.
            sampling_metadata (Ensemble):
                Sampling metadata (i.e. ensemble name, mckernel type, etc)
            nwalkers (int):
                Number of walkers used to generate chain. Default is 1
        """
        self.num_sites = num_sites
        self.sublattices = sublattices
        self.natural_parameters = natural_parameters
        self.metadata = {} if sampling_metadata is None else sampling_metadata
        self._num_energy_coefs = num_energy_coefs
        self.total_mc_steps = 0
        self._nsamples = 0
        self._chain = np.empty((0, nwalkers, num_sites), dtype=int)
        self._features = np.empty((0, nwalkers, len(natural_parameters)))
        self._enthalpy = np.empty((0, nwalkers))
        self._temperature = np.empty((0, nwalkers))
        self._accepted = np.zeros((0, nwalkers), dtype=int)
        self._bias = np.empty((0, nwalkers))
        self._time = np.empty((0, nwalkers))  # For time stamping
        self.aux_checkpoint = None
        self._backend = None  # for streaming

    @property
    def num_samples(self):
        """Get the total number of samples."""
        return self._nsamples

    @property
    def shape(self):
        """Get the shape of the samples in chain."""
        return self._chain.shape[1:]

    def sampling_efficiency(self, discard=0, flat=True):
        """Return the sampling efficiency for chains."""
        total_accepted = self._accepted[discard:].sum(axis=0)
        efficiency = total_accepted/(self.total_mc_steps - discard)
        if flat:
            efficiency = efficiency.mean()
        return efficiency

    def get_bias(self, discard=0, thin_by=1, flat=True):
        """Get all bias from samples."""
        bias = self._bias[discard + thin_by - 1::thin_by]
        if flat:
            bias = self._flatten(bias)
        return bias

    def get_time(self, discard=0, thin_by=1, flat=True):
        """Get all time stamps from samples.

        For performance evaluation.
        """
        time = self._time[discard + thin_by - 1::thin_by]
        if flat:
            time = self._flatten(time)
        return time

    def get_occupancies(self, discard=0, thin_by=1, flat=True):
        """Get an occupancy chain from samples."""
        chain = self._chain[discard + thin_by - 1::thin_by]
        if flat:
            chain = self._flatten(chain)
        return chain

    def get_temperatures(self, discard=0, thin_by=1, flat=True):
        """Get the generalized entalpy changes from samples in chain."""
        temps = self._temperature[discard + thin_by - 1::thin_by]
        if flat:
            temps = self._flatten(temps)
        return temps

    def get_enthalpies(self, discard=0, thin_by=1, flat=True):
        """Get the generalized entalpy changes from samples in chain."""
        enthalpies = self._enthalpy[discard + thin_by - 1::thin_by]
        if flat:
            enthalpies = self._flatten(enthalpies)
        return enthalpies

    def get_feature_vectors(self, discard=0, thin_by=1, flat=True):
        """Get the feature vector changes from samples in chain."""
        feats = self._features[discard + thin_by - 1::thin_by]
        if flat:
            feats = self._flatten(feats)
        return feats

    def get_energies(self, discard=0, thin_by=1, flat=True):
        """Get the energies from samples in chain."""
        features = self.get_feature_vectors(discard, thin_by, flat=False)
        energies = np.array([np.dot(self.natural_parameters[:self._num_energy_coefs],  # noqa
                             features[:, :self._num_energy_coefs].T)
                             for features in features])
        if flat:
            energies = self._flatten(energies)
        return energies

    def get_sublattice_compositions(self, sublattice, discard=0, thin_by=1,
                                    flat=True):
        """Get the compositions of a specific sublattice."""
        counts = self.get_sublattice_species_counts(sublattice, discard,
                                                    thin_by, flat)
        return counts/len(sublattice.sites)

    def get_compositions(self, discard=0, thin_by=1, flat=True):
        """Get the compositions for each occupancy in the chain."""
        counts = self.get_species_counts(discard, thin_by, flat)
        return {spec: count/self.shape[1] for spec, count in counts.items()}

    def zero_bias_ratio(self, discard=0, thin_by=1, flat=True):
        """Count ratio of state with zero bias.

        Use in biased walk implementation of charge neutrality, etc.
        """
        bias = self.get_bias(discard, thin_by, flat)
        return np.sum(bias == 0)/len(bias)

    def mean_enthalpy(self, discard=0, thin_by=1, flat=True,
                      cut_offbias=False):
        """Get the mean generalized enthalpy.

        Average will be reweighted by bias.
        If enabled cut_offbias, will not count states whose bias are not 0.
        Only used in constraint sampling.
        """
        hs = self.get_enthalpies(discard, thin_by, flat)
        bias = self.get_bias(discard, thin_by, flat)
        mask = (bias == 0) if cut_offbias else np.ones(bias.shape)
        return np.average(hs, axis=0, weights=np.exp(bias) * mask)

    def enthalpy_variance(self, discard=0, thin_by=1, flat=True,
                          cut_offbias=False):
        """Get the variance in enthalpy."""
        hs = self.get_enthalpies(discard, thin_by, flat)
        bias = self.get_bias(discard, thin_by, flat)
        mask = (bias == 0) if cut_offbias else np.ones(bias.shape)
        hs_av = np.average(hs, axis=0, weights=np.exp(bias) * mask)
        return np.average((hs - hs_av)**2, axis=0,
                          weights=np.exp(bias) * mask)

    def mean_energy(self, discard=0, thin_by=1, flat=True,
                    cut_offbias=False):
        """Calculate the mean energy from samples."""
        es = self.get_energies(discard, thin_by, flat)
        bias = self.get_bias(discard, thin_by, flat)
        mask = (bias == 0) if cut_offbias else np.ones(bias.shape)
        return np.average(es, axis=0, weights=np.exp(bias) * mask)

    def energy_variance(self, discard=0, thin_by=1, flat=True,
                        cut_offbias=False):
        """Calculate the variance of sampled energies."""
        es = self.get_energies(discard, thin_by, flat)
        bias = self.get_bias(discard, thin_by, flat)
        mask = (bias == 0) if cut_offbias else np.ones(bias.shape)
        es_av = np.average(es, axis=0, weights=np.exp(bias) * mask)
        return np.average((es - es_av)**2, axis=0,
                          weights=np.exp(bias) * mask)

    def mean_feature_vector(self, discard=0, thin_by=1, flat=True,
                            cut_offbias=False):
        """Get the mean feature vector from samples."""
        vs = self.get_feature_vectors(discard, thin_by, flat)
        bias = self.get_bias(discard, thin_by, flat)
        mask = (bias == 0) if cut_offbias else np.ones(bias.shape)
        weights = np.exp(bias) * mask
        weights = np.repeat(weights.reshape((*vs.shape[:-1], -1)),
                            vs.shape[-1], axis=-1)
        return np.average(vs, axis=0, weights=weights)

    def feature_vector_variance(self, discard=0, thin_by=1, flat=True,
                                cut_offbias=False):
        """Get the variance of feature vector elements."""
        vs = self.get_feature_vectors(discard, thin_by, flat)
        bias = self.get_bias(discard, thin_by, flat)
        mask = (bias == 0) if cut_offbias else np.ones(bias.shape)
        weights = np.exp(bias) * mask
        weights = np.repeat(weights.reshape((*vs.shape[:-1], -1)),
                            vs.shape[-1], axis=-1)
        # Weights reshaped as feature vectors.
        vs_av = np.average(vs, axis=0, weights=weights)
        return np.average((vs - vs_av)**2, axis=0, weights=weights)

    def mean_composition(self, discard=0, thin_by=1, flat=True,
                         cut_offbias=False):
        """Get mean composition for all species regardless of sublattice."""
        comps = self.get_compositions(discard, thin_by, flat)
        bias = self.get_bias(discard, thin_by, flat)
        mask = (bias == 0) if cut_offbias else np.ones(bias.shape)
        return {spec: np.average(comp, axis=0, weights=np.exp(bias) * mask)
                for spec, comp in comps.items()}

    def composition_variance(self, discard=0, thin_by=1, flat=True,
                             cut_offbias=False):
        """Get the variance in composition of all species."""
        comps = self.get_compositions(discard, thin_by, flat)
        bias = self.get_bias(discard, thin_by, flat)
        mask = (bias == 0) if cut_offbias else np.ones(bias.shape)
        return {spec:
                np.average((comp -
                            np.average(comp, axis=0,
                                       weights=np.exp(bias) * mask))**2,
                           axis=0, weights=np.exp(bias) * mask)
                for spec, comp in comps.items()}

    def mean_sublattice_composition(self, sublattice, discard=0, thin_by=1,
                                    flat=True, cut_offbias=False):
        """Get the mean composition of a specific sublattice."""
        xs = self.get_sublattice_compositions(sublattice, discard,
                                              thin_by, flat)
        bias = self.get_bias(discard, thin_by, flat)
        mask = (bias == 0) if cut_offbias else np.ones(bias.shape)
        weights = np.exp(bias) * mask
        weights = np.repeat(weights.reshape((*xs.shape[:-1], -1)),
                            xs.shape[-1], axis=-1)
        return np.average(xs, axis=0, weights=weights)

    def sublattice_composition_variance(self, sublattice, discard=0, thin_by=1,
                                        flat=True, cut_offbias=False):
        """Get the varience in composition of a specific sublattice."""
        xs = self.get_sublattice_compositions(sublattice, discard,
                                              thin_by, flat)
        bias = self.get_bias(discard, thin_by, flat)
        mask = (bias == 0) if cut_offbias else np.ones(bias.shape)
        weights = np.exp(bias) * mask
        weights = np.repeat(weights.reshape((*xs.shape[:-1], -1)),
                            xs.shape[-1], axis=-1)

        xs_av = np.average(xs, axis=0, weights=weights)
        return np.average((xs - xs_av)**2, axis=0, weights=weights)

    def get_minimum_enthalpy(self, discard=0, thin_by=1, flat=True,
                             cut_offbias=False):
        """Get the minimum energy from samples."""
        bias = self.get_bias(discard, thin_by, flat)
        mask = (bias == 0) if cut_offbias else np.ones(bias.shape)
        hs = self.get_enthalpies(discard, thin_by, flat)
        hs[mask == 0] = np.inf
        return hs.min(axis=0)

    def get_minimum_enthalpy_occupancy(self, discard=0, thin_by=1, flat=True,
                                       cut_offbias=False):
        """Find the occupancy with minimum energy from samples."""
        bias = self.get_bias(discard, thin_by, flat)
        mask = (bias == 0) if cut_offbias else np.ones(bias.shape)
        hs = self.get_enthalpies(discard, thin_by, flat)
        hs[mask == 0] = np.inf

        inds = hs.argmin(axis=0)

        if flat:
            occus = self.get_occupancies(discard, thin_by, flat)[inds]
        else:
            occus = self.get_occupancies(discard, thin_by, flat)[inds, np.arange(self.shape[0])]  # noqa
        return occus

    def get_minimum_energy(self, discard=0, thin_by=1, flat=True,
                           cut_offbias=False):
        """Get the minimum energy from samples."""
        bias = self.get_bias(discard, thin_by, flat)
        mask = (bias == 0) if cut_offbias else np.ones(bias.shape)
        es = self.get_energies(discard, thin_by, flat)
        es[mask == 0] = np.inf
        return es.min(axis=0)

    def get_minimum_energy_occupancy(self, discard=0, thin_by=1, flat=True,
                                     cut_offbias=False):
        """Find the occupancy with minimum energy from samples."""
        bias = self.get_bias(discard, thin_by, flat)
        mask = (bias == 0) if cut_offbias else np.ones(bias.shape)
        es = self.get_energies(discard, thin_by, flat)
        es[mask == 0] = np.inf

        inds = es.argmin(axis=0)
        if flat:
            occus = self.get_occupancies(discard, thin_by, flat)[inds]
        else:
            occus = self.get_occupancies(discard, thin_by, flat)[inds, np.arange(self.shape[0])]  # noqa
        return occus

    def get_species_counts(self, discard=0, thin_by=1, flat=True):
        """Get the species counts for each occupancy in the chain."""
        samples = (self.num_samples - discard) // thin_by
        shape = self.shape[0]*samples if flat else (self.shape[0], samples)
        counts = defaultdict(lambda: np.zeros(shape=shape))
        for sublattice in self.sublattices:
            subcounts = self.get_sublattice_species_counts(sublattice, discard,
                                                           thin_by, flat)
            for species, count in zip(sublattice.species, subcounts.T):
                counts[species] += count
        return counts

    def get_sublattice_species_counts(self, sublattice, discard=0, thin_by=1,
                                      flat=True):
        """Get the counts of each species in a sublattices.

        Returns:
            ndarray: where last axis is the count for each species in same
                     order as the underlying site space.
        """
        if sublattice not in self.sublattices:
            raise ValueError('Sublattice provided is not recognized.\n'
                             'Provide one included in the sublattices '
                             'attribute of this SampleContainer.')
        occu_chain = self.get_occupancies(discard, thin_by, flat=False)
        counts = np.zeros((*occu_chain.shape[:-1], len(sublattice.site_space)))
        #  This can probably be re-written in a clean/faster way
        for i, occupancies in enumerate(occu_chain):
            for j, occupancy in enumerate(occupancies):
                codes, count = np.unique(occupancy[sublattice.sites],
                                         return_counts=True)
                # check for zero counts
                if len(codes) != len(sublattice.sites):
                    n = len(sublattice.site_space)
                    missed = list(set(range(n)) - set(codes))
                    codes = np.append(codes, missed)
                    count = np.append(count, len(missed) * [0])

                counts[i][j] = count[codes.argsort()]  # order them accordingly
        if flat:
            counts = self._flatten(counts)
        return counts

    def save_sample(self, accepted, temperature, occupancies, bias, times,
                    enthalpy, features, thinned_by):
        """Save a sample from the generated chain.

        Args:
            accepted (ndarray):
                array of total acceptances.
            temperature (ndarray)
                array of temperatures at which samples were taken.
            occupancies (ndarray):
                array of occupancies
            bias (ndarray):
                array of bias
            times (ndarray):
                array of times
            enthalpy (ndarray):
                array of generalized enthalpy changes
            features (ndarray):
                array of feature vector changes
            thinned_by (int):
                the amount that the sampling was thinned by. Used to update
                the total mc iterations.
        """
        self._accepted[self._nsamples, :] = accepted
        self._temperature[self._nsamples, :] = temperature
        self._chain[self._nsamples, :, :] = occupancies
        self._bias[self._nsamples, :] = bias
        self._time[self._nsamples, :] = times
        self._enthalpy[self._nsamples, :] = enthalpy
        self._features[self._nsamples, :, :] = features
        self._nsamples += 1
        self.total_mc_steps += thinned_by

    def clear(self):
        """Clear all samples from container."""
        nwalkers, num_sites = self.shape
        self.total_mc_steps = 0
        self._nsamples = 0
        self._chain = np.empty((0, nwalkers, num_sites), dtype=int)
        self._bias = np.empty((0, nwalkers))
        self._time = np.empty((0, nwalkers))
        self._features = np.empty((0, nwalkers, len(self.natural_parameters)))
        self._enthalpy = np.empty((0, nwalkers))
        self._temperature = np.empty((0, nwalkers))
        self._accepted = np.zeros((0, nwalkers), dtype=int)

    def allocate(self, nsamples):
        """Allocate more space in arrays for more samples."""
        arr = np.empty((nsamples, *self._chain.shape[1:]), dtype=int)
        self._chain = np.append(self._chain, arr, axis=0)
        arr = np.empty((nsamples, *self._bias.shape[1:]))
        self._bias = np.append(self._bias, arr, axis=0)
        arr = np.empty((nsamples, *self._time.shape[1:]))
        self._time = np.append(self._time, arr, axis=0)
        arr = np.empty((nsamples, *self._features.shape[1:]))
        self._features = np.append(self._features, arr, axis=0)
        arr = np.empty((nsamples, *self._enthalpy.shape[1:]))
        self._enthalpy = np.append(self._enthalpy, arr, axis=0)
        arr = np.empty((nsamples, *self._temperature.shape[1:]))
        self._temperature = np.append(self._temperature, arr, axis=0)
        arr = np.zeros((nsamples, *self._accepted.shape[1:]), dtype=int)
        self._accepted = np.append(self._accepted, arr, axis=0)

    def flush_to_backend(self, backend):
        """Flush current samples and trace to backend file."""
        start = backend["chain"].attrs["nsamples"]
        end = len(self._chain) + start
        backend["accepted"][start:end, :] = self._accepted
        backend["temperature"][start:end, :] = self._temperature
        backend["chain"][start:end, :, :] = self._chain
        backend["bias"][start:end, :] = self._bias
        backend["time"][start:end, :] = self._time
        backend["enthalpy"][start:end, :] = self._enthalpy
        backend["features"][start:end, :, :] = self._features
        backend["chain"].attrs["total_mc_steps"] += self.total_mc_steps
        backend["chain"].attrs["nsamples"] += self._nsamples
        backend.flush()
        self.total_mc_steps = 0
        self._nsamples = 0

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
        if h5py is None:
            raise h5err

        if os.path.isfile(file_path):
            backend = self._check_backend(file_path)
            chain = backend["chain"]
            available = len(chain) - chain.attrs["nsamples"]
            # this probably fails since maxshape is not set.
            if available < alloc_nsamples:
                self._grow_backend(backend, alloc_nsamples - available)
        else:
            backend = h5py.File(file_path, "w", libver='latest')
            self._init_backed(backend, alloc_nsamples)

        if swmr_mode:
            backend.swmr_mode = swmr_mode
        return backend

    def _check_backend(self, file_path):
        """Check if existing backend file is populated correctly."""
        backend = h5py.File(file_path, mode="r+", libver='latest')
        if self.shape != backend["chain"].shape[1:]:
            shape = backend['chain'].shape[1:]
            backend.close()
            raise RuntimeError(f"Backend file {file_path} has incompatible "
                               f"dimensions {self.shape}, "
                               f"{shape}.")
        return backend

    def _init_backed(self, backend, nsamples):
        """Initialize a backend file."""
        sublattices = [sublatt.as_dict() for sublatt in self.sublattices]
        backend.create_dataset("sublattices", data=json.dumps(sublattices))
        backend["sublattices"].attrs["num_sites"] = self.num_sites
        backend.create_dataset("natural_parameters",
                               data=self.natural_parameters)
        backend["natural_parameters"].attrs["num_energy_coefs"] = self._num_energy_coefs  # noqa
        backend.create_dataset("sampling_metadata",
                               data=json.dumps(self.metadata))
        backend.create_dataset("chain", (nsamples, *self.shape))
        backend["chain"].attrs["nsamples"] = 0
        backend["chain"].attrs["total_mc_steps"] = 0
        backend.create_dataset("bias",
                               (nsamples, *self._bias.shape[1:]))
        backend.create_dataset("time",
                               (nsamples, *self._time.shape[1:]))
        backend.create_dataset("accepted",
                               (nsamples, *self._accepted.shape[1:]))
        backend.create_dataset("temperature",
                               (nsamples, *self._temperature.shape[1:]))
        backend.create_dataset("enthalpy",
                               (nsamples, *self._enthalpy.shape[1:]))
        backend.create_dataset("features",
                               (nsamples, *self._features.shape[1:]))

    def _grow_backend(self, backend, nsamples):
        """Extend space available in a backend file."""
        backend["chain"].resize((backend["chain"].shape[0] + nsamples,
                                 *backend["chain"].shape[1:]))
        backend["bias"].resize((backend["chain"].shape[0] + nsamples,
                                *self._bias.shape[1:]))
        backend["time"].resize((backend["chain"].shape[0] + nsamples,
                                *self._time.shape[1:]))
        backend["accepted"].resize((backend["chain"].shape[0] + nsamples,
                                    *self._accepted.shape[1:]))
        backend["temperature"].resize((backend["chain"].shape[0] + nsamples,
                                       *self._temperature.shape[1:]))
        backend["enthalpy"].resize((backend["chain"].shape[0] + nsamples,
                                    *self._enthalpy.shape[1:]))
        backend["features"].resize((backend["chain"].shape[0] + nsamples,
                                    *self._features.shape[1:]))

    @staticmethod
    def _flatten(chain):
        """Flatten values in chain with multiple walkers."""
        s = list(chain.shape[1:])
        s[0] = np.prod(chain.shape[:2])
        return chain.reshape(s)

    def __len__(self):
        """Return the number of samples."""
        return self._nsamples

    def as_dict(self):
        """Get Json-serialization dict representation.

        Returns:
            MSONable dict
        """
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             'num_sites': self.num_sites,
             'sublattices': [s.as_dict() for s in self.sublattices],
             'natural_parameters': self.natural_parameters,
             'metadata': self.metadata,
             'num_energy_coefs': self._num_energy_coefs,
             'total_mc_steps': self.total_mc_steps,
             'nsamples': self._nsamples,
             'chain': self._chain.tolist(),
             'bias': self._bias.tolist(),
             'time': self._time.tolist(),
             'features': self._features.tolist(),
             'enthalpy': self._enthalpy.tolist(),
             'accepted': self._accepted.tolist(),
             'aux_checkpoint': self.aux_checkpoint}
        # TODO need to think how to genrally serialize the aux checkpoint
        return d

    @classmethod
    def from_dict(cls, d):
        """Instantiate a SampleContainer from dict representation.

        Args:
            d (dict):
                dictionary representation.
        Returns:
            Sublattice
        """
        sublattices = [Sublattice.from_dict(s) for s in d['sublattices']]
        container = cls(d['num_sites'], sublattices,
                        d['natural_parameters'], d['num_energy_coefs'],
                        d['metadata'])
        container._nsamples = np.array(d['nsamples'])
        container.total_mc_steps = d['total_mc_steps']
        container._chain = np.array(d['chain'], dtype=int)
        container._bias = np.array(d['bias'])
        container._time = np.array(d['time'])
        container._features = np.array(d['features'])
        container._enthalpy = np.array(d['enthalpy'])
        container._accepted = np.array(d['accepted'], dtype=int)
        return container

    @classmethod
    def from_hdf5(cls, file_path, swmr_mode=False):
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
        if h5py is None:
            raise h5err

        with h5py.File(file_path, "r", swmr=swmr_mode) as f:
            # Check if written states matches the size of datasets
            nsamples = f["chain"].attrs["nsamples"]
            if len(f["chain"]) > nsamples:
                warnings.warn("The hdf5 file provided appears to be from an "
                              f"unifinished MC run.\n Only {nsamples} of "
                              f"{len(f['chain'])} samples have been written.",
                              UserWarning)

            sublattices = [Sublattice.from_dict(s) for s in
                           json.loads(f["sublattices"][()])]
            container = cls(f["sublattices"].attrs["num_sites"],
                            sublattices=sublattices,
                            natural_parameters=f["natural_parameters"][()],
                            num_energy_coefs=f["natural_parameters"].attrs["num_energy_coefs"],  # noqa
                            sampling_metadata=json.loads(f["sampling_metadata"][()]),  # noqa
                            nwalkers=f["chain"].shape[1])
            container._chain = f["chain"][:nsamples]
            container._bias = f["bias"][:nsamples]
            container._time = f["time"][:nsamples]
            container._accepted = f["accepted"][:nsamples]
            container._temperature = f["temperature"][:nsamples]
            container._enthalpy = f["enthalpy"][:nsamples]
            container._features = f["features"][:nsamples]
            container._nsamples = nsamples
            container.total_mc_steps = f["chain"].attrs["total_mc_steps"]
        return container
