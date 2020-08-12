"""Implementation of SampleContainer class

A SampleContainer is used to store data from Monte Carlo sampling simulation.
It has some minimimal methods and properties useful to start analyzing the
samples.
"""

__author__ = "Luis Barroso-Luque"

import os
from datetime import datetime
from collections import defaultdict
import numpy as np

from monty.json import MSONable
from smol.moca.ensemble.sublattice import Sublattice


class SampleContainer(MSONable):
    """A SampleContainter class stores Monte Carlo simulation samples.

    It also provides some minor functionality to get sample statistics.
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
        temperature (float):
            temperature of ensemble that was sampled.
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
            dictionary of metadata from the MC run that generated the samples.
    """
    def __init__(self, temperature, num_sites, sublattices, natural_parameters,
                 num_energy_coefs, ensemble_metadata=None, nwalkers=1):
        """Initialize a sample container.

        Args:
            temperature (float):
                Temperature of the ensemble.
            num_sites (int):
                Total number of sites in supercell of the ensemble.
            sublattices (list of Sublattice):
                Sublattices of the ensemble sampled.
            natural_parameters (ndarray):
                array of natural parameters used in the ensemble.
            num_energy_coefs (int):
                the number of coeficients in the natural parameters that
                correspond to the energy only.
            ensemble_metadata (Ensemble):
                Metadata of the ensemble that was sampled.
            nwalkers (int):
                Number of walkers used to generate chain. Default is 1
        """

        self.temperature = temperature
        self.num_sites = num_sites
        self.sublattices = sublattices
        self.natural_parameters = natural_parameters
        self.metadata = {} if ensemble_metadata is None else ensemble_metadata
        self._num_energy_coefs = num_energy_coefs
        self.total_mc_steps = 0
        self._nsamples = 0
        self._chain = np.empty((0, nwalkers, num_sites), dtype=int)
        self._feature_blob = np.empty((0, nwalkers,
                                       len(natural_parameters)))
        self._enthalpy = np.empty((0, nwalkers))
        self._accepted = np.zeros((0, nwalkers), dtype=int)

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

    def get_occupancies(self, discard=0, thin_by=1, flat=True):
        """Get an occupancy chain from samples."""
        chain = self._chain[discard + thin_by - 1::thin_by]
        if flat:
            chain = self._flatten(chain)
        return chain

    def get_enthalpies(self, discard=0, thin_by=1, flat=True):
        """Get the generalized entalpy changes from samples in chain."""
        chain = self._enthalpy[discard + thin_by - 1::thin_by]
        if flat:
            chain = self._flatten(chain)
        return chain

    def get_feature_vectors(self, discard=0, thin_by=1, flat=True):
        """Get the feature vector changes from samples in chain."""
        chain = self._feature_blob[discard + thin_by - 1::thin_by]
        if flat:
            chain = self._flatten(chain)
        return chain

    def get_energies(self, discard=0, thin_by=1, flat=True):
        """Get the energies from samples in chain."""
        feature_blob = self.get_feature_vectors(discard, thin_by, flat=False)
        energies = np.array([np.dot(self.natural_parameters[:self._num_energy_coefs],  # noqa
                             features[:, :self._num_energy_coefs].T)
                             for features in feature_blob])
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

    def mean_enthalpy(self, discard=0, thin_by=1, flat=True):
        """Get the mean generalized enthalpy."""
        return self.get_enthalpies(discard, thin_by, flat).mean(axis=0)

    def enthalpy_variance(self, discard=0, thin_by=1, flat=True):
        """Get the variance in enthalpy"""
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

    def mean_sublattice_composition(self, sublattice, discard=0, thin_by=1,
                                    flat=True):
        """Get the mean composition of a specific sublattice."""
        return self.get_sublattice_compositions(sublattice, discard,
                                                thin_by, flat).mean(axis=0)

    def sublattice_composition_variance(self, sublattice, discard=0, thin_by=1,
                                        flat=True):
        """Get the varience in composition of a specific sublattice."""
        return self.get_sublattice_compositions(sublattice, discard,
                                                thin_by, flat).var(axis=0)

    def get_minimum_enthalpy(self, discard=0, thin_by=1,  flat=True):
        """Get the minimum energy from samples."""
        return self.get_enthalpies(discard, thin_by, flat).min(axis=0)

    def get_minimum_enthalpy_occupancy(self, discard=0, thin_by=1, flat=True):
        """Find the occupancy with minimum energy from samples."""
        inds = self.get_enthalpies(discard, thin_by, flat).argmin(axis=0)
        if flat:
            occus = self.get_occupancies(discard, thin_by, flat)[inds]
        else:
            occus = self.get_occupancies(discard, thin_by, flat)[inds, np.arange(self.shape[0])]  # noqa
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
            occus = self.get_occupancies(discard, thin_by, flat)[inds, np.arange(self.shape[0])]  # noqa
        return occus

    def get_species_counts(self, discard=0, thin_by=1, flat=True):
        """Get the species counts for each occupancy in the chain."""
        samples = self.num_samples // thin_by
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

    def save_sample(self, accepted, occupancies, enthalpy, feature_blob,
                    thinned_by):
        """Save a sample from the generated chain

        Args:
            accepted (ndarray):
                array of total acceptances.
            occupancies (ndarray):
                array of occupancies
            enthalpy (ndarray):
                array of generalized enthalpy changes
            feature_blob (ndarray):
                array of feature vector changes
            thinned_by (int):
                the amount that the sampling was thinned by. Used to update
                the total mc iterations.
        """
        self._chain[self._nsamples, :, :] = occupancies
        self._enthalpy[self._nsamples, :] = enthalpy
        self._feature_blob[self._nsamples, :, :] = feature_blob
        self._accepted[self._nsamples, :] = accepted
        self._nsamples += 1
        self.total_mc_steps += thinned_by

    def clear(self):
        """Clear all samples from container."""
        nwalkers, num_sites = self.shape
        self.total_mc_steps = 0
        self._nsamples = 0
        self._chain = np.empty((0, nwalkers, num_sites), dtype=int)
        self._feature_blob = np.empty((0, nwalkers,
                                       len(self.natural_parameters)))
        self._enthalpy = np.empty((0, nwalkers))
        self._accepted = np.zeros((0, nwalkers), dtype=int)

    def allocate(self, nsamples):
        """allocate more space in arrays for more samples."""
        arr = np.empty((nsamples, *self._chain.shape[1:]), dtype=int)
        self._chain = np.append(self._chain, arr, axis=0)
        arr = np.empty((nsamples, *self._feature_blob.shape[1:]))
        self._feature_blob = np.append(self._feature_blob, arr, axis=0)
        arr = np.empty((nsamples, *self._enthalpy.shape[1:]))
        self._enthalpy = np.append(self._enthalpy, arr, axis=0)
        arr = np.zeros((nsamples, *self._accepted.shape[1:]), dtype=int)
        self._accepted = np.append(self._accepted, arr, axis=0)

    # TODO write this up
    def stream(self, file_path=None):
        if file_path is None:
            now = datetime.now()
            file_name = 'moca-samples-' + now.strftime('%Y-%m-%d-%H%M%S%f')
            file_path = os.path.join(os.getcwd(), file_name + '.json')

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
        d = {'temperature': self.temperature,
             'num_sites': self.num_sites,
             'sublattices': [s.as_dict() for s in self.sublattices],
             'natural_parameters': self.natural_parameters,
             'metadata': self.metadata,
             'num_energy_coefs': self._num_energy_coefs,
             'total_mc_steps': self.total_mc_steps,
             'nsamples': self._nsamples,
             'chain': self._chain.tolist(),
             'feature_blob': self._feature_blob.tolist(),
             'enthalpy': self._enthalpy.tolist(),
             'accepted': self._accepted.tolist()}
        return d

    @classmethod
    def from_dict(cls, d):
        """Instantiate a sublattice from dict representation.

        Args:
            d (dict):
                dictionary representation.
        Returns:
            Sublattice
        """
        sublattices = [Sublattice.from_dict(s) for s in d['sublattices']]
        container = cls(d['temperature'], d['num_sites'], sublattices,
                        d['natural_parameters'], d['num_energy_coefs'],
                        d['metadata'])
        container._nsamples = np.array(d['nsamples'])
        container._chain = np.array(d['chain'], dtype=int)
        container._feature_blob = np.array(d['feature_blob'])
        container._enthalpy = np.array(d['enthalpy'])
        container._accepted = np.array(d['accepted'], dtype=int)
        return container
