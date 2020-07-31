"""Implementation of SampleContainer class

A SampleContainer is used to store data from Monte Carlo sampling simulation.
It has some minimimal methods and properties useful to start analyzing the
samples.
"""

__author__ = "Luis Barroso-Luque"

import os
from datetime import datetime
import numpy as np

from monty.json import MSONable


class SampleContainer(MSONable):
    """A SampleContainter class stores Monte Carlo simulation samples.

    It also provides some minor functionality to get sample statistics.

    Attributes:
        temperature (float):
            temperature of ensemble that was sampled.
        num_sites (int):
            Size of system (usually in number of prims in supercell, but
            can be anything representative i.e. number of sites)
        sublattices (list of Sublattice)
            Sublattices of the ensemble sampled.
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
            sublattices (list of Sublattice)
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
        self.__nsamples = 0
        self.__chain = np.empty((0, nwalkers, num_sites), dtype=int)
        self.__feature_blob = np.empty((0, nwalkers,
                                        len(natural_parameters)))
        self.__enthalpy = np.empty((0, nwalkers))
        self.__accepted = np.zeros(nwalkers, dtype=int)

    @property
    def num_samples(self):
        """Get the total number of samples."""
        return self.__nsamples

    @property
    def shape(self):
        """Get the shape of the samples in chain."""
        return self.__chain.shape[1:]

    @property
    def sampling_efficiency(self, flat=True):
        """Return the sampling efficiency for each chain."""
        efficiency = self.__accepted / self.total_mc_steps
        if flat:
            efficiency = efficiency.mean()
        return efficiency

    def get_occupancies(self, discard=0, thin_by=1, flat=False):
        """Get an occupancy chain from samples."""
        chain = self.__chain[discard + thin_by - 1::thin_by]
        if flat:
            chain = self._flatten(chain)
        return chain

    def get_enthalpies(self, discard=0, thin_by=1, flat=False):
        """Get the generalized entalpy changes from samples in chain"""
        chain = self.__enthalpy[discard + thin_by - 1::thin_by]
        if flat:
            chain = self._flatten(chain)
        return chain

    def get_feature_vectors(self, discard=0, thin_by=1, flat=False):
        """Get the feature vector changes from samples in chain"""
        chain = self.__feature_blob[discard + thin_by - 1::thin_by]
        if flat:
            chain = self._flatten(chain)
        return chain

    def get_energies(self, discard=0, thin_by=1, flat=False):
        """Get the energies from samples in chain."""
        feature_blob = self.get_feature_vectors(discard, thin_by)
        energies = np.array([np.dot(self.natural_parameters[:self._num_energy_coefs],  # noqa
                             features.T) for features in feature_blob])
        if flat:
            energies = self._flatten(energies)
        return energies

    def mean_enthalpy(self, discard=0, thin_by=1, flat=False):
        """Get the mean generalized enthalpy."""
        return self.get_enthalpies(discard, thin_by, flat).mean(axis=0)

    def enthalpy_variance(self, discard=0, thin_by=1, flat=False):
        """Get the variance in enthalpy"""
        return np.var(self.get_enthalpies(discard, thin_by, flat), axis=0)

    def mean_energy(self, discard=0, thin_by=1, flat=False):
        """Calculate the mean energy from samples."""
        return self.get_energies(discard, thin_by, flat).mean(axis=0)

    def energy_variance(self, discard=0, thin_by=1, flat=False):
        """Calculate the variance of sampled energies."""
        return np.var(self.get_energies(discard, thin_by, flat), axis=0)

    def mean_feature_vector(self, discard=0, thin_by=1, flat=False):
        """Get the mean feature vector from samples."""
        return self.get_feature_vectors(discard, thin_by, flat).mean(axis=0)

    def feature_vector_variance(self, discard=0, thin_by=1, flat=False):
        """Get the variance of feature vector elements."""
        return np.var(self.get_feature_vectors(discard, thin_by, flat), axis=0)

    def mean_composition(self, discard=0, thin_by=1, flat=False):
        """Get the mean composition for all species."""
        return

    def composition_variance(self, discard=0, thin_by=1, flat=False):
        """Get the variance in composition of all species."""
        return

    def sublattice_composition(self, sublattice, discard=0, thin_by=1,
                               flat=False):
        """Get the compositions of a specific sublattice."""
        return

    def sublattice_composition_variance(self, sublattice, discard=0, thin_by=1,
                                        flat=False):
        """Get the varience in composition of a specific sublattice."""
        return

    def heat_capacity(self, discard=0, thin_by=1, flat=False):
        """Get the heat capacity."""
        return

    def get_minimum_energy(self, flat=True):
        """Get the minimum energy from samples."""
        return

    def get_minimum_energy_occupancy(self, flat=True):
        """Find the occupancy with minimum energy from samples."""
        return

    def get_minimum_enthalpy(self, flat=True):
        """Get the minimum energy from samples."""
        return

    def get_minimum_enthalpy_occupancy(self, flat=True):
        """Find the occupancy with minimum energy from samples."""
        return

    def save_sample(self, accepted, occupancies, delta_enthalpy,
                    delta_feature_blob):
        """Save a sample from the generated chain

        Args:
            accepted (ndarray):
                boolean array of acceptances.
            occupancies (ndarray):
                array of occupancies
            delta_enthalpy (ndarray):
                array of generalized enthalpy changes
            delta_feature_blob (ndarray):
                array of feature vector changes
        """
        self.__chain[self.__nsamples, :, :] = occupancies
        self.__enthalpy[self.__nsamples, :] = delta_enthalpy
        self.__feature_blob[self.__nsamples, :, :] = delta_feature_blob
        self.__accepted += accepted
        self.__nsamples += 1

    def clear(self):
        """Clear all samples from container."""
        nwalkers, num_sites = self.shape
        self.total_mc_steps = 0
        self.__nsamples = 0
        self.__chain = np.empty((0, nwalkers, num_sites), dtype=int)
        self.__feature_blob = np.empty((0, nwalkers,
                                        len(self.natural_parameters)))
        self.__enthalpy = np.empty((0, nwalkers))
        self.__accepted = np.zeros(nwalkers, dtype=int)

    def allocate(self, nsamples):
        """allocate more space in arrays for more samples."""
        arr = np.empty((nsamples, *self.__chain.shape[1:]), dtype=int)
        self.__chain = np.append(self.__chain, arr, axis=0)
        arr = np.empty((nsamples, *self.__feature_blob.shape[1:]))
        self.__feature_blob = np.append(self.__feature_blob, arr, axis=0)
        arr = np.empty((nsamples, *self.__enthalpy.shape[1:]))
        self.__enthalpy = np.append(self.__enthalpy, arr, axis=0)

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
        return self.__nsamples

    def as_dict(self):
        pass

    def from_dict(cls, d):
        pass
