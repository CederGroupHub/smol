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

    Attributes:
        temperature (float):
            temperature of ensemble that was sampled.
        num_sites (int):
            Size of system (usually in number of prims in supercell, but
            can be anything representative i.e. number of sites)
        sublattices (list of Sublattice)
            Sublattices of the ensemble sampled.
        total_iterations (int)
            Number of iterations used in sampling
        metadata (dict):
            dictionary of metadata from the MC run that generated the samples.
    """

    # To get sublattices, ensemble name, temperature, usher type

    def __init__(self, temperature, num_sites, sublattices,
                 natural_parameters, ensemble_metadata=None, num_walkers=1):
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
            ensemble_metadata (Ensemble):
                Metadata of the ensemble that was sampled.
            num_walkers (int):
                Number of walkers used to generate chain. Default is 1
        """

        self.temperature = temperature
        self.num_sites = num_sites
        self.sublattices = sublattices
        self.natural_parameters = natural_parameters
        self.metadata = {} if ensemble_metadata is None else ensemble_metadata
        self.total_iterations = 0
        self.__nsamples = 0
        self.__chain = np.empty((0, num_walkers, num_sites), dtype=int)
        self.__feature_blob = np.empty((0, num_walkers,
                                        len(natural_parameters)))
        self.__enthalpy = np.empty((0, num_walkers, 1))
        self.__accepted = np.zeros(num_walkers)

    @property
    def num_samples(self):
        """Get the total number of samples."""
        return self.__nsamples

    @property
    def shape(self):
        """Get the shape of the samples in chain."""
        return self.__chain.shape[1:]

    def get_occupancy_chain(self, thin_by=1, discard=0, flat=False):
        """Get an occupancy chain from samples."""
        # TODO implement flat part
        return self.__chain[discard + thin_by - 1::thin_by]

    def mean_energy(self):
        """Calculate the mean energy from samples."""
        return

    def energy_variance(self):
        """Calculate the variance of sampled energies."""
        return

    def mean_composition(self):
        """Get the mean composition for all species."""
        return

    def composition_variance(self):
        """Get the variance in composition of all species."""
        return

    def sublattice_composition(self, sublattice):
        """Get the compositions of a specific sublattice."""
        return

    def sublattice_composition_variance(self, sublattice):
        """Get the varience in composition of a specific sublattice."""
        return

    def mean_features(self):
        """Get the mean feature vector from samples."""
        return

    def features_variance(self):
        """Get the variance of feature vector elements."""
        return

    def mean_enthalpy(self):
        """Get the mean generalized enthalpy."""
        return

    def enthalpy_variance(self):
        """Get the variance in enthalpy"""
        return

    def heat_capacity(self):
        """Get the heat capacity."""
        return

    def get_minimum_energy(self):
        """Get the minimum energy from samples."""
        return

    def get_minimum_energy_occupancy(self):
        """Find the occupancy with minimum energy from samples."""
        return

    def save_sample(self, accepted, occupancies, feature_blob, enthalpies):
        """Save a sample from the generated chain

        Args:
            accepted (ndarray):
                boolean array of acceptances.
            occupancies (ndarray):
                array of occupancies
            feature_blob (ndarray):
                array of feature vectors
            enthalpies (ndarray):
                array of enthalpies
        """
        self.__chain[self.__nsamples, :, :] = occupancies
        self.__feature_blob[self.__nsamples, :, :] = feature_blob
        self.__enthalpy[self.__nsamples, :] = enthalpies
        self.__accepted += accepted
        self.__nsamples += 1

    def allocate(self, nsamples):
        """allocate more space in arrays for more samples."""
        arr = np.empty((nsamples, *self.__chain.shape[1:]), dtype=int)
        self.__chain = np.append(self.__chain, arr, axis=0)
        arr = np.empty((nsamples, *self.__feature_blob.shape[1:]))
        self.__feature_blob = np.append(self.__feature_blob, arr, axis=0)
        arr = np.empty((nsamples, *self.__enthalpy.shape[1:]), dtype=int)
        self.__enthalpy = np.append(self.__enthalpy, arr, axis=0)

    def stream(self, file_path=None):
        if file_path is None:
            now = datetime.now()
            file_name = 'moca-samples-' + now.strftime('%Y-%m-%d-%H%M%S%f')
            file_path = os.path.join(os.getcwd(), file_name + '.json')

    def as_dict(self):
        pass

    def from_dict(cls, d):
        pass
