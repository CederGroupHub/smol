"""Implementation of SampleContainer class

A SampleContainer is used to store data from Monte Carlo sampling simulation.
It has some minimimal methods and properties useful to start analyzing the
samples.
"""

__author__ = "Luis Barroso-Luque"

import os
from datetime import datetime
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
        metadata (dict):
            dictionary of metadata from the MC run that generated the samples.
    """

    # To get sublattices, ensemble name, temperature, usher type

    def __init__(self, temperature, num_sites, sublattices,
                 natural_parameters, ensemble_metadata=None):
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
        """

        self.temperature = temperature
        self.num_sites = num_sites
        self.sublattices = sublattices
        self.metadata = {} if ensemble_metadata is None else ensemble_metadata
        self._chains = np

    @property
    def num_samples(self):
        """Get the total number of samples."""
        return

    def get_mean_energy(self):
        """Calculate the mean energy from samples."""
        return

    def get_energy_variance(self):
        """Calculate the variance of sampled energies."""
        return

    def get_minimum_energy(self):
        """Get the minimum energy from samples."""
        return

    def get_minimum_energy_occupancy(self):
        """Find the occupancy with minimum energy from samples."""
        return

    def get_mean_composition(self):
        """Get the mean composition for all species."""
        return

    def get_composition_variance(self):
        """Get the variance in composition of all species."""
        return

    def get_sublattice_composition(self, sublattice):
        """Get the compositions of a specific sublattice."""
        return

    def get_sublattice_composition_variance(self, sublattice):
        """Get the varience in composition of a specific sublattice."""
        return

    def get_mean_features(self):
        """Get the mean feature vector from samples."""
        return

    def get_features_variance(self):
        """Get the variance of feature vector elements."""
        return

    def stream(self, file_path=None):
        if file_path is None:
            now = datetime.now()
            file_name = 'moca-samples-' + now.strftime('%Y-%m-%d-%H%M%S%f')
            file_path = os.path.join(os.getcwd(), file_name + '.json')

    def as_dict(self):
        pass

    def from_dict(cls, d):
        pass
