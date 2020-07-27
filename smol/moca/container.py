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
        metadata (dict):
            dictionary of metadata from the MC run that generated the samples.
    """

    def __init__(self, temperature, ensemble_name, system_description,
                 system_size, metadata=None):
        """Initialize a sample container.

        Args:
            temperature (float):
                Temperature of the ensemble.
            ensemble_name (str):
                Name or description of ensemble sampled.
            system_description (str):
                Description of system sampled
            system_size (int):
                Size of system (usually in number of prims in supercell, but
                can be anything representative i.e. number of sites)
        """

        self.temperature = temperature
        self.ensemble_name = ensemble_name
        self.system_description = system_description
        self.system_size = system_size
        self.metadata = {} if metadata is None else metadata

        self._total_steps = 0
        self._accepted_steps = 0

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
