import random
from copy import deepcopy
import json
import numpy as np
from math import exp
from abc import ABC, abstractmethod
from pymatgen.transformations.standard_transformations import \
    OrderDisorderedStructureTransformation

# TODO it would be great to use the design paradigm of observers to extract
#  a variety of information during a montecarlo run


class BaseEnsemble(ABC):
    """
    Base Class for Monte Carlo Ensembles.
    """

    def __init__(self, processor, save_interval, initial_occupancy=None,
                 sublattices=None, seed=None):
        """
        Args:
            processor (Processor Class):
                A processor that can compute the change in a property given
                a set of flips.
            save_interval (int):
                interval of steps to save the current occupancy and property
            inital_occupancy (array):
                Initial occupancy vector. If none is given then a random one
                will be used.
            sublattices (dict): optional
                dictionary with keys identifying the active sublattices
                (i.e. "anion" or the bits in that sublattice
                "['Li+', 'Vacancy']"), the values should be a dictionary
                with two items {'sites': array with the site indices for all
                sites corresponding to that sublattice in the occupancy vector,
                'bits': tuple of bits (allowed species) in sublattice}
                All sites in a sublattice need to have the same bits/species
                allowed.
            seed (int): optional
                seed for random number generator
        """

        if initial_occupancy is None:
            struct = processor.subspace.structure.copy()
            scmatrix = processor.supercell_matrix
            struct.make_supercell(scmatrix)
            odt = OrderDisorderedStructureTransformation()
            struct = odt.apply_transformation(struct)
            initial_occupancy = processor.subspace.occupancy_from_structure(struct, scmatrix)  # noqa

        if sublattices is None:
            sublattices = {str(bits):
                           {'sites': np.array([i for i, b in
                                               enumerate(initial_occupancy)
                                               if b in bits]),
                            'bits': bits}
                           for bits in processor.unique_bits}

        self.processor = processor
        self.save_interval = save_interval
        self.num_atoms = len(initial_occupancy)
        self._sublattices = sublattices
        self._init_occupancy = processor.encode_occupancy(initial_occupancy)
        self._occupancy = self._init_occupancy.copy()
        self._energy = processor.compute_property(self._occupancy)
        self._step = 0
        self._ssteps = 0
        self._data = []

        # Set and save the seed for random. This allows reproducible results.
        if seed is None:
            seed = random.randint(1, 1E24)

        self._seed = seed
        random.seed(seed)

    @property
    def occupancy(self):
        return self.processor.decode_occupancy(self._occupancy)

    @property
    def initial_occupancy(self):
        return self.processor.decode_occupancy(self._init_occupancy)

    @property
    def energy(self):
        return deepcopy(self._energy)

    @property
    def current_structure(self):
        return self.processor.structure_from_occupancy(self._occupancy)

    @property
    def current_step(self):
        return self._step

    @property
    def accepted_steps(self):
        return self._ssteps

    @property
    def data(self):
        return self._data

    @property
    def seed(self):
        return self._seed

    def run(self, iterations, sublattice_name=None):
        """
        Samples the ensembles for the given number of iterations. Sampling at
        the provided intervals???

        Args:
            iterations (int):
                Total number of monte carlo steps to attempt
        """

        write_loops = iterations//self.save_interval
        if iterations % self.save_interval > 0:
            write_loops += 1

        start_step = self.current_step

        for _ in range(write_loops):
            remaining = iterations - self.current_step + start_step
            no_interrupt = min(remaining, self.save_interval)

            for i in range(no_interrupt):
                success = self._attempt_step(sublattice_name)
                self._ssteps += success

            self._step += no_interrupt
            self._save_data()

    def reset(self):
        """
        Resets the ensemble by returning it to its initial state. This will
        also clear the data.
        """
        self._occupancy = self._init_occupancy.copy()
        self._step, self._ssteps = 0, 0
        self._data = []

    def dump(self, filename):
        """
        Write data into a text file in json format, and clear data
        """
        with open(filename, 'a') as fp:
            json.dump(self.data, fp)

        self._data = []

    @abstractmethod
    def _attempt_step(self, sublattice_name):
        """
        Attempts a MC step and returns 0, 1 based on whether the step was
        accepted or not.
        """
        pass

    def _get_current_data(self):
        """
        Method to extract the ensembles data from the current state. Should
        return a dict with current data.

        Returns: ensembles data
            dict
        """
        return {}

    @staticmethod
    def _accept(delta_e, beta=1.0):
        """
        Evaluate the metropolis acceptance criterion

        Args:
            delta_e (float):
                potential change
            beta (float):
                1/kBT

        Returns:
            bool
        """
        return True if delta_e < 0 else exp(-beta*delta_e) >= random.random()

    def _save_data(self):
        """
        Save the current sample and properties
        Args:
            step (int):
                Current montecarlo step
        """
        data = self._get_current_data()
        data['step'] = self.current_step
        self._data.append(data)
