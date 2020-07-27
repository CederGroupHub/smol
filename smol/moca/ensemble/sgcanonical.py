"""Implementation of Semi-Grand Canonical Ensemble Classes.

These are used to run Monte Carlo sampling for fixed number of sites but
variable concentration of species.

Two classes are different SGC ensemble implemented:
* MuSemiGrandEnsemble - for which relative chemical potentials are fixed
* FuSemiGrandEnsemble - for which relative fugacity fractions are fixed.
"""

__author__ = "Luis Barroso-Luque"

import random
from abc import ABCMeta, abstractmethod
from math import log, exp
import numpy as np

from monty.json import MSONable
from smol.moca.processor.base import Processor
from smol.moca.ensemble.base import Ensemble


class BaseSemiGrandEnsemble(Ensemble, metaclass=ABCMeta):
    """Abstract Semi-Grand Canonical Base Ensemble.

    Total number of species are fixed but composition of "active" (with partial
    occupancies) sublattices is allowed to change.

    This class can not be instantiated. See MuSemiGrandEnsemble and
    FuSemiGrandEnsemble below.
    """
    valid_move_types = ('flip', )

    def __init__(self, processor, temperature, sublattices=None,
                 sublattice_probabilities=None, move_type=None):
        """Initialize BaseSemiGrandEnsemble.

        Args:
            processor (Processor):
                A processor that can compute the change in a property given
                a set of flips. See moca.processor
            temperature (float):
                Temperature of ensemble
            sublattices (list of Sublattice): optional
                list of Lattice objects representing sites in the processor
                supercell with same site spaces.
            sublattice_probabilities (list of float): optional
                list of probability to pick a site from a specific sublattice
            move_type (str):
                string specifying the type of MCMC move for the SGC ensemble.
        """
        if move_type is None:
            move_type = 'flip'
        elif move_type is not None and move_type not in self.valid_move_types:
            raise ValueError(f'Provided move type {move_type} is not a valid '
                             'option for a Semigrand ensemble. Valid options '
                             f'are {self.valid_move_types}.')

        super().__init__(processor, temperature, move_type=move_type,
                         sublattices=sublattices,
                         sublattice_probabilities=sublattice_probabilities)
        self.__params = np.append(self.processor.coefs, 1.0)

    @property
    def exponential_parameters(self):
        """Get the vector of exponential parameters."""
        return self.__params


class MuSemiGrandEnsemble(BaseSemiGrandEnsemble):
    """Relative chemical potential based SemiGrand Ensemble.

    A Semi-Grand Canonical Ensemble for Monte Carlo Simulations where species
    chemical potentials are predefined. Note that in the SGC Ensemble
    implemented here, only the differences in chemical potentials with
    respect to a reference species on each sublattice are fixed, and not the
    absolute values. To obtain the absolute values you must calculate the
    reference chemical potential and then simply subtract it from the given
    values.
    """

    def __init__(self, processor, temperature, chemical_potentials,
                 sublattices=None, sublattice_probabilities=None,
                 move_type=None):
        """Initialize MuSemiGrandEnsemble.

        Args:
            processor (Processor):
                A processor that can compute the change in a property given
                a set of flips. See moca.processor
            temperature (float):
                Temperature of ensemble
            chemical_potentials (dict):
                dictionary with species names and chemical potentials.
            sublattices (list of Sublattice): optional
                list of Lattice objects representing sites in the processor
                supercell with same site spaces.
            sublattice_probabilities (list of float): optional
                list of probability to pick a site from a specific sublattice
            move_type (str):
                string specifying the type of MCMC move for the SGC ensemble.
        """
        super().__init__(processor, temperature, sublattices,
                         sublattice_probabilities, move_type)

        # check that species are valid
        species = [sp for sps in processor.unique_site_spaces for sp in sps]
        for sp in chemical_potentials.keys():
            if sp not in species:
                raise ValueError(f'Species {sp} in provided chemical '
                                 'potentials is not an allowed species in the '
                                 f'system: {species}')
        for sp in species:
            if sp not in chemical_potentials.keys():
                raise ValueError(f'Species {sp} was not assigned a chemical '
                                 ' potential, a value must be provided.')

        # TODO create a map to get chemical potentials in update functions quick
        self._mu_table = None

    def compute_sufficient_statistics(self, occupancy):
        """Compute the sufficient statistics for a give occupancy.

        In the canonical case it is just the feature vector.

        Args:
            occupancy (ndarray):
                encoded occupancy string

        Returns:
            ndarray: vector of sufficient statistics
        """
        feature_vector = self.processor.compute_feature_vector(occupancy)
        chemical_part = None  # TODO get this thing
        return -self.beta * np.append(feature_vector, chemical_part)

    def compute_sufficient_statistics_change(self, occupancy, move):
        """Return the change in the sufficient statistics vector from a move.

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            move (list of tuple):
                A sequence of moves given my the MCMCMove.propose.

        Returns:
            ndarray: difference in vector of sufficient statistics
        """
        delta_feature = self.processor.compute_feature_vector_change(occupancy,
                                                                     move)
        delta_mu = (self._mu_table[move[0]][occupancy[move[0]]]
                    - self._mu_table[move[0]][move[1]])  # -delta mu actually
        return -self.beta * np.append(delta_feature, delta_mu)  # prellocate to improve speed


class FuSemiGrandEnsemble(BaseSemiGrandEnsemble, MSONable):
    """Fugacity fraction SemiGrandEnsemble.

    A Semi-Grand Canonical Ensemble for Monte Carlo simulations where the
    species fugacity ratios are set constant. This implicitly sets the chemical
    potentials, albeit for a specific temperature. Since one species per
    sublattice is the reference species, to calculate actual fugacities the
    reference fugacity must be computed as an ensemble average and all other
    fugacities can then be calculated. From the fugacities and the set
    temperature the corresponding chemical potentials can then be calculated.
    """
    def __init__(self, processor, temperature, fugacity_fractions=None,
                 sublattices=None, sublattice_probabilities=None,
                 move_type=None):
        """Initialize MuSemiGrandEnsemble.

        Args:
            processor (Processor):
                A processor that can compute the change in a property given
                a set of flips. See moca.processor
            temperature (float):
                Temperature of ensemble
            fugacity_fractions (sequence of dicts): optional
                dictionary of species name and fugacity fraction for each
                sublattice (ie think of it as the sublattice concentrations
                for random structure). If not given this will be taken from the
                prim structure used in the cluster subspace.
            sublattices (list of Sublattice): optional
                list of Lattice objects representing sites in the processor
                supercell with same site spaces.
            sublattice_probabilities (list of float): optional
                list of probability to pick a site from a specific sublattice
            move_type (str):
                string specifying the type of MCMC move for the SGC ensemble.
        """
        super().__init__(processor, temperature, sublattices,
                         sublattice_probabilities, move_type)

        # TODO remove option to pass fugacity fractions, always use the prim
        if fugacity_fractions is not None:
            # check that species are valid
            species = [sp for sps in processor.unique_site_spaces
                       for sp in sps]
            for sublatt in fugacity_fractions:
                if sum([f for f in sublatt.values()]) != 1:
                    raise ValueError('Fugacity ratios must add to one.')
                for sp in sublatt.keys():
                    if sp not in species:
                        raise ValueError(f'Species {sp} in provided fugacity '
                                         'ratios is not a species in the'
                                         f'expansion: {species}')

        # TODO create a map to get fugacities in update functions quick
        self._fu_table = None

    def compute_sufficient_statistics(self, occupancy):
        """Compute the sufficient statistics for a give occupancy.

        In the canonical case it is just the feature vector.

        Args:
            occupancy (ndarray):
                encoded occupancy string

        Returns:
            ndarray: vector of sufficient statistics
        """
        feature_vector = self.processor.compute_feature_vector(occupancy)
        chemical_part = None  # TODO get this thing
        return -self.beta * np.append(feature_vector, chemical_part)

    def compute_sufficient_statistics_change(self, occupancy, move):
        """Return the change in the sufficient statistics vector from a move.

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            move (list of tuple):
                A sequence of moves given my the MCMCMove.propose.

        Returns:
            ndarray: difference in vector of sufficient statistics
        """
        delta_feature = self.processor.compute_feature_vector_change(occupancy,
                                                                     move)
        delta_feature *= -self.beta
        delta_log_fu = log(self._fu_table[move[0]][move[1]] /
                           self._fu_table[move[0]][occupancy[move[0]]])
        return np.append(delta_feature, delta_log_fu)  # prellocate to improve speed
