"""Implementation of Semi-Grand Canonical Ensemble Classes.

These are used to run Monte Carlo sampling for fixed number of sites but
variable concentration of species.

Two classes are different SGC ensemble implemented:
* MuSemiGrandEnsemble - for which relative chemical potentials are fixed
* FuSemiGrandEnsemble - for which relative fugacity fractions are fixed.
"""

__author__ = "Luis Barroso-Luque"

from abc import abstractmethod
from math import log, prod
import numpy as np

from monty.json import MSONable
from smol.moca.processor.base import Processor
from smol.moca.ensemble.base import Ensemble


class BaseSemiGrandEnsemble(Ensemble):
    """Abstract Semi-Grand Canonical Base Ensemble.

    Total number of species are fixed but composition of "active" (with partial
    occupancies) sublattices is allowed to change.

    This class can not be instantiated. See MuSemiGrandEnsemble and
    FuSemiGrandEnsemble below.
    """
    valid_mcmc_ushers = ('Flipper',)

    def __init__(self, processor, temperature, sublattices=None):
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
        """
        super().__init__(processor, temperature, sublattices=sublattices)
        self.__params = np.append(self.processor.coefs, -1.0)

    @property
    def natural_parameters(self):
        """Get the vector of natural parameters.

        For SGC an extra -1 is added for the chemical part of the LT.
        """
        return self.__params

    @abstractmethod
    def compute_chemical_work(self, occupancy):
        """Compute the chemical work term"""
        return []

    def compute_feature_vector(self, occupancy):
        """Compute the feature vector for a give occupancy.

        In the canonical case it is just the feature vector from the processor.

        Args:
            occupancy (ndarray):
                encoded occupancy string

        Returns:
            ndarray: feature vector
        """
        feature_vector = self.processor.compute_feature_vector(occupancy)
        chemical_work = self.compute_chemical_work(occupancy)
        return np.append(feature_vector, chemical_work)  # prellocate to improve speed


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
                 sublattices=None):
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
        """
        super().__init__(processor, temperature, sublattices)

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

        self.__mus = chemical_potentials
        self._mu_table = self._build_mu_table(chemical_potentials)
        self.thermo_boundaries = {'chemical-potentials':
                                  self.chemical_potentials}

    @property
    def chemical_potentials(self):
        """Get the chemical potentials for species in system."""
        return self.__mus

    @chemical_potentials.setter
    def chemical_potentials(self, value):
        """Set the chemical potentials and update table"""
        if not all(val in self.__mus.keys() for val in value.keys()):
            raise ValueError('Chemical potentials given are missing species. '
                             'Values must be given for each of the following:'
                             f' {self.__mus.keys()}')
        self.__mus = value
        self._mu_table = self._build_mu_table(value)

    def compute_feature_vector_change(self, occupancy, step):
        """Return the change in the feature vector from a step.

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            step (list of tuple):
                A sequence of flips given my the MCMCUsher.propose_step

        Returns:
            ndarray: difference in feature vector
        """
        delta_feature = self.processor.compute_feature_vector_change(occupancy,
                                                                     step)
        delta_mu = sum(self._mu_table[f[0]][f[1]]
                       - self._mu_table[f[0]][occupancy[f[0]]] for f in step)
        return np.append(delta_feature, delta_mu)  # prellocate to improve speed

    def compute_chemical_work(self, occupancy):
        """Compute sum of mu * N for given occupancy"""
        return sum(self._mu_table[site][species]
                   for site, species in enumerate(occupancy))

    def _build_mu_table(self, chemical_potentials):
        """Build an array for chemical potentials for all sites in system.

        Rows represent sites and columns species. This allows quick evaluation
        of chemical potential changes from flips. Not that the total number
        of columns will be the number of species in the largest site space. For
        smaller site spaces the values at those rows are meaningless and will
        be given values of 0. Also rows representing sites with not partial
        occupancy will have all 0 values and should never be used.
        """
        num_cols = max(len(site_space) for site_space
                       in self.processor.unique_site_spaces)
        table = np.zeros((self.num_sites, num_cols))
        for sublatt in self.sublattices:
            ordered_pots = [chemical_potentials[sp] for sp in sublatt.species]
            table[sublatt.sites, :len(ordered_pots)] = ordered_pots
        return table


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
                 sublattices=None):
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
                prim structure used in the cluster subspace. Needs to be in
                the same order as the corresponding sublattice.
            sublattices (list of Sublattice): optional
                list of Lattice objects representing sites in the processor
                supercell with same site spaces.
        """
        super().__init__(processor, temperature, sublattices)

        if fugacity_fractions is not None:
            # check that species are valid
            for fus, sublatt in zip(fugacity_fractions, self.sublattices):
                if sum([fu for fu in fus.values()]) != 1:
                    raise ValueError('Fugacity ratios must add to one.')
                if not all(fu in sublatt.species for fu in fus.keys()):
                    raise ValueError('Fugacity fractions given are missing or '
                                     'not valid species. Values must be given '
                                     'for each of the following: '
                                     f'{[f.keys() for f in self.__fus]}')

        self.__fus = fugacity_fractions
        self._fu_table = self._build_fu_table(fugacity_fractions)
        self.thermo_boundaries = {'fugacity-fractions':
                                  self.fugacity_fractions}

    @property
    def fugacity_fractions(self):
        """Get the fugacity fractions for species in system."""
        return self.__fus

    @fugacity_fractions.setter
    def fugacity_fractions(self, value):
        """Set the fugacity fractions and update table"""
        if not all(sum(fus.values()) == 1 for fus in value):
            raise ValueError('Fugacity ratios must add to one.')
        for (fus, vals) in zip(self.__fus, value):
            if not all(val in fus.keys() for val in vals.keys()):
                raise ValueError('Fugacity fractions given are missing or not '
                                 'valid species. Values must be given for each'
                                 ' of the following: '
                                 f'{[f.keys() for f in self.__fus]}')
        self.__fus = value
        self._fu_table = self._build_fu_table(value)

    def compute_feature_vector_change(self, occupancy, step):
        """Return the change in the sufficient statistics vector from a step.

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            step (list of tuple):
                A sequence of flips given my the MCMCUsher.propose_step

        Returns:
            ndarray: difference in vector of sufficient statistics
        """
        delta_feature = self.processor.compute_feature_vector_change(occupancy,
                                                                     step)
        delta_log_fu = log(prod(self._fu_table[f[0]][f[1]] /
                                self._fu_table[f[0]][occupancy[f[0]]]
                                for f in step))
        return np.append(delta_feature, delta_log_fu)  # prellocate to improve speed

    def compute_chemical_work(self, occupancy):
        """Compute log of product of fugacities for given occupancy"""
        return log(prod(self._fu_table[site][species]
                        for site, species in enumerate(occupancy)))

    def _build_fu_table(self, fugacity_fractions):
        """Build an array for fugacity fractions for all sites in system.

        Rows represent sites and columns species. This allows quick evaluation
        of fugacity fraction changes from flips. Not that the total number
        of columns will be the number of species in the largest site space. For
        smaller site spaces the values at those rows are meaningless and will
        be given values of 0. Also rows representing sites with not partial
        occupancy will have all 0 values and should never be used.
        """
        num_cols = max(len(site_space) for site_space
                       in self.processor.unique_site_spaces)
        table = np.zeros((self.num_sites, num_cols))
        for fus, sublatt in zip(self.sublattices, fugacity_fractions):
            ordered_fus = [fugacity_fractions[sp] for sp in sublatt.species]
            table[sublatt.sites, :len(ordered_fus)] = ordered_fus
        return table
