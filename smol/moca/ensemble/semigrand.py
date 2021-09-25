"""Implementation of Semi-Grand Canonical Ensemble Classes.

These are used to run Monte Carlo sampling for fixed number of sites but
variable concentration of species.

Two classes are different SGC ensemble implemented:
* MuSemiGrandEnsemble - for which relative chemical potentials are fixed
* FuSemiGrandEnsemble - for which relative fugacity fractions are fixed.
*
"""

__author__ = "Luis Barroso-Luque"

from abc import abstractmethod
from math import log
from collections import Counter
import numpy as np
from copy import deepcopy

from monty.json import MSONable
from pymatgen.core import Species, DummySpecies, Element

from smol.cofe.space.domain import get_species, Vacancy
from smol.moca.processor.base import Processor
from smol.moca.ensemble.sublattice import Sublattice

from .base import Ensemble
from ..utils.math_utils import GCD_list
from ..utils.occu_utils import (occu_to_species_stat,
                                delta_ccoords_from_step)
from ..comp_space import CompSpace


class BaseSemiGrandEnsemble(Ensemble):
    """Abstract Semi-Grand Canonical Base Ensemble.

    Total number of species are fixed but composition of "active" (with partial
    occupancies) sublattices is allowed to change.

    This class can not be instantiated. See :class:`MuSemiGrandEnsemble` and
    :class:`FuSemiGrandEnsemble` below.
    """

    valid_mcmc_steps = ('flip', 'table-flip', 'subchain-walk')

    def __init__(self, processor, sublattices=None):
        """Initialize BaseSemiGrandEnsemble.

        Args:
            processor (Processor):
                A processor that can compute the change in a property given
                a set of flips. See moca.processor
            sublattices (list of Sublattice): optional
                list of Lattice objects representing sites in the processor
                supercell with same site spaces. Active sublattices only.
        """
        super().__init__(processor, sublattices=sublattices)
        self._params = np.append(self.processor.coefs, -1.0)

    @property
    def natural_parameters(self):
        """Get the vector of natural parameters.

        For SGC an extra -1 is added for the chemical part of the LT.
        """
        return self._params

    @abstractmethod
    def compute_chemical_work(self, occupancy):
        """Compute the chemical work term."""
        return []

    def compute_feature_vector(self, occupancy):
        """Compute the feature vector for a given occupancy.

        In the semigrand case it is the feature vector and the chemical work
        term.

        Args:
            occupancy (ndarray):
                encoded occupancy string

        Return:
            ndarray: feature vector
        """
        feature_vector = self.processor.compute_feature_vector(occupancy)
        chemical_work = self.compute_chemical_work(occupancy)
        # prellocate to improve speed
        return np.append(feature_vector, chemical_work)


class MuSemiGrandEnsemble(BaseSemiGrandEnsemble, MSONable):
    """Relative chemical potential based SemiGrand Ensemble.

    A Semi-Grand Canonical Ensemble for Monte Carlo Simulations where species
    relative chemical potentials are predefined. Note that in the SGC Ensemble
    implemented here, only the differences in chemical potentials with
    respect to a reference species on each sublattice are fixed, and not the
    absolute values. To obtain the absolute values you must calculate the
    reference chemical potential and then simply subtract it from the given
    values.

    Attributes:
        thermo_boundaries (dict):
            dict of chemical potentials.
    """

    def __init__(self, processor, chemical_potentials,
                 sublattices=None):
        """Initialize MuSemiGrandEnsemble.

        Args:
            processor (Processor):
                A processor that can compute the change in a property given
                a set of flips.
            chemical_potentials (dict):
                Dictionary with species and chemical potentials.
            sublattices (list of Sublattice): optional
                List of Sublattice objects representing sites in the processor
                supercell with same site spaces. Active sublattices only.
        """
        super().__init__(processor, sublattices)

        # check that species are valid
        chemical_potentials = {get_species(k): v for k, v
                               in chemical_potentials.items()}
        species = set([sp for sps in processor.unique_site_spaces
                       for sp in sps])
        for sp in chemical_potentials.keys():
            if sp not in species:
                raise ValueError(f'Species {sp} in provided chemical '
                                 'potentials is not an allowed species in the '
                                 f'system: {species}')
        for sp in species:
            if sp not in chemical_potentials.keys():
                raise ValueError(f'Species {sp} was not assigned a chemical '
                                 ' potential, a value must be provided.')

        self._mus = chemical_potentials
        self._mu_table = self._build_mu_table(chemical_potentials)
        self.thermo_boundaries = {'chemical-potentials':
                                  {str(k): v for k, v
                                   in chemical_potentials.items()}}

    @property
    def chemical_potentials(self):
        """Get the chemical potentials for species in system."""
        return self._mus

    @chemical_potentials.setter
    def chemical_potentials(self, value):
        """Set the chemical potentials and update table."""
        for sp, count in Counter(map(get_species, value.keys())).items():
            if count > 1:
                raise ValueError(
                    f"{count} values of the chemical potential for the same "
                    f"species {sp} were provided.\n Make sure the dictionary "
                    "you are using has only string keys or only Species "
                    "objects as keys."
                )
        value = {get_species(k): v for k, v in value.items()}
        if set(value.keys()) != set(self._mus.keys()):
            raise ValueError('Chemical potentials given are missing species. '
                             'Values must be given for each of the following:'
                             f' {self._mus.keys()}')
        self._mus = value
        self._mu_table = self._build_mu_table(value)

    def compute_feature_vector_change(self, occupancy, step):
        """Return the change in the feature vector from a given step.

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            step (list of tuple):
                A sequence of flips given my the MCMCUsher.propose_step

        Return:
            ndarray: difference in feature vector
        """
        delta_feature = self.processor.compute_feature_vector_change(occupancy,
                                                                     step)
        delta_mu = sum(self._mu_table[f[0]][f[1]]
                       - self._mu_table[f[0]][occupancy[f[0]]] for f in step)
        # prellocate to improve speed
        return np.append(delta_feature, delta_mu)

    def compute_chemical_work(self, occupancy):
        """Compute sum of mu * N for given occupancy."""
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
            ordered_pots = [chemical_potentials[sp]
                            for sp in sublatt.site_space]
            table[sublatt.sites, :len(ordered_pots)] = ordered_pots
        return table

    def as_dict(self):
        """Get Json-serialization dict representation.

        Return:
            MSONable dict
        """
        d = super().as_dict()
        d['chemical_potentials'] = tuple((s.as_dict(), c) for s, c
                                         in self.chemical_potentials.items())
        return d

    @classmethod
    def from_dict(cls, d):
        """Instantiate a MuSemiGrandEnsemble from dict representation.

        Return:
            CanonicalEnsemble
        """
        sl_dicts = d.get('sublattices')
        sublattices = ([Sublattice.from_dict(s) for s in sl_dicts] if
                       sl_dicts is not None else None)

        chemical_potentials = {}
        for sp, c in d['chemical_potentials']:
            if ("oxidation_state" in sp
                    and Element.is_valid_symbol(sp["element"])):
                sp = Species.from_dict(sp)
            elif "oxidation_state" in sp:
                if sp['@class'] == 'Vacancy':
                    sp = Vacancy.from_dict(sp)
                else:
                    sp = DummySpecies.from_dict(sp)
            else:
                sp = Element(sp["element"])
            chemical_potentials[sp] = c
        return cls(Processor.from_dict(d['processor']),
                   chemical_potentials=chemical_potentials,
                   sublattices=sublattices)


class FuSemiGrandEnsemble(BaseSemiGrandEnsemble, MSONable):
    """Fugacity fraction SemiGrandEnsemble.

    A Semi-Grand Canonical Ensemble for Monte Carlo simulations where the
    species fugacity ratios are set constant. This implicitly sets the chemical
    potentials, albeit for a specific temperature. Since one species per
    sublattice is the reference species, to calculate actual fugacities the
    reference fugacity must be computed as an ensemble average and all other
    fugacities can then be calculated. From the fugacities and the set
    temperature the corresponding chemical potentials can then be calculated.

    Attributes:
        thermo_boundaries (dict):
            dictionary of fugacity fractions.
    """

    def __init__(self, processor, fugacity_fractions=None,
                 sublattices=None):
        """Initialize MuSemiGrandEnsemble.

        Args:
            processor (Processor):
                A processor that can compute the change in a property given
                a set of flips. See moca.processor
            fugacity_fractions (sequence of dicts): optional
                Dictionary of species name and fugacity fraction for each
                sublattice (.i.e think of it as the sublattice concentrations
                for random structure). If not given this will be taken from the
                prim structure used in the cluster subspace. Needs to be in
                the same order as the corresponding sublattice.
            sublattices (list of Sublattice): optional
                list of Sublattice objects representing sites in the processor
                supercell with same site spaces.
        """
        super().__init__(processor, sublattices)

        if fugacity_fractions is not None:
            # check that species are valid
            fugacity_fractions = [{get_species(k): v for k, v in sub.items()}
                                  for sub in fugacity_fractions]
            for fus, sublatt in zip(fugacity_fractions, self.sublattices):
                if sum([fu for fu in fus.values()]) != 1:
                    raise ValueError('Fugacity ratios must add to one.')
                if set(sublatt.site_space) != set(fus.keys()):
                    raise ValueError('Fugacity fractions given are missing or '
                                     'not valid species. Values must be given '
                                     'for each of the following: '
                     f'{[sublatt.site_space for sublatt in self.sublattices]}')  # noqa
        else:
            fugacity_fractions = [dict(sublatt.site_space) for sublatt
                                  in self.sublattices]
        self._fus = fugacity_fractions
        self._fu_table = self._build_fu_table(fugacity_fractions)
        self.thermo_boundaries = {'fugacity-fractions':
                                  [{str(k): v for k, v in sub.items()}
                                   for sub in fugacity_fractions]}

    @property
    def fugacity_fractions(self):
        """Get the fugacity fractions for species in system."""
        return self._fus

    @fugacity_fractions.setter
    def fugacity_fractions(self, value):
        """Set the fugacity fractions and update table."""
        for sub in value:
            for sp, count in Counter(map(get_species, sub.keys())).items():
                if count > 1:
                    raise ValueError(
                        f"{count} values of the fugacity for the same "
                        f"species {sp} were provided.\n Make sure the "
                        "dictionaries you are using have only "
                        "string keys or only Species objects as keys."
                    )

        value = [{get_species(k): v for k, v in sub.items()} for sub in value]
        if not all(sum(fus.values()) == 1 for fus in value):
            raise ValueError('Fugacity ratios must add to one.')
        for (fus, vals) in zip(self._fus, value):
            if set(fus.keys()) != set(vals.keys()):
                raise ValueError('Fugacity fractions given are missing or not '
                                 'valid species. Values must be given for each'
                                 ' of the following: '
                                 f'{[f.keys() for f in self._fus]}')
        self._fus = value
        self._fu_table = self._build_fu_table(value)

    def compute_feature_vector_change(self, occupancy, step):
        """Compute the change in the feature vector from a given step.

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            step (list of tuple):
                A sequence of flips given my the MCMCUsher.propose_step

        Return:
            ndarray: difference in feature vector
        """
        delta_feature = self.processor.compute_feature_vector_change(occupancy,
                                                                     step)
        # python > 3.8 has math.prod that works on generator
        delta_log_fu = sum(log(self._fu_table[f[0]][f[1]] /
                               self._fu_table[f[0]][occupancy[f[0]]])
                           for f in step)
        # prellocate to improve speed
        return np.append(delta_feature, delta_log_fu)

    def compute_chemical_work(self, occupancy):
        """Compute log of product of fugacities for given occupancy."""
        # python > 3.8 has math.prod
        return sum(log(self._fu_table[site][species])
                   for site, species in enumerate(occupancy))

    def _build_fu_table(self, fugacity_fractions):
        """Build an array for fugacity fractions for all sites in system.

        Rows represent sites and columns species. This allows quick evaluation
        of fugacity fraction changes from flips. Not that the total number
        of columns will be the number of species in the largest site space. For
        smaller site spaces the values at those rows are meaningless and will
        be given values of 1. Also rows representing sites with not partial
        occupancy will have all 1 values and should never be used.
        """
        num_cols = max(len(site_space) for site_space
                       in self.processor.unique_site_spaces)
        table = np.ones((self.num_sites, num_cols))
        for fus, sublatt in zip(fugacity_fractions, self.sublattices):
            ordered_fus = [fus[sp] for sp in sublatt.site_space]
            table[sublatt.sites, :len(ordered_fus)] = ordered_fus
        return table

    def as_dict(self):
        """Get Json-serialization dict representation.

        Return:
            MSONable dict
        """
        d = super().as_dict()
        d['fugacity_fractions'] = [tuple((sp.as_dict(), fu)
                                         for sp, fu in fus.items())
                                   for fus in self.fugacity_fractions]
        return d

    @classmethod
    def from_dict(cls, d):
        """Instantiate a FuSemiGrandEnsemble from dict representation.

        Return:
            FuSemiGrandEnsemble
        """
        sl_dicts = d.get('sublattices')
        sublattices = ([Sublattice.from_dict(s) for s in sl_dicts] if
                       sl_dicts is not None else None)

        fugacity_fractions = []
        for sublatt in d['fugacity_fractions']:
            fus = {}
            for sp, fu in sublatt:
                if ("oxidation_state" in sp
                        and Element.is_valid_symbol(sp["element"])):
                    sp = Species.from_dict(sp)
                elif "oxidation_state" in sp:
                    if sp['@class'] == 'Vacancy':
                        sp = Vacancy.from_dict(sp)
                    else:
                        sp = DummySpecies.from_dict(sp)
                else:
                    sp = Element(sp["element"])
                fus[sp] = fu
            fugacity_fractions.append(fus)
        return cls(Processor.from_dict(d['processor']),
                   fugacity_fractions=fugacity_fractions,
                   sublattices=sublattices)


# ChargeNeutralSemiGrandEnsemble no longer needs a separate class.
class DiscChargeNeutralSemiGrandEnsemble(BaseSemiGrandEnsemble, MSONable):
    """Sublattice disctiminative charge neutral semigrand ensemble.

    This is used to examine ground states convergence of cluster expansion
    only.

    Discriminating the same specie on different sublattices is necessary
    when computing ground state occupancies, because when talking about
    ground states we actually care about internal distribution of species
    on each sublattice.

    A non-discriminative charge neutral ensemble is not necessary, as you
    only need to specify step_type = 'charge-neutral-flip' when intializing
    sampler with a MuSemiGrandEnsemble.

    You are not recommended to use this ensemble for other purposes!
    """

    valid_mcmc_steps = ('table-flip', 'subchain-walk')
    # Biased walk is not allowed in Disc ensemble because intermediate
    # states would appear in chain, and the chemical works of them can
    # not be defined under constrained coordinates.

    def __init__(self, processor, mu, sublattices=None):
        """Initialize DiscChargeNeutralSemiGrandEnsemble.

        Args:
            processor (Processor):
                A processor that can compute the change in a property given
                a set of flips. See moca.processor
            mu (1D arrayLike):
                chemical potentials on sublattice discriminative, contrained
                coordinates.
            sublattices (list of Sublattice): optional
                list of Sublattice objects representing sites in the processor
                supercell with same site spaces. Must be active sublattices.
        """
        super().__init__(processor, sublattices)

        # Must use complete sublattices, instead of sublattices generated by
        # expansion structure.
        self.mu = np.array(mu)

        self.bits = [sl.species for sl in self.all_sublattices]
        self.sc_sublat_list = [sl.sites for sl in self.all_sublattices]
        self.sc_sl_sizes = [len(sl_sites) for sl_sites in self.sc_sublat_list]

        self.sc_size = GCD_list(self.sc_sl_sizes)
        self.sl_sizes = [sl_size // self.sc_size
                         for sl_size in self.sc_sl_sizes]

        self._compspace = CompSpace(self.bits, self.sl_sizes)
        if len(self.mu) != self._compspace.dim:
            raise ValueError("Chemcial potential dimension mismatch \
                             compositional space!")

    def compute_chemical_work(self, occupancy):
        """Compute the chemical work.

        Args:
            occupancy(ndarray):
                encoded occupancy string.
        Return:
            float: chemcial work of the occupancy.
        """
        compstat = occu_to_species_stat(occupancy,
                                        self.all_sublattices)
        ccoord = self._compspace.translate_format(compstat,
                                                  from_format='compstat',
                                                  to_format='constr',
                                                  sc_size=self.sc_size)
        return np.dot(ccoord, self.mu)

    def compute_feature_vector_change(self, occupancy, step):
        """Compute change of feature vector from a proposed step.

        Currently only supports a single flip in the table, so do not
        combine multiple flips!

        Args:
            occupancy (ndarray):
                encoded occupancy string.
            step (list[tuple(int,int)]):
                A sequence of flips given my the MCMCUsher.propose_step

        Return:
            ndarray: difference in feature vector
        """
        delta_feature = self.processor.compute_feature_vector_change(occupancy,
                                                                     step)

        # Compute flip direction in the compositional space.
        delta_ccoords = delta_ccoords_from_step(occupancy,
                                                step,
                                                self._compspace,
                                                self.all_sublattices)

        delta_mu = np.dot(delta_ccoords, self.mu)

        return np.append(delta_feature, delta_mu)

    def as_dict(self):
        """Serialize object into dict.

        Return:
            dict.
        """
        d = super().as_dict()
        d['mu'] = self.mu.tolist()
        return d

    @classmethod
    def from_dict(cls, d):
        """Initialize from a dict.

        Args:
            d(dict):
                dictionary containing all neccessary info to initialize.
        Return:
            DiscChargeNeutralSemiGrandEnsemble.
        """
        sl_dicts = d.get('sublattices')
        sublattices = ([Sublattice.from_dict(s) for s in sl_dicts] if
                       sl_dicts is not None else None)

        return cls(Processor.from_dict(d['processor']),
                   mu=d['mu'],
                   sublattices=sublattices)
