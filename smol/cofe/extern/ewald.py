"""Implementation of EwaldTerm.

Provides the functionality to fit an Ewald term as an additional feature in a
cluster expansion as proposed refs below to improve convergence of expansions
of ionic materials.
Chapter 4.6 of W.D. Richard's thesis.
W. D. Richards, et al., Energy Environ. Sci., 2016, 9, 3272–3278
"""

__author__ = "William Davidson Richard, Luis Barroso-Luque"

import numpy as np
from pymatgen.analysis.ewald import EwaldSummation
from monty.json import MSONable
from base import PairwiseTerms
    
    

class EwaldTerm(PairwiseTerms, MSONable):
    """EwaldTerm Class that can be added to a ClusterSubspace.

    This class can be used as an external term added to a ClusterSubspace
    using the add_external_term method. Doing so allows to introduce Ewald
    electrostatic interaction energies as an additional feature to include
    when fitting a cluster expansion. This usually helps to increase accuracy
    and reduce complexity (number of orbits) required to fit ionic materials.
    """

    ewald_term_options = ('total', 'real', 'reciprocal', 'point')

    def __init__(self, eta=None, real_space_cut=None, recip_space_cut=None,
                 use_term='total'):
        """Initialize EwaldTerm.

        Input parameters are standard input parameters to pymatgen
        EwaldSummation. See class documentation for more information.

        Args:
            eta (float):
                Parameter to override the EwaldSummation default screening
                parameter eta. If not given it is determined automatically.
            real_space_cut (float):
                Real space cutoff radius. Determined automagically if not
                given.
            recip_space_cut (float):
                Reciprocal space cutoff radius. Determined automagically if not
                given.
            use_term (str): optional
                the Ewald expansion term to use as an additional term in the
                expansion. Options are total, real, reciprocal, point.
        """
        super().__init__()
        self.eta = eta
        self.real_space_cut = real_space_cut
        self.recip_space_cut = recip_space_cut
        if use_term not in self.ewald_term_options:
            raise AttributeError(f'Provided use_term {use_term} is not a valid'
                                 'option. Please use one of '
                                 f'{self.ewald_term_options}.')
        self.use_term = use_term


    def value_from_occupancy(self, occu, structure):
        """Obtain the Ewald interaction energy.

        The size of the given structure in terms of prims.
        The computed electrostatic interactions do not include the charged
        cell energy (which is only important for charged structures). See
        the pymatgen EwaldSummation class for further details.
        For more details also look at
        https://doi.org/10.1017/CBO9780511805769.034 (pp. 499–511).

        Args:
            occu (array):
                occupation vector for the given structure
            structure (Structure):
                pymatgen Structure for which to compute the electrostatic
                interactions,
        Returns:
            float
        """
        ewald_structure, ewald_inds = self.get_pairwise_structure(structure)
        ewald_summation = EwaldSummation(ewald_structure, self.real_space_cut,
                                         self.recip_space_cut, eta=self.eta)
        ewald_matrix = self.get_ewald_matrix(ewald_summation)
        ew_occu = self.get_pairwise_occu(occu, ewald_matrix.shape[0], ewald_inds)
        return np.array([np.sum(ewald_matrix[ew_occu, :][:, ew_occu])])

    def get_ewald_matrix(self, ewald_summation):
        """Get the corresponding Ewald matrix for the given term to use.

        Args:
            ewald_summation (EwaldSummation):
                pymatgen EwaldSummation instance
        Returns:
            ndarray: ewald matrix for corresponding term
        """
        if self.use_term == 'total':
            matrix = ewald_summation.total_energy_matrix
        elif self.use_term == 'reciprocal':
            matrix = ewald_summation.reciprocal_space_energy_matrix
        elif self.use_term == 'real':
            matrix = ewald_summation.real_space_energy_matrix
        else:
            matrix = np.diag(ewald_summation.point_energy_matrix)
        return matrix

    def as_dict(self) -> dict:
        """
        Get Json-serialization dict representation.

        Returns:
            dict: MNSONable dict
        """
        d = {'@module': self.__class__.__module__,
             '@class': self.__class__.__name__,
             'eta': self.eta,
             'real_space_cut': self.real_space_cut,
             'recip_space_cut': self.recip_space_cut,
             'use_term': self.use_term}
        return d

    @classmethod
    def from_dict(cls, d):
        """Create EwaldTerm from msonable dict.

        (Over-kill here since only EwaldSummation params are saved).
        """
        return cls(eta=d['eta'], real_space_cut=d['real_space_cut'],
                   recip_space_cut=d['recip_space_cut'],
                   use_term=d['use_term'])
