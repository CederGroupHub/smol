"""
Provides the functionality to fit an Ewald term as an additional feature in a
cluster expansion as proposed refs below to improve convergence of expansions
of ionic materials.
Chapter 4.6 of W.D. Richard's thesis.
W. D. Richards, et al., Energy Environ. Sci., 2016, 9, 3272–3278
"""

import numpy as np
from pymatgen import Structure, PeriodicSite
from pymatgen.analysis.ewald import EwaldSummation
from smol.cofe.configspace.utils import get_bits


class EwaldTerm():
    """
    This class can be used as an external term added to a ClusterSubspace
    using the add_external_term method. Doing so allows to introduce Ewald
    electrostatic interaction energies as an additional feature to include
    when fitting a cluster expansion. This usually helps to increase accuracy
    and reduce complexity (number of orbits) required to fit ionic materials.
    """

    def __init__(self, eta=None):
        """
        Args:
            eta (float):
                parameter to override the EwaldSummation default eta.
        """
        self.eta = eta

    def corr_from_occupancy(self, occu, structure, size):
        """
        Obtains the Ewald interaction energy normalized by the given size
        (which should be the size of the given structure in terms of prims)
        The computed electrostatic interactions do not include the charged
        cell energy (which is only important for charged structures). See
        the pymatgen EwaldSummation class for further details.
        Args:
            occu (array):
                occupation vector for the given structure
            structure (Structure):
                pymatgen Structure for which to compute the electrostatic
                interactions,
            size (int):
                the size of the given structure in number of prim cells
        Returns:
            float
        """
        ewald_structure, ewald_inds = self._get_ewald_structure(structure)
        ewald = EwaldSummation(ewald_structure, eta=self.eta)
        ewald_matrix = ewald.total_energy_matrix
        ew_occu = self._get_ewald_occu(occu, len(ewald_structure), ewald_inds)
        return np.array([np.sum(ewald_matrix[ew_occu, :][:, ew_occu])])/size

    @staticmethod
    def _get_ewald_occu(occu, num_ewald_sites, ewald_inds):
        i_inds = ewald_inds[np.arange(len(occu)), occu]
        # instead of this line:
        #   i_inds = i_inds[i_inds != -1]
        # just make b_inds one longer than it needs to be and don't return
        # the last value
        b_inds = np.zeros(num_ewald_sites + 1, dtype=np.bool)
        b_inds[i_inds] = True
        return b_inds[:-1]

    @staticmethod
    def _get_ewald_structure(structure):
        """
        Gets the structure contributing to Ewald summation (removes vacancies)
        and the corresponding indices to those sites from the original
        structure
        Args:
            structure:
                Structure to compute Ewald interaction energy from
        Returns:
            Structure, array
        """
        struct_bits = get_bits(structure)
        nbits = np.array([len(b) - 1 for b in struct_bits])
        ewald_inds, ewald_sites = [], []
        for bits, site in zip(struct_bits, structure):
            inds = np.zeros(max(nbits) + 1) - 1
            for i, b in enumerate(bits):
                if b == 'Vacancy':  # skip vacancies
                    continue
                inds[i] = len(ewald_sites)
                ewald_sites.append(PeriodicSite(b, site.frac_coords,
                                                site.lattice))
            ewald_inds.append(inds)
        ewald_inds = np.array(ewald_inds, dtype=np.int)
        ewald_structure = Structure.from_sites(ewald_sites)

        return ewald_structure, ewald_inds
