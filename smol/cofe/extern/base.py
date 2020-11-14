"""Abstract Base class for calculating Pairwise Interactions."""

__author__ = "Charles Yang"


import numpy as np
from abc import ABC
from pymatgen import Structure, PeriodicSite
from smol.cofe.space.domain import get_allowed_species, Vacancy


class PairwiseTerms(ABC):
    """Abstract Base Class for calculating pair-wise terms.

    Attributes:    
    """
    
    def __init__():
        return

    @staticmethod
    def get_pairwise_structure(structure):
        """Get the structure and site indices contributing to pairwise summation.

        Creates a structure with overlapping sites for each species in the
        corresponding site space. This is used to construct single Pairwise
        matrices for all possible configurations. The pairwise_inds array is a
        2D array where the first element is the index of the site in a
        supercell and the second is the encoded species occupancy

        Removes vacancies and the corresponding indices to those sites from the
        original structure.

        Args:
            structure:
                Structure to compute pairwise interactions from

        Returns:
            tuple: (Structure, array) structure and indices of sites
        """
        site_spaces = get_allowed_species(structure)
        nbits = np.array([len(b) - 1 for b in site_spaces])
        pairwise_inds, pairwise_sites = [], []
        for space, site in zip(site_spaces, structure):
            # allocate array with all -1 for vacancies
            inds = np.zeros(max(nbits) + 1) - 1
            for i, b in enumerate(space):
                if isinstance(b, Vacancy):  # skip vacancies
                    continue
                inds[i] = len(pairwise_sites)
                pairwise_sites.append(PeriodicSite(b, site.frac_coords,
                                                site.lattice))
            pairwise_inds.append(inds)

        pairwise_inds = np.array(pairwise_inds, dtype=np.int)
        pairwise_structure = Structure.from_sites(pairwise_sites)

        return pairwise_structure, pairwise_inds

    @staticmethod
    def get_pairwise_occu(occu, num_pairwise_sites, pairwise_inds):
        """Get the pairwise indices from a given encoded occupancy string.

        The pairwise indices indicate matrix elements (ie species) corresponding
        to the given occupancy.

        Args:
            occu (ndarray):
                encoded encoded occupancy string
            num_pairwise_sites (int):
                number of total pairwise sites. This is the size of the pairwise interaction
                matrix, the sum of the size of all site spaces for all sites
                in the cell.
            pairwise_inds (ndarray):
                pairwise indices for the corresponding species in the occupancy
                array.

        Returns:
            ndarray: indices of species in ewald matrix
        """
        i_inds = pairwise_inds[np.arange(len(occu)), occu]
        # instead of this line:
        #   i_inds = i_inds[i_inds != -1]
        # just make b_inds one longer than it needs to be and don't return
        # the last value
        b_inds = np.zeros(num_pairwise_sites + 1, dtype=np.bool)
        b_inds[i_inds] = True
        return b_inds[:-1]
