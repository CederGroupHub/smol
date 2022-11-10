"""Implementation of EwaldTerm.

Provides the functionality to fit an Ewald term as an additional feature in a
cluster expansion as proposed refs below to improve convergence of expansions
of ionic materials.
Chapter 4.6 of W.D. Richard's thesis.
W. D. Richards, et al., Energy Environ. Sci., 2016, 9, 3272–3278
"""

__author__ = "William Davidson Richard, Luis Barroso-Luque"

import numpy as np
from monty.json import MSONable
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core import PeriodicSite, Structure

from smol.cofe.space.domain import Vacancy, get_allowed_species


class EwaldTerm(MSONable):
    """EwaldTerm class that can be added to a ClusterSubspace.

    This class can be used as an external term added to a ClusterSubspace
    using the add_external_term method. Doing so introduces Ewald
    electrostatic interaction energies as an additional feature to include
    when fitting a cluster expansion. This usually helps to increase accuracy
    and reduce complexity (number of orbits) required to fit ionic materials.
    """

    ewald_term_options = ("total", "real", "reciprocal", "point")

    def __init__(
        self, eta=None, real_space_cut=None, recip_space_cut=None, use_term="total"
    ):
        """Initialize EwaldTerm.

        Input parameters are standard input parameters to pymatgen
        EwaldSummation. See class documentation for more information.

        Args:
            eta (float):
                parameter to override the EwaldSummation default screening
                parameter eta. If not given it is determined automatically.
            real_space_cut (float):
                real space cutoff radius. Determined automatically if not
                given.
            recip_space_cut (float):
                reciprocal space cutoff radius. Determined automatically if not
                given.
            use_term (str): optional
                the Ewald expansion term to use as an additional term in the
                expansion. Options are total, real, reciprocal, point.
        """
        self.eta = eta
        self.real_space_cut = real_space_cut
        self.recip_space_cut = recip_space_cut
        if use_term not in self.ewald_term_options:
            raise AttributeError(
                f"Provided use_term {use_term} is not a valid option. "
                f"Please use one of {self.ewald_term_options}."
            )
        self.use_term = use_term

    @staticmethod
    def get_ewald_structure(structure):
        """Get the structure and site indices contributing to Ewald summation.

        Creates a structure with overlapping sites for each species in the
        corresponding site space. This is used to construct single Ewald
        matrices for all possible configurations. The ewald_inds array is a
        2D array where the first element is the index of the site in a
        supercell and the second is the encoded species occupancy

        Removes vacancies and the corresponding indices to those sites from the
        original structure.

        Args:
            structure:
                Structure to compute Ewald interaction energy from

        Returns:
            tuple: (Structure, array) structure and indices of sites
        """
        site_spaces = get_allowed_species(structure)
        nbits = np.array([len(b) - 1 for b in site_spaces])
        ewald_inds, ewald_sites = [], []
        for space, site in zip(site_spaces, structure):
            # allocate array with all -1 for vacancies
            inds = np.zeros(max(nbits) + 1) - 1
            for i, spec in enumerate(space):
                if isinstance(spec, Vacancy):  # skip vacancies
                    continue
                inds[i] = len(ewald_sites)
                ewald_sites.append(PeriodicSite(spec, site.frac_coords, site.lattice))
            ewald_inds.append(inds)
        ewald_inds = np.array(ewald_inds, dtype=int)
        ewald_structure = Structure.from_sites(ewald_sites)

        return ewald_structure, ewald_inds

    @staticmethod
    def get_ewald_occu(occu, num_ewald_sites, ewald_inds):
        """Get the Ewald indices from a given encoded occupancy string.

        The Ewald indices indicate matrix elements (i.e. species) corresponding
        to the given occupancy.

        Args:
            occu (ndarray):
                encoded occupancy string
            num_ewald_sites (int):
                number of total Ewald sites. This is the size of the Ewald
                matrix, the sum of the size of all site spaces for all sites
                in the cell.
            ewald_inds (ndarray):
                Ewald indices for the corresponding species in the occupancy
                array.

        Returns:
            ndarray: indices of species in ewald matrix
        """
        i_inds = ewald_inds[np.arange(len(occu)), occu]
        # instead of this line:
        #   i_inds = i_inds[i_inds != -1]
        # just make b_inds one longer than it needs to be and don't return
        # the last value
        b_inds = np.zeros(num_ewald_sites + 1, dtype=bool)
        b_inds[i_inds] = True
        return b_inds[:-1]

    def value_from_occupancy(self, occu, structure):
        """Obtain the Ewald interaction energy.

        The size of the given structure is in terms of prims.
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
                interactions
        Returns:
            float
        """
        ewald_structure, ewald_inds = self.get_ewald_structure(structure)
        ewald_summation = EwaldSummation(
            ewald_structure, self.real_space_cut, self.recip_space_cut, eta=self.eta
        )
        ewald_matrix = self.get_ewald_matrix(ewald_summation)
        ew_occu = self.get_ewald_occu(occu, ewald_matrix.shape[0], ewald_inds)
        return np.array([np.sum(ewald_matrix[ew_occu, :][:, ew_occu])])

    def get_ewald_matrix(self, ewald_summation):
        """Get the corresponding Ewald matrix for the given term to use.

        Args:
            ewald_summation (EwaldSummation):
                pymatgen EwaldSummation instance
        Returns:
            ndarray: ewald matrix for corresponding term
        """
        if self.use_term == "total":
            matrix = ewald_summation.total_energy_matrix
        elif self.use_term == "reciprocal":
            matrix = ewald_summation.reciprocal_space_energy_matrix
        elif self.use_term == "real":
            matrix = ewald_summation.real_space_energy_matrix
        else:
            matrix = np.diag(ewald_summation.point_energy_matrix)
        return matrix

    def __str__(self):
        """Pretty print EwaldTerm."""
        prop_str = ""
        for prop, val in self.__dict__.items():
            if val is not None:
                prop_str += f"{prop}={val}"
        return f"EwaldTerm({prop_str})"

    def __repr__(self):
        """Get EwaldTerm summary."""
        return f"EwaldTerm({self.use_term})"

    def as_dict(self) -> dict:
        """
        Get Json-serialization dict representation.

        Returns:
            dict: MSONable dict
        """
        ewald_d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "eta": self.eta,
            "real_space_cut": self.real_space_cut,
            "recip_space_cut": self.recip_space_cut,
            "use_term": self.use_term,
        }
        return ewald_d

    @classmethod
    def from_dict(cls, d):
        """Create EwaldTerm from MSONable dict.

        (Over-kill here since only EwaldSummation params are saved).
        """
        return cls(
            eta=d["eta"],
            real_space_cut=d["real_space_cut"],
            recip_space_cut=d["recip_space_cut"],
            use_term=d["use_term"],
        )
