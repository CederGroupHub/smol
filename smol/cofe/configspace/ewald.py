"""
Provides the functionality to fit an Ewald term as a fictitious ECI in a
cluster expansion as proposed by William Richards & Daniil Kitchaev to improve
convergence of expansions of ionic materials
"""

from itertools import combinations
import numpy as np
import warnings
from pymatgen import Structure, PeriodicSite
from pymatgen.analysis.ewald import EwaldSummation


class EwaldTerm():
    """
    Class cooked up to remove the functions required to fit an ewald term from
    the supercel This concept can be extended to other terms, but then the best
    would probably be to write a base class for extra cluster expansion terms
    and then subclass that with any other terms people can cook up Baseclass
    could simply have an abstract classmethod the returns True if a supercell
    is needed for it to be constructed
    """

    def __init__(self, cluster_supercell, eta=None, use_inv_r=False):
        """
        Args:
            cluster_supercell (ClusterSupercell):
                ClusterSupercell for which to evaluate Ewald sum
            use_inv_r (bool):
                experimental feature that allows fitting to arbitrary 1/r
                interactions between specie-site combinations.
            eta (float):
                parameter to override the EwaldSummation default eta.
                Usually only necessary if use_inv_r=True
        """

        self.cluster_supercell = cluster_supercell
        self.eta = eta
        self.use_inv_r = use_inv_r

        # lazily generate the difficult ewald parts
        self.ewald_inds = []
        ewald_sites = []

        if use_inv_r and eta is None:
            warnings.warn('Be careful, you might need to change eta to get '
                          'properly converged electrostatic energies. This is '
                          'not well tested', RuntimeWarning)

        for bits, s in zip(self.cluster_supercell.bits,
                           self.cluster_supercell.supercell):
            inds = np.zeros(max(self.cluster_supercell.nbits) + 1) - 1
            for i, b in enumerate(bits):
                if b == 'Vacancy':
                    # inds.append(-1)
                    continue
                inds[i] = len(ewald_sites)
                ewald_sites.append(PeriodicSite(b, s.frac_coords, s.lattice))
            self.ewald_inds.append(inds)

        self.ewald_inds = np.array(self.ewald_inds, dtype=np.int)
        self._ewald_structure = Structure.from_sites(ewald_sites)
        self._ewald_matrix = None
        self._partial_ems = None
        self._all_ewalds = None
        self._range = np.arange(len(self.cluster_supercell.nbits))

    @classmethod
    def corr_from_occu(cls, occu, cluster_supercell, eta=None,
                       use_inv_r=False):
        return cls(cluster_supercell, eta, use_inv_r)._get_ewald_eci(occu)

    @property
    def all_ewalds(self):
        if self._all_ewalds is None:
            ms = [self.ewald_matrix]
            if self.use_inv_r:
                ms += self.partial_ems
            self._all_ewalds = np.array(ms)
        return self._all_ewalds

    @property
    def ewald_matrix(self):
        if self._ewald_matrix is None:
            self._ewald = EwaldSummation(self._ewald_structure,
                                         eta=self.eta)
            self._ewald_matrix = self._ewald.total_energy_matrix
        return self._ewald_matrix

    @property
    def partial_ems(self):
        if self._partial_ems is None:
            # There seems to be an issue with SpacegroupAnalyzer such that
            # making a supercell can actually reduce the symmetry operations,
            # so we're going to group the ewald matrix by the equivalency in
            # self.cluster_indices
            equiv_orb_inds = []
            ei = self.ewald_inds
            n_inds = len(self.ewald_matrix)
            for orb, inds in self.cluster_supercell.cluster_indices:
                # only want the point terms, which should be first
                if len(orb.bits) > 1:
                    break
                equiv = ei[inds[:, 0]]  # inds is 2d, but these are point terms
                for inds in equiv.T:
                    if inds[0] > -1:
                        b = np.zeros(n_inds, dtype=np.int)
                        b[inds] = 1
                        equiv_orb_inds.append(b)

            self._partial_ems = []
            for x in equiv_orb_inds:
                mask = x[None, :] * x[:, None]
                self._partial_ems.append(self.ewald_matrix * mask)
            for x, y in combinations(equiv_orb_inds, r=2):
                mask = x[None, :] * y[:, None]
                # for the love of god don't use a += here,
                # or you will forever regret it
                mask = mask.T + mask
                self._partial_ems.append(self.ewald_matrix * mask)
        return self._partial_ems

    def _get_ewald_occu(self, occu):
        i_inds = self.ewald_inds[self._range, occu]

        # instead of this line:
        #   i_inds = i_inds[i_inds != -1]
        # just make b_inds one longer than it needs to be and don't return
        # the last value
        b_inds = np.zeros(len(self._ewald_structure) + 1, dtype=np.bool)
        b_inds[i_inds] = True
        return b_inds[:-1]

    def _get_ewald_eci(self, occu):
        # This is a quick fix for occu being a list of species strings now.
        # Could be better?
        occu = self.cluster_supercell.encode_occu(occu)
        inds = self._get_ewald_occu(occu)
        ecis = [np.sum(self.ewald_matrix[inds, :][:, inds])/self.cluster_supercell.size]  # noqa

        if self.use_inv_r:
            for m in self.partial_ems:
                ecis.append(np.sum(m[inds, :][:, inds])/self.cluster_supercell.size)  # noqa

        return np.array(ecis)

    def _get_ewald_diffs(self, new_occu, occu):
        inds = self._get_ewald_occu(occu)
        new_inds = self._get_ewald_occu(new_occu)
        diff = inds != new_inds
        both = inds & new_inds
        add = new_inds & diff
        sub = inds & diff

        ms = [self.ewald_matrix]
        if self.use_inv_r:
            ms += self.partial_ems

        diffs = []
        for m in ms:
            ma = m[add]
            ms = m[sub]
            v = np.sum(ma[:, add]) - np.sum(ms[:, sub]) + \
                (np.sum(ma[:, both]) - np.sum(ms[:, both])) * 2

            diffs.append(v / self.cluster_supercell.supercell.size)

        return diffs
