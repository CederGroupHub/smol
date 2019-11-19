from __future__ import division
import warnings
from collections import defaultdict
import logging
import numpy as np
from pymatgen import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from .utils import NotFittedError


def _pd(structures, energies, ce):
    """
    Generate a phase diagram with the structures and energies
    """
    entries = []

    for s, e in zip(structures, energies):
        entries.append(PDEntry(s.composition.element_composition, e))

    max_e = max(entries, key=lambda e: e.energy_per_atom).energy_per_atom + 1000
    for el in ce.structure.composition.keys():
        entries.append(PDEntry(Composition({el: 1}).element_composition, max_e))

    return PhaseDiagram(entries)


def _energies_above_composition(structures, energies):
    min_e = defaultdict(lambda: np.inf)
    for s, e in zip(structures, energies):
        comp = s.composition.reduced_composition
        if e / len(s) < min_e[comp]:
            min_e[comp] = e / len(s)
    e_above = []
    for s, e in zip(structures, energies):
        comp = s.composition.reduced_composition
        e_above.append(e / len(s) - min_e[comp])
    return np.array(e_above)


def _energies_above_hull(pd, structures, energies):
    e_above_hull = []
    for s, e in zip(structures, energies):
        e_above_hull.append(pd.get_e_above_hull(PDEntry(s.composition.element_composition, e)))
    return np.array(e_above_hull)


class ClusterExpansion(object):
    _pd_input = None
    _pd_ce = None
    _e_above_hull_input = None
    _e_above_hull_ce = None

    def __init__(self, datawrangler, solver='cvxopt_l1', weights=None, max_dielectric=None,  ecis=None):
        """
        Fit ECI's to obtain a cluster expansion. This init function takes in all possible arguments,
        but its much simpler to use one of the factory classmethods below,
        e.g. EciGenerator.unweighted

        Args:
            datawrangler: A StructureWrangler object to provide the fitting data and processing
            max_dielectric: constrain the dielectric constant to be positive and below the
                supplied value (note that this is also affected by whether the primitive
                cell is the correct size)
            solver: solver, current options are cvxopt_l1, bregman_l1, gs_preserve
        """
        self.wrangler = datawrangler
        self.solver = solver
        self.max_dielectric = max_dielectric
        self.ecis = ecis
        self.weights = np.ones(len(self.wrangler.structures)) if weights is None else weights

        # do least squares for comparison
        # x = np.linalg.lstsq(self.feature_matrix, self.normalized_energies)[0]
        # ls_err = np.dot(self.feature_matrix, x) - self.normalized_energies
        # logging.info('least squares rmse: {}'.format(np.sqrt(np.sum(ls_err ** 2) / len(self.feature_matrix))))

    #TODO The next @properties should be removed and an analysis/tools module with hull stuff should do this
    @property
    def pd_input(self):
        if self._pd_input is None:
            self._pd_input = _pd(self.structures, self.energies, self.datawranger.cs)
        return self._pd_input

    @property
    def pd_ce(self):
        if self._pd_ce is None:
            self._pd_ce = _pd(self.structures, self.datawranger.cs_energies, self.datawranger.cs)
        return self._pd_ce

    @property
    def e_above_hull_input(self):
        if self._e_above_hull_input is None:
            self._e_above_hull_input = _energies_above_hull(self.pd_input, self.structures, self.energies)
        return self._e_above_hull_input

    @property
    def e_above_hull_ce(self):
        if self._e_above_hull_ce is None:
            self._e_above_hull_ce = _energies_above_hull(self.pd_ce, self.structures, self.datawranger.cs_energies)
        return self._e_above_hull_ce

    def fit(self, f, mu, override_solver=False):
        #TODO change this so that the CE gets a solver class, implement it and also allow to bring in sklearn classes...
        """
        Returns the A matrix and f vector for the bregman
        iterations, given weighting parameters
        """
        A_in = self.wrangler.feature_matrix.copy()
        f_in = self.wrangler.propert_vector.copy()

        self.ecis = self._solve_weighted(A_in, f_in, self.weights, mu, override_solver=override_solver)

        if self.max_dielectric is not None:
            if self.wrangler.cs.use_ewald is False:
                warnings.warn('The StructureWrangler.use_ewald is False can not constrain by max_dieletric'
                                    ' This will be ignored', RuntimeWarning)
                return

            if self.datawranger.cs.use_inv_r:
                warnings.warn('Cant use inv_r with max dielectric. This has not been implemented yet. '
                               'inv_r will be ignored.', RuntimeWarning)

            if self.ecis[-1] < 1 / self.max_dielectric:
                f_in -= A_in[:, -1] / self.max_dielectric
                A_in[:, -1] = 0
                ecis = self._solve_weighted(A_in, f_in, self.weights, mu, override_solver=override_solver)
                ecis[-1] = 1 / self.max_dielectric

        self.ecis = ecis

    def predict(self, structure):
        if self.ecis is None:
            raise NotFittedError('This ClusterExpansion has not been fitted yet. '
                                 'Run ClusterExpansion.fit to do so.')
        scaled_corr = self.datawrangler.clustersubspace.corr_from_structure(structure)
        return np.dot(scaled_corr, self.ecis)

    def _calc_cv_score(self, mu, A, f, weights, k=5):
        """
        Args:
            mu: weight of error in bregman
            A: sensing matrix (scaled appropriately)
            f: data to fit (scaled appropriately)
            k: number of partitions

        Partition the sample into k partitions, calculate out-of-sample
        variance for each of these partitions, and add them together
        """
        logging.info('starting cv score calculations for mu: {}, k: {}'.format(mu, k))
        # generate random partitions
        partitions = np.tile(np.arange(k), len(f) // k + 1)
        np.random.shuffle(partitions)
        partitions = partitions[:len(f)]

        ssr = 0
        ssr_uw = 0
        for i in range(k):
            ins = (partitions != i)  # in the sample for this iteration
            oos = (partitions == i)  # out of the sample for this iteration

            ecis = self._fit(A[ins], f[ins], weights[ins], mu, override_solver=True)
            res = (np.dot(A[oos], ecis) - f[oos]) ** 2
            ssr += np.sum(res * weights[oos]) / np.average(weights[oos])
            ssr_uw += np.sum(res)

        logging.info(
            'cv rms_error: {} (weighted) {} (unweighted)'.format(np.sqrt(ssr / len(f)), np.sqrt(ssr_uw / len(f))))
        cv = 1 - ssr / np.sum((f - np.average(f)) ** 2)
        return cv

    def print_ecis(self):
        if self.ecis is None:
            raise NotFittedError('This ClusterExpansion has not been fitted yet. '
                                 'Run ClusterExpansion.fit to do so.')

        corr = np.zeros(self.wrangler.cs.n_bit_orderings)
        corr[0] = 1  # zero point cluster
        cluster_std = np.std(self.feature_matrix, axis=0)
        for orbit in self.datawranger.cs.orbits:
            print(orbit, len(orbit.bits) - 1, orbit.sc_b_id)
            print('bit    eci    cluster_std    eci*cluster_std')
            for i, bits in enumerate(orbit.bit_combos):
                eci = self.ecis[orbit.sc_b_id + i]
                c_std = cluster_std[orbit.sc_b_id + i]
                print(bits, eci, c_std, eci * c_std)
        print(self.ecis)

    @classmethod
    def from_dict(cls, d):
        return cls(cluster_expansion=ClusterExpansion.from_dict(d['cluster_expansion']),
                   structures=[Structure.from_dict(s) for s in d['structures']],
                   energies=np.array(d['energies']), max_dielectric=d.get('max_dielectric'),
                   max_ewald=d.get('max_ewald'), supercell_matrices=d['supercell_matrices'],
                   mu=d.get('mu'), ecis=d.get('ecis'), feature_matrix=d.get('feature_matrix'),
                   solver=d.get('solver', 'cvxopt_l1'), weights=d['weights'])

    def as_dict(self):
        return {'cluster_expansion': self.datawranger.cs.as_dict(),
                'structures': [s.as_dict() for s in self.structures],
                'energies': self.energies.tolist(),
                'supercell_matrices': [cs.supercell_matrix.tolist() for cs in self.supercells],
                'max_dielectric': self.max_dielectric,
                'max_ewald': self.wrangler.max_ewald,
                'mu': self.mu,
                'ecis': self.ecis.tolist(),
                'feature_matrix': self.feature_matrix.tolist(),
                'weights': self.weights.tolist(),
                'solver': self.solver,
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__}
