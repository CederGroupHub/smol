from __future__ import division
import warnings
from collections import defaultdict
import numpy as np
from pymatgen import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from .utils import NotFittedError
from .regression.estimator import BaseEstimator


class ClusterExpansion(object):
    _pd_input = None
    _pd_ce = None
    _e_above_hull_input = None
    _e_above_hull_ce = None

    def __init__(self, datawrangler, estimator=None, max_dielectric=None, ecis=None):
        """
        Fit ECI's to obtain a cluster expansion. This init function takes in all possible arguments,
        but its much simpler to use one of the factory classmethods below,
        e.g. EciGenerator.unweighted

        Args:
            datawrangler: A StructureWrangler object to provide the fitting data and processing
            max_dielectric: constrain the dielectric constant to be positive and below the
                supplied value (note that this is also affected by whether the primitive
                cell is the correct size)
            estimator: Estimator or sklearn model
        """
        self.wrangler = datawrangler
        self.estimator = estimator
        self.max_dielectric = max_dielectric
        self.ecis = ecis

        if self.estimator is None:
            if self.ecis is None:
                raise AttributeError('No estimator or ecis were given. One of them needs to be provided')
            self.estimator = BaseEstimator()
            self.estimator.coef_ = self.ecis

        # do least squares for comparison
        # x = np.linalg.lstsq(self.feature_matrix, self.normalized_energies)[0]
        # ls_err = np.dot(self.feature_matrix, x) - self.normalized_energies
        # logging.info('least squares rmse: {}'.format(np.sqrt(np.sum(ls_err ** 2) / len(self.feature_matrix))))

    #TODO The next @properties should be removed and an analysis/tools module with hull stuff should do this
    @property
    def pd_input(self):
        if self._pd_input is None:
            self._pd_input = _pd(self.structures, self.energies, self.wrangler.cs)
        return self._pd_input

    @property
    def pd_ce(self):
        if self._pd_ce is None:
            self._pd_ce = _pd(self.structures, self.wrangler.cs_energies, self.wrangler.cs)
        return self._pd_ce

    @property
    def e_above_hull_input(self):
        if self._e_above_hull_input is None:
            self._e_above_hull_input = _energies_above_hull(self.pd_input, self.structures, self.energies)
        return self._e_above_hull_input

    @property
    def e_above_hull_ce(self):
        if self._e_above_hull_ce is None:
            self._e_above_hull_ce = _energies_above_hull(self.pd_ce, self.structures, self.wrangler.cs_energies)
        return self._e_above_hull_ce

    def fit(self, *args, **kwargs):
        """
        Fits the cluster expansion using the given estimator's fit function
        args, kwargs are the arguments and keyword arguments taken by the Estimator.fit function
        """
        A_in = self.wrangler.feature_matrix.copy()
        y_in = self.wrangler.normalized_properties.copy()

        self.estimator.fit(A_in, y_in, *args, **kwargs)
        try:
            self.ecis = self.estimator.coef_
        except AttributeError:
            msg = f'The provided estimator does not provide fit coefficients for ECIS: {self.estimator}'
            warnings.warn(msg)

        if self.max_dielectric is not None:
            if self.wrangler.cs.use_ewald is False:
                warnings.warn('The StructureWrangler.use_ewald is False can not constrain by max_dieletric'
                                    ' This will be ignored', RuntimeWarning)
                return

            if self.wrangler.cs.use_inv_r:
                warnings.warn('Cant use inv_r with max dielectric. This has not been implemented yet. '
                               'inv_r will be ignored.', RuntimeWarning)

            if self.ecis[-1] < 1 / self.max_dielectric:
                y_in -= A_in[:, -1] / self.max_dielectric
                A_in[:, -1] = 0
                self.estimator.fit(A_in, y_in, *args, **kwargs)
                self.ecis[-1] = 1 / self.max_dielectric

    def predict(self, structures, normalized=False):
        structures = structures if type(structures) == list else [structures]
        corrs = []
        for structure in structures:
            size, corr = self.wrangler.cs.size_corr_from_structure(structure)
            if not normalized:
                corr *= size
            corrs.append(corr)

        return self.estimator.predict(np.array(corrs))

    def print_ecis(self):
        if self.ecis is None:
            raise NotFittedError('This ClusterExpansion has no ECIs available.'
                                 'If it has not been fitted yet, run ClusterExpansion.fit to do so.'
                                 'Otherwise you may have chosen an estimator that does not provide them:'
                                 f'{self.estimator}.')

        corr = np.zeros(self.wrangler.cs.n_bit_orderings)
        corr[0] = 1  # zero point cluster
        cluster_std = np.std(self.feature_matrix, axis=0)
        for orbit in self.wrangler.cs.orbits:
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
        return {'cluster_expansion': self.wrangler.cs.as_dict(),
                'structures': [s.as_dict() for s in self.structures],
                'energies': self.energies.tolist(),
                'supercell_matrices': [cs.supercell_matrix.tolist() for cs in self.supercells],
                'max_dielectric': self.max_dielectric,
                'max_ewald': self.wrangler.max_ewald,
                'mu': self.mu,
                'ecis': self.ecis.tolist(),
                'feature_matrix': self.feature_matrix.tolist(),
                'weights': self.weights.tolist(),
                'solver': self.estimator,
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__}


#TODO need to refactor this into a tools module and remove the corresponding ClusterExpansion attributes
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


def weight_by_e_above_hull(cls, cluster_expansion, structures, energies, mu=None, max_dielectric=None,
                           max_ewald=None, temperature=2000, solver='cvxopt_l1'):
    pd = _pd(structures, energies, cluster_expansion)
    e_above_hull = _energies_above_hull(pd, structures, energies)
    weights = np.exp(-e_above_hull / (0.00008617 * temperature))

    return cls(cluster_expansion=cluster_expansion, structures=structures, energies=energies,
               mu=mu, weights=weights, max_dielectric=max_dielectric, max_ewald=max_ewald, solver=solver)


def weight_by_e_above_comp(cls, cluster_expansion, structures, energies, mu=None, max_dielectric=None,
                           max_ewald=None, temperature=2000, solver='cvxopt_l1'):
    e_above_comp = _energies_above_composition(structures, energies)
    weights = np.exp(-e_above_comp / (0.00008617 * temperature))

    return cls(cluster_expansion=cluster_expansion, structures=structures, energies=energies,
               mu=mu, weights=weights, max_dielectric=max_dielectric, max_ewald=max_ewald, solver=solver)
