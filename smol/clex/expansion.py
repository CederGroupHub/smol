from __future__ import division
import warnings
from collections import defaultdict
from itertools import chain
import logging
import numpy as np
from matplotlib import pylab as plt
from pymatgen import Composition, Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
#from ..compressive.bregman import split_bregman
from ..cluster_expansion.ce import ClusterExpansion


#TODO This needs to be simplified to only hold the cluster expansion (ie ECI) and include methods to use and analyze it
#TODO Also should have the fit methods

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


class LightFittedEciGenerator(object):

    def __init__(self, cluster_expansion, ecis):
        """
        :param cluster_expansion: cluster expansion used to fit original EciGenerator
        :param ecis: already fitted list of ECIs from an EciGenerator
        """
        self.ce = cluster_expansion
        self.ecis = ecis

    def structure_energy(self, structure):
        return self.ce.structure_energy(structure, self.ecis)

    @classmethod
    def from_eg(cls, eg):
        """
        Make a LightFittedEciGenerator from a fitted EciGenerator object
        """
        return cls(cluster_expansion=eg.ce, ecis=eg.ecis)

    @classmethod
    def from_dict(cls, d):
        return cls(cluster_expansion=ClusterExpansion.from_dict(d['cluster_expansion']), ecis=d.get('ecis'))

    def as_dict(self):
        return {'cluster_expansion': self.ce.as_dict(),
                'ecis': self.ecis.tolist(),
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__}


class EciGenerator(object):
    _pd_input = None
    _pd_ce = None
    _e_above_hull_input = None
    _e_above_hull_ce = None

    def __init__(self, cluster_expansion, structures, energies, weights,
                 mu=None, max_dielectric=None, max_ewald=None,
                 solver='cvxopt_l1', supercell_matrices=None,
                 ecis=None, feature_matrix=None):
        """
        Fit ECIs to a cluster expansion. This init function takes in all possible arguments,
        but its much simpler to use one of the factory classmethods below,
        e.g. EciGenerator.unweighted

        Args:
            cluster_expansion: A ClusterExpansion object
            structures: list of Structure objects
            energies: list of total (non-normalized) energies
            weights: list of weights for the optimization.
            mu: mu to use in the split_bregman, otherwise optimal value is calculated
                by CV optimization
            max_dielectric: constrain the dielectric constant to be positive and below the
                supplied value (note that this is also affected by whether the primitive
                cell is the correct size)
            max_ewald: filter the input structures to only use those with low electrostatic
                energies (no large charge separation in cell). This energy is referenced to the lowest
                value at that composition. Note that this is before the division by the relative dielectric
                 constant and is per primitive cell in the cluster exapnsion -- 1.5 eV/atom seems to be a
                 reasonable value for dielectric constants around 10.
            solver: solver, current options are cvxopt_l1, bregman_l1, gs_preserve
            supercell_matrices, ecis, feature_matrix: Options used by from_dict to speed up
                initialization from json. It shouldn't ever be necessary to specify these.
        """
        self.ce = cluster_expansion
        self.solver = solver
        self.max_dielectric = max_dielectric
        self.max_ewald = max_ewald

        # Match all input structures to cluster expansion
        self.items = []
        supercell_matrices = supercell_matrices or [None] * len(structures)
        fm_rows = feature_matrix or [None] * len(structures)
        for s, e, m, w, fm_row in zip(structures, energies, supercell_matrices, weights, fm_rows):
            try:
                if m is None:
                    m = self.ce.supercell_matrix_from_structure(s)
                sc = self.ce.supercell_from_matrix(m)
                if fm_row is None:
                    fm_row = sc.corr_from_structure(s)
            except Exception:
                logging.debug('Unable to match {} with energy {} to supercell'
                              ''.format(s.composition, e))
                if self.ce.supercell_size not in ['volume', 'num_sites', 'num_atoms'] \
                        and s.composition[self.ce.supercell_size] == 0:
                    logging.warn('Specie {} not in {}'.format(self.ce.supercell_size, s.composition))
                continue
            self.items.append({'structure': s,
                               'energy': e,
                               'weight': w,
                               'supercell': sc,
                               'features': fm_row,
                               'size': sc.size})

        if self.ce.use_ewald and self.max_ewald is not None:
            if self.ce.use_inv_r:
                raise NotImplementedError('cant use inv_r with max ewald yet')

            min_e = defaultdict(lambda: np.inf)
            for i in self.items:
                c = i['structure'].composition.reduced_composition
                if i['features'][-1] < min_e[c]:
                    min_e[c] = i['features'][-1]

            items = []
            for i in self.items:
                r_e = i['features'][-1] - min_e[i['structure'].composition.reduced_composition]
                if r_e > self.max_ewald:
                    logging.debug('Skipping {} with energy {}, ewald energy is {}'
                                  ''.format(i['structure'].composition, i['energy'], r_e))
                else:
                    items.append(i)
            self.items = items

        logging.info("Matched {} of {} structures".format(len(self.items),
                                                          len(structures)))

        # do least squares for comparison
        x = np.linalg.lstsq(self.feature_matrix, self.normalized_energies)[0]
        ls_err = np.dot(self.feature_matrix, x) - self.normalized_energies
        logging.info('least squares rmse: {}'.format(np.sqrt(np.sum(ls_err ** 2) / len(self.feature_matrix))))

        # calculate optimum mu
        self.mu = mu or self.get_optimum_mu(self.feature_matrix, self.normalized_energies, self.weights)
        # actually fit the cluster expansion
        if ecis is None:
            self.ecis = self._fit(self.feature_matrix, self.normalized_energies, self.weights, self.mu)
        else:
            self.ecis = np.array(ecis)

        # calculate the results of the fitting
        self.normalized_ce_energies = np.dot(self.feature_matrix, self.ecis)
        self.ce_energies = self.normalized_ce_energies * self.sizes
        self.normalized_error = self.normalized_ce_energies - self.normalized_energies

        self.rmse = np.average(self.normalized_error ** 2) ** 0.5
        logging.info('rmse (in-sample): {}'.format(self.rmse))

    @property
    def structures(self):
        return [i['structure'] for i in self.items]

    @property
    def energies(self):
        return np.array([i['energy'] for i in self.items])

    @property
    def weights(self):
        return np.array([i['weight'] for i in self.items])

    @property
    def supercells(self):
        return np.array([i['supercell'] for i in self.items])

    @property
    def feature_matrix(self):
        return np.array([i['features'] for i in self.items])

    @property
    def sizes(self):
        return np.array([i['size'] for i in self.items])

    @property
    def normalized_energies(self):
        return self.energies / self.sizes

    @classmethod
    def unweighted(cls, cluster_expansion, structures, energies, mu=None, max_dielectric=None,
                   max_ewald=None, solver='cvxopt_l1'):
        weights = np.ones(len(structures))
        return cls(cluster_expansion=cluster_expansion, structures=structures, energies=energies,
                   mu=mu, weights=weights, max_dielectric=max_dielectric, max_ewald=max_ewald, solver=solver)

    @classmethod
    def weight_by_e_above_hull(cls, cluster_expansion, structures, energies, mu=None, max_dielectric=None,
                               max_ewald=None, temperature=2000, solver='cvxopt_l1'):
        pd = _pd(structures, energies, cluster_expansion)
        e_above_hull = _energies_above_hull(pd, structures, energies)
        weights = np.exp(-e_above_hull / (0.00008617 * temperature))

        return cls(cluster_expansion=cluster_expansion, structures=structures, energies=energies,
                   mu=mu, weights=weights, max_dielectric=max_dielectric, max_ewald=max_ewald, solver=solver)

    @classmethod
    def weight_by_e_above_comp(cls, cluster_expansion, structures, energies, mu=None, max_dielectric=None,
                               max_ewald=None, temperature=2000, solver='cvxopt_l1'):
        e_above_comp = _energies_above_composition(structures, energies)
        weights = np.exp(-e_above_comp / (0.00008617 * temperature))

        return cls(cluster_expansion=cluster_expansion, structures=structures, energies=energies,
                   mu=mu, weights=weights, max_dielectric=max_dielectric, max_ewald=max_ewald, solver=solver)

    @property
    def pd_input(self):
        if self._pd_input is None:
            self._pd_input = _pd(self.structures, self.energies, self.ce)
        return self._pd_input

    @property
    def pd_ce(self):
        if self._pd_ce is None:
            self._pd_ce = _pd(self.structures, self.ce_energies, self.ce)
        return self._pd_ce

    @property
    def e_above_hull_input(self):
        if self._e_above_hull_input is None:
            self._e_above_hull_input = _energies_above_hull(self.pd_input, self.structures, self.energies)
        return self._e_above_hull_input

    @property
    def e_above_hull_ce(self):
        if self._e_above_hull_ce is None:
            self._e_above_hull_ce = _energies_above_hull(self.pd_ce, self.structures, self.ce_energies)
        return self._e_above_hull_ce

    def get_scatterplot(self, xaxis='e_above_hull_input', yaxis='e_above_hull_ce'):
        """
        plots two attributes. Some useful pairs:
            xaxis='e_above_hull_input', yaxis='e_above_hull_ce'
            xaxis='normalized_energies', yaxis='normalized_ce_energies'
            xaxis='e_above_hull_input', yaxis='normalized_error'
        """
        plt.scatter(self.__getattribute__(xaxis), self.__getattribute__(yaxis))
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        return plt

    def get_optimum_mu(self, A, f, weights, k=5, min_mu=0.1, max_mu=6):
        """
        Finds the value of mu that maximizes the cv score
        """
        mus = list(np.logspace(min_mu, max_mu, 10))
        cvs = [self._calc_cv_score(mu, A, f, weights, k) for mu in mus]

        for _ in range(2):
            i = np.nanargmax(cvs)
            if i == len(mus) - 1:
                warnings.warn('Largest mu chosen. You should probably increase the basis set')
                break

            mu = (mus[i] * mus[i + 1]) ** 0.5
            mus[i + 1:i + 1] = [mu]
            cvs[i + 1:i + 1] = [self._calc_cv_score(mu, A, f, weights, k)]

            mu = (mus[i - 1] * mus[i]) ** 0.5
            mus[i:i] = [mu]
            cvs[i:i] = [self._calc_cv_score(mu, A, f, weights, k)]

        self.mus = mus
        self.cvs = cvs
        logging.info('best cv score: {}'.format(np.nanmax(self.cvs)))
        return mus[np.nanargmax(cvs)]

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

    def _fit(self, A, f, weights, mu, override_solver=False):
        """
        Returns the A matrix and f vector for the bregman
        iterations, given weighting parameters
        """
        A_in = A.copy()
        f_in = f.copy()

        ecis = self._solve_weighted(A_in, f_in, weights, mu, override_solver=override_solver)

        if self.ce.use_ewald and self.max_dielectric is not None:
            if self.ce.use_inv_r:
                raise NotImplementedError('cant use inv_r with max dielectric yet')
            if ecis[-1] < 1 / self.max_dielectric:
                f_in -= A_in[:, -1] / self.max_dielectric
                A_in[:, -1] = 0
                ecis = self._solve_weighted(A_in, f_in, weights, mu, override_solver=override_solver)
                ecis[-1] = 1 / self.max_dielectric

        return ecis

    def _solve_weighted(self, A, f, weights, mu, override_solver=False):
        A_w = A * weights[:, None] ** 0.5
        f_w = f * weights ** 0.5

        if self.solver == 'cvxopt_l1' or override_solver:
            return self._solve_cvxopt(A_w, f_w, mu)
        elif self.solver == 'bregman_l1':
            return self._solve_bregman(A_w, f_w, mu)
        elif self.solver == 'gs_preserve':
            return self._solve_gs_preserve(A_w, f_w, mu)

    def _solve_cvxopt(self, A, f, mu):
        """
        A and f should already have been adjusted to account for weighting
        """
        from .l1regls import l1regls, solvers
        solvers.options['show_progress'] = False
        from cvxopt import matrix
        A1 = matrix(A)
        b = matrix(f * mu)
        return (np.array(l1regls(A1, b)) / mu).flatten()

    #def _solve_bregman(self, A, f, mu):
     #   return split_bregman(A, f, MaxIt=1e5, tol=1e-7, mu=mu, l=1, quiet=True)

    def _solve_gs_preserve(self, A, f, mu):
        from cvxopt import matrix
        from cvxopt import solvers
        solvers.options['show_progress'] = False
        ehull = list(self.e_above_hull_input)
        structure_index_at_hull = [i for (i, e) in enumerate(ehull) if e < 1e-5]
        # structure_index_at_hull = list(np.argwhere(self._e_above_hull_input < 1e-5)[0])

        logging.info("removing duplicated correlation entries")
        corr_in = np.array(A)
        duplicated_correlation_set = []
        for i in range(len(corr_in)):
            if i not in structure_index_at_hull:
                for j in structure_index_at_hull:
                    if np.max(np.abs(corr_in[i] - corr_in[j])) < 1e-6:
                        logging.info("structure ", i, " has the same correlation as hull structure ", j)
                        duplicated_correlation_set.append(i)

        reduce_composition_at_hull = [
            self.structures[i].composition.element_composition.reduced_composition.element_composition for
            i in structure_index_at_hull]  # Wenxuan added

        print(reduce_composition_at_hull)

        corr_in = np.array(self.feature_matrix)  # verify this line

        engr_in = np.array(self.normalized_energies)
        engr_in.shape = (len(engr_in), 1)

        weights_tmp = self.weights.copy()

        for i in duplicated_correlation_set:
            weights_tmp[i] = 0

        weight_vec = np.array(weights_tmp)  # verify this line
        # print ("weight_vec is ", weight_vec, "weight_vec.shape is is ",weight_vec.shape)

        weight_matrix = np.diag(weight_vec.transpose())

        N_corr = corr_in.shape[1]  # verify this line

        P_corr_part = 2 * ((weight_matrix.dot(corr_in)).transpose()).dot(corr_in)

        P = np.lib.pad(P_corr_part, ((0, N_corr), (0, N_corr)), mode='constant', constant_values=0)

        q_corr_part = -2 * ((weight_matrix.dot(corr_in)).transpose()).dot(engr_in)
        q_z_part = np.ones((N_corr, 1)) / mu  # CHANGED BY WILL from * mu (to be consistent with cvxopt_l1)

        q = np.concatenate((q_corr_part, q_z_part), axis=0)

        G_1 = np.concatenate((np.identity(N_corr), -np.identity(N_corr)), axis=1)
        G_2 = np.concatenate((-np.identity(N_corr), -np.identity(N_corr)), axis=1)
        G_3 = np.concatenate((G_1, G_2), axis=0)
        h_3 = np.zeros((2 * N_corr, 1))

        for i in range(len(corr_in)):
            # print (i,self.concentrations[i])
            # print(i)
            if i not in structure_index_at_hull and i not in duplicated_correlation_set:
                # print (i,"is not in hull_idx")

                reduced_comp = self.structures[
                    i].composition.element_composition.reduced_composition.element_composition
                # print(reduced_comp)
                if reduced_comp in reduce_composition_at_hull:  ## in hull composition

                    hull_idx = reduce_composition_at_hull.index(reduced_comp)
                    global_index = structure_index_at_hull[hull_idx]

                    # G_3_new_line=np.concatenate((corr_in[global_index]-corr_in[i],np.zeros((self.N_corr))),axis=1)
                    G_3_new_line = np.concatenate((corr_in[global_index] - corr_in[i], np.zeros((N_corr))))

                    G_3_new_line.shape = (1, 2 * N_corr)
                    G_3 = np.concatenate((G_3, G_3_new_line), axis=0)
                    small_error = np.array(-1e-3)
                    small_error.shape = (1, 1)
                    h_3 = np.concatenate((h_3, small_error), axis=0)

                else:  # out of hull composition

                    ele_comp_now = self.structures[
                        i].composition.element_composition.reduced_composition.element_composition
                    decomposition_now = self.pd_input.get_decomposition(ele_comp_now)
                    new_vector = -corr_in[i]
                    for decompo_keys, decompo_values in decomposition_now.iteritems():
                        reduced_decompo_keys = decompo_keys.composition.element_composition.reduced_composition.element_composition
                        index_1 = reduce_composition_at_hull.index(reduced_decompo_keys)
                        vertex_index_global = structure_index_at_hull[index_1]
                        new_vector = new_vector + decompo_values * corr_in[vertex_index_global]

                    G_3_new_line = np.concatenate((new_vector, np.zeros(
                        N_corr)))  # G_3_new_line=np.concatenate((new_vector,np.zeros((self.N_corr))),axis=1)

                    G_3_new_line.shape = (1, 2 * N_corr)
                    G_3 = np.concatenate((G_3, G_3_new_line), axis=0)

                    small_error = np.array(-1e-3)
                    small_error.shape = (1, 1)
                    h_3 = np.concatenate((h_3, small_error), axis=0)

            elif i in structure_index_at_hull:
                if self.structures[i].composition.element_composition.is_element:
                    # print ("structure i=",i,"is an element")
                    continue

                # print ("i in self.structure_index_at_hull self.structure_index_at_hull and i is ",i)
                global_index = i
                hull_idx = structure_index_at_hull.index(i)
                # print ("hull_idx is ", hull_idx)
                ele_comp_now = self.structures[
                    i].composition.element_composition.reduced_composition.element_composition
                # print ("reduced_comp is ",ele_comp_now)
                entries_new = []
                for j in structure_index_at_hull:
                    if not j == i:
                        entries_new.append(
                            PDEntry(self.structures[j].composition.element_composition, self.energies[j]))

                for el in self.ce.structure.composition.keys():
                    entries_new.append(PDEntry(Composition({el: 1}).element_composition,
                                               max(self.normalized_energies) + 1000))

                pd_new = PhaseDiagram(entries_new)
                # print ("pd_new is")
                # print pd_new
                vertices_list_new_pd = list(set(chain.from_iterable(pd_new.facets)))
                # print ("vertices_list_new_pd is",vertices_list_new_pd)
                vertices_red_composition_list_new_pd = [
                    pd_new.qhull_entries[i].composition.element_composition.reduced_composition.element_composition
                    for i in vertices_list_new_pd]
                # print ("vertices_red_composition_list_new_pd is",vertices_red_composition_list_new_pd)

                decomposition_now = pd_new.get_decomposition(ele_comp_now)

                new_vector = corr_in[i]

                abandon = False
                for decompo_keys, decompo_values in decomposition_now.iteritems():
                    # print ("decompo_keys is", decompo_keys)
                    # print ("decompo_values is", decompo_values)
                    reduced_decompo_keys = decompo_keys.composition.element_composition.reduced_composition.element_composition
                    # print ("reduced_decompo_keys is ",reduced_decompo_keys)
                    if not reduced_decompo_keys in reduce_composition_at_hull:
                        # print ("the structure decompose into arbitarily introduced structure, we will abandon it")
                        abandon = True
                        break

                    index_1 = reduce_composition_at_hull.index(reduced_decompo_keys)
                    # print ("index_1 is",index_1)
                    vertex_index_global = structure_index_at_hull[index_1]
                    new_vector = new_vector - decompo_values * corr_in[vertex_index_global]
                if abandon:
                    continue

                # G_3_new_line=np.concatenate((new_vector,np.zeros((self.N_corr))),axis=1)
                G_3_new_line = np.concatenate((new_vector, np.zeros(N_corr)))

                G_3_new_line.shape = (1, 2 * N_corr)
                G_3 = np.concatenate((G_3, G_3_new_line), axis=0)
                small_error = np.array(-1e-3)
                small_error.shape = (1, 1)
                h_3 = np.concatenate((h_3, small_error), axis=0)

        P_matrix = matrix(P)
        q_matrix = matrix(q)
        G_3_matrix = matrix(G_3)
        h_3_matrix = matrix(h_3)

        # formulation is min 1/2 x'Px+ q'x s.t.: Gx<=h, Ax=b

        sol = solvers.qp(P_matrix, q_matrix, G_3_matrix, h_3_matrix)

        # ecis = np.zeros(N_corr)  # check this
        return np.array(sol['x'])[:N_corr, 0]

    def get_mu_plot(self):
        ax = plt.subplot(111)
        ax.scatter(self.mus, self.cvs)
        ax.set_xscale('log')
        return plt

    def print_ecis(self):
        corr = np.zeros(self.ce.n_bit_orderings)
        corr[0] = 1  # zero point cluster
        cluster_std = np.std(self.feature_matrix, axis=0)
        for sc in self.ce.orbit:
            print(sc, len(sc.bits) - 1, sc.sc_b_id)
            print('bit    eci    cluster_std    eci*cluster_std')
            for i, bits in enumerate(sc.bit_combos):
                eci = self.ecis[sc.sc_b_id + i]
                c_std = cluster_std[sc.sc_b_id + i]
                print(bits, eci, c_std, eci * c_std)
        print(self.ecis)

    def structure_energy(self, structure):
        return self.ce.structure_energy(structure, self.ecis)

    @classmethod
    def from_dict(cls, d):
        return cls(cluster_expansion=ClusterExpansion.from_dict(d['cluster_expansion']),
                   structures=[Structure.from_dict(s) for s in d['structures']],
                   energies=np.array(d['energies']), max_dielectric=d.get('max_dielectric'),
                   max_ewald=d.get('max_ewald'), supercell_matrices=d['supercell_matrices'],
                   mu=d.get('mu'), ecis=d.get('ecis'), feature_matrix=d.get('feature_matrix'),
                   solver=d.get('solver', 'cvxopt_l1'), weights=d['weights'])

    def as_dict(self):
        return {'cluster_expansion': self.ce.as_dict(),
                'structures': [s.as_dict() for s in self.structures],
                'energies': self.energies.tolist(),
                'supercell_matrices': [cs.supercell_matrix.tolist() for cs in self.supercells],
                'max_dielectric': self.max_dielectric,
                'max_ewald': self.max_ewald,
                'mu': self.mu,
                'ecis': self.ecis.tolist(),
                'feature_matrix': self.feature_matrix.tolist(),
                'weights': self.weights.tolist(),
                'solver': self.solver,
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__}
