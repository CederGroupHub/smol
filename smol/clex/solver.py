#TODO Here go all the solver functions ok. Build them up out of fragments of the old eci_fit file
import numpy as np

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


def unweighted(cls, cluster_expansion, structures, energies, mu=None, max_dielectric=None,
               max_ewald=None, solver='cvxopt_l1'):
    weights = np.ones(len(structures))
    return cls(cluster_expansion=cluster_expansion, structures=structures, energies=energies,
               mu=mu, weights=weights, max_dielectric=max_dielectric, max_ewald=max_ewald, solver=solver)


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

#def _solve_bregman(self, A, f, mu):
 #   return split_bregman(A, f, MaxIt=1e5, tol=1e-7, mu=mu, l=1, quiet=True)

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
