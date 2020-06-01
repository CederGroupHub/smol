"""
Ground State Preserving solver from Wenxhuan Huang (implemented by Daniil?)
"""

import logging
import numpy as np
from pymatgen import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from cvxopt import matrix, solvers
from .base import BaseEstimator

# TODO gspreserve solve need to fix calls to e_above_hull was in the
#  this is really broken


class GSPreserveEstimator(BaseEstimator):
    """
    Estimator implementing WH's gs preserve fit.
    """

    def __init__(self):
        super().__init__()

    def _solve(self, A, f, mu):
        solvers.options['show_progress'] = False
        ehull = list(self.e_above_hull_input)
        struct_index_at_hull = [i for (i, e) in enumerate(ehull) if e < 1e-5]
        # struct_index_at_hull = list(
        # np.argwhere(self._e_above_hull_input < 1e-5)[0])

        logging.info("removing duplicated correlation entries")
        corr_in = np.array(A)
        duplicated_corr_set = []
        for i in range(len(corr_in)):
            if i not in struct_index_at_hull:
                for j in struct_index_at_hull:
                    if np.max(np.abs(corr_in[i] - corr_in[j])) < 1e-6:
                        logging.info("structure ", i,
                                     " has the same correlation as hull"
                                     "structure ", j)
                        duplicated_corr_set.append(i)

        r_comp_at_hull = [self.structures[i].composition.element_composition.
                          reduced_composition.element_composition
                          for i in struct_index_at_hull]  # Wenxuan added

        print(r_comp_at_hull)

        corr_in = np.array(self.feature_matrix)  # verify this line

        engr_in = np.array(self.normalized_energies)
        engr_in.shape = (len(engr_in), 1)

        weights_tmp = self.weights.copy()

        for i in duplicated_corr_set:
            weights_tmp[i] = 0

        weight_vec = np.array(weights_tmp)  # verify this line
        # print ("weight_vec is ", weight_vec, "weight_vec.shape is is ",
        #        weight_vec.shape)

        weight_mat = np.diag(weight_vec.transpose())

        N_corr = corr_in.shape[1]  # verify this line

        P_corr_part = 2*((weight_mat.dot(corr_in)).transpose()).dot(corr_in)

        P = np.lib.pad(P_corr_part, ((0, N_corr), (0, N_corr)),
                       mode='constant', constant_values=0)

        q_corr_part = -2*((weight_mat.dot(corr_in)).transpose()).dot(engr_in)
        # CHANGED BY WILL from * mu (to be consistent with cvxopt_l1)
        q_z_part = np.ones((N_corr, 1)) / mu

        q = np.concatenate((q_corr_part, q_z_part), axis=0)

        G_1 = np.concatenate((np.identity(N_corr), -np.identity(N_corr)),
                             axis=1)
        G_2 = np.concatenate((-np.identity(N_corr), -np.identity(N_corr)),
                             axis=1)
        G_3 = np.concatenate((G_1, G_2),
                             axis=0)
        h_3 = np.zeros((2 * N_corr, 1))

        for i in range(len(corr_in)):
            if i not in struct_index_at_hull and i not in duplicated_corr_set:

                reduced_comp = self.structures[i].composition.\
                    element_composition.reduced_composition.element_composition

                if reduced_comp in r_comp_at_hull:  # in hull composition

                    hull_idx = r_comp_at_hull.index(reduced_comp)
                    global_index = struct_index_at_hull[hull_idx]

                    # G_3_new_line=np.concatenate((corr_in[global_index]
                    # -corr_in[i],np.zeros((self.N_corr))),axis=1)
                    G_3_new_line = np.concatenate((corr_in[global_index] -
                                                   corr_in[i],
                                                   np.zeros((N_corr))))

                    G_3_new_line.shape = (1, 2 * N_corr)
                    G_3 = np.concatenate((G_3, G_3_new_line), axis=0)
                    small_error = np.array(-1e-3)
                    small_error.shape = (1, 1)
                    h_3 = np.concatenate((h_3, small_error), axis=0)

                else:  # out of hull composition

                    ele_comp_now = self.structures[i].composition.\
                        element_composition.reduced_composition.\
                        element_composition
                    decomp_now = self.pd_input.get_decomposition(ele_comp_now)
                    new_vector = -corr_in[i]
                    for decomp_keys, decomp_vals in decomp_now.iteritems():
                        reduced_decompo_keys = decomp_keys.composition.\
                            element_composition.reduced_composition.\
                            element_composition
                        index_1 = r_comp_at_hull.index(reduced_decompo_keys)
                        vertex_index_global = struct_index_at_hull[index_1]
                        new_vector += decomp_vals*corr_in[vertex_index_global]

                    G_3_new_line = np.concatenate((new_vector,
                                                   np.zeros(N_corr)))
                    # G_3_new_line = np.concatenate((new_vector,
                    #                       np.zeros((self.N_corr))),axis=1)

                    G_3_new_line.shape = (1, 2 * N_corr)
                    G_3 = np.concatenate((G_3, G_3_new_line), axis=0)

                    small_error = np.array(-1e-3)
                    small_error.shape = (1, 1)
                    h_3 = np.concatenate((h_3, small_error), axis=0)

            elif i in struct_index_at_hull:
                if self.structures[i].composition.\
                        element_composition.is_element:
                    # print ("structure i=",i,"is an element")
                    continue

                global_index = i
                hull_idx = struct_index_at_hull.index(i)
                ele_comp_now = self.structures[i].composition.\
                    element_composition.reduced_composition.element_composition
                entries_new = []
                for j in struct_index_at_hull:
                    if not j == i:
                        entry = PDEntry(self.structures[j].composition.
                                        element_composition,
                                        self.properties[j])
                        entries_new.append(entry)

                for el in self.ce.structure.composition.keys():
                    entry = PDEntry(Composition({el: 1}).element_composition,
                                    max(self.normalized_energies) + 1000)
                    entries_new.append(entry)

                pd_new = PhaseDiagram(entries_new)

                # vertices_list_new_pd = list(
                #  set(chain.from_iterable(pd_new.facets)))

                # vertices_red_composition_list_new_pd = [
                #    pd_new.qhull_entries[i].composition.element_composition.
                #        reduced_composition.element_composition
                #    for i in vertices_list_new_pd]

                decomp_now = pd_new.get_decomposition(ele_comp_now)

                new_vector = corr_in[i]

                abandon = False
                for decomp_keys, decomp_vals in decomp_now.iteritems():

                    reduced_decompo_keys = decomp_keys.composition.\
                        element_composition.reduced_composition.\
                        element_composition

                    if reduced_decompo_keys not in r_comp_at_hull:
                        abandon = True
                        break

                    index_1 = r_comp_at_hull.index(reduced_decompo_keys)
                    vertex_index_global = struct_index_at_hull[index_1]
                    new_vector -= decomp_vals*corr_in[vertex_index_global]
                if abandon:
                    continue

                # G_3_new_line=np.concatenate((new_vector,
                # np.zeros((self.N_corr))),axis=1)
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
