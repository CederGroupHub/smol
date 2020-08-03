import warnings
import numpy as np
import pickle
import matplotlib.pyplot as plt
from cvxopt import matrix
from gurobipy import *
from l1regls import l1regls, solvers
import itertools

"""
In this optimization method file, some terminology shall be defined here:

A: refers to the feature matrix with size m*n (m: number of structures, n: number of features)
f: refers to the scalar property (e.g. DFT normalized energies), in size of m*1


"""

def rmse(ecis, A, f):
    e = np.dot(A, ecis)
    #     print(e)
    return np.average((e - f) ** 2) ** 0.5


def l1_optimize(A, f, mu, weights=None):
    """
    Definition of mu in l1_regularized solver is different from normal expression

    Normal way: |y-y_pre|^2+ mu|ecis
    Convex solver way: mu|Ax-b|+|x|

    so take the inverse of mu in region 1e-9 to 1e1 to accomodate

    """
    mu = 1 / mu

    if weights is None:
        weights = np.ones(len(f))

    A_w = A * weights[:, None] ** 0.5
    f_w = f * weights ** 0.5

    solvers.options['show_progress'] = False
    A1 = matrix(A)
    b = matrix(f * mu)
    ecis = (np.array(l1regls(A1, b)) / mu).flatten()
    return ecis


def l1_optimize_ecis(A, f, mu, use_Ewald = False):

    ecis = l1_optimize(A=A, f = f, mu = mu)
    dielectrict = 1/ecis[-1]

    if use_Ewald:

        if (dielectrict > 100) | (dielectrict < 0):
            A_new = A[:,:-1]
            f_new = f - A[:,-1]*0.01
            ecis_new = l1_optimize(A = A_new, f= f_new, mu = mu).tolist()
            ecis_new.append(0.01)
            ecis = np.array(ecis_new)
        return np.array(ecis)
    else:
        return np.array(ecis)



def ridge_optimize(A, f, mu):
    m, d = A.shape
    inv = np.linalg.pinv(np.dot(np.transpose(A), A) + mu * np.eye(d))
    ecis = np.dot(np.dot(inv, np.transpose(A)), f)
    return ecis


def Bayes_optimize(A, f, cov):
    m, d = A.shape
    inv = np.linalg.pinv(np.dot(np.transpose(A), A) + cov)
    ecis = np.dot(np.dot(inv, np.transpose(A)), f)
    return ecis


def l0l1_optimize(A, f, mu0, mu1, M=100.0, cutoff=300, if_Ewald = True):
    """
    Brute force solving by gurobi. Cutoff in 300s.
    """
    n = A.shape[0]
    d = A.shape[1]
    ATA = A.T @ A
    fTA = f.T @ A

    l1l0 = Model()
    w = l1l0.addVars(d, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    z0 = l1l0.addVars(d, vtype=GRB.BINARY)
    z1 = l1l0.addVars(d)
    for i in range(d):
        l1l0.addConstr(M * z0[i] >= w[i])
        l1l0.addConstr(M * z0[i] >= (-1.0 * w[i]))
        l1l0.addConstr(z1[i] >= w[i])
        l1l0.addConstr(z1[i] >= (-1.0 * w[i]))
    # if if_Ewald:
    #     l1l0.addConstr(w[d-1]>= 0.01)
    # Cost function
    L = QuadExpr()
    for i in range(d):
        L = L + mu0 * z0[i]
        L = L + mu1 * z1[i]
        L = L - 2 * fTA[i] * w[i]
        for j in range(d):
            L = L + w[i] * w[j] * ATA[i][j]

    l1l0.setObjective(L, GRB.MINIMIZE)
    l1l0.setParam(GRB.Param.TimeLimit, cutoff)
    l1l0.setParam(GRB.Param.PSDTol, 1e-5)  # Set a larger PSD tolerance to ensure success
    l1l0.setParam(GRB.Param.OutputFlag, 0)
    # Using the default algorithm, and shut gurobi up.
    l1l0.update()
    l1l0.optimize()
    w_opt = np.array([w[v_id].x for v_id in w])
    return w_opt

def l0l1_optimize_ecis(A, f, mu1, mu0, M = 100.0, use_Ewald = False):

    ecis = l0l1_optimize(A = A, f = f, M = M, mu1= mu1, mu0= mu0)

    if use_Ewald:
        dielectrict = 1/ecis[-1]
    #     print(dielectrict)
        if (dielectrict > 100) | (dielectrict < 0):
            A_new = A[:,:-1]
            f_new = f - A[:,-1]*0.01
            ecis_new = l0l1_optimize(A = A_new, f = f_new, mu1= mu1, mu0= mu0, M = M).tolist()
            ecis_new.append(0.01)
            ecis = np.array(ecis_new)
        return np.array(ecis)
    else:
        return np.array(ecis)






def l0l1_optimize_fixing(A, f, mu0, mu1, indices, epsilon, M=100.0, cutoff=300):
    """
    indices indicate training points with constrain.
    Brute force solving by gurobi. Cutoff in 300s.
    """
    n = A.shape[0]
    d = A.shape[1]
    ATA = A.T @ A
    fTA = f.T @ A

    l1l0 = Model()
    w = l1l0.addVars(d, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    z0 = l1l0.addVars(d, vtype=GRB.BINARY)
    z1 = l1l0.addVars(d)
    for i, idx in enumerate(indices):
        constrain = 0
        for j in range(d):
            constrain += A[idx, j] * w[j]
        # print(constrain)
        l1l0.addConstr((constrain - f[idx]) <= epsilon)
        l1l0.addConstr((constrain - f[idx]) >= -epsilon)

    for i in range(d):
        l1l0.addConstr(M * z0[i] >= w[i])
        l1l0.addConstr(M * z0[i] >= (-1.0 * w[i]))
        l1l0.addConstr(z1[i] >= w[i])
        l1l0.addConstr(z1[i] >= (-1.0 * w[i]))
    # Cost function
    L = QuadExpr()
    for i in range(d):
        L = L + mu0 * z0[i]
        L = L + mu1 * z1[i]
        L = L - 2 * fTA[i] * w[i]
        for j in range(d):
            L = L + w[i] * w[j] * ATA[i][j]

    l1l0.setObjective(L, GRB.MINIMIZE)
    l1l0.setParam(GRB.Param.TimeLimit, cutoff)
    l1l0.setParam(GRB.Param.PSDTol, 1e-5)  # Set a larger PSD tolerance to ensure success
    l1l0.setParam(GRB.Param.OutputFlag, 0)
    # Using the default algorithm, and shut gurobi up.
    l1l0.update()
    l1l0.optimize()
    w_opt = np.array([w[v_id].x for v_id in w])
    return w_opt


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.04, **cbar_kw)
    #     cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", )

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    #     # Rotate the tick labels and set their alignment.
    #     plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
    #              rotation_mode="anchor")
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, np.round(data[i, j], decimals=2),
                           ha="center", va="center", color="w", fontsize=20)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


class l1_solver(object):
    """docstring for l1_solver."""

    def __init__(self, A, f, mu=None, use_Ewald = False):
        super(l1_solver, self).__init__()
        self.A = A
        self.f = f
        self.ecis = np.zeros(A.shape[0])
        self.use_Ewald = use_Ewald
        if mu != None:
            self.mu = mu

    def calc_cv_score_l1(self, mu, k=5):
        """
            Args:
                mu: weight of error in bregman
                A: sensing matrix (scaled appropriately)
                f: data to fit (scaled appropriately)
                k: number of partitions

            Partition the sample into k partitions, calculate out-of-sample
            variance for each of these partitions, and add them together
            """
        # logging.info('starting cv score calculations for mu: {}, k: {}'.format(mu, k))
        # generate random partitions
        A = self.A
        f = self.f

        partitions = np.tile(np.arange(k), len(f) // k + 1)
        np.random.shuffle(partitions)
        partitions = partitions[:len(f)]
        ssr = 0

        for i in range(k):
            ins = (partitions != i)  # in the sample for this iteration
            oos = (partitions == i)  # out of the sample for this iteration

            ecis = l1_optimize_ecis(A=A[ins], f=f[ins], mu=mu, use_Ewald=self.use_Ewald)

            res = (np.dot(A[oos], ecis) - f[oos]) ** 2
            ssr += np.average(res)

        cv = ssr / k
        return cv

    def get_optimal_mu(self, k=5, min_order=-10, max_order=1):
        """
        calculate the optimal mu from l1-regularized least square fitting

        regularization is uniform with mu * eye(n)

        """
        mus = list(np.logspace(min_order, max_order, int(max_order - min_order) + 1))
        A = self.A
        f = self.f
        print(mus)
        cvs = [self.calc_cv_score_l1(mu=mu, k=k) for mu in mus]

        for _ in range(2):
            i = np.nanargmax(cvs)
            if i == len(mus) - 1:
                # warnings.warn('Largest mu chosen. You should probably increase the basis set')
                break

            mu = (mus[i] * mus[i + 1]) ** 0.5
            mus[i + 1:i + 1] = [mu]
            cvs[i + 1:i + 1] = [self.calc_cv_score_l1(mu=mu, k=k)]

            mu = (mus[i - 1] * mus[i]) ** 0.5
            mus[i:i] = [mu]
            cvs[i:i] = [self.calc_cv_score_l1(mu=mu, k=k)]

        self.mu = mus[np.nanargmin(cvs)]
        return self.mu

    def optimize(self, mu=None):
        self.get_optimal_mu()
        print(self.mu)
        self.ecis = l1_optimize(A=self.A, f=self.f, mu=self.mu)
        return self.ecis


class l0l1_solver(object):
    """docstring for l0l1_solver."""

    def __init__(self, A, f, mu0=None, mu1=None, M = 100.0, use_Ewald = False):
        super(l0l1_solver, self).__init__()
        self.A = A
        self.f = f
        self.ecis = np.zeros(A.shape[0])
        self.cv_grid = None
        self.use_Ewald = use_Ewald
        self.M = M
        if (mu0 != None) and (mu1 != None):
            self.mu0 = mu0
            self.mu1 = mu1

    def calc_cv_score_l0l1(self, mu0, mu1, k=5):
        """
        Args:
            mu: weight of error in bregman
            A: sensing matrix (scaled appropriately)
            f: data to fit (scaled appropriately)
            k: number of partitions

        Partition the sample into k partitions, calculate out-of-sample
        variance for each of these partitions, and add them together
        """
        # logging.info('starting cv score calculations for mu: {}, k: {}'.format(mu, k))
        # generate random partitions
        A = self.A
        f = self.f
        partitions = np.tile(np.arange(k), len(f) // k + 1)
        np.random.shuffle(partitions)
        partitions = partitions[:len(f)]
        ssr = 0

        for i in range(k):
            ins = (partitions != i)  # in the sample for this iteration
            oos = (partitions == i)  # out of the sample for this iteration

            ecis = l0l1_optimize_ecis(A=A[ins,:], f=f[ins], mu0=mu0, mu1=mu1, M = self.M, use_Ewald=self.use_Ewald)
            #             print(A[oos])
            #             print(ecis)

            res = (np.dot(A[oos], ecis) - f[oos]) ** 2
            #             print(res)
            ssr += np.average(res)

        cv = ssr / k
        return cv

    def grid_optimal_mu_l0l1(self, k=5, min_order=-9, max_order=1):
        """
        generate a cv score grid with mu0 and mu1

        regularization is uniform with mu * eye(n)

        """

        n = int(max_order - min_order) + 1
        mu0s = np.logspace(min_order, max_order, n)
        mu1s = np.logspace(min_order, max_order, n)
        # print(mus)
        #     cvs = [calc_cv_score_l1(mu, A, f, k) for mu in mus]

        cv_grid = np.zeros([n, n])

        for ii, mu0 in enumerate(mu0s):
            for jj, mu1 in enumerate(mu1s):
                cv_tot = []
                for trial in range(20):
                    cv = self.calc_cv_score_l0l1(mu0=mu0, mu1=mu1, k=5)
                    cv_tot.append(cv)

                cv_grid[ii, jj] = np.average(cv_tot)

        i, j = np.unravel_index(cv_grid.argmin(), cv_grid.shape)
        self.cv_grid = cv_grid
        return cv_grid, mu0s[i], mu1s[j]

    def plot_cv_grid(self, k=5, min_order=-9, max_order=1, filename = None):
        """

        :param k: k-fold
        :param min_order: minimum order in logrithm search
        :param max_order: maximum order in logrithm search
        :param filename: filename to be saved as a pdf file.  should be './xxx.pdf'
        :return:
        """
        if self.cv_grid is None:
            cv_grid, _, _ = self.grid_optimal_mu_l0l1(k=5, min_order=min_order, max_order=max_order)
        else:
            cv_grid =self.cv_grid
        mu0s = np.logspace(min_order, max_order, int(max_order - min_order) + 1)
        mu1s = np.logspace(min_order, max_order, int(max_order - min_order) + 1)

        fig, ax = plt.subplots(figsize=(20, 20))

        im, cbar = heatmap(np.sqrt(cv_grid) * 500, mu0s, mu1s, ax=ax,
                           cmap="bwr", )
        # cbar.remove()
        # texts = annotate_heatmap(im, valfmt="{x:.1f} t")
        # cbar.ax.tick_params(labelsize=25)
        cbar.set_clim(4, 10)
        cbar.remove()
        ax.set_xlabel("mu0")
        ax.set_ylabel("mu1")
        ax.xaxis.set_tick_params(labelsize=25)
        ax.yaxis.set_tick_params(labelsize=25)

        fig.tight_layout()
        if filename == None:
            plt.savefig('./l0l1_cvgrid.pdf')
        else:
            plt.savefig(filename)
        plt.show()

    def optimize(self, mu0, mu1):
        self.mu0 = mu0
        self.mu1 = mu1
        self.ecis = l0l1_optimize(A=self.A, f=self.f, mu0=self.mu0, mu1=self.mu1, M=self.M)
        return self.ecis


class ridge_solver(object):
    def __init__(self, A, f):
        self.A = A
        self.f = f

    def calc_cv_score_ridge(self, mu, k=5):
        """
            Args:
                mu: regularization term for Bayes prior distribution
                A: sensing matrix (scaled appropriately)
                f: data to fit (scaled appropriately)
                k: number of partitions

            Partition the sample into k partitions, calculate out-of-sample
            variance for each of these partitions, and add them together
        """
        A = self.A
        f = self.f
        partitions = np.tile(np.arange(k), len(f) // k + 1)
        np.random.shuffle(partitions)
        partitions = partitions[:len(f)]
        ssr = 0

        for i in range(k):
            ins = (partitions != i)  # in the sample for this iteration
            oos = (partitions == i)  # out of the sample for this iteration

            ecis = ridge_optimize(A=A[ins], f=f[ins], mu=mu)

            res = (np.dot(A[oos], ecis) - f[oos]) ** 2
            ssr += np.average(res)

        cv = ssr / k
        return cv


    def calculate_gcv_score_ridge(self, mu):
        A= self.A
        f = self.f
        m, d = A.shape
        ecis = ridge_optimize(A=A, f=f, mu=mu)
        rss = np.average((np.dot(A, ecis) - f) ** 2)
        inv = np.linalg.pinv(np.dot(np.transpose(A), A) + mu * np.eye(d))

        domi = np.trace(np.eye(m) - np.dot(np.dot(A, inv), np.transpose(A)))
        GCV = np.sqrt(m * rss) / domi
        return GCV









def get_optimal_mu_l2(A, f, weights, k=5, min_mu=-10, max_mu=0.1):
    """
    calculate the optimal mu from l2-regularized least square fitting

    regularization is uniform with mu * eye(n)

    """
    mus = list(np.logspace(min_mu, max_mu, 20))
    print(mus)
    cvs = [calc_cv_score_l2(mu, A, f, weights, k) for mu in mus]

    for _ in range(2):
        i = np.nanargmax(cvs)
        if i == len(mus) - 1:
            warnings.warn('Largest mu chosen. You should probably increase the basis set')
            break

        mu = (mus[i] * mus[i + 1]) ** 0.5
        mus[i + 1:i + 1] = [mu]
        cvs[i + 1:i + 1] = [calc_cv_score_l2(mu, A, f, weights, k)]

        mu = (mus[i - 1] * mus[i]) ** 0.5
        mus[i:i] = [mu]
        cvs[i:i] = [calc_cv_score_l2(mu, A, f, weights, k)]

    return mus[np.nanargmax(cvs)]



def get_optimal_gmu_l2(A, f, min_mu=-9, max_mu=0):
    """
    calculate the optimal generalized mu from l2-regularized least square fitting

    regularization is uniform with mu * eye(n)

    """
    mus = list(np.logspace(min_mu, max_mu, 10))
    print(mus)
    cvs = [gcv_score_l2(A=A, f=f, mu=mu) for mu in mus]

    for _ in range(2):
        i = np.nanargmax(cvs)
        if i == len(mus) - 1:
            warnings.warn('Largest mu chosen. You should probably increase the basis set')
            break

        mu = (mus[i] * mus[i + 1]) ** 0.5
        mus[i + 1:i + 1] = [mu]
        cvs[i + 1:i + 1] = [gcv_score_l2(A=A, f=f, mu=mu)]

        mu = (mus[i - 1] * mus[i]) ** 0.5
        mus[i:i] = [mu]
        cvs[i:i] = [gcv_score_l2(A=A, f=f, mu=mu)]

    return mus[np.nanargmin(cvs)], cvs


def cluster_properties(ce):
    """
    return max radius and number of atoms in cluster

    input ce is cluster expansion class object
    """
    cluster_n = [0]
    cluster_r = [0]
    for sc in ce.symmetrized_clusters:
        for j in range(len(sc.bit_combos)):
            cluster_n.append(sc.bit_combos[j].shape[1])
            cluster_r.append(sc.max_radius)
    return np.array(cluster_n), np.array(cluster_r)  # without ewald term


def gcv_score_Bayes(A, f, cov):
    """

    """
    m, d = A.shape
    ecis = ridge_optimize(A=A, f=f, cov=cov)
    rss = np.average((np.dot(A, ecis) - f) ** 2)
    inv = np.linalg.pinv(np.dot(np.transpose(A), A) + cov)

    domi = np.trace(np.eye(m) - np.dot(np.dot(A, inv), np.transpose(A)))
    GCV = np.sqrt(m * rss) / domi
    return GCV


def get_optimal_gamma(ce, A, f):
    cluster_n, cluster_r = cluster_properties(ce=ce)
    gamma0 = np.logspace(-9, 0, 10)
    gammas = [np.append(gamma0, 0), np.logspace(-4, 0, 5), np.logspace(-4, 0, 5),
              np.linspace(0, 10, 5), np.linspace(0, 10, 5)]
    gammas = list(itertools.product(*gammas))
    # test_list = np.array(test_list)
    gcvs = np.zeros(len(gammas))

    for i in range(len(gammas)):
        gamma = gammas[i]
        #         print(gamma[3])
        #         print(cluster_n*gamma[3])
        regu = gamma[0] * (cluster_r * gamma[1] + gamma[2] + 1) ** (cluster_n * gamma[3] + gamma[4])

        regu = np.append(regu, gamma[0])
        #         ecis_i = l2_Bayessian(A=A, f=f, cov = np.diag(regu))

        gcv = gcv_score_Bayes(A=A, f=f, cov=np.diag(regu))
        gcvs[i] = gcv


    print(np.min(gcvs))
    opt_gamma = gammas[np.nanargmin(gcvs)]

    regu = opt_gamma[0] * (cluster_r * opt_gamma[1] + opt_gamma[2] + 1) ** (cluster_n * opt_gamma[3] + opt_gamma[4])
    regu = np.append(regu, opt_gamma[0])

    return opt_gamma, regu


def regu_matrix(cluster_n, cluster_r, opt_gamma):
    regu = opt_gamma[0] * (cluster_r * opt_gamma[1] + opt_gamma[2] + 1) ** (cluster_n * opt_gamma[3] + opt_gamma[4])
    regu = np.append(regu, opt_gamma[0])
    return np.diag(regu)


if __name__ == '__main__':
    with open('./LMCT_fullCE/ecis_generated/feature_matrix', 'rb') as fp: feature_matrix = pickle.load(fp)
    with open('./LMCT_fullCE/ecis_generated/normalized_energies', 'rb') as fp: normalized_energies = pickle.load(fp)

    # l1_solver = l1_solver(A = feature_matrix, f= normalized_energies)
    # # l1_solver.get_optimal_mu()
    # ecis = l1_solver.optimize()

    # l0l1_solver = l0l1_solver(A=feature_matrix, f=normalized_energies)
    # l0l1_solver.plot_cv_grid()
    # print(ecis)
    error = 1/1000
    w_opt = l0l1_optimize_fixing(A=feature_matrix, f=normalized_energies, indices=range(900,920), mu0=1e-3, mu1=1e-4, epsilon=error)

    print(w_opt)
    print(rmse(w_opt, feature_matrix, normalized_energies)*500)
    print(np.dot(w_opt, feature_matrix[901]) - normalized_energies[901], error)

