import warnings
import numpy as np
import itertools

"""
In this optimization method file, some terminology shall be defined here:

A: refers to the feature matrix with size m*n (m: number of structures, n: number of features)
f: refers to the scalar property (e.g. DFT normalized energies), in size of m*1


"""



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
