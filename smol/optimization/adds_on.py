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

def rmse(ecis, A, f):
    e = np.dot(A, ecis)
    #     print(e)
    return np.average((e - f) ** 2) ** 0.5
