import os
import numpy as np
import matplotlib.pyplot as plt


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

def cluster_properties(subsapce):
    """
    return max radius and number of atoms in cluster

    input ce is cluster expansion class object

    subspace: a Cluster_subspace object defined in smol
    """
    cluster_n = [0]
    cluster_r = [0]
    for sc in subsapce.iterorbits():
        for j in range(len(sc.bit_combos)):
            cluster_n.append(sc.bit_combos[j].shape[1])
            cluster_r.append(sc.radius)
    return np.array(cluster_n), np.array(cluster_r)  # without ewald term


def rmse(ecis, A, f):
    """
    Calculating the RMSE of fitting in CE
    :param ecis:
    :param A:
    :param f:
    :return:
    """
    e = np.dot(A, ecis)
    #     print(e)
    return np.average((e - f) ** 2) ** 0.5

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("Folder exists")

