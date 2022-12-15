"""General tools to be used with cluster expansion and related models."""

__author__ = "Luis Barroso-Luque"

import numpy as np

from smol._math import yield_hermite_normal_forms


def gen_supercell_matrices(sizes, symmops, tol=1e-5):
    """Generate all symmetrically distinct supercell matrices of given sizes.

    Matrices are given in Hermite normal form following the following work:

    * https://link.aps.org/doi/10.1103/PhysRevB.77.224115
    * https://link.aps.org/doi/10.1103/PhysRevB.80.014120

    Args:
        sizes (Sequence of int):
            sizes of the supercell in multiples of the primitive cell
        symmops (list of SymmOps):
            symmetry operations
        tol (float): optional
            tolerance for checking if a matrix is symmetrically distinct
    Returns:
        list of ndarray: list of supercell matrices with given size
    """
    supercell_matrices = []
    for size in sizes:
        for hnf in yield_hermite_normal_forms(size):
            for symop in symmops:
                hnf_rot = np.linalg.inv(hnf) @ symop.rotation_matrix
                for scm in supercell_matrices:
                    unimod = hnf_rot @ scm
                    if (abs(unimod - np.round(unimod)) < tol).all():
                        break
                else:
                    continue
                break
            else:
                supercell_matrices.append(hnf)

    return supercell_matrices
