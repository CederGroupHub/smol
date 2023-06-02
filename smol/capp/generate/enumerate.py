"""Tools for exhaustive enumeration of symmetrically distinct supercell matrices."""

__author__ = "Luis Barroso-Luque"


import numpy as np

from smol.utils.math import yield_hermite_normal_forms


def enumerate_supercell_matrices(size, symmops, tol=1e-5):
    """Generate all symmetrically distinct supercell matrices of a given size.

    Matrices are given in Hermite normal form following the following work:

    * https://link.aps.org/doi/10.1103/PhysRevB.77.224115
    * https://link.aps.org/doi/10.1103/PhysRevB.80.014120

    Args:
        size (int):
            size of the supercell in multiples of the primitive cell
        symmops (list of SymmOps):
            symmetry operations
        tol (float): optional
            tolerance for checking if a matrix is symmetrically distinct
    Returns:
        list of ndarray: list of supercell matrices with given size
    """
    supercell_matrices = []
    for hnf in yield_hermite_normal_forms(size):
        for symop in symmops:
            hnf_rot = np.linalg.inv(hnf) @ symop.rotation_matrix
            for scm in supercell_matrices:
                unimod = hnf_rot @ scm.T
                if (abs(unimod - np.round(unimod)) < tol).all():
                    break
            else:
                continue
            break
        else:
            supercell_matrices.append(hnf.T)  # supercells in pmg are transpose of hnf

    return supercell_matrices
