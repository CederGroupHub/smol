"""General helper math utilities.

To be used internally in other modules.
"""

__author__ = "Luis Barroso-Luque"


from itertools import product

import numpy as np


def yield_hermite_normal_forms(determinant):
    """Yield all hermite normal form matrices with given determinant.

    Args:
        determinant (int):
            determinant to find the hermite normal form of

    Returns:
        list of ndarray: list of hermite normal forms with given determinant
    """
    for a in filter(lambda x: determinant % x == 0, range(1, determinant + 1)):
        quotient = determinant // a
        for c in filter(lambda x: quotient % x == 0, range(1, determinant // a + 1)):
            f = quotient // c
            for b, d, e in product(range(0, c), range(0, f), range(0, f)):
                yield np.array([[a, 0, 0], [b, c, 0], [d, e, f]], dtype=int)


def yield_supercell_matrices(size, symmops):
    """Yield all symmetrically distinct supercell matrices with given size.

    Matrices are given in Hermite normal form following the following work:

    * https://link.aps.org/doi/10.1103/PhysRevB.77.224115
    * https://link.aps.org/doi/10.1103/PhysRevB.80.014120

    Args:
        size (int):
            size of the supercell
        symmops (list of SymmOps):
            symmetry operations
    Returns:
        list of ndarray: list of supercell matrices with given size
    """
    for hnf in yield_hermite_normal_forms(size):
        for symop in symmops:
            yield symop.apply(hnf)
