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
            determinant of hermite normal forms to be yielded

    Yields:
        ndarray: hermite normal form matrix with given determinant
    """
    for a in filter(lambda x: determinant % x == 0, range(1, determinant + 1)):
        quotient = determinant // a
        for c in filter(lambda x: quotient % x == 0, range(1, determinant // a + 1)):
            f = quotient // c
            for b, d, e in product(range(0, c), range(0, f), range(0, f)):
                yield np.array([[a, 0, 0], [b, c, 0], [d, e, f]], dtype=int)
