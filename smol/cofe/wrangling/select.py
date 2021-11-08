"""Tools for training structure selection."""

from warnings import warn
import random
import numpy as np
from scipy.linalg import lu
from scipy.stats import multivariate_normal, multinomial

__author__ = "Luis Barroso-Luque"


def full_row_rank_select(feature_matrix, tol=1E-15, nrows=None):
    """Choose a (maximally) full rank subset of rows in a feature matrix.

    This method is for underdetermined systems.
    columns/features > rows/structures

    Args:
        feature_matrix (ndarray):
            feature matrix to select rows/structure from.
        tol (float): optional
            tolerance to use to determine the pivots in upper triangular matrix
            of the LU decomposition.
        nrows (int): optional
            Number of rows to include. If None given, will include the maximum
            possible number of rows.
    Returns:
        list of int: list with indices of rows that form a full rank system.
    """
    nrows = nrows if nrows is not None else feature_matrix.shape[0]
    P, L, U = lu(feature_matrix.T)
    sample_ids = np.unique((abs(U) > tol).argmax(axis=1))[:nrows]
    if len(sample_ids) > np.linalg.matrix_rank(feature_matrix[:nrows]):
        warn("More samples than the matrix rank where selected.\n"
             "The selection will not actually be full rank.\n"
             "Decrease tolerance to ensure full rank indices are selected.\n")
    return list(sample_ids)


def gaussian_select(feature_matrix, num_samples):
    """V. Vanilla structure selection to increase incoherence.

    Sequentially pick samples with feature vector that most closely aligns
    with a sampled random gaussian vector on the unit sphere.

    This works much better when the number of rows in the feature matrix
    is much larger than the number of samples requested.

    Args:
        feature_matrix (ndarray):
            feature matrix to select rows/structure from.
        num_samples (int):
            number of samples/rows/structures to select.

    Returns:
        list of int: list with indices of rows that align with Gaussian samples
    """
    M = feature_matrix[:, 1:].copy()  # exclude first column (empty cluster)
    m, n = M.shape
    M /= np.linalg.norm(M, axis=1).reshape(m, 1)
    gauss = multivariate_normal(mean=np.zeros(n)).rvs(size=num_samples)
    gauss /= np.linalg.norm(gauss, axis=1).reshape(num_samples, 1)
    sample_ids, compliment = [], list(range(m))
    for vec in gauss:
        sample_ids.append(
            compliment[np.argmin(np.linalg.norm(M[compliment] - vec, axis=1))])
        compliment.remove(sample_ids[-1])
    return sample_ids


def composition_select(composition_vector, composition, cell_sizes,
                       num_samples):
    """Structure selection based on composition multinomial probability.

    Note: this function is needs quite a bit of tweaking to get nice samples.

    Args:
        composition_vector (ndarray):
            N (samples) by n (components) with the composition of the samples
            to select from.
        composition (ndarray):
            array for the center composition to sample around.
        cell_sizes (int or Sequence):
            size of unit cell sizes or size of supercells used to set the
            number of variables in the multinomial distribution.
        num_samples (int):
            number of samples to return. Note that if the number is too high
            compared to the total number of samples (or coverage of the space)
            it may take very long to return if at all, and the samples will not
            be very representative of the multinomial distributions.
    Returns:
        list of int: list with indices of the composition vector corresponding
                     to selected samples.
    """
    unique_compositions = np.unique(composition_vector, axis=1)
    # Sample structures biased by composition
    if isinstance(cell_sizes, int):
        cell_sizes = np.array(len(composition_vector) * [cell_sizes, ])
    else:
        cell_sizes = np.array(cell_sizes)

    dists = {size: multinomial(size, composition)
             for size in np.unique(cell_sizes)}
    max_probs = {size: dist.pmf(size * unique_compositions).max()
                 for size, dist in dists.items()}
    sample_ids = [
        i for i, (size, comp) in enumerate(zip(cell_sizes, composition_vector))
        if dists[size].pmf(size*comp) >= max_probs[size]*random.random()]
    while len(sample_ids) <= num_samples:
        sample_ids += [
            i for i, (size, comp) in enumerate(
                zip(cell_sizes, composition_vector))
            if dists[size].pmf(size*comp) >= max_probs[size]*random.random()
            and i not in sample_ids]
    return sample_ids[:num_samples]
