import numpy as np
import numpy.testing as npt

from smol.cofe.wrangling.select import (
    composition_select,
    full_row_rank_select,
    gaussian_select,
)


def test_full_row_rank_select(rng):
    for shape in ((100, 100), (100, 200), (100, 500)):
        matrix = 10 * rng.random(shape)
        # test for set sizes
        for n in range(5, shape[0] // 2):
            inds = full_row_rank_select(feature_matrix=matrix, nrows=n)
            assert max(inds) <= shape[0]
            assert len(inds) == n

        # test for max possible samples
        inds = full_row_rank_select(feature_matrix=matrix)
        assert max(inds) <= shape[0]
        # add 1 for numerical slack
        assert len(inds) + 1 >= np.linalg.matrix_rank(matrix)


def test_gaussian_select(rng):
    # not an amazing test, but something is something
    matrix = 10 * rng.random((3000, 1000)) - 2
    inds = gaussian_select(matrix, 300)
    std = matrix[inds].std(axis=0)
    assert len(np.unique(inds)) == 300
    assert np.all(abs(matrix[inds].mean(axis=0)) <= 1.5 * std)
    assert matrix[inds].mean() <= matrix.mean()


def test_composition_select(rng):
    n = 5000
    cell_sizes = rng.choice((5, 10, 15, 20), size=n)
    species = [
        np.array([rng.integers(0, 3) for _ in range(size)]) for size in cell_sizes
    ]
    concentrations = np.array(
        [
            [sum(s == 0) / len(s), sum(s == 1) / len(s), sum(s == 2) / len(s)]
            for s in species
        ]
    )
    inds = composition_select(concentrations, [0.2, 0.2, 0.6], cell_sizes, 200)

    # initial concentrations should be .3, .3, .3
    npt.assert_array_almost_equal(
        concentrations.mean(axis=0), [1 / 3.0, 1 / 3.0, 1 / 3.0], decimal=2
    )
    sample_avg = concentrations[inds].mean(axis=0)
    # very loose criteria here
    assert sample_avg[0] < 1 / 3
    assert sample_avg[1] < 1 / 3
    assert sample_avg[2] > 1 / 3
