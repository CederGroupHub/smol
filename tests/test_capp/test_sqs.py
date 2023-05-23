import numpy as np
import pytest

from smol.capp.generate import StochasticSQSGenerator
from smol.utils.math import integerize_vector


@pytest.fixture(params=["correlation", "cluster-interaction"], scope="package")
def generator(structure, cluster_cutoffs, request):
    _, supercell_size = integerize_vector(
        [val for val in structure.composition.values()]
    )
    generator = StochasticSQSGenerator.from_structure(
        structure,
        cluster_cutoffs,
        supercell_size=supercell_size,
        feature_type=request.param,
    )
    return generator


def test_generate_get_sqs(generator):
    assert generator.num_structures == 0
    generator.generate(1000, max_save_num=100)
    assert 0 < generator.num_structures <= 100

    traces = list(generator._sqs_deque)
    assert all(
        tr1.enthalpy >= tr2.enthalpy for tr1, tr2 in zip(traces[:-1], traces[1:])
    )

    assert len(generator.get_best_sqs()) == 1
    assert (
        len(generator.get_best_sqs(generator.num_structures, remove_duplicates=False))
        == generator.num_structures
    )
    assert (
        len(generator.get_best_sqs(generator.num_structures))
        <= generator.num_structures
    )
    assert (
        len(generator.get_best_sqs(generator.num_structures, reduction_algorithm="LLL"))
        <= generator.num_structures
    )


def test_bad_generator(cluster_subspace):
    with pytest.raises(ValueError):
        StochasticSQSGenerator(cluster_subspace, 2, feature_type="blah")

    with pytest.raises(ValueError):
        StochasticSQSGenerator(
            cluster_subspace, 2, target_vector=np.ones(len(cluster_subspace) - 2)
        )

    with pytest.raises(ValueError):
        StochasticSQSGenerator(
            cluster_subspace, 2, target_weights=np.ones(len(cluster_subspace) + 2)
        )

    with pytest.raises(ValueError):
        StochasticSQSGenerator(
            cluster_subspace, 2, supercell_matrices=[np.ones((4, 3))]
        )

    with pytest.raises(ValueError):
        StochasticSQSGenerator(
            cluster_subspace, 2, supercell_matrices=[np.random.random((3, 3))]
        )
