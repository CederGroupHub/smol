import numpy as np
import pytest

from smol.capp.generate import generate_random_ordered_occupancy
from smol.cofe import ClusterSubspace
from smol.moca import ClusterExpansionProcessor


@pytest.mark.parametrize(
    "composition, charge_neutral", [(True, False), (None, True), (None, False)]
)
def test_generate_random_ordered_occupancy(
    single_structure, composition, charge_neutral
):
    subspace = ClusterSubspace.from_cutoffs(
        single_structure, cutoffs={}, supercell_size="volume"
    )
    scm = np.eye(3)
    scm[0, 0] = 5
    scm[1, 1] = 2
    processor = ClusterExpansionProcessor(subspace, scm, np.ones(len(subspace)))

    if composition is True:
        composition = processor.cluster_subspace.structure.composition

    for _ in range(5):
        occu = generate_random_ordered_occupancy(
            processor, composition=composition, charge_neutral=charge_neutral
        )
        structure = processor.structure_from_occupancy(occu)
        assert structure.is_ordered

        if composition is not None:
            assert (
                structure.composition.fractional_composition
                == composition.fractional_composition
            )

        if charge_neutral is True:
            assert structure.charge == 0
