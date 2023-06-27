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
    scm[1, 1] = 4
    processor = ClusterExpansionProcessor(subspace, scm, np.ones(len(subspace)))
    print(processor.num_sites)

    if composition is True:
        composition = [sl.composition for sl in processor.get_sublattices()]
        print([sl.sites.shape for sl in processor.get_sublattices()])
        print(composition)

    for _ in range(5):
        occu = generate_random_ordered_occupancy(
            processor, composition=composition, charge_neutral=charge_neutral
        )
        structure_ = processor.structure_from_occupancy(occu)
        assert structure_.is_ordered

        if composition is not None:
            assert (
                structure_.composition.fractional_composition
                == single_structure.composition.fractional_composition
            )

        if charge_neutral is True:
            assert structure_.charge == 0
