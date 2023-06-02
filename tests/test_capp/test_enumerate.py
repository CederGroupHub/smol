import numpy as np
import numpy.testing as npt
import pytest
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from smol.capp.generate import enumerate_supercell_matrices


@pytest.mark.parametrize("size", range(2, 9, 2))
def test_enumerate_supercell_matrices(structure, size):
    symops = SpacegroupAnalyzer(structure).get_symmetry_operations()
    scms = enumerate_supercell_matrices(size, symops)

    # assert determinants are correct
    for scm in scms:
        assert np.linalg.det(scm) == pytest.approx(size)

    # make sure that all the scms are unique
    assert len(np.unique(scms, axis=0)) == len(scms)

    # check that no two matrices are related by symmetry
    for scm in scms:
        for symop in symops:
            rot = np.linalg.inv(scm.T) @ symop.rotation_matrix
            equiv = [(abs(rot @ m.T - np.round(rot @ m.T)) < 1e-5).all() for m in scms]
            assert sum(equiv) <= 1  # at most one matrix is equivalent (ie itself)
            if sum(equiv) == 1:  # make sure the equivalent is actually the same
                equiv_scm = scms[equiv.index(True)]
                npt.assert_allclose(scm, equiv_scm)
