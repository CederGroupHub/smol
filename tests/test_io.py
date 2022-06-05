import os

from smol.cofe import ClusterSubspace, StructureWrangler
from smol.io import load_work, save_work
from smol.moca import ClusterExpansionProcessor, Ensemble


def test_save_load_work(single_canonical_ensemble, tmpdir):
    processor = single_canonical_ensemble.processor
    subspace = processor.cluster_subspace
    wrangler = StructureWrangler(processor)
    file_path = os.path.join(tmpdir, "smol.mson")
    save_work(file_path, subspace, wrangler, processor, single_canonical_ensemble)
    assert os.path.isfile(file_path)
    work_dict = load_work(file_path)
    assert len(work_dict) == 4
    for name, obj in work_dict.items():
        assert name == obj.__class__.__name__
        assert type(obj) in (
            ClusterSubspace,
            StructureWrangler,
            ClusterExpansionProcessor,
            Ensemble,
        )
    os.remove(file_path)
