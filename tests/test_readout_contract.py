import json
from types import SimpleNamespace
import numpy as np
import pytest
from train.readout_contract import capture_readout, resolve_readout, sha
from train.features import build_provider


def test_cache_completion_identity_and_row_contract(tmp_path):
    compounds = tmp_path/'data/compounds/compounds.csv'
    compounds.parent.mkdir(parents=True); compounds.write_text('row_idx,smiles\n0,C\n1,CC\n')
    cache = tmp_path/'features/cls.npy'; cache.parent.mkdir()
    np.save(cache, np.ones((2,384),np.float32))
    manifest = {'status':'building','contract':{'compound_sha256':sha(compounds),'rows':2},
                'output_sha256':{'cls':sha(cache)}}
    p = cache.parent/'manifest.json'; p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='not complete'): capture_readout(cache,tmp_path)
    manifest['status'] = 'complete'; p.write_text(json.dumps(manifest))
    record = capture_readout(cache,tmp_path)
    assert resolve_readout(record,tmp_path) == cache
    assert capture_readout(None,tmp_path) is None
    assert resolve_readout(None,tmp_path) is None
    compounds.write_text('row_idx,smiles\n0,CC\n1,C\n')
    with pytest.raises(ValueError, match='compound rows'): resolve_readout(record,tmp_path)
    np.save(cache,np.zeros((2,384),np.float32))
    with pytest.raises(ValueError, match='extraction manifest'): capture_readout(cache,tmp_path)


def test_provider_uses_selected_readout_without_changing_other_blocks(tmp_path):
    for name in ['morgan','atompair','toptorsion']:
        np.save(tmp_path/f'{name}_fps_2048.npy',np.ones((2,2048),np.float32))
    np.save(tmp_path/'chemberta_384.npy',np.zeros((2,384),np.float32))
    np.save(tmp_path/'cls.npy',np.full((2,384),9,np.float32))
    d = np.zeros((2,20),np.float32)
    old = build_provider(tmp_path,d)[[1,0]]
    new = build_provider(tmp_path,d,chemberta_path=tmp_path/'cls.npy')[[1,0]]
    np.testing.assert_array_equal(old[:,:6164],new[:,:6164])
    assert (old[:,6164:]==0).all() and (new[:,6164:]==9).all()


def test_ensemble_refuses_a_single_matrix_for_different_readouts(monkeypatch):
    import inference.fusion_predictor as module
    monkeypatch.setattr(module,'FusionPredictor',lambda path,device:SimpleNamespace(chemberta_feature_contract=path))
    with pytest.raises(ValueError,match='one encoder readout'):
        module.FusionEnsemble(['mean','cls'])
