import numpy as np
import torch
from model.revision import RevisionConcat
from model.latent_fusion import CachedConcat
from train.revision_protocol import permute_observed_labels,permute_label_rows


def test_whole_row_null_moves_missingness_and_preserves_joint_targets():
    y=np.arange(240,dtype=np.float32).reshape(30,8);y[::3,::2]=np.nan
    folds={f:np.arange(i*10,(i+1)*10) for i,f in enumerate(['train','val','test'])}
    out,audit=permute_label_rows(y,folds,20260428)
    assert sum(a['availability_changed'] for a in audit)>0
    for indices in folds.values():
        original={tuple(np.nan_to_num(r,nan=-1)) for r in y[indices]}
        shuffled={tuple(np.nan_to_num(r,nan=-1)) for r in out[indices]}
        assert original==shuffled


def test_control_preserves_fold_marginals_and_missingness():
    y=np.arange(240,dtype=np.float32).reshape(30,8);y[::3,::2]=np.nan
    folds={f:np.arange(i*10,(i+1)*10) for i,f in enumerate(['train','val','test'])}
    out,audit=permute_observed_labels(y,folds,20260428)
    assert np.array_equal(np.isnan(y),np.isnan(out))
    assert all(sum(a['changed'] for a in audit if a['fold']==f)>0 for f in folds)
    for rows in folds.values():
        for c in range(8):
            np.testing.assert_array_equal(np.sort(y[rows,c]),np.sort(out[rows,c]))
    np.testing.assert_array_equal(out,permute_observed_labels(y,folds,20260428)[0])


def test_iks_extension_preserves_existing_initial_function():
    torch.set_num_threads(1);torch.manual_seed(42)
    baseline=CachedConcat(chemberta_dim=583,attention='none').eval()
    torch.manual_seed(42);revision=RevisionConcat().eval()
    x=torch.randn(3,6747)
    with torch.no_grad():
        for h,value in baseline(x).items():
            torch.testing.assert_close(value,revision(x)[h],rtol=0,atol=0)
        p=revision.predict(x)['iks_blocker']
    assert len(revision.heads)==8 and ((p>=0)&(p<=1)).all()
    assert sum(p.numel() for p in revision.parameters())==3820680
