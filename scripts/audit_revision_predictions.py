"""Replay revised checkpoints and audit target identity and probability ensembling."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from model.revision import RevisionConcat
from train.train_revision import read_labels,read_split,LABEL_TO_HEAD,REG_COLS,REG_HEAD_IDX
from train.revision_protocol import permute_observed_labels,permute_label_rows
from train.features import build_provider
from train.fusion_preprocessing import transform_descriptors
from train.readout_contract import sha


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs_revision_v3'))
    p.add_argument('--arms',nargs='+',default=['final','yrandom']);p.add_argument('--out',type=Path,default=Path('reports/paper_revision/v3_inference_audit.json'));a=p.parse_args()
    torch.set_num_threads(1);d=np.load('data/features/descriptors_20.npy');real=read_labels(Path('data/labels/labels_v3.csv'),len(d))
    old=read_labels(Path('data/labels/labels_v2.csv'),len(d));changed=~np.all((old==real)|(np.isnan(old)&np.isnan(real)),axis=1)
    records=[];ensembles=[]
    for arm in a.arms:
        for split in ['tan70_v1_1','tan60_v1_1']:
            folds=read_split(Path('data/splits')/(split+'.csv'));parts={f:[] for f in ['val','test']};targets={}
            for seed in range(42,47):
                root=a.root/arm/split/f'seed_{seed}';ck=torch.load(root/'model.pt',map_location='cpu',weights_only=False)
                assert ck['labels']=='labels_v3.csv' and ck['encoder_dim']==583 and ck['attention']=='none'
                target=real
                if ck['y_randomize'] is not None:
                    function=permute_label_rows if ck['permutation_protocol']=='joint_label_rows_within_fold' else permute_observed_labels
                    target,_=function(real,folds,ck['y_randomize'])
                    assert json.loads((root/'loss_tensor_audit.json').read_text())['loss_tensor_matches_permuted_scaled_labels']
                model=RevisionConcat().eval();model.load_state_dict(ck['model_state_dict'],strict=True)
                provider=build_provider(Path('data/features'),transform_descriptors((d-ck['descriptor_scaler_means'])/ck['descriptor_scaler_stds'],ck['descriptor_transform']),
                    chemberta_path=Path('data/features')/('revision_'+split)/'combined.npy')
                for fold in parts:
                    f=np.load(root/f'predictions_{fold}_revision.npz');idx=f['idx'];np.testing.assert_array_equal(idx,folds[fold]);np.testing.assert_array_equal(f['labels'],target[idx])
                    selected=np.unique(np.r_[np.linspace(0,len(idx)-1,512,dtype=int),np.flatnonzero(np.isfinite(real[idx,7])|changed[idx])])
                    values=model(torch.from_numpy(provider[idx[selected]]));actual=[]
                    for c,h in LABEL_TO_HEAD.items():
                        v=values[h]
                        if c in REG_COLS:
                            i=REG_HEAD_IDX[h];v=v*model.reg_std[i]+model.reg_mean[i]
                        actual.append(v.numpy())
                    actual=np.stack(actual,1);np.testing.assert_allclose(actual,f['raw'][selected],atol=1e-4,rtol=1e-5)
                    pred=f['raw'].copy()
                    for c in LABEL_TO_HEAD:
                        if c not in REG_COLS:pred[:,c]=expit(pred[:,c])
                    parts[fold].append(pred);targets[fold]=target[idx]
                    records.append(dict(arm=arm,split=split,seed=seed,fold=fold,rows=len(selected),
                        max_absolute_error=float(np.abs(actual-f['raw'][selected]).max()),checkpoint_sha256=sha(root/'model.pt')))
            for fold,pred in parts.items():
                root=a.root/arm/split;f=np.load(root/f'ensemble_{fold}_revision.npz');meta=json.loads((root/f'ensemble_{fold}_revision.json').read_text())
                truth=real[folds[fold]] if meta.get('truth','').startswith('Untouched') else targets[fold]
                np.testing.assert_array_equal(f['labels'],truth)
                expected=np.mean(pred,0);np.testing.assert_allclose(f['predictions'],expected,atol=2e-7,rtol=0)
                np.testing.assert_array_equal(f['predictions'][:,REG_COLS],expected[:,REG_COLS])
                ensembles.append(dict(arm=arm,split=split,fold=fold,max_absolute_error=float(np.abs(f['predictions']-expected).max())))
    a.out.write_text(json.dumps(dict(status='passed',records=records,ensembles=ensembles,
        sampling='512 evenly spaced fold positions plus every labelled IKs and every v2-to-v3 changed molecule',
        tolerance='Raw checkpoint1e-4 absolute/1e-5 relative cross-platform; probability mean2e-7; inverse-scaled regression means exact'),indent=2))
    print('passed',len(records),'checkpoint/fold checks')


if __name__=='__main__':main()
