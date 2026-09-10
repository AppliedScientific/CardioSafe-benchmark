"""Score revised checkpoints, retaining per-seed logits and probability ensembles."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from model.revision_baselines import build_baseline_model
from train.features import build_provider
from train.fusion_preprocessing import transform_descriptors
from train.readout_contract import resolve_readout,sha
from train.train_revision import read_split,read_labels,LABEL_TO_HEAD,REG_COLS,REG_HEAD_IDX
from train.revision_protocol import permute_observed_labels,permute_label_rows


@torch.inference_mode()
def predict(run, fold):
    ck=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
    assert ck['revision_version']==1
    model=build_baseline_model(mode=ck['attention'],encoder_dim=ck['encoder_dim'],chemberta=ck['chemberta']).eval()
    if 'active_heads' in ck:
        for head in list(model.heads):
            if head not in ck['active_heads']:del model.heads[head]
    model.load_state_dict(ck['model_state_dict'],strict=True)
    d=np.load('data/features/descriptors_20.npy')
    d=transform_descriptors((d-ck['descriptor_scaler_means'])/ck['descriptor_scaler_stds'],ck['descriptor_transform'])
    provider=build_provider(Path('data/features'),d,chemberta_path=resolve_readout(ck['chemberta_cache'],Path.cwd()))
    folds=read_split(Path('data/splits')/(ck['split']+'.csv'));idx=folds[fold]
    y=read_labels(Path('data/labels')/ck['labels'],len(provider))
    if ck['y_randomize'] is not None:
        permute=permute_label_rows if ck.get('permutation_protocol')=='joint_label_rows_within_fold' else permute_observed_labels
        y,_=permute(y,folds,ck['y_randomize'])
    chunks=[]
    for start in range(0,len(idx),1024):
        out=model(torch.from_numpy(provider[idx[start:start+1024]]))
        parts=[]
        for c,h in LABEL_TO_HEAD.items():
            if h not in out:
                parts.append(np.full(len(idx[start:start+1024]),np.nan,dtype=np.float32));continue
            value=out[h]
            if c in REG_COLS:
                k=REG_HEAD_IDX[h];value=value*model.reg_std[k]+model.reg_mean[k]
            parts.append(value.numpy())
        chunks.append(np.stack(parts,axis=1))
    values=np.concatenate(chunks)
    np.savez_compressed(run/f'predictions_{fold}_revision.npz',idx=idx,labels=y[idx],raw=values)
    return idx,y[idx],values,dict(seed=ck['seed'],epoch=ck['epoch'],checkpoint_sha256=sha(run/'model.pt'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',type=Path,required=True);p.add_argument('--fold',choices=['val','test'],required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(42,47)));a=p.parse_args();torch.set_num_threads(1)
    predictions=[];records=[];idx=y=None
    for seed in a.seeds:
        ii,yy,values,record=predict(a.arm/f'seed_{seed}',a.fold)
        if idx is not None:
            assert np.array_equal(idx,ii) and np.array_equal(y,yy,equal_nan=True)
        idx,y=ii,yy
        for c in LABEL_TO_HEAD:
            if c not in REG_COLS:values[:,c]=expit(values[:,c])
        predictions.append(values);records.append(record)
    np.savez_compressed(a.arm/f'ensemble_{a.fold}_revision.npz',idx=idx,labels=y,predictions=np.mean(predictions,axis=0))
    (a.arm/f'ensemble_{a.fold}_revision.json').write_text(json.dumps(dict(seeds=a.seeds,fold=a.fold,records=records,
        classification='mean sigmoid probabilities',regression='mean inverse-scaled pChEMBL',heads=LABEL_TO_HEAD),indent=2))


if __name__=='__main__':main()
