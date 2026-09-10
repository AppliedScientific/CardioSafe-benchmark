"""Verify backed-up control labels, checkpoint inference and probability ensembling."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from model.revision import build_revision_model
from train.train_revision import read_labels,read_split,LABEL_TO_HEAD,REG_COLS,REG_HEAD_IDX
from train.revision_protocol import permute_observed_labels
from train.features import build_provider
from train.fusion_preprocessing import transform_descriptors
from train.readout_contract import sha


@torch.inference_mode()
def main():
    torch.set_num_threads(1);root=Path('runs_revision_controls');records=[];ensemble_errors=[]
    descriptors=np.load('data/features/descriptors_20.npy')
    labels=read_labels(Path('data/labels/labels_v2.csv'),len(descriptors))
    for split in ['tan70_v1_1','tan60_v1_1']:
        folds=read_split(Path('data/splits')/(split+'.csv'));target,permutation=permute_observed_labels(labels,folds,20260428)
        all_predictions={fold:[] for fold in ['val','test']}
        for seed in range(42,47):
            run=root/split/f'seed_{seed}';ck=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
            model=build_revision_model(encoder_dim=583).eval();model.load_state_dict(ck['model_state_dict'],strict=True)
            desc=transform_descriptors((descriptors-ck['descriptor_scaler_means'])/ck['descriptor_scaler_stds'],ck['descriptor_transform'])
            provider=build_provider(Path('data/features'),desc,chemberta_path=Path('data/features')/('revision_'+split)/'combined.npy')
            loss_audit=json.loads((run/'loss_tensor_audit.json').read_text());assert loss_audit['loss_tensor_matches_permuted_scaled_labels']
            for fold in ['val','test']:
                saved=np.load(run/f'predictions_{fold}_revision.npz')
                assert np.array_equal(saved['idx'],folds[fold]);np.testing.assert_array_equal(saved['labels'],target[folds[fold]])
                selected=np.unique(np.r_[np.linspace(0,len(saved['idx'])-1,512,dtype=int),np.where(np.isfinite(saved['labels'][:,7]))[0]])
                out=model(torch.from_numpy(provider[saved['idx'][selected]]));parts=[]
                for c,h in LABEL_TO_HEAD.items():
                    v=out[h]
                    if c in REG_COLS:v=v*model.reg_std[REG_HEAD_IDX[h]]+model.reg_mean[REG_HEAD_IDX[h]]
                    parts.append(v.numpy())
                actual=np.stack(parts,1);error=float(np.abs(actual-saved['raw'][selected]).max())
                np.testing.assert_allclose(actual,saved['raw'][selected],atol=1e-4,rtol=1e-5)
                p=saved['raw'].copy()
                for c in LABEL_TO_HEAD:
                    if c not in REG_COLS:p[:,c]=expit(p[:,c])
                all_predictions[fold].append(p)
                records.append(dict(split=split,seed=seed,fold=fold,inference_rows=len(selected),max_absolute_error=error,
                    loss_tensor_verified=True,labels_match_permutation=True,checkpoint_sha256=sha(run/'model.pt')))
        for fold,values in all_predictions.items():
            ensemble=np.load(root/split/f'ensemble_{fold}_revision.npz')
            expected=np.mean(values,axis=0)
            ensemble_errors.append(dict(split=split,fold=fold,max_absolute_error=float(np.abs(ensemble['predictions']-expected).max())))
            # expit float32 differs by a few ULPs between the Linux/x86 and
            # macOS/ARM math libraries. Regression means should match exactly.
            np.testing.assert_allclose(ensemble['predictions'],expected,atol=2e-7,rtol=0)
            np.testing.assert_array_equal(ensemble['predictions'][:,REG_COLS],expected[:,REG_COLS])
    result=dict(status='passed',records=records,ensemble_errors=ensemble_errors,
        ensemble='Mean probability match within 2e-7 cross-platform float32 tolerance; regression ensemble exact')
    Path('reports/paper_revision/control_inference_audit.json').write_text(json.dumps(result,indent=2));print('passed',len(records),'checkpoint/fold audits')


if __name__=='__main__':main()
