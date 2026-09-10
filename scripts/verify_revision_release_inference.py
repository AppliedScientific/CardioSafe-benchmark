"""Replay final ensembles from raw feature blocks through the release adapter."""
from pathlib import Path
import json
import numpy as np
import torch
from inference.revision_predictor import RevisionEnsemble
from train.train_revision import LABEL_TO_HEAD,CLS_COLS
from train.readout_contract import sha


def main():
    torch.set_num_threads(1);report=[]
    fp=[np.load('data/features/'+p,mmap_mode='r') for p in ['morgan_fps_2048.npy','atompair_fps_2048.npy','toptorsion_fps_2048.npy','descriptors_20.npy']]
    cls=np.load('data/features/chemberta_mtr_readouts/cls.npy',mmap_mode='r');mtr=np.load('data/features/chemberta_mtr_cls_predictions/predictions.npy',mmap_mode='r')
    for split in ['tan70_v1_1','tan60_v1_1']:
        checkpoints=[Path('runs_revision_v3/final')/split/f'seed_{s}/model.pt' for s in range(42,47)]
        ensemble=RevisionEnsemble(checkpoints,Path('data/features')/('revision_'+split)/'manifest.json')
        for fold in ['val','test']:
            reference=np.load(Path('runs_revision_v3/final')/split/f'ensemble_{fold}_revision.npz')
            positions=np.unique(np.r_[np.linspace(0,len(reference['idx'])-1,512,dtype=int),np.flatnonzero(np.isfinite(reference['labels'][:,7]))])
            ids=reference['idx'][positions];chemical=np.concatenate([x[ids] for x in fp],axis=1)
            predictions=ensemble.predict_features(chemical,cls[ids],mtr[ids],batch_size=128)
            errors={}
            for col,head in LABEL_TO_HEAD.items():
                truth=reference['predictions'][positions,col];np.testing.assert_allclose(predictions[head],truth,atol=1e-5,rtol=1e-5)
                errors[head]=float(np.max(np.abs(predictions[head]-truth)))
            report.append(dict(split=split,fold=fold,n=len(ids),max_abs_error=errors,checkpoint_sha256={str(p):sha(p) for p in checkpoints}))
    Path('reports/paper_revision/release_inference_audit.json').write_text(json.dumps(dict(status='passed',checks=report),indent=2));print('Release feature inference passed',len(report),'fold checks')


if __name__=='__main__':main()
