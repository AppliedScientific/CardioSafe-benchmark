"""Compare randomized-label fits with their untrained initialization on real labels."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from model.revision import build_revision_model
from model.fusion_factory import parameter_digest
from train.train_revision import read_labels,read_split,LABEL_TO_HEAD,REG_COLS
from train.features import build_provider
from train.fusion_preprocessing import transform_descriptors
from scripts.revision_metrics import EndpointMetrics


@torch.inference_mode()
def main():
    torch.set_num_threads(1);d=np.load('data/features/descriptors_20.npy')
    y=read_labels(Path('data/labels/labels_v2.csv'),len(d));records=[]
    for split in ['tan70_v1_1','tan60_v1_1']:
        idx=read_split(Path('data/splits')/(split+'.csv'))['test'];initial=[];trained=[]
        for seed in range(42,47):
            run=Path('runs_revision_controls')/split/f'seed_{seed}'
            ck=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
            torch.manual_seed(seed);np.random.seed(seed)
            model=build_revision_model(encoder_dim=583).eval()
            expected=json.loads((run/'provenance.json').read_text())['initial_parameter_sha256']
            actual=parameter_digest(model)
            assert expected==actual,(split,seed,'Initialization differs',expected,actual)
            model.reg_mean.copy_(ck['model_state_dict']['reg_mean']);model.reg_std.copy_(ck['model_state_dict']['reg_std'])
            desc=transform_descriptors((d-ck['descriptor_scaler_means'])/ck['descriptor_scaler_stds'],ck['descriptor_transform'])
            provider=build_provider(Path('data/features'),desc,chemberta_path=Path(ck['chemberta_cache']['path']))
            chunks=[]
            for start in range(0,len(idx),1024):
                out=model(torch.from_numpy(provider[idx[start:start+1024]]))
                chunks.append(np.stack([out[h].numpy() for h in LABEL_TO_HEAD.values()],1))
            p=np.concatenate(chunks);saved=np.load(run/'predictions_test_revision.npz');assert np.array_equal(idx,saved['idx'])
            q=saved['raw'].copy()
            for c in LABEL_TO_HEAD:
                if c not in REG_COLS:p[:,c]=expit(p[:,c]);q[:,c]=expit(q[:,c])
            initial.append(p);trained.append(q)
            dest=Path('runs_revision_untrained')/split/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/'classification_initialization.npz',idx=idx,predictions=p)
            for state,pred in [('initialization',p),('yrandom_trained',q)]:
                for c,h in LABEL_TO_HEAD.items():
                    if c in REG_COLS:continue
                    mask=np.isfinite(y[idx,c]);v=pred[mask,c]
                    records.append(dict(split=split,seed=str(seed),state=state,head=h,n=int(mask.sum()),
                        **EndpointMetrics(y[idx,c][mask],v,False)(),probability_mean=float(v.mean()),probability_sd=float(v.std()),
                        init_hash_verified=True))
            print(split,seed,'initialization verified',flush=True)
        for state,ps in [('initialization',initial),('yrandom_trained',trained)]:
            pred=np.mean(ps,0)
            for c,h in LABEL_TO_HEAD.items():
                if c in REG_COLS:continue
                mask=np.isfinite(y[idx,c]);v=pred[mask,c]
                records.append(dict(split=split,seed='ensemble',state=state,head=h,n=int(mask.sum()),
                    **EndpointMetrics(y[idx,c][mask],v,False)(),probability_mean=float(v.mean()),probability_sd=float(v.std()),
                    init_hash_verified=True))
    with Path('reports/paper_revision/tables/phase1_control_initialization_both.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


if __name__=='__main__':main()
