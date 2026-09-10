"""Score saved randomized-fit predictions against untouched real fold labels.

Different TRAIN/VAL permutations across seeds are allowed; all ensemble members
are evaluated against the same real test targets. No labels enter inference.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from train.train_revision import read_labels,REG_COLS
from train.readout_contract import sha


def export(arm,out,labels_path,fold):
    labels=read_labels(labels_path,334444);predictions=[];records=[];idx=None
    for seed in range(42,47):
        p=arm/f'seed_{seed}'/f'predictions_{fold}_revision.npz';d=np.load(p)
        if idx is not None:assert np.array_equal(idx,d['idx'])
        idx=d['idx'];values=d['raw'].copy()
        for c in range(values.shape[1]):
            if c not in REG_COLS:values[:,c]=expit(values[:,c])
        predictions.append(values);records.append(dict(seed=seed,prediction_source=str(p),sha256=sha(p)))
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/f'ensemble_{fold}_revision.npz',idx=idx,labels=labels[idx],predictions=np.mean(predictions,0))
    (out/f'ensemble_{fold}_revision.json').write_text(json.dumps(dict(records=records,labels=str(labels_path),
        labels_sha256=sha(labels_path),truth='Untouched real fold labels',
        classification='Mean sigmoid probabilities',regression='Mean inverse-scaled predictions'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--fold',choices=['val','test'],required=True)
    a=p.parse_args();export(a.arm,a.out,a.labels,a.fold)
