"""Ensemble already-exported probabilities and unscaled predictions (tree/GNN arms)."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from train.train_revision import CLS_COLS
from train.readout_contract import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',type=Path,required=True);p.add_argument('--fold',choices=['val','test'],required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(42,47)));a=p.parse_args()
    idx=labels=None;arrays=[];hashes={}
    for seed in a.seeds:
        path=a.arm/f'seed_{seed}'/f'predictions_{a.fold}_revision.npz';f=np.load(path)
        if idx is not None:assert np.array_equal(idx,f['idx']) and np.array_equal(labels,f['labels'],equal_nan=True)
        idx=f['idx'];labels=f['labels']
        if 'raw' in f.files:
            values=f['raw'].copy()
            for col in CLS_COLS:values[:,col]=expit(values[:,col])
        else:values=f['predictions']
        arrays.append(values);hashes[str(seed)]=sha(path)
    np.savez_compressed(a.arm/f'ensemble_{a.fold}_revision.npz',idx=idx,labels=labels,predictions=np.mean(arrays,axis=0))
    (a.arm/f'ensemble_{a.fold}_revision.json').write_text(json.dumps(dict(seeds=a.seeds,source_sha256=hashes,
        classification='mean probabilities',regression='mean unscaled pChEMBL'),indent=2))


if __name__=='__main__':main()
