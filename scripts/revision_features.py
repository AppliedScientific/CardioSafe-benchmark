"""Fit the 199 predicted-property feature scales separately on each split's TRAIN."""
import argparse
import json
from pathlib import Path
import numpy as np
from train.readout_contract import sha, capture_readout
from train.train_revision import read_split


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',required=True,choices=['tan70_v1_1','tan60_v1_1']);a=p.parse_args()
    source=Path('data/features/chemberta_mtr_cls_predictions')
    m=json.loads((source/'manifest.json').read_text())
    assert m['status']=='complete' and sha(source/'predictions.npy')==m['output_sha256']['predictions']
    cls_path=Path('data/features/chemberta_mtr_readouts/cls.npy')
    cls_contract=capture_readout(cls_path,Path.cwd())
    pred=np.load(source/'predictions.npy',mmap_mode='r');cls=np.load(cls_path,mmap_mode='r')
    split_path=Path('data/splits')/(a.split+'.csv');rows=read_split(split_path)['train']
    out=Path('data/features')/('revision_'+a.split)
    out.mkdir(exist_ok=False)
    mean=pred[rows].mean(axis=0,dtype=np.float64);scale=np.maximum(pred[rows].std(axis=0,dtype=np.float64),1e-6)
    combined=np.lib.format.open_memmap(out/'combined.npy',mode='w+',dtype=np.float32,shape=(len(cls),583))
    for start in range(0,len(cls),4096):
        end=min(start+4096,len(cls));combined[start:end,:384]=cls[start:end]
        combined[start:end,384:]=(pred[start:end]-mean)/scale
        assert np.isfinite(combined[start:end]).all()
    combined.flush()
    manifest=dict(status='complete',contract=dict(rows=len(cls),width=583,compound_sha256=cls_contract['compound_sha256'],
        split=a.split,split_sha256=sha(split_path),source_sha256=sha(__file__)),
        input_cls=cls_contract,input_predictions_sha256=sha(source/'predictions.npy'),
        mean=mean.tolist(),scale=scale.tolist(),fit_fold='train',fit_n=len(rows),
        output_sha256=dict(combined=sha(out/'combined.npy')))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(a.split,manifest['output_sha256'],flush=True)


if __name__=='__main__':main()
