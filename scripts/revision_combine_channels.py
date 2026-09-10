"""Join four independently trained channel models into one comparable prediction file."""
import argparse
import json
from pathlib import Path
import numpy as np
from train.revision_supervision import CHANNEL_COLUMNS
from train.readout_contract import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fold',choices=['val','test'],required=True);a=p.parse_args()
    idx=labels=pred=None;hashes={}
    for channel,cols in CHANNEL_COLUMNS.items():
        path=a.root/channel/f'ensemble_{a.fold}_revision.npz';f=np.load(path)
        if idx is None:
            idx=f['idx'];labels=f['labels'];pred=np.full_like(f['predictions'],np.nan)
        else:
            assert np.array_equal(idx,f['idx']) and np.array_equal(labels,f['labels'],equal_nan=True)
        assert np.isfinite(f['predictions'][:,cols]).all()
        pred[:,cols]=f['predictions'][:,cols];hashes[channel]=sha(path)
    assert np.isfinite(pred).all();a.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out/f'ensemble_{a.fold}_revision.npz',idx=idx,labels=labels,predictions=pred)
    (a.out/f'ensemble_{a.fold}_revision.json').write_text(json.dumps(dict(source_sha256=hashes,
        method='Four independently trained channel ensembles; no shared learned parameters between channels'),indent=2))


if __name__=='__main__':main()
