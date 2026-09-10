"""Create graph-only comparator inputs on the immutable global compound order."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from train.revision_graphs import compact_graph,CHOICES,SIZES
from train.readout_contract import sha


def main():
    out=Path('data/features/revision_graphs');out.mkdir(exist_ok=False);started=time.monotonic()
    frame=pd.read_csv('data/compounds/compounds.csv');assert np.array_equal(frame.row_idx,np.arange(len(frame)))
    names=['atoms','mass','edges','bonds'];handles={name:(out/(name+'.raw')).open('wb') for name in names}
    atom_offsets=[0];edge_offsets=[0]
    try:
        for i,smi in enumerate(frame.smiles):
            arrays=compact_graph(smi)
            for name,arr in zip(names,arrays):arr.tofile(handles[name])
            atom_offsets.append(atom_offsets[-1]+len(arrays[0]));edge_offsets.append(edge_offsets[-1]+len(arrays[2]))
            if i%50000==0:print(i,round(time.monotonic()-started),flush=True)
    finally:
        for handle in handles.values():handle.close()
    for name,dtype,width in [('atoms',np.uint8,7),('mass',np.float32,1),('edges',np.uint32,2),('bonds',np.uint8,14)]:
        raw=out/(name+'.raw');arr=np.memmap(raw,dtype=dtype,mode='r')
        arr=arr if width==1 else arr.reshape(-1,width);np.save(out/(name+'.npy'),arr);del arr;raw.unlink()
    np.save(out/'atom_offsets.npy',np.asarray(atom_offsets,dtype=np.int64));np.save(out/'edge_offsets.npy',np.asarray(edge_offsets,dtype=np.int64))
    manifest=dict(status='complete',rows=len(frame),atoms=atom_offsets[-1],directed_edges=edge_offsets[-1],
        source_sha256=sha(__file__),feature_source_sha256=sha('train/revision_graphs.py'),compound_sha256=sha('data/compounds/compounds.csv'),
        atom_feature_width=133,bond_feature_width=14,atomic_choices=CHOICES,atomic_category_widths=SIZES,
        output_sha256={p.name:sha(p) for p in out.glob('*.npy')},elapsed_seconds=time.monotonic()-started)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
