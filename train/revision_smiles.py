"""Compact, row-aligned SMILES token cache and dynamic padded batches."""
import json
from pathlib import Path
import numpy as np
import torch


class SmilesTokens:
    def __init__(self,path=Path('data/features/revision_smiles_ids')):
        self.ids=np.load(path/'ids.npy',mmap_mode='r');self.offsets=np.load(path/'offsets.npy',mmap_mode='r')
        self.manifest=json.loads((path/'manifest.json').read_text())
        assert self.manifest['status']=='complete'
        self.lengths=np.diff(self.offsets);self.pad=self.manifest['pad_id']

    def batch(self,rows,device):
        lengths=self.lengths[rows];width=int(lengths.max())
        ids=np.full((len(rows),width),self.pad,dtype=np.int64);mask=np.zeros_like(ids)
        for i,row in enumerate(rows):
            n=int(lengths[i]);ids[i,:n]=self.ids[self.offsets[row]:self.offsets[row+1]];mask[i,:n]=1
        return torch.from_numpy(ids).to(device),torch.from_numpy(mask).to(device)

    def ordered_batches(self,rows,batch_size,rng=None,bucket_size=4096):
        """Shuffle rows, then length-sort local buckets; shuffle whole batches.

        Local bucketing limits padding without globally ordering optimization by
        molecular length. Each row occurs once; the short tail is retained.
        """
        rows=np.asarray(rows)
        order=rows.copy() if rng is None else rng.permutation(rows)
        batches=[]
        for start in range(0,len(order),bucket_size):
            bucket=order[start:start+bucket_size]
            bucket=bucket[np.argsort(self.lengths[bucket],kind='stable')]
            batches.extend(bucket[i:i+batch_size] for i in range(0,len(bucket),batch_size))
        if rng is not None:rng.shuffle(batches)
        return batches
