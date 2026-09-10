"""Identify and verify optional frozen encoder readouts used by experiments."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def capture_readout(path, repo):
    if path is None: return None
    path, repo = Path(path).resolve(), Path(repo).resolve()
    relative = str(path.relative_to(repo))
    manifest = json.loads((path.parent/'manifest.json').read_text())
    if manifest['status'] != 'complete': raise ValueError('Encoder cache is not complete')
    record = {'path':relative,'sha256':sha(path),'readout':path.stem,
              'compound_sha256':manifest['contract']['compound_sha256']}
    if record['sha256'] != manifest['output_sha256'][path.stem]:
        raise ValueError('Encoder cache does not match extraction manifest')
    if record['compound_sha256'] != sha(repo/'data/compounds/compounds.csv'):
        raise ValueError('Encoder cache uses different compound rows')
    array = np.load(path, mmap_mode='r')
    width = manifest['contract'].get('width', 384)
    if not isinstance(width, int) or width < 1:
        raise ValueError('Encoder width must be a positive integer')
    if array.shape != (manifest['contract']['rows'],width) or array.dtype != np.float32:
        raise ValueError('Encoder cache must contain aligned float32 vectors of the declared width')
    if width != 384:
        record['width'] = width
    return record


def resolve_readout(record, repo):
    if record is None: return None
    path = Path(repo)/record['path']
    actual = capture_readout(path,repo)
    if actual != record: raise ValueError('Checkpoint encoder readout contract does not match cache')
    return path
