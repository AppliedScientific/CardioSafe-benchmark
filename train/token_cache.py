"""Masked, row-identified shards of frozen token representations.

A shard includes every non-padding token, including special tokens, to preserve
ChemBERTaEncoder's existing mean-pooling contract. It is usable only once its
manifest is published. Float16 storage is intentional; readers return float32.
"""
import json
from pathlib import Path
import numpy as np
from train.readout_contract import sha


def create_packed_arrays(path, lengths, width=384):
    """Allocate packed storage with native-int NPY shapes (NumPy 2 compatible)."""
    lengths = np.asarray(lengths)
    if (lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer)
            or not len(lengths) or np.any(lengths <= 0) or width < 1):
        raise ValueError('Expected positive integer token lengths and width')
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    np.save(path/'row_ids.npy', np.arange(len(lengths), dtype=np.int64))
    np.save(path/'offsets.npy', offsets)
    arrays = {k: np.load(path/f'{k}.npy', mmap_mode='r') for k in ('row_ids', 'offsets')}
    count = int(offsets[-1])
    for name, dtype, shape in [('token_ids', np.int32, (count,)),
                               ('values', np.float16, (count, int(width)))]:
        arrays[name] = np.lib.format.open_memmap(path/f'{name}.npy', mode='w+', dtype=dtype, shape=shape)
    return arrays


def write_shard(path, row_ids, hidden, token_ids, mask):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    raw_rows = np.asarray(row_ids)
    if raw_rows.ndim != 1 or not np.issubdtype(raw_rows.dtype, np.integer):
        raise ValueError('Row identities must be integers')
    rows = raw_rows.astype(np.int64)
    h, ids, valid = np.asarray(hidden), np.asarray(token_ids), np.asarray(mask)
    if valid.dtype != bool and not np.isin(valid, [0, 1]).all():
        raise ValueError('Mask must be binary')
    valid = valid.astype(bool)
    if (h.ndim != 3 or ids.shape != h.shape[:2] or valid.shape != ids.shape
            or rows.shape != (len(h),) or len(np.unique(rows)) != len(rows)
            or np.any(rows < 0) or not valid.any(axis=1).all()):
        raise ValueError('Invalid token geometry, rows, or empty molecule')
    if not np.isfinite(h[valid]).all() or np.max(np.abs(h[valid])) > np.finfo(np.float16).max:
        raise ValueError('Non-finite or unrepresentable valid token values')
    path.mkdir(parents=True)
    arrays = {'row_ids': rows, 'offsets': np.r_[0, np.cumsum(valid.sum(1))].astype(np.int64),
              'token_ids': ids[valid].astype(np.int32), 'values': h[valid].astype(np.float16)}
    records = {}
    for name, value in arrays.items():
        p = path / (name + '.npy')
        np.save(p, value)
        records[p.name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                           'sha256': sha(p)}
    manifest = {'schema': 1, 'rows': len(rows), 'width': h.shape[-1],
                'tokens': int(valid.sum()), 'files': records,
                'mask_policy': 'all non-padding tokens, including special tokens'}
    temporary = path / 'manifest.json.partial'
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(path / 'manifest.json')
    return manifest


class TokenShard:
    def __init__(self, path, verify_hashes=True):
        self.path = Path(path)
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        if self.manifest.get('status', 'complete') != 'complete':
            raise ValueError('Token cache is not complete')
        for name, record in self.manifest['files'].items():
            p = self.path / name
            if verify_hashes and sha(p) != record['sha256']:
                raise ValueError(f'Token cache hash mismatch: {p}')
            value = np.load(p, mmap_mode='r')
            if list(value.shape) != record['shape'] or str(value.dtype) != record['dtype']:
                raise ValueError(f'Token cache geometry mismatch: {p}')
        self.rows = np.load(self.path / 'row_ids.npy', mmap_mode='r')
        self.offsets = np.load(self.path / 'offsets.npy', mmap_mode='r')
        self.values = np.load(self.path / 'values.npy', mmap_mode='r')
        self.ids = np.load(self.path / 'token_ids.npy', mmap_mode='r')
        if (len(set(self.rows)) != len(self.rows) or self.offsets.shape != (len(self.rows)+1,)
                or self.offsets[0] != 0 or self.offsets[-1] != len(self.values)
                or not (np.diff(self.offsets) > 0).all() or len(self.ids) != len(self.values)):
            raise ValueError('Malformed token cache index')
        self.lookup = {int(row): i for i, row in enumerate(self.rows)}

    def batch(self, row_ids):
        rows = np.asarray(row_ids)
        if rows.ndim != 1 or (rows.size and not np.issubdtype(rows.dtype, np.integer)):
            raise ValueError('Row identities must be integers')
        positions = [self.lookup[int(row)] for row in rows]  # Unknown row fails closed.
        lengths = [int(self.offsets[i+1]-self.offsets[i]) for i in positions]
        width = self.values.shape[1]
        values = np.zeros((len(positions), max(lengths, default=0), width), np.float32)
        ids = np.zeros(values.shape[:2], np.int32)
        mask = np.zeros(values.shape[:2], bool)
        for j, (i, n) in enumerate(zip(positions, lengths)):
            sl = slice(self.offsets[i], self.offsets[i+1])
            values[j, :n], ids[j, :n], mask[j, :n] = self.values[sl], self.ids[sl], True
        return values, ids, mask


def open_token_cache(path, repo, expected=None):
    """Verify content and compound identity once per consumer, before any batches."""
    if path is None:
        if expected is not None: raise ValueError('Checkpoint requires a token cache')
        return None, None
    path, repo = Path(path).resolve(), Path(repo).resolve()
    cache = TokenShard(path)
    contract = cache.manifest['contract']
    if contract['compound_sha256'] != sha(repo/'data/compounds/compounds.csv'):
        raise ValueError('Token cache uses different compound rows')
    if ('old_mean_sha256' in contract
            and contract['old_mean_sha256'] != sha(repo/'data/features/chemberta_384.npy')):
        raise ValueError('Token cache was built against a different mean representation')
    if (cache.values.ndim != 2 or cache.values.shape[1] != 384
            or not np.array_equal(cache.rows, np.arange(contract['rows']))):
        raise ValueError('Expected full row-aligned molecular token cache')
    record = {'path':str(path.relative_to(repo)), 'manifest_sha256':sha(path/'manifest.json')}
    if expected is not None and record != expected:
        raise ValueError('Checkpoint token cache contract changed')
    return cache, record
