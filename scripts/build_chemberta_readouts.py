"""Extract frozen MTR first-token and regression-head features, preserving old caches."""
import argparse
import csv
import datetime
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import transformers

EXPECTED_WEIGHTS = '7882205587d3985c07e425632d0e57ba0fc2fc778e3b47aab02aef9b7961407f'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def publish(path, data):
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('data/features/chemberta_mtr_readouts'))
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.batch, args.threads) < 1: p.error('Batch and threads must be positive')
    torch.set_num_threads(args.threads)
    weights = args.model/'pytorch_model.bin'
    if sha(weights) != EXPECTED_WEIGHTS: raise ValueError('Unexpected pretrained weights')
    compounds = list(csv.DictReader(Path('data/compounds/compounds.csv').open()))
    assert all(int(row['row_idx']) == i for i, row in enumerate(compounds))
    n = len(compounds)
    contract = {'weights_sha256':EXPECTED_WEIGHTS,'compound_sha256':sha('data/compounds/compounds.csv'),
                'source_sha256':sha(__file__),'rows':n,'width':384,'max_length':512,
                'old_mean_sha256':sha('data/features/chemberta_384.npy')}
    manifest = args.out/'manifest.json'
    if args.out.exists():
        if not args.resume: raise FileExistsError('Use --resume only for this incomplete extraction')
        state = json.loads(manifest.read_text())
        if state['contract'] != contract: raise ValueError('Resume contract changed')
        if state['status'] == 'complete': raise FileExistsError('Extraction already complete')
        cls, head = [np.load(args.out/f'{k}.npy', mmap_mode='r+') for k in ['cls','head']]
    else:
        args.out.mkdir(parents=True)
        cls, head = [np.lib.format.open_memmap(args.out/f'{k}.npy', mode='w+',
                     dtype=np.float32, shape=(n,384)) for k in ['cls','head']]
        state = {'status':'building','contract':contract,'completed_rows':0,
                 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'old_mean_max_abs_error':0.,'elapsed_seconds':0.,'labels_read':False,
                 'readouts':{'cls':'last_hidden_state[:,0,:]',
                            'head':'relu(regression.dense(last_hidden_state[:,0,:])); pretrained weights, dropout disabled'},
                 'runtime':{'torch':torch.__version__,'transformers':transformers.__version__,
                            'threads':args.threads,'batch':args.batch}}
        publish(manifest,state)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    pretrained = torch.load(weights, map_location='cpu', weights_only=True)
    w, b = pretrained['regression.dense.weight'], pretrained['regression.dense.bias']
    assert w.shape == (384,384) and b.shape == (384,)
    existing = np.load('data/features/chemberta_384.npy', mmap_mode='r')
    assert existing.shape == (n,384)
    last = time.perf_counter()
    start_row = state['completed_rows']
    for start in range(start_row, n, args.batch):
        end = min(start+args.batch,n)
        inputs = tokenizer([c['smiles'] for c in compounds[start:end]], padding=True,
                           truncation=True, max_length=512, return_tensors='pt')
        hidden = model(**inputs).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        mean = (hidden*mask).sum(1)/mask.sum(1)
        error = float(np.max(np.abs(mean.numpy()-existing[start:end])))
        if not np.isfinite(hidden.numpy()).all() or error >= 1e-3:
            raise ValueError(f'Feature reproduction failed at rows {start}:{end}: {error}')
        cls[start:end] = hidden[:,0].numpy()
        head[start:end] = F.relu(F.linear(hidden[:,0],w,b)).numpy()
        state['old_mean_max_abs_error'] = max(state['old_mean_max_abs_error'],error)
        if end == n or end-start_row >= 4096 and (end-start_row) % 4096 < args.batch:
            cls.flush(); head.flush()
            state['completed_rows'] = end
            state['elapsed_seconds'] += time.perf_counter()-last
            last = time.perf_counter()
            publish(manifest,state)
            print(json.dumps({'rows':end,'total':n,'seconds':state['elapsed_seconds'],
                              'mean_error':state['old_mean_max_abs_error']}),flush=True)
    state.update({'status':'complete','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'output_sha256':{k:sha(args.out/f'{k}.npy') for k in ['cls','head']}})
    publish(manifest,state)
    print(json.dumps(state,indent=2),flush=True)


if __name__ == '__main__':main()
