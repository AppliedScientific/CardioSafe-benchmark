"""Apply the pinned pretrained MTR head to verified final-layer first-token states.

Dropout is disabled, exactly as in evaluation. Prediction coordinates remain in
the pretrained target space; no unsupported physical-unit names are assigned.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.build_chemberta_readouts import EXPECTED_WEIGHTS, publish
from train.readout_contract import capture_readout, sha
from train.train_fusion import read_split


def predict(cls, weights):
    hidden = F.relu(F.linear(cls, weights['regression.dense.weight'],
                            weights['regression.dense.bias']))
    logits = F.linear(hidden, weights['regression.out_proj.weight'],
                      weights['regression.out_proj.bias'])
    # Authors' inference forward returns unnormalize_logits when labels=None.
    return logits * weights['norm_std'] + weights['norm_mean']


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('data/features/chemberta_mtr_cls_predictions'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert sha(args.weights) == EXPECTED_WEIGHTS, 'Unexpected pretrained checkpoint'
    weights = torch.load(args.weights, map_location='cpu', weights_only=True)
    assert weights['regression.dense.weight'].shape == (384, 384)
    assert weights['regression.out_proj.weight'].shape == (199, 384), 'Inspect actual output width before proceeding'
    assert weights['regression.out_proj.bias'].shape == (199,)
    assert weights['norm_mean'].shape == weights['norm_std'].shape == (199,)
    assert (weights['norm_std'] > 0).all()
    source = Path('data/features/chemberta_mtr_readouts/cls.npy')
    record = capture_readout(source, Path.cwd())
    cls = np.load(source, mmap_mode='r')
    old_head_path = source.parent/'head.npy'
    capture_readout(old_head_path, Path.cwd())
    old_head = np.load(old_head_path, mmap_mode='r')
    args.out.mkdir(parents=True, exist_ok=False)
    pred = np.lib.format.open_memmap(args.out/'predictions.npy', mode='w+', dtype=np.float32, shape=(len(cls), 199))
    max_error = 0.
    for start in range(0, len(cls), 4096):
        x = torch.from_numpy(np.array(cls[start:start+4096]))
        y = predict(x, weights).numpy()
        independent = F.linear(torch.from_numpy(np.array(old_head[start:start+4096])),
                               weights['regression.out_proj.weight'], weights['regression.out_proj.bias']).numpy()
        independent = independent * weights['norm_std'].numpy() + weights['norm_mean'].numpy()
        np.testing.assert_allclose(y, independent, atol=1e-4, rtol=1e-5)
        max_error = max(max_error, float(np.max(np.abs(y-independent))))
        assert np.isfinite(y).all()
        pred[start:start+len(x)] = y
    pred.flush()
    split = Path('data/splits/tan70_v1_1.csv')
    train = read_split(split)['train']
    mean = np.mean(pred[train], axis=0, dtype=np.float64)
    std = np.std(pred[train], axis=0, dtype=np.float64)
    scale = np.maximum(std, 1e-6)
    combined = np.lib.format.open_memmap(args.out/'combined.npy', mode='w+', dtype=np.float32, shape=(len(cls), 583))
    for start in range(0, len(cls), 4096):
        end = min(start+4096, len(cls))
        combined[start:end, :384] = cls[start:end]
        combined[start:end, 384:] = (pred[start:end]-mean)/scale
        assert np.isfinite(combined[start:end]).all()
    combined.flush()
    publish(args.out/'manifest.json', dict(status='complete',
        contract=dict(rows=len(cls), width=583, compound_sha256=record['compound_sha256'],
                      weights_sha256=EXPECTED_WEIGHTS, source_sha256=sha(__file__), split_sha256=sha(split)),
        input_readout=record, output_sha256={k:sha(args.out/f'{k}.npy') for k in ['combined','predictions']},
        layout='384 raw final-layer first-token coordinates + 199 TRAIN-standardized pretrained MTR predictions',
        raw_predictions_width=199, pretrained_unnormalization=True,
        prediction_mean=mean.tolist(), prediction_scale=scale.tolist(),
        scaling_fit_rows=len(train), labels_read=False, cached_head_reproduction_max_error=max_error))


if __name__ == '__main__':
    main()
