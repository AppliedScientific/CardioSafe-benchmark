"""Nested training-only CaV label budgets, shared across architectures and seeds."""
import hashlib
import numpy as np


def apply_cav_budget(labels, train_idx, fraction=1., seed=20260908):
    if not 0 < fraction <= 1:
        raise ValueError('CaV label fraction must be in (0, 1]')
    columns = [5, 6]
    eligible = np.asarray(train_idx)[np.isfinite(labels[np.ix_(train_idx, columns)]).any(1)]
    order = np.random.default_rng(seed).permutation(eligible)
    keep = np.sort(order[:int(np.ceil(len(order)*fraction))])
    remove = order[len(keep):]
    masked = labels.copy()
    masked[np.ix_(remove, columns)] = np.nan
    metadata = {'fraction':fraction, 'seed':seed, 'eligible_training_molecules':len(eligible),
                'retained_training_molecules':len(keep),
                'retained_rows_sha256':hashlib.sha256(keep.astype('<i8').tobytes()).hexdigest(),
                'original_label_counts':np.isfinite(labels[np.ix_(train_idx, columns)]).sum(0).tolist(),
                'retained_label_counts':np.isfinite(masked[np.ix_(train_idx, columns)]).sum(0).tolist(),
                'columns':['cav12_pchembl','cav12_blocker']}
    return masked, keep, metadata
