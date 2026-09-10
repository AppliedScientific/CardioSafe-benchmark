"""Label-only interventions for single-channel and hERG-volume comparisons."""
import hashlib
import numpy as np

CHANNEL_COLUMNS={'herg':(0,1,2),'nav15':(3,4),'cav12':(5,6),'iks':(7,)}


def select_supervision(labels,train_idx,channel='all',herg_fraction=1.,seed=20260428):
    if not 0<herg_fraction<=1:raise ValueError('hERG fraction must be in (0,1]')
    if channel!='all' and herg_fraction!=1:raise ValueError('Volume sweep is defined for multi-task training')
    active=tuple(range(8)) if channel=='all' else CHANNEL_COLUMNS[channel]
    out=labels.copy();inactive=[c for c in range(8) if c not in active];out[:,inactive]=np.nan
    candidates=train_idx[np.isfinite(labels[train_idx,:3]).any(axis=1)]
    order=np.random.default_rng(seed).permutation(candidates)
    keep=np.sort(order[:int(np.ceil(len(order)*herg_fraction))])
    if herg_fraction<1:
        drop=np.setdiff1d(candidates,keep);out[np.ix_(drop,[0,1,2])]=np.nan
    # Freeze the final model's original eligible TRAIN population throughout the
    # hERG label sweep, including rows whose only targets are subsequently masked.
    # Originally entirely unlabelled rows were absent from the final-model fits.
    selected=(train_idx[np.isfinite(labels[train_idx]).any(axis=1)] if channel=='all'
              else train_idx[np.isfinite(out[train_idx]).any(axis=1)])
    record=dict(channel=channel,active_columns=active,herg_fraction=herg_fraction,seed=seed,
        original_training_rows=len(train_idx),training_rows=len(selected),herg_candidate_rows=len(candidates),
        training_rows_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
        retained_herg_rows=len(keep),retained_herg_rows_sha256=hashlib.sha256(keep.tobytes()).hexdigest(),
        label_cells_by_column=np.isfinite(out[selected]).sum(axis=0).tolist(),
        feature_scaling='Shared global TRAIN statistics for all supervision arms',
        input_exposure='Original any-label TRAIN population held fixed across hERG fractions; channel-labelled rows for single-channel models',
        selection='Channel-normalized validation wRMSE; IKs-only uses validation BCE')
    return out,selected,active,record
