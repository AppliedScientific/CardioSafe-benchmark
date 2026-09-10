"""Label controls preserve per-fold label availability and class balance."""
import numpy as np


def permute_label_rows(labels,folds,seed):
    """Joint row null: destroy feature–target AND feature–availability association.

    Whole label vectors move together within each fixed partition, preserving
    cross-head associations, missingness patterns and all fold marginals.
    """
    out=labels.copy();rng=np.random.default_rng(seed);audit=[]
    for fold in ['train','val','test']:
        rows=folds[fold];source=rng.permutation(rows);out[rows]=labels[source]
        assert np.array_equal(np.sort(source),np.sort(rows))
        for col in range(labels.shape[1]):
            before=labels[rows,col];after=out[rows,col]
            np.testing.assert_array_equal(np.sort(before),np.sort(after))
            both=np.isfinite(before)&np.isfinite(after)
            audit.append(dict(fold=fold,column=col,n=int(np.isfinite(before).sum()),
                changed=int(np.sum(before[both]!=after[both])),
                availability_changed=int(np.sum(np.isfinite(before)!=np.isfinite(after))),
                protocol='joint_label_rows_within_fold'))
    return out,audit


def permute_observed_labels(labels, folds, seed):
    out=labels.copy();rng=np.random.default_rng(seed);audit=[]
    for fold in ['train','val','test']:
        rows=folds[fold]
        for col in range(labels.shape[1]):
            observed=rows[np.isfinite(labels[rows,col])]
            before=labels[observed,col]
            after=rng.permutation(before)
            out[observed,col]=after
            assert np.array_equal(np.sort(before),np.sort(after))
            audit.append(dict(fold=fold,column=col,n=len(observed),changed=int(np.sum(before!=after))))
    assert np.array_equal(np.isnan(out),np.isnan(labels))
    return out,audit
