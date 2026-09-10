"""TRAIN-only, unwhitened PCA for encoder input-width controls."""
import numpy as np


def fit_projection(features, rows, width=384, chunk=4096):
    """Merge float64 centered covariances; never inspect rows outside `rows`."""
    rows = np.asarray(rows)
    if features.ndim != 2 or rows.ndim != 1 or rows.dtype.kind not in 'iu':
        raise ValueError('Expected a feature matrix and integer row indices')
    if len(rows)<2 or len(np.unique(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=len(features)):
        raise ValueError('Fit rows must be unique, in range, and contain at least two molecules')
    if not 1<=width<=features.shape[1] or chunk<1:
        raise ValueError('Invalid projection width or chunk size')
    mean=np.zeros(features.shape[1],dtype=np.float64)
    scatter=np.zeros((len(mean),len(mean)),dtype=np.float64)
    count=0
    for start in range(0,len(rows),chunk):
        batch=np.asarray(features[rows[start:start+chunk]],dtype=np.float64)
        if not np.isfinite(batch).all():raise ValueError('Nonfinite training encoder features')
        batch_mean=batch.mean(0)
        centered=batch-batch_mean
        delta=batch_mean-mean
        total=count+len(batch)
        scatter+=centered.T@centered+np.outer(delta,delta)*(count*len(batch)/total)
        mean+=delta*(len(batch)/total)
        count=total
    covariance=scatter/(count-1)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    order=np.argsort(eigenvalues)[::-1]
    eigenvalues=eigenvalues[order]
    components=eigenvectors[:,order[:width]].T
    # A saved basis is authoritative; this removes ordinary eigenvector sign ambiguity.
    pivots=np.argmax(np.abs(components),axis=1)
    components*=np.where(components[np.arange(width),pivots]<0,-1.,1.)[:,None]
    variance=np.maximum(eigenvalues,0)
    if variance.sum()<=0:raise ValueError('Training encoder features have no variance')
    return dict(mean=mean,components=components,eigenvalues=eigenvalues,
                retained_variance_fraction=float(variance[:width].sum()/variance.sum()),fit_rows=count)


def project(features, state):
    values=np.asarray(features,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=state['mean'].shape[0]:
        raise ValueError('Encoder input width does not match the fitted projection')
    if not np.isfinite(values).all():raise ValueError('Nonfinite encoder features')
    return ((values-state['mean'])@state['components'].T).astype(np.float32)
