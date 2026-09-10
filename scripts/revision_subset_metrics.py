"""Common endpoint metrics for explicitly defined evaluation subsets."""
import numpy as np
from scripts.revision_metrics import EndpointMetrics


def endpoint_rows(truth,predictions,*,head,split,cohort,regression=False,reference='final',bootstrap=1000,fold='test'):
    y=np.asarray(truth);n=len(y)
    assert np.isfinite(y).all()
    assert all(len(p)==n and np.isfinite(p).all() for p in predictions.values())
    common=dict(head=head,split=split,fold=fold,cohort=cohort,n=n,
        prevalence=float(y.mean()) if n and not regression else None)
    if n<5:
        return [dict(model=a,**common,metric='not_scored',estimate=None,ci_lower=None,ci_upper=None,
            bootstrap_valid=0,comparison='n<5',uncertainty_note='Insufficient labelled compounds') for a in predictions]
    engines={a:EndpointMetrics(y,p,regression) for a,p in predictions.items()}
    points={a:e() for a,e in engines.items()};samples={a:{k:[] for k in v} for a,v in points.items()}
    rng=np.random.default_rng(20260428)
    for _ in range(bootstrap):
        w=rng.multinomial(n,np.full(n,1/n))
        for a,e in engines.items():
            for k,v in e(w).items():samples[a][k].append(v)
    rows=[]
    for a,point in points.items():
        for k,value in point.items():
            for comparison in ['absolute']+(['delta_vs_'+reference] if a!=reference and reference in points else []):
                values=np.asarray(samples[a][k]);estimate=value
                if comparison!='absolute':values=values-np.asarray(samples[reference][k]);estimate-=points[reference][k]
                finite=values[np.isfinite(values)];ci=np.quantile(finite,[.025,.975]) if len(finite) else [None,None]
                rows.append(dict(model=a,**common,metric=k,estimate=float(estimate) if np.isfinite(estimate) else None,
                    ci_lower=ci[0],ci_upper=ci[1],bootstrap_valid=len(finite),comparison=comparison,
                    uncertainty_note='Very small minority class; conditional percentile intervals may be degenerate'
                        if not regression and min(y.sum(),n-y.sum())<5 else ''))
    return rows
