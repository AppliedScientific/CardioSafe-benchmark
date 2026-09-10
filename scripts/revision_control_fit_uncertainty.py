"""Control-ensemble sensitivity to fitted-model resampling, not molecule resampling.

Exact enumeration of five-fit bootstrap multiplicities (126 combinations) with
multinomial probabilities. This is descriptive with only five independent fits.
"""
import itertools,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scripts.revision_metrics import EndpointMetrics
from train.train_revision import CLS_COLS,LABEL_TO_HEAD


def weighted_quantile(v,w,qs):
    order=np.argsort(v);v=np.asarray(v)[order];w=np.asarray(w)[order];cdf=np.cumsum(w)/np.sum(w)
    return np.interp(qs,cdf,v)


def main():
    combinations=np.array([c for c in itertools.product(range(6),repeat=5) if sum(c)==5]);assert len(combinations)==126
    probabilities=np.array([math.factorial(5)/np.prod([math.factorial(int(i)) for i in c])/5**5 for c in combinations]);np.testing.assert_allclose(probabilities.sum(),1)
    rows=[];geometries=[]
    for split in ['tan70_v1_1','tan60_v1_1']:
        real=np.load(Path('runs_revision_v3/final')/split/'ensemble_test_revision.npz')
        for arm in ['yrandom','yrandom_rows']:
            root=Path('runs_revision_independent')/arm/split
            fits=[]
            for seed in range(42,47):
                f=np.load(root/f'seed_{seed}/predictions_test_revision.npz');np.testing.assert_array_equal(f['idx'],real['idx']);fits.append(expit(f['raw']))
            for c in CLS_COLS:
                mask=np.isfinite(real['labels'][:,c]);y=real['labels'][mask,c];pred=np.stack([p[mask,c] for p in fits]);samples=[]
                for counts in combinations:
                    p=counts@pred/5;samples.append(EndpointMetrics(y,p,False)()['auc_roc'])
                center=pred.mean(0);lo,hi=weighted_quantile(samples,probabilities,[.025,.975])
                rows.append(dict(model=arm,head=LABEL_TO_HEAD[c],split=split,n=len(y),metric='auc_roc',estimate=EndpointMetrics(y,center,False)()['auc_roc'],ci_lower=lo,ci_upper=hi,uncertainty='Five-fit resampling only; exact126 multiplicities; no molecule resampling; descriptive and unstable with five fits',weighted_fraction_auc_le_half=float(probabilities[np.asarray(samples)<=.5].sum()),ensemble_probability_sd=float(center.std())))
                for i,seed in enumerate(range(42,47)):
                    geometries.append(dict(model=arm,head=LABEL_TO_HEAD[c],split=split,seed=seed,probability_mean=float(pred[i].mean()),probability_sd=float(pred[i].std()),positive_minus_negative_mean=float(pred[i,y==1].mean()-pred[i,y==0].mean()),correlation_with_ensemble=float(np.corrcoef(pred[i],center)[0,1])))
    out=Path('reports/paper_revision/tables');pd.DataFrame(rows).to_csv(out/'phase1_control_fit_bootstrap_both.csv',index=False);pd.DataFrame(geometries).to_csv(out/'phase1_control_ensemble_geometry_both.csv',index=False)
    print(pd.DataFrame(rows)[lambda d:d['head']=='herg_blocker_10um'].to_string(index=False))


if __name__=='__main__':main()
