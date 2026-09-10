"""Common revision metric schema; paired observed-molecule bootstrap per endpoint."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from train.train_revision import LABEL_TO_HEAD,REG_COLS,WRMSE_WEIGHT


def correlation(x,y,w):
    n=w.sum();mx=w@x/n;my=w@y/n
    xx=x-mx;yy=y-my;den=np.sqrt((w@(xx*xx))*(w@(yy*yy)))
    return float(w@(xx*yy)/den) if den>0 else np.nan


class EndpointMetrics:
    def __init__(self,y,p,regression):
        self.y=np.asarray(y,dtype=np.float64);self.p=np.asarray(p,dtype=np.float64);self.regression=regression
        self.order=np.argsort(self.p,kind='stable')
        self.groups=np.r_[0,1+np.flatnonzero(np.diff(self.p[self.order]))]
        if regression:
            self.yorder=np.argsort(self.y,kind='stable')
            self.ygroups=np.r_[0,1+np.flatnonzero(np.diff(self.y[self.yorder]))]

    @staticmethod
    def weighted_ranks(w,order,groups):
        counts=np.add.reduceat(w[order],groups);ranks=np.cumsum(counts)-counts/2+.5
        expanded=np.repeat(ranks,np.diff(np.r_[groups,len(w)]));out=np.empty(len(w));out[order]=expanded
        return out

    def __call__(self,w=None):
        y,p=self.y,self.p;w=np.ones(len(y)) if w is None else np.asarray(w,dtype=np.float64);n=w.sum()
        if self.regression:
            return dict(rmse=float(np.sqrt(w@((p-y)**2)/n)),mae=float(w@np.abs(p-y)/n),
                pearson_r=correlation(y,p,w),spearman_rho=correlation(
                    self.weighted_ranks(w,self.yorder,self.ygroups),self.weighted_ranks(w,self.order,self.groups),w))
        pos=w*y;neg=w*(1-y);P=pos.sum();N=neg.sum();call=p>=.5
        tp=pos@call;fp=neg@call;tn=N-fp;fn=P-tp
        denom=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        pg=np.add.reduceat(pos[self.order],self.groups);ng=np.add.reduceat(neg[self.order],self.groups)
        auc=float(pg@(np.cumsum(ng)-ng/2)/(P*N)) if P and N else np.nan
        pc=np.cumsum(pg[::-1]);total=np.cumsum((pg+ng)[::-1])
        precision=np.divide(pc,total,out=np.zeros_like(pc),where=total>0)
        ap=float(pg[::-1]@precision/P) if P else np.nan
        return dict(auc_roc=auc,mcc=float((tp*tn-fp*fn)/denom) if denom else 0.,
                    sensitivity=float(tp/P) if P else np.nan,specificity=float(tn/N) if N else np.nan,
                    accuracy=float((tp+tn)/n),auprc=ap)


def summarize(arms,split,fold,reference,bootstrap=1000):
    rows=[];idx=y=None;pred={}
    for name,path in arms.items():
        f=np.load(path/f'ensemble_{fold}_revision.npz')
        if idx is not None:
            assert np.array_equal(idx,f['idx']) and np.array_equal(y,f['labels'],equal_nan=True)
        idx,y=f['idx'],f['labels'];pred[name]=f['predictions']
        assert pred[name].shape==y.shape and np.isfinite(pred[name]).all()
    for c,head in LABEL_TO_HEAD.items():
        mask=np.isfinite(y[:,c]);truth=y[mask,c];n=len(truth)
        if n<5:
            for a in pred:
                rows.append(dict(model=a,head=head,split=split,fold=fold,metric='not_scored',
                    estimate=None,ci_lower=None,ci_upper=None,n=n,prevalence=None,bootstrap_valid=0,comparison='n<5'))
            continue
        engines={a:EndpointMetrics(truth,p[mask,c],c in REG_COLS) for a,p in pred.items()}
        point={a:e() for a,e in engines.items()};samples={a:{k:[] for k in values} for a,values in point.items()}
        rng=np.random.default_rng(20260428)
        for _ in range(bootstrap):
            counts=rng.multinomial(n,np.full(n,1/n))
            for a,engine in engines.items():
                for k,v in engine(counts).items():samples[a][k].append(v)
        for a in engines:
            for k,value in point[a].items():
                for comparison in ['absolute']+(['delta_vs_'+reference] if a!=reference else []):
                    values=np.array(samples[a][k]);estimate=value
                    if comparison!='absolute':
                        values=values-np.array(samples[reference][k]);estimate-=point[reference][k]
                    valid=values[np.isfinite(values)];ci=np.quantile(valid,[.025,.975]) if len(valid) else [None,None]
                    rows.append(dict(model=a,head=head,split=split,fold=fold,metric=k,
                        estimate=float(estimate) if np.isfinite(estimate) else None,
                        ci_lower=ci[0],ci_upper=ci[1],n=n,
                        prevalence=float(truth.mean()) if c not in REG_COLS else None,
                        bootstrap_valid=len(valid),comparison=comparison))
    # Resample whole molecules jointly for the aggregate, retaining cross-head
    # missingness and correlation. This is not the sum of marginal CI bounds.
    columns=[c for c in REG_COLS if LABEL_TO_HEAD[c] in WRMSE_WEIGHT]
    if all(np.isfinite(y[:,c]).sum()>=5 for c in columns):
        valid=np.isfinite(y[:,columns]);weights=np.array([WRMSE_WEIGHT[LABEL_TO_HEAD[c]] for c in columns])
        error={a:np.where(valid,(p[:,columns]-y[:,columns])**2,0.) for a,p in pred.items()}
        def aggregate(counts):
            denominators=counts@valid
            return {a:float(weights@np.sqrt((counts@e)/denominators)) if np.all(denominators>0) else np.nan for a,e in error.items()}
        point=aggregate(np.ones(len(y)));samples={a:[] for a in pred};rng=np.random.default_rng(20260428)
        for _ in range(bootstrap):
            for a,v in aggregate(rng.multinomial(len(y),np.full(len(y),1/len(y)))).items():samples[a].append(v)
        for a in pred:
            for comparison in ['absolute']+(['delta_vs_'+reference] if a!=reference else []):
                values=np.asarray(samples[a]);estimate=point[a]
                if comparison!='absolute':values=values-np.asarray(samples[reference]);estimate-=point[reference]
                finite=values[np.isfinite(values)];ci=np.quantile(finite,[.025,.975]) if len(finite) else [None,None]
                rows.append(dict(model=a,head='primary_regression_aggregate',split=split,fold=fold,metric='wrmse',
                    estimate=estimate,ci_lower=ci[0],ci_upper=ci[1],n=int(valid.any(axis=1).sum()),
                    prevalence=None,bootstrap_valid=len(finite),comparison=comparison))
    for row in rows:
        prevalence=row.get('prevalence');n=row['n']
        row['uncertainty_note']=('Very small minority class; percentile bootstrap can be degenerate and must not be used as evidence of precise discrimination.'
            if prevalence is not None and min(prevalence*n,(1-prevalence)*n)<5 else '')
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',action='append',required=True);p.add_argument('--split',required=True)
    p.add_argument('--fold',default='test');p.add_argument('--reference',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();arms={name:Path(path) for name,path in (v.split('=',1) for v in a.arm)}
    rows=summarize(arms,a.split,a.fold,a.reference);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    a.out.with_suffix('.json').write_text(json.dumps(dict(bootstrap=1000,seed=20260428,
        method='Paired endpoint observed-molecule bootstrap; aggregate wRMSE resamples all fold molecules jointly; fixed fitted ensemble',
        auprc='average precision',threshold=.5,multiple_comparison_adjustment=False,
        caveat='No scaffold, training-seed or model-selection uncertainty; undefined draws excluded and counted'),indent=2))


if __name__=='__main__':main()
