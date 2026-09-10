"""Branch CKA/rank, five-seed linear probes and fixed UMAP visualizations."""
import csv
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from scripts.revision_model_io import load_final
from scripts.revision_subset_metrics import endpoint_rows
from train.train_revision import read_split,read_labels,LABEL_TO_HEAD,CLS_COLS


def cka(x,y):
    x=x.astype(float);y=y.astype(float);x-=x.mean(0);y-=y.mean(0)
    return float(np.linalg.norm(x.T@y,'fro')**2/(np.linalg.norm(x.T@x,'fro')*np.linalg.norm(y.T@y,'fro')))


def rank(x):
    x=x.astype(float);x-=x.mean(0)
    singular=np.linalg.svd(x,compute_uv=False);singular=singular[singular>1e-12]
    p=singular/singular.sum();eigen=singular**2
    return dict(effective_rank=float(np.exp(-np.sum(p*np.log(p)))),participation_ratio=float(eigen.sum()**2/np.sum(eigen**2)))


@torch.inference_mode()
def extract(model,provider,rows):
    values={k:[] for k in ['fingerprints','descriptors','chemberta']}
    for start in range(0,len(rows),1024):
        x=torch.from_numpy(provider[rows[start:start+1024]])
        values['fingerprints'].append(model.fp_branch(x[:,:6144]).numpy())
        values['descriptors'].append(model.desc_branch(x[:,6144:6164]).numpy())
        values['chemberta'].append(model.chemberta_branch(x[:,6164:]).numpy())
    return {k:np.concatenate(v) for k,v in values.items()}


def write(path,rows):
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    torch.set_num_threads(3);out=Path('reports/paper_revision');geometry=[];convergence=[];manifest=[]
    labels=read_labels(Path('data/labels/labels_v3.csv'),334444)
    store=Path('runs_revision_probes');store.mkdir(exist_ok=True)
    for split in ['tan70_v1_1','tan60_v1_1']:
        folds=read_split(Path('data/splits')/(split+'.csv'));train,test=folds['train'],folds['test']
        sample=np.sort(np.random.default_rng(20260428).choice(len(test),min(4096,len(test)),replace=False))
        final=np.load(Path('runs_revision_v3/final')/split/'ensemble_test_revision.npz');assert np.array_equal(final['idx'],test)
        ensemble={k:[] for k in ['fingerprints','descriptors','chemberta']}
        for seed in range(42,47):
            model,provider,ck=load_final(Path('runs_revision_v3/final')/split/f'seed_{seed}')
            training=extract(model,provider,train);testing=extract(model,provider,test)
            for name,x in testing.items():geometry.append(dict(split=split,seed=seed,branch=name,n=len(sample),width=x.shape[1],**rank(x[sample])))
            names=list(testing)
            for i,one in enumerate(names):
                for two in names[i+1:]:
                    manifest.append(dict(split=split,seed=seed,branch_a=one,branch_b=two,n=len(sample),linear_cka=cka(testing[one][sample],testing[two][sample])))
            if seed==42:
                np.savez_compressed(store/(split+'_umap_inputs.npz'),idx=test[sample],labels=labels[test[sample]],
                    chemical=np.concatenate([testing['fingerprints'][sample],testing['descriptors'][sample]],1),
                    chemberta=testing['chemberta'][sample],shared=np.concatenate([v[sample] for v in testing.values()],1))
            for name,x in training.items():
                predictions=np.full((len(test),8),np.nan,dtype=np.float32)
                for col in CLS_COLS:
                    observed=np.isfinite(labels[train,col]);truth=labels[train[observed],col]
                    assert len(np.unique(truth))==2
                    scaler=StandardScaler().fit(x[observed]);xx=scaler.transform(x[observed]);tt=scaler.transform(testing[name])
                    probe=LogisticRegression(C=1.,class_weight='balanced',solver='lbfgs',max_iter=1000)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always',ConvergenceWarning);probe.fit(xx,truth)
                    predictions[:,col]=probe.predict_proba(tt)[:,1]
                    np.savez_compressed(store/f'{split}_{name}_{col}_seed{seed}_probe.npz',
                        coefficient=probe.coef_,intercept=probe.intercept_,mean=scaler.mean_,scale=scaler.scale_)
                    convergence.append(dict(split=split,seed=seed,branch=name,head=LABEL_TO_HEAD[col],train_n=len(truth),
                        iterations=int(probe.n_iter_[0]),convergence_warning=any(issubclass(w.category,ConvergenceWarning) for w in caught)))
                ensemble[name].append(predictions)
                np.savez_compressed(store/f'{split}_{name}_seed{seed}.npz',idx=test,predictions=predictions)
            print(split,seed,'probes complete',flush=True)
        rows=[]
        for col in CLS_COLS:
            observed=np.isfinite(labels[test,col]);pred={'final':final['predictions'][observed,col]}
            pred.update({'probe_'+k:np.mean(v,0)[observed,col] for k,v in ensemble.items()})
            rows.extend(endpoint_rows(labels[test[observed],col],pred,head=LABEL_TO_HEAD[col],split=split,cohort='learned_branch_linear_probes'))
        write(out/'tables'/f'phase4_linear_probes_{split}.csv',rows)
    write(out/'tables/phase4_branch_rank_both.csv',geometry);write(out/'tables/phase4_branch_cka_both.csv',manifest)
    write(out/'tables/phase4_probe_convergence_both.csv',convergence)
    (out/'notes/phase4_representation_protocol.json').write_text(json.dumps(dict(
        probe='Independent train-only standardization and balanced L2 logistic regression C1/max_iter1000 per learned branch, head and seed; mean test probabilities',
        geometry='Uniform4096 test molecules, seed20260428; each fitted initialization reported separately',
        effective_rank='exp(entropy of normalized singular values of centered branch matrix)',
        participation_ratio='Squared sum of covariance eigenvalues divided by sum of squared eigenvalues',
        cka='Centered linear CKA; descriptive similarity does not establish predictive redundancy',
        query_analysis='Inapplicable to final concat: no query tokens or channel attention contexts'),indent=2))


if __name__=='__main__':main()
