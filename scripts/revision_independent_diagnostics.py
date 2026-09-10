"""Separate label-permutation variability from conditional molecule uncertainty."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from scripts.revision_metrics import EndpointMetrics
from train.train_revision import read_labels,read_split,LABEL_TO_HEAD,CLS_COLS,REG_COLS
from train.readout_contract import sha


def main():
    real=read_labels(Path('data/labels/labels_v3.csv'),334444);records=[];geometry=[];hashes={}
    for split in ['tan70_v1_1','tan60_v1_1']:
        folds=read_split(Path('data/splits')/(split+'.csv'))
        for arm in ['yrandom','yrandom_rows']:
            for seed in range(42,47):
                run=Path('runs_revision_independent')/arm/split/f'seed_{seed}'
                provenance=json.loads((run/'provenance.json').read_text())
                old=json.loads((Path('runs_revision_controls')/split/f'seed_{seed}/provenance.json').read_text())
                assert provenance['initial_parameter_sha256']==old['initial_parameter_sha256']
                for fold in ['val','test']:
                    path=run/f'predictions_{fold}_revision.npz';f=np.load(path);idx=f['idx'];hashes[str(path)]=sha(path);np.testing.assert_array_equal(idx,folds[fold])
                    p=f['raw'].copy()
                    for c in CLS_COLS:p[:,c]=expit(p[:,c])
                    for truth_name,truth in [('real',real[idx]),('independently_shuffled',f['labels'])]:
                        for col,head in LABEL_TO_HEAD.items():
                            mask=np.isfinite(truth[:,col]);n=int(mask.sum())
                            if n<5:continue
                            metrics=EndpointMetrics(truth[mask,col],p[mask,col],col in REG_COLS)()
                            for metric,value in metrics.items():records.append(dict(model=arm,head=head,split=split,fold=fold,seed=seed,permutation_seed=20260429+seed-42,truth=truth_name,metric=metric,estimate=value,n=n,ci_lower=None,ci_upper=None))
                    if fold=='test':
                        initial_path=Path('runs_revision_untrained')/split/f'seed_{seed}/classification_initialization.npz';initial=np.load(initial_path);np.testing.assert_array_equal(initial['idx'],idx);hashes[str(initial_path)]=sha(initial_path)
                        for c in CLS_COLS:
                            mask=np.isfinite(real[idx,c]);a=initial['predictions'][mask,c];b=p[mask,c];y=real[idx[mask],c]
                            geometry.append(dict(model=arm,head=LABEL_TO_HEAD[c],split=split,seed=seed,n=len(a),trained_auc=EndpointMetrics(y,b,False)()['auc_roc'],initial_auc=EndpointMetrics(y,a,False)()['auc_roc'],initial_probability_sd=float(a.std()),trained_probability_sd=float(b.std()),trained_probability_mean=float(b.mean()),initial_trained_spearman=float(spearmanr(a,b).statistic),initial_parameter_hash_verified=True))
    out=Path('reports/paper_revision/tables');d=pd.DataFrame(records);d.to_csv(out/'phase1_independent_per_seed_both.csv',index=False);pd.DataFrame(geometry).to_csv(out/'phase1_independent_initialization_both.csv',index=False)
    grouped=d.groupby(['model','head','split','fold','truth','metric'],dropna=False).estimate.agg(['mean','std','min','max','count']).reset_index()
    grouped.rename(columns={'std':'between_fit_sd','count':'finite_fits'}).to_csv(out/'phase1_independent_between_fit_both.csv',index=False)
    (out/'phase1_independent_diagnostic_protocol.json').write_text(json.dumps(dict(input_sha256=hashes,permutations={str(s):20260429+s-42 for s in range(42,47)},note='Five independent permutations and five paired initialization seeds; these two variation sources are not separately estimated. Mean/SD/range across five fitted controls is descriptive, not a calibrated permutation p-value. Never average differently shuffled target matrices into one truth matrix. Untrained classification arrays were generated with verified Linux initial parameter hashes.'),indent=2))


if __name__=='__main__':main()
