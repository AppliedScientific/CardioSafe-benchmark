"""Final-model similarity, measured-potency and operational-error bands."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scripts.revision_subset_metrics import endpoint_rows
from train.train_revision import LABEL_TO_HEAD,REG_COLS
from train.readout_contract import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs_revision_v3/final'));a=p.parse_args()
    out=Path('reports/paper_revision/tables');hashes={};errors=[]
    for split,cache in [('tan70_v1_1','test_train_tanimoto'),('tan60_v1_1','revision_tan60_tanimoto')]:
        fpath=a.root/split/'ensemble_test_revision.npz';spath=Path('data/features')/cache/'nearest_train.npz'
        f=np.load(fpath);s=np.load(spath);assert np.array_equal(f['idx'],s['idx'])
        hashes[str(fpath)]=sha(fpath);hashes[str(spath)]=sha(spath)
        y=f['labels'];pred=f['predictions'];sim=s['similarity'];rows=[]
        cohorts={'similarity_0.0_0.3':sim<.3,'similarity_0.3_0.5':(sim>=.3)&(sim<.5),
                 'similarity_0.5_0.7':(sim>=.5)&(sim<=.7)}
        for col,head in LABEL_TO_HEAD.items():
            observed=np.isfinite(y[:,col])
            for name,mask in cohorts.items():
                select=observed&mask
                rows.extend(endpoint_rows(y[select,col],{'final':pred[select,col]},head=head,split=split,
                    cohort=name,regression=col in REG_COLS))
            if col in REG_COLS:continue
            false_negative=observed&(y[:,col]>.5)&(pred[:,col]<.5)
            false_positive=observed&(y[:,col]<.5)&(pred[:,col]>=.5)
            regcol={1:0,2:0,4:3,6:5}.get(col)
            if regcol is None:continue
            potency=y[:,regcol];known=observed&np.isfinite(potency)
            potency_bands={'pchembl_lt4.5':potency<4.5,'pchembl_4.5_5':(potency>=4.5)&(potency<5),
                'pchembl_5_6':(potency>=5)&(potency<=6),'pchembl_6_6.5':(potency>6)&(potency<=6.5),'pchembl_gt6.5':potency>6.5}
            for name,mask in potency_bands.items():
                select=known&mask;n=int(select.sum())
                rows.extend(endpoint_rows(y[select,col],{'final':pred[select,col]},head=head,split=split,cohort=name))
                rows.extend(endpoint_rows(y[select,regcol],{'final':pred[select,regcol]},head=LABEL_TO_HEAD[regcol],split=split,
                    cohort=name+'_paired_to_'+head,regression=True))
                errors.append(dict(split=split,head=head,band=name,n=n,positive_n=int((y[select,col]>.5).sum()),
                    false_negative_n=int((select&false_negative).sum()),false_positive_n=int((select&false_positive).sum()),
                    all_labelled_n=int(observed.sum()),all_classification_errors_n=int((false_negative|false_positive).sum()),
                    errors_with_observed_potency_n=int(((false_negative|false_positive)&known).sum()),
                    caveat='Potency-band inference applies only to molecules with measured potency; screen-only labels cannot be assigned a pChEMBL band'))
        with (out/f'phase4_domain_{split}.csv').open('w',newline='') as file:
            w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        print(split,'complete',flush=True)
    with (out/'phase4_potency_error_counts_both.csv').open('w',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(errors[0]));w.writeheader();w.writerows(errors)
    (out/'phase4_domain_protocol.json').write_text(json.dumps(dict(bootstrap=1000,seed=20260428,input_sha256=hashes,
        similarity='Exact max Morgan2048/radius2 Tanimoto to global TRAIN; no comparator-pool substitution',
        potency='Observed assay potency, not predicted potency; 5<=pChEMBL<=6 corresponds to1–10 micromolar',
        caveat='Conditional exploratory subgroup analyses; no multiplicity correction; n<5 not scored'),indent=2))


if __name__=='__main__':main()
