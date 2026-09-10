"""Pair models on identical molecules before/after comparator-specific overlap removal."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.revision_subset_metrics import endpoint_rows
from train.train_revision import read_labels,LABEL_TO_HEAD
from train.readout_contract import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--final-root',type=Path,default=Path('runs_revision_v3/final'))
    p.add_argument('--out',type=Path,default=Path('reports/paper_revision/tables'));a=p.parse_args()
    paths={'ctoxpred2':Path('data/comparators/revision_ctox/predictions.csv'),
           'cardiogenai':Path('data/comparators/revision_cga/predictions.csv')}
    frames={k:pd.read_csv(v).set_index('row_idx',verify_integrity=True) for k,v in paths.items()}
    overlap_path=Path('data/comparators/revision_overlap/overlap.csv');overlap=pd.read_csv(overlap_path)
    labels=read_labels(Path('data/labels/labels_v3.csv'),334444)
    compounds=pd.read_csv('data/compounds/compounds.csv').set_index('row_idx')
    coverage=[];input_hashes={str(v):sha(v) for v in paths.values()};input_hashes[str(overlap_path)]=sha(overlap_path)
    for split in ['tan70_v1_1','tan60_v1_1']:
        path=a.final_root/split/'ensemble_test_revision.npz';f=np.load(path);idx=f['idx'];rows=[]
        input_hashes[str(path)]=sha(path)
        np.testing.assert_array_equal(f['labels'],labels[idx]);assert np.isfinite(f['predictions']).all()
        for channel,column in [('herg',1),('nav15',4),('cav12',6)]:
            observed=np.isfinite(labels[idx,column]);ids=idx[observed];y=labels[ids,column]
            pred={'final':f['predictions'][observed,column]}
            evidence=overlap.loc[overlap['head'].eq(channel)].set_index('row_idx',verify_integrity=True).loc[ids]
            clean={};valid={}
            for tag,frame in frames.items():
                selected=frame.loc[ids]
                assert np.array_equal(selected.inchikey.to_numpy(),compounds.loc[ids,'inchikey'].to_numpy())
                pred[tag]=selected[channel+'_prob'].to_numpy()
                valid[tag]=np.isfinite(pred[tag])
                fp=evidence[tag+'_fp99'].to_numpy(dtype=bool);key=evidence[tag+'_standardized_key_overlap'].to_numpy(dtype=bool)
                clean[tag]={'fp99':~fp,'identity':~(fp|key)}
                coverage.append(dict(split=split,head=LABEL_TO_HEAD[column],model=tag,labelled_test_n=len(ids),
                    predictions_valid_n=int(valid[tag].sum()),fp99_overlap_n=int(fp.sum()),standardized_identity_overlap_n=int(key.sum()),
                    identity_only_extra_n=int((key&~fp).sum()),clean_fp99_n=int((valid[tag]&~fp).sum()),
                    clean_identity_n=int((valid[tag]&~(fp|key)).sum())))
            common=np.logical_and.reduce(list(valid.values()))
            cohorts={'all_models_full':(common,list(pred)),
                'all_models_fp99_clean':(common&clean['ctoxpred2']['fp99']&clean['cardiogenai']['fp99'],list(pred)),
                'all_models_identity_clean':(common&clean['ctoxpred2']['identity']&clean['cardiogenai']['identity'],list(pred))}
            for tag in frames:
                cohorts[tag+'_pair_full']=(valid[tag],['final',tag])
                for policy in ['fp99','identity']:cohorts[tag+'_pair_'+policy+'_clean']=(valid[tag]&clean[tag][policy],['final',tag])
            for name,(mask,models) in cohorts.items():
                rows.extend(endpoint_rows(y[mask],{m:pred[m][mask] for m in models},head=LABEL_TO_HEAD[column],split=split,cohort=name))
            print(split,channel,'common',int(common.sum()),flush=True)
        destination=a.out/f'phase4_comparators_{split}.csv'
        with destination.open('w',newline='') as out:
            w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (a.out/'phase4_comparator_coverage_both.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
    (a.out/'phase4_comparator_protocol.json').write_text(json.dumps(dict(
        bootstrap=1000,seed=20260428,threshold=.5,auprc='Average precision',input_sha256=input_hashes,
        masks='Primary historical Morgan2048/r2 Tanimoto>=0.99; separate standardized-identity union sensitivity',
        pairing='Common valid, labelled molecules within each cohort; comparator-specific and common-union clean sets both reported',
        delta='Comparator minus final on same molecules; filtering changes cohort so absolute before/after difference is not a model effect',
        limitation='Conditional fitted-model molecule bootstrap; no multiplicity correction or training/pretraining uncertainty',
        comparator_labels='CToxPred2 pretrained DNN with100 MC-dropout passes; CardioGenAI graph+fingerprint+transformer classifier'),indent=2))


if __name__=='__main__':main()
