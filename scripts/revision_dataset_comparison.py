"""Compare dataset sizes without treating unlike endpoint collections as equivalent."""
import csv
from pathlib import Path
import pandas as pd


def main():
    rows=[];labels=pd.read_csv('data/labels/labels_v3.csv');ct=pd.read_csv('data/curation/revision/ctoxpred_records.csv')
    for h,display,cols,binary in [('herg','hERG',['herg_pchembl','herg_blocker_10um','herg_blocker_1um'],'herg_blocker_10um'),
        ('nav15','NaV1.5',['nav15_pchembl','nav15_blocker'],'nav15_blocker'),('cav12','CaV1.2',['cav12_pchembl','cav12_blocker'],'cav12_blocker'),
        ('iks','IKs',['iks_blocker'],'iks_blocker')]:
        rows.append(dict(dataset='CardioSafe revision v3',channel=display,n=int(labels[cols].notna().any(axis=1).sum()),
            n_binary=int(labels[binary].notna().sum()),count_semantics='Unique standardized compounds with any channel label',
            policy='Mixed potency/binding proxies, informative censoring and screen votes; targeted EC50 exclusions',source='data/labels/MANIFEST_v3.json'))
        if h!='iks':
            d=ct[(ct.channel==h)&~ct.source_file.str.contains('_60.csv',regex=False)]
            rows.append(dict(dataset='CToxPred cached development + Eval70',channel=display,n=d.standardized_inchikey.nunique(),n_binary='',
                count_semantics=f'Our standardization of {len(d)} curated export rows; no Eval60 double counting',
                policy='Compound-level pIC50 summaries; exact individual assay metadata unavailable',source='https://github.com/issararab/CToxPred/tree/main/data/raw'))
    for h,n in [('hERG',22720),('NaV1.5',2209),('CaV1.2',883)]:
        rows.append(dict(dataset='CardioGenAI published train+validation+test',channel=h,n=n,n_binary='',
            count_semantics='Sum of published split datapoint counts; not re-standardized here',
            policy='pIC50 collection with aggregation/filtering and 10uM classification threshold',source='https://pmc.ncbi.nlm.nih.gov/articles/PMC11881490/'))
    rows.append(dict(dataset='hERGCentral recovered export',channel='hERG',n=306671,n_binary='',
        count_semantics='306893 screen rows after our identity standardization',policy='Primary electrophysiology screen; not an IC50-only potency collection',
        source='https://pmc.ncbi.nlm.nih.gov/articles/PMC3232635/'))
    out=Path('reports/paper_revision/tables/phase2_published_dataset_comparison.csv')
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


if __name__=='__main__':main()
