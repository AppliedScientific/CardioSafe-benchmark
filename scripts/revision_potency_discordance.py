"""Describe independently pooled binary/potency agreement, without relabelling."""
from pathlib import Path
import numpy as np
import pandas as pd
from train.readout_contract import sha
import json


def main():
    rows=[];details=[];hashes={}
    for split in ['tan70_v1_1','tan60_v1_1']:
        path=Path('runs_revision_v3/final')/split/'ensemble_test_revision.npz';f=np.load(path);y=f['labels'];hashes[str(path)]=sha(path)
        for cls,reg,threshold,head in [(1,0,5,'herg_blocker_10um'),(2,0,6,'herg_blocker_1um'),(4,3,5,'nav15_blocker'),(6,5,5,'cav12_blocker')]:
            observed=np.isfinite(y[:,cls])&np.isfinite(y[:,reg]);discordant=observed&((y[:,reg]>threshold)!=(y[:,cls]>.5));exact=observed&(y[:,reg]==threshold)
            rows.append(dict(model='final',head=head,split=split,n=int(observed.sum()),metric='binary_vs_potency_threshold_discordance',estimate=float(discordant.sum()/observed.sum()),ci_lower=None,ci_upper=None,discordant_n=int(discordant.sum()),exact_threshold_n=int(exact.sum()),exact_threshold_binary_positive_n=int((exact&(y[:,cls]>.5)).sum()),threshold_pchembl=threshold))
            for i in np.flatnonzero(discordant):details.append(dict(split=split,row_idx=int(f['idx'][i]),head=head,binary_label=int(y[i,cls]),pooled_pchembl=float(y[i,reg]),threshold_pchembl=threshold))
    out=Path('reports/paper_revision/tables');pd.DataFrame(rows).to_csv(out/'phase4_potency_label_discordance_both.csv',index=False);pd.DataFrame(details).to_csv(out/'phase4_potency_label_discordant_rows_both.csv',index=False)
    (out/'phase4_potency_label_discordance_both.json').write_text(json.dumps(dict(input_sha256=hashes,note='Descriptive census of fixed test labels; no model-performance CI. Binary majority votes can include censored/percent-inhibition or screen assays; pooled exact-potency medians are a different summary. Discordance does not automatically establish a curation error.'),indent=2))


if __name__=='__main__':main()
