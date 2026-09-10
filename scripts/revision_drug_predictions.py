"""Score the corrected panel and retain every initialization's prediction."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.revision_model_io import load_final
from train.train_revision import LABEL_TO_HEAD,REG_COLS
from train.readout_contract import sha


@torch.inference_mode()
def main():
    torch.set_num_threads(3);panel=pd.read_csv('reports/paper_revision/tables/phase4_drug_panel_identity.csv')
    records=[];hashes={}
    for split,group in panel.groupby('split',sort=False):
        rows=group.row_idx.to_numpy();predictions=[]
        for seed in range(42,47):
            root=Path('runs_revision_v3/final')/split/f'seed_{seed}'
            model,provider,ck=load_final(root);values=model.predict(torch.from_numpy(provider[rows]))
            predictions.append(np.stack([values[h].numpy() for h in LABEL_TO_HEAD.values()],1))
            hashes[str(root/'model.pt')]=sha(root/'model.pt')
        stack=np.stack(predictions)
        for i,r in enumerate(group.to_dict('records')):
            for col,head in LABEL_TO_HEAD.items():
                observed=r[head+'_observed'];record=dict(compound=r['compound'],row_idx=r['row_idx'],inchikey=r['inchikey'],
                    split=split,fold=r['fold'],head=head,observed=observed if np.isfinite(observed) else None,
                    prediction=float(stack[:,i,col].mean()),seed_sd=float(stack[:,i,col].std(ddof=1)),
                    unit='pChEMBL' if col in REG_COLS else 'sigmoid blocker score (uncalibrated)',
                    evaluation_role=r['evaluation_role'],identity_source=r['identity_source'])
                for j,seed in enumerate(range(42,47)):record['seed_'+str(seed)]=float(stack[j,i,col])
                records.append(record)
    out=Path('reports/paper_revision/tables/phase4_drug_predictions_both.csv')
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    out.with_suffix('.json').write_text(json.dumps(dict(checkpoint_sha256=hashes,
        seed_sd='Descriptive variability across five initialization fits, not a calibrated prediction interval',
        limitation='No test-set drugs in this named panel; no clinical safety labels or external validation claim'),indent=2))


if __name__=='__main__':main()
