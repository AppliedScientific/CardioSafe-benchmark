"""Score completed sparse-channel baselines and explicitly aligned IC50 cohorts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scripts.revision_subset_metrics import endpoint_rows
from train.train_revision import LABEL_TO_HEAD,REG_COLS
from train.readout_contract import sha


def main():
    root=Path('reports/paper_revision');audit=json.loads((root/'baseline_inference_audit_sparse.json').read_text());assert audit['status']=='passed'
    transfer=[];sensitivity=[];hashes={}
    for split in ['tan70_v1_1','tan60_v1_1']:
        path=Path('runs_revision_v3/final')/split/'ensemble_test_revision.npz';final=np.load(path);hashes[str(path)]=sha(path)
        for channel,cols in [('nav15',(3,4)),('cav12',(5,6)),('iks',(7,))]:
            path=Path('runs_revision_sparse')/('single_'+channel)/split/'ensemble_test_revision.npz';single=np.load(path);hashes[str(path)]=sha(path)
            np.testing.assert_array_equal(single['idx'],final['idx']);np.testing.assert_array_equal(single['labels'],final['labels'])
            for col in cols:
                mask=np.isfinite(final['labels'][:,col]);transfer.extend(endpoint_rows(final['labels'][mask,col],{'single_channel':single['predictions'][mask,col],'final':final['predictions'][mask,col]},head=LABEL_TO_HEAD[col],split=split,cohort='v3_observed_'+channel,regression=col in REG_COLS,reference='single_channel'))
        path=Path('runs_revision_sparse/ic50')/split/'ensemble_test_revision.npz';strict=np.load(path);hashes[str(path)]=sha(path)
        np.testing.assert_array_equal(strict['idx'],final['idx'])
        for name,truth in [('v3_all_labels',final['labels']),('strict_ic50_labels',strict['labels'])]:
            for col,head in LABEL_TO_HEAD.items():
                mask=np.isfinite(truth[:,col]);sensitivity.extend(endpoint_rows(truth[mask,col],{'final':final['predictions'][mask,col],'ic50_only':strict['predictions'][mask,col]},head=head,split=split,cohort=name,regression=col in REG_COLS))
    pd.DataFrame(transfer).to_csv(root/'tables/phase3_sparse_transfer_both.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(root/'tables/phase3_sparse_ic50_sensitivity_both.csv',index=False)
    (root/'tables/phase3_sparse_comparison_protocol.json').write_text(json.dumps(dict(status='complete_for_sparse_block',source_sha256=hashes,bootstrap=1000,seed=20260428,
        transfer_delta='Final multi-task minus single-channel on identical observed TEST molecules',
        sensitivity_delta='IC50-only minus final, against the SAME target matrix within each named reference cohort',
        limitations='Conditional fixed-ensemble intervals; unadjusted multiplicity. Single-channel optimization-step counts differ. IC50 restriction changes coverage. IKs has one TEST positive per split.'),indent=2))
    print('Sparse comparisons complete',len(transfer),len(sensitivity),flush=True)

if __name__=='__main__':main()
