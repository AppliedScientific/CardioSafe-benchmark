"""Final baseline matrix and IC50 sensitivity scored against explicit common truths."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.revision_metrics import summarize
from scripts.revision_subset_metrics import endpoint_rows
from train.revision_supervision import CHANNEL_COLUMNS
from train.train_revision import LABEL_TO_HEAD,REG_COLS
from train.readout_contract import sha

ROOTS={'chemical':'runs_revision_dense/chemical','static':'runs_revision_dense/static',
       'herg_10':'runs_revision_dense/herg_10','herg_25':'runs_revision_dense/herg_25','herg_50':'runs_revision_dense/herg_50',
       'molformer':'runs_revision_encoder/molformer','rf':'runs_revision_trees/rf','xgb':'runs_revision_trees/xgb',
       'dmpnn':'runs_revision_neural/dmpnn','finetune':'runs_revision_neural/finetune','frozen_matched':'runs_revision_neural/frozen_matched'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--available-only',action='store_true');a=parser.parse_args();outputs=Path('reports/paper_revision/tables');missing=[];hashes={}
    for split in ['tan70_v1_1','tan60_v1_1']:
        arms={'final':Path('runs_revision_v3/final')/split}
        for name,path in ROOTS.items():
            root=Path(path)/split;file=root/'ensemble_test_revision.npz'
            if file.exists():arms[name]=root
            else:missing.append(str(file))
        final=np.load(arms['final']/'ensemble_test_revision.npz');channel_files={}
        for channel in CHANNEL_COLUMNS:
            path=Path('runs_revision_dense' if channel=='herg' else 'runs_revision_sparse')/('single_'+channel)/split/'ensemble_test_revision.npz'
            if path.exists():channel_files[channel]=path
            else:missing.append(str(path))
        if len(channel_files)==4:
            pred=np.full_like(final['predictions'],np.nan)
            for channel,path in channel_files.items():
                f=np.load(path);np.testing.assert_array_equal(f['idx'],final['idx']);np.testing.assert_array_equal(f['labels'],final['labels']);pred[:,CHANNEL_COLUMNS[channel]]=f['predictions'][:,CHANNEL_COLUMNS[channel]];hashes[str(path)]=sha(path)
            assert np.isfinite(pred).all();root=Path('runs_revision_single_combined')/split;root.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(root/'ensemble_test_revision.npz',idx=final['idx'],labels=final['labels'],predictions=pred);arms['single_channel']=root
            (root/'ensemble_test_revision.json').write_text(json.dumps(dict(method='Four independent channel ensembles, each with five seeds; no learned parameters shared across channels',sources={k:str(v) for k,v in channel_files.items()}),indent=2))
        ic50_path=Path('runs_revision_sparse/ic50')/split/'ensemble_test_revision.npz'
        if not ic50_path.exists():missing.append(str(ic50_path))
        if missing and not a.available_only:raise FileNotFoundError('Required baseline outputs missing: '+', '.join(missing))
        table=summarize(arms,split,'test','final');pd.DataFrame(table).to_csv(outputs/f'phase3_baselines_{split}.csv',index=False)
        if 'finetune' in arms and 'frozen_matched' in arms:
            matched=summarize({name:arms[name] for name in ['frozen_matched','finetune']},split,'test','frozen_matched')
            pd.DataFrame(matched).to_csv(outputs/f'phase3_adaptation_matched_{split}.csv',index=False)
        for root in arms.values():path=root/'ensemble_test_revision.npz';hashes[str(path)]=sha(path)
        if ic50_path.exists():
            strict=np.load(ic50_path);np.testing.assert_array_equal(strict['idx'],final['idx']);hashes[str(ic50_path)]=sha(ic50_path);rows=[]
            for truth_name,truth in [('v3_all_labels',final['labels']),('strict_ic50_labels',strict['labels'])]:
                for col,head in LABEL_TO_HEAD.items():
                    mask=np.isfinite(truth[:,col]);rows.extend(endpoint_rows(truth[mask,col],{'final':final['predictions'][mask,col],'ic50_only':strict['predictions'][mask,col]},head=head,split=split,cohort=truth_name,regression=col in REG_COLS))
            pd.DataFrame(rows).to_csv(outputs/f'phase3_ic50_sensitivity_{split}.csv',index=False)
        print(split,'scored',list(arms),flush=True)
    (outputs/'phase3_baseline_protocol.json').write_text(json.dumps(dict(input_sha256=hashes,missing_outputs=missing,complete=not missing,bootstrap=1000,seed=20260428,
        comparison='Model minus final; higher classification metrics/correlations are better; lower errors are better',
        single_channel='Four channel models, not one shared multitask model',
        adaptation='Fine-tuned versus frozen_matched uses direct paired differences with frozen_matched as reference, matching the neural batching protocol; separate from comparisons against locked final.',
        ic50='Both models evaluated against the SAME chosen target matrix on the SAME molecules; v3 and strict IC50 reference targets reported as separate cohorts. Do not compare metric values across reference cohorts as a model delta.',
        partial_reporting=a.available_only),indent=2))


if __name__=='__main__':main()
