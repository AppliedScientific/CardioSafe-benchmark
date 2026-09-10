"""Match released splits to archived source artifacts without changing assignments."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np


def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    asi=Path(__file__).resolve().parents[2]/'asi';cache=asi/'data/ion-channel-cache'
    source=asi/'services/bio-intelligence/src/service_bio_intelligence/training'
    sys.path.insert(0,str(asi/'services/bio-intelligence/src'))
    from service_bio_intelligence.training.build_tan_splits_v2 import build_split_v2
    with open('data/compounds/compounds.csv') as f:compounds=list(csv.DictReader(f))
    smiles=json.loads((cache/'smiles.json').read_text())
    assert smiles==[r['smiles'] for r in compounds]
    loo=np.load(cache/'max_sim_loo.npy');assert len(loo)==len(smiles)
    forced=[int(r['row_idx']) for r in compounds if r['inchikey'][:14] in ('RWTNPBWLLIMQHL','GUGOEEXESWIERI')]
    results={}
    for split,cutoff in [('tan70',.7),('tan60',.6)]:
        old=np.load(cache/f'splits_{split}_v2.npz')
        with open(f'data/splits/{split}.csv') as f:released=list(csv.DictReader(f))
        with open(f'data/splits/{split}_v1_1.csv') as f:current=list(csv.DictReader(f))
        old_folds={k:np.array([int(r['row_idx']) for r in released if r['fold']==k]) for k in ['train','val','test']}
        final_folds={k:np.array([int(r['row_idx']) for r in current if r['fold']==k]) for k in ['train','val','test']}
        for k in old_folds:assert np.array_equal(np.sort(old[k]),np.sort(old_folds[k]))
        rebuilt=build_split_v2(loo,cutoff,forced)
        assert all(np.array_equal(x,old[k]) for x,k in zip(rebuilt,['train','val','test']))
        unassigned=sorted(set(range(len(smiles)))-set(np.concatenate(rebuilt).tolist()))
        original_eval=np.union1d(old['val'],old['test'])
        outliers=original_eval[loo[original_eval]>=cutoff]
        final_eval=np.union1d(final_folds['val'],final_folds['test'])
        final_outliers=final_eval[loo[final_eval]>=cutoff]
        results[split]=dict(source_split=str(cache/f'splits_{split}_v2.npz'),
            source_sha256=sha(cache/f'splits_{split}_v2.npz'),released_equals_source_all_folds=True,
            source_algorithm_reproduces_assignments_exactly=True,
            unassigned_rows=[dict(compounds[i],max_sim_loo=float(loo[i])) for i in unassigned],
            unassigned_explanation='Odd augmented evaluation-pool remainder is omitted by two equal-size half slices in build_split_v2; not an unsatisfiable similarity conflict.',
            original_counts={k:len(v) for k,v in old_folds.items()},final_counts={k:len(v) for k,v in final_folds.items()},
            original_unassigned=len(smiles)-sum(map(len,old_folds.values())),
            final_unassigned=len(smiles)-sum(map(len,final_folds.values())),
            original_eval_above_global_loo_cutoff=[dict(compounds[i],max_sim_loo=float(loo[i])) for i in outliers],
            final_eval_above_global_loo_cutoff=[dict(compounds[i],max_sim_loo=float(loo[i])) for i in final_outliers])
    report=dict(compounds=len(smiles),source_smiles_exact_order_match=True,
        source_smiles_sha256=sha(cache/'smiles.json'),max_sim_loo_sha256=sha(cache/'max_sim_loo.npy'),
        source_code_sha256={n:sha(source/n) for n in ['build_tan_splits.py','build_tan_splits_v2.py','build_tan_splits_per_head.py']},
        algorithm='Global leave-one-out Morgan Tanimoto novelty pool, deterministic 50/50 evaluation split, forced validation compounds; later v1.1 train-to-val corrections.',
        unsupported_claim='753 removed by reconciliation of per-channel components: no corresponding removal in the released tan70 assignment, which assigns all 334444 compounds.',
        caution='Stored LOO values establish source provenance, not an independent recomputation of every similarity. Forced validation compounds are exceptions to the LOO novelty rule; direct train similarity must be audited separately.',
        splits=results)
    Path('reports/paper_revision/notes/phase2_split_provenance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:{f:v for f,v in r.items() if 'outliers' not in f and 'above' not in f} for k,r in results.items()},indent=2))


if __name__=='__main__':main()
