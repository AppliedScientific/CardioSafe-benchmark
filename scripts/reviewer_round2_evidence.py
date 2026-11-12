"""Prepare reproducible round-2 evidence from the locked saved fits; no training.

Run from any checkout:
python scripts/reviewer_round2_evidence.py --source /path/to/full/artifact/checkout --out reports/reviewer_round2

Requires Python 3.11+ and numpy, pandas, scipy, scikit-learn and torch. The source checkout must contain
the companion v3 labels, prediction arrays and descriptor/property caches.
"""
import argparse
import gzip
import hashlib
import io
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
import sklearn
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

HEADS = ['herg_pchembl', 'herg_blocker_10um', 'herg_blocker_1um',
         'nav15_pchembl', 'nav15_blocker', 'cav12_pchembl', 'cav12_blocker', 'iks_blocker']
REG = [0, 3, 5]
CLS = [1, 2, 4, 6, 7]
SEEDS = [42, 43, 44, 45, 46]
DESC = {'molecular_weight': 0, 'clogp': 1, 'tpsa': 2,
        'aromatic_ring_count': 6, 'basic_pka': 14, 'cation_fraction_ph7_4': 16}


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def values(y, p, regression):
    if regression:
        return {'rmse': float(np.sqrt(np.mean((p-y)**2))),
                'mae': float(np.mean(np.abs(p-y)))}
    calls = p >= .5
    pos, neg = y == 1, y == 0
    both = bool(pos.any() and neg.any())
    return {'auroc': float(roc_auc_score(y, p)) if both else np.nan,
            'auprc': float(average_precision_score(y, p)) if pos.any() else np.nan,
            'mcc': float(matthews_corrcoef(y, calls)) if both else np.nan,
            'false_negative_rate': float((~calls[pos]).mean()) if pos.any() else np.nan,
            'false_positive_rate': float(calls[neg].mean()) if neg.any() else np.nan}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.source.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'date': '2026-11-12', 'reviewed_branch_commit': '398731342ded43488c2359d840cf22bad8105918',
                'python': platform.python_version(), 'numpy': np.__version__,
                'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'torch': torch.__version__,
                'input_sha256': {}, 'checks': {}, 'protocol': {
                    'ensemble': 'Mean sigmoid probabilities; mean already inverse-scaled regression predictions.',
                    'stability': 'All 31 nonempty subsets of five fixed fits per split; descriptive ranges, not confidence intervals or independent replicates.',
                    'descriptor_groups': 'Four bins defined by global TRAIN quartiles before test error grouping; ties may produce empty/unequal bins. All six selected descriptors and all endpoints are reported.',
                    'descriptor_inference': 'Exploratory conditional summaries; no hypothesis tests, multiplicity claims, causal claims, or validated domain boundary.',
                    'copied_tables': 'Archived text tables are copied with LF line endings; CSV cells are unchanged.',
                    'data': 'Lossless numeric join of v3 labels to standardized compound identities and both global fold assignments; no relabelling.'}}
    def track(rel):
        p = root / rel
        manifest['input_sha256'][rel] = sha(p)
        return p
    compounds = pd.read_csv(track('data/compounds/compounds.csv'))
    labels = pd.read_csv(track('data/labels/labels_v3.csv'))
    assert len(compounds) == len(labels) == 334444
    assert compounds.row_idx.is_unique and labels.row_idx.is_unique
    assert compounds.inchikey.is_unique
    assert np.array_equal(compounds.row_idx, np.arange(len(compounds)))
    data = compounds.merge(labels, on=['row_idx', 'inchikey'], validate='one_to_one')
    assert len(data) == len(compounds)
    descriptors = np.load(track('data/features/descriptors_20.npy'), mmap_mode='r')
    assert descriptors.shape == (len(data), 20)
    assert np.isfinite(descriptors).all()
    props = np.load(track('data/features/chemberta_mtr_cls_predictions/predictions.npy'), mmap_mode='r')
    assert props.shape == (len(data), 199)
    stability, domain, scaler_rows, checkpoint_rows = [], [], [], []
    for split in ['tan70_v1_1', 'tan60_v1_1']:
        folds = pd.read_csv(track('data/splits/'+split+'.csv'))
        assert folds.row_idx.is_unique
        assert set(folds.fold) == ({'train', 'val', 'test'} if split.startswith('tan70') else {'train', 'val', 'test', 'excluded'})
        data = data.merge(folds.rename(columns={'fold': split}), on=['row_idx','inchikey'], how='left', validate='one_to_one')
        data[split] = data[split].fillna('unassigned')
        train = folds.loc[folds.fold == 'train', 'row_idx'].to_numpy()
        meta = json.loads(track('data/features/revision_'+split+'/manifest.json').read_text())
        mean = np.mean(props[train], axis=0, dtype=np.float64)
        scale = np.maximum(np.std(props[train], axis=0, dtype=np.float64), 1e-6)
        assert meta['fit_fold'] == 'train' and meta['fit_n'] == len(train)
        np.testing.assert_allclose(mean, meta['mean'], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(scale, meta['scale'], rtol=1e-10, atol=1e-10)
        scaler_rows.append({'split': split, 'fit_n': len(train),
                            'mean_max_abs_difference': float(np.max(np.abs(mean-meta['mean']))),
                            'scale_max_abs_difference': float(np.max(np.abs(scale-meta['scale']))),
                            'split_sha256_matches': meta['contract']['split_sha256'] == sha(root/'data/splits'/f'{split}.csv')})
        run = 'runs_revision_v3/final/'+split
        ensemble = np.load(track(run+'/ensemble_test_revision.npz'))
        idx, y, final = ensemble['idx'], ensemble['labels'], ensemble['predictions']
        assert np.array_equal(np.sort(idx), np.sort(folds.loc[folds.fold == 'test','row_idx']))
        np.testing.assert_allclose(y, data.set_index('row_idx').loc[idx, HEADS].to_numpy(), rtol=0, atol=1e-6, equal_nan=True)
        per_seed = []
        train_desc = np.asarray(descriptors[train], dtype=np.float32)
        desc_mean, desc_std = train_desc.mean(0), train_desc.std(0)
        desc_std[desc_std < 1e-8] = 1.0
        train_labels = data.set_index('row_idx').loc[train, HEADS].to_numpy(dtype=np.float32)
        for seed in SEEDS:
            f = np.load(track(run+f'/seed_{seed}/predictions_test_revision.npz'))
            assert np.array_equal(f['idx'], idx)
            np.testing.assert_array_equal(f['labels'], y)
            p = f['raw'].copy()
            p[:, CLS] = expit(p[:, CLS])
            assert np.isfinite(p).all()
            per_seed.append(p)
            checkpoint = torch.load(track(run+f'/seed_{seed}/model.pt'), map_location='cpu', weights_only=False)
            np.testing.assert_allclose(desc_mean, checkpoint['descriptor_scaler_means'], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(desc_std, checkpoint['descriptor_scaler_stds'], rtol=1e-6, atol=1e-6)
            for col in REG:
                v = train_labels[:,col]; v = v[np.isfinite(v)]
                np.testing.assert_allclose([v.mean(), v.std()], checkpoint['reg_scalers'][col], rtol=1e-6, atol=1e-6)
            for col in CLS:
                v = train_labels[:,col]; v = v[np.isfinite(v)]; positive = (v>.5).sum()
                expected = float((len(v)-positive)/positive) if positive else 1.0
                np.testing.assert_allclose(expected, checkpoint['class_weights'][col], rtol=1e-6)
            provenance = json.loads(track(run+f'/seed_{seed}/provenance.json').read_text())
            checkpoint_rows.append({'split':split, 'seed':seed, 'descriptor_scalers_match_train':True,
                'regression_scalers_match_train':True, 'class_weights_match_train':True,
                'labels':checkpoint['labels'], 'descriptor_transform':checkpoint['descriptor_transform'],
                'epoch':checkpoint['epoch'], 'python':provenance['python'], 'torch':provenance['torch'],
                'platform':provenance['platform']})
        per_seed = np.stack(per_seed)
        np.testing.assert_allclose(per_seed.mean(axis=0), final, rtol=1e-6, atol=1e-6)
        manifest['checks'][split] = {'test_rows': len(idx), 'prediction_ensemble_max_abs_difference':
                                     float(np.max(np.abs(per_seed.mean(axis=0)-final))), 'labels_match_v3': True}
        for k in range(1, 6):
            for subset in itertools.combinations(range(5), k):
                p = per_seed[list(subset)].mean(axis=0)
                rmse = {}
                for col, head in enumerate(HEADS):
                    mask = np.isfinite(y[:, col])
                    metrics = values(y[mask,col], p[mask,col], col in REG)
                    if col in REG: rmse[col] = metrics['rmse']
                    for metric, value in metrics.items():
                        stability.append({'split':split,'ensemble_size':k,'seeds':','.join(str(SEEDS[i]) for i in subset),
                                          'head':head,'metric':metric,'value':value,
                                          'n':int(mask.sum()),'n_positive':int((y[mask,col]==1).sum()) if col in CLS else None})
                stability.append({'split':split,'ensemble_size':k,'seeds':','.join(str(SEEDS[i]) for i in subset),
                                  'head':'weighted_regression','metric':'rmse',
                                  'value':.45*rmse[0]+.15*rmse[3]+.40*rmse[5],
                                  'n':int(np.isfinite(y[:,REG]).any(axis=1).sum()),'n_positive':None})
        for name, dc in DESC.items():
            edges = np.quantile(descriptors[train,dc], [.25,.5,.75])
            groups = np.searchsorted(edges, descriptors[idx,dc], side='right')
            for bin_idx in range(4):
                for col, head in enumerate(HEADS):
                    mask = (groups == bin_idx) & np.isfinite(y[:,col])
                    n = int(mask.sum())
                    row = {'split':split,'descriptor':name,'train_q25':edges[0],'train_q50':edges[1],
                           'train_q75':edges[2],'bin':bin_idx+1,'head':head,'n':n,
                           'n_positive':int((y[mask,col]==1).sum()) if col in CLS else None,
                           'status':'exploratory' if n>=5 else 'n<5_not_scored'}
                    if n >= 5:
                        row.update(values(y[mask,col], final[mask,col], col in REG))
                        if col in CLS:
                            calls = final[mask,col]>=.5
                            row['false_negatives']=int(((y[mask,col]==1)&~calls).sum())
                            row['false_positives']=int(((y[mask,col]==0)&calls).sum())
                    domain.append(row)
        print(split, 'verified; seed and descriptor summaries complete', flush=True)
    (out/'checkpoint_scaler_audit.json').write_text(json.dumps({'checks':checkpoint_rows, 'tolerance':'rtol=1e-6, atol=1e-6 for scalers'},indent=2)+'\n')
    seed_df = pd.DataFrame(stability)
    seed_df.to_csv(out/'seed_subsets.csv', index=False)
    summary = seed_df.groupby(['split','head','metric','ensemble_size'], dropna=False).value.agg(['count','mean','std','min','max']).reset_index()
    summary.to_csv(out/'seed_stability_summary.csv', index=False)
    pd.DataFrame(domain).to_csv(out/'descriptor_error_groups.csv', index=False)
    pd.DataFrame(scaler_rows).to_csv(out/'pretrained_scaler_audit.csv', index=False)
    gz = out/'Additional_file_1_cardiosafe_v3.csv.gz'
    with gzip.GzipFile(filename='', mode='wb', fileobj=gz.open('wb'), mtime=0) as binary:
        with io.TextIOWrapper(binary, encoding='utf-8', newline='') as text:
            data.to_csv(text,index=False,na_rep='',float_format='%.17g')
    reread = pd.read_csv(gz, float_precision='round_trip')
    assert list(reread.columns) == list(data.columns)
    pd.testing.assert_frame_equal(reread, data, check_exact=True)
    manifest['checks']['supplement_exact_roundtrip'] = True
    manifest['dataset'] = {'rows':len(data),'label_source_sha256':sha(root/'data/labels/labels_v3.csv'),
                           'columns':list(data.columns),'missing_labels':'empty CSV cell; never a negative label',
                           'counts':{h:{'labelled':int(data[h].notna().sum()),
                                        'positive':int((data[h]==1).sum()) if i in CLS else None} for i,h in enumerate(HEADS)},
                           'fold_counts':{s:{k:int(v) for k,v in data[s].value_counts().items()} for s in ['tan70_v1_1','tan60_v1_1']}}
    for rel in ['reports/paper_revision/tables/phase4_domain_tan70_v1_1.csv',
                'reports/paper_revision/tables/phase4_domain_tan60_v1_1.csv',
                'reports/paper_revision/tables/phase4_potency_error_counts_both.csv',
                'reports/paper_revision/tables/phase4_domain_protocol.json',
                'reports/paper_revision/notes/phase2_split_provenance.json']:
        (out/Path(rel).name).write_text(track(rel).read_text())
    manifest['output_sha256'] = {p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='evidence_manifest.json'}
    manifest['script_sha256'] = sha(Path(__file__))
    (out/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('Finished:', out, 'supplement bytes', gz.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
