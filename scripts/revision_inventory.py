"""Freeze the chosen architecture and audit released compound/label/split identities."""
import csv
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from model.revision import build_revision_model as build_model

OUT = Path('reports/paper_revision')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--labels',type=Path,default=Path('data/labels/labels_v3.csv'));args=parser.parse_args()
    compound_path=Path('data/compounds/compounds.csv')
    label_path=args.labels
    compounds=list(csv.DictReader(compound_path.open()))
    labels=list(csv.DictReader(label_path.open()))
    assert len(compounds)==len(labels)==334444
    for i,(c,y) in enumerate(zip(compounds,labels)):
        assert int(c['row_idx'])==int(y['row_idx'])==i and c['inchikey']==y['inchikey']
    assert len(set(c['inchikey'] for c in compounds))==len(compounds)
    assert len(set(c['smiles'] for c in compounds))==len(compounds)
    channels={'hERG':['herg_pchembl','herg_blocker_10um','herg_blocker_1um'],
              'NaV1.5':['nav15_pchembl','nav15_blocker'],
              'CaV1.2':['cav12_pchembl','cav12_blocker'],'IKs':['iks_blocker']}
    counts=[];coverage=[];split_records=[]
    for split in ['tan70','tan60']:
        path=Path(f'data/splits/{split}_v1_1.csv')
        raw=list(csv.DictReader(path.open()))
        assert len(raw)==len(compounds)
        assert len(set(int(r['row_idx']) for r in raw))==len(compounds)
        by_idx={int(r['row_idx']):r for r in raw}
        key_folds={}
        for i,c in enumerate(compounds):
            r=by_idx[i];assert r['inchikey']==c['inchikey']
            key_folds.setdefault(c['inchikey'],set()).add(r['fold'])
        assert all(len(v)==1 for v in key_folds.values())
        original={int(r['row_idx']):r['fold'] for r in csv.DictReader(Path(f'data/splits/{split}.csv').open())}
        changes=[dict(row_idx=i,before=original[i],after=r['fold']) for i,r in by_idx.items() if original[i]!=r['fold']]
        split_records.append(dict(split=split,version='v1_1',sha256=sha(path),
            fold_counts={f:sum(r['fold']==f for r in raw) for f in ['train','val','test','excluded']},
            cross_partition_duplicate_inchikeys=sum(len(v)>1 for v in key_folds.values()),
            changes_from_original=changes, note='Identity audit; does not independently recompute every Tanimoto pair.'))
        for fold in ['train','val','test']:
            indices=[i for i in by_idx if by_idx[i]['fold']==fold]
            for channel,heads in channels.items():
                coverage.append(dict(label_version=label_path.stem,split=split,fold=fold,channel=channel,
                    n_unique_labelled_compounds=sum(any(labels[i][h]!='' for h in heads) for i in indices)))
                for head in heads:
                    if 'pchembl' in head:continue
                    values=[int(float(labels[i][head])) for i in indices if labels[i][head]!='']
                    assert all(v in (0,1) for v in values)
                    n=len(values);positive=sum(values)
                    counts.append(dict(label_version=label_path.stem,split=split,fold=fold,head=head,n=n,n_positive=positive,
                        n_negative=n-positive,prevalence=positive/n if n else None,
                        modelled=True))
    write_csv(OUT/'tables/phase2_class_balance_both.csv',counts)
    write_csv(OUT/'tables/phase2_channel_coverage_both.csv',coverage)
    model=build_model(encoder_dim=583).eval()
    ck=torch.load('runs_phase10_mtr_predictions/cls_predictions/seed_42/model.pt',map_location='cpu',weights_only=False)
    missing,unexpected=model.load_state_dict(ck['model_state_dict'],strict=False)
    assert not unexpected and all(n.startswith('heads.iks_blocker.') for n in missing)
    shapes=[]
    def hook(name):
        def record(module, inputs, output):
            shapes.append(dict(layer=name,type=type(module).__name__,input=str(tuple(inputs[0].shape)),
                output=str(tuple(output.shape)),parameters=sum(p.numel() for p in module.parameters(recurse=False)),
                dropout=getattr(module,'p','')))
        return record
    handles=[module.register_forward_hook(hook(name)) for name,module in model.named_modules() if name and not list(module.children())]
    with torch.no_grad():model(torch.zeros(2,6747))
    for handle in handles:handle.remove()
    # CachedConcat computes auxiliary embeddings as well; report each shared module once.
    unique={r['layer']:r for r in shapes}
    assert sum(r['parameters'] for r in unique.values())==sum(p.numel() for p in model.parameters())==3820680
    write_csv(OUT/'tables/phase4_architecture_layers.csv',list(unique.values()))
    lock=dict(decision='User-selected final architecture; not selected by significant superiority',
        input_dim=6747,chemical_dim=6164,first_token_dim=384,mtr_predictions_dim=199,
        chemberta_branch_input_dim=583,concat_dim=448,trainable_parameters=3820680,
        attention='none',layer2=False,l1000=False,heads=list(model.heads),
        seeds=list(range(42,47)),labels_file=str(label_path),labels_sha256=sha(label_path),compounds_sha256=sha(compound_path),
        splits=split_records,seven_output_reference_checkpoint_hashes={str(p):sha(p) for p in Path('runs_phase10_mtr_predictions/cls_predictions').glob('seed_*/model.pt')},
        source_hashes={str(p):sha(p) for d in ['model','train'] for p in Path(d).glob('*.py')},
        status='Architecture fixed; revision training/analyses and manuscript handback in progress',
        ensemble_for_revision='mean sigmoid probability; mean inverse-scaled regression',
        prior_experiments='averaged logits for classification; recompute before reporting revision metrics')
    (OUT/'architecture_lock.json').write_text(json.dumps(lock,indent=2))
    (OUT/'notes/phase2_identity_audit.json').write_text(json.dumps(split_records,indent=2))
    print(json.dumps(dict(rows=len(compounds),parameters=lock['trainable_parameters'],splits=split_records),indent=2))


if __name__=='__main__':main()
