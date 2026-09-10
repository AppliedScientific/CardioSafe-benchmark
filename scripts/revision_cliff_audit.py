"""Check every historical cliff compound against the current global partitions."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,default=Path('data/supplementary/note_s2_per_split_manifest.csv'))
    parser.add_argument('--tag',default='historical');args=parser.parse_args()
    RDLogger.DisableLog('rdApp.*')
    path=args.manifest
    with path.open() as f:cliffs=list(csv.DictReader(f))
    with open('data/compounds/compounds.csv') as f:compounds=list(csv.DictReader(f))
    keys={r['inchikey']:int(r['row_idx']) for r in compounds}
    source=Path('data/features/morgan_fps_2048.npy');fps=np.load(source,mmap_mode='r')
    packed=np.packbits(fps.astype(np.uint8),axis=1,bitorder='little')
    bits=[DataStructs.CreateFromBinaryText(r.tobytes()) for r in packed]
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    for i in np.random.default_rng(20260428).choice(len(bits),128,replace=False):
        actual=generator.GetFingerprint(Chem.MolFromSmiles(compounds[i]['smiles']))
        assert actual==bits[i],f'Fingerprint cache mismatch at {i}'
    salt=SaltRemover();uncharger=rdMolStandardize.Uncharger();records=[]
    for split,cutoff in [('tan70_v1_1',.7),('tan60_v1_1',.6)]:
        with open('data/splits/'+split+'.csv') as f:folds={int(r['row_idx']):r['fold'] for r in csv.DictReader(f)}
        hold=np.array([i for i in range(len(bits)) if folds[i] in ('val','test')]);holdbits=[bits[i] for i in hold]
        for c in cliffs:
            m=Chem.MolFromSmiles(c['smiles']);m=salt.StripMol(m)
            m=uncharger.uncharge(m);Chem.SanitizeMol(m);key=Chem.MolToInchiKey(m);idx=keys.get(key)
            fp=generator.GetFingerprint(m);sims=np.asarray(DataStructs.BulkTanimotoSimilarity(fp,holdbits));j=int(sims.argmax())
            include=bool(sims[j]<cutoff and (idx is None or folds[idx]=='train'))
            historical=c['in_'+split[:5]+'_cliff_set']=='True'
            records.append(dict(compound=c['compound'],split=split,inchikey=key,row_idx=idx,
                existing_fold=folds[idx] if idx is not None else 'external_to_pool',
                max_tanimoto_to_val_test=float(sims[j]),nearest_holdout_row_idx=int(hold[j]),
                nearest_holdout_fold=folds[int(hold[j])],cutoff=cutoff,eligible=include,
                historical_included=historical,eligibility_agrees=include==historical,
                pair_id=c['pair_id'],role=c['role'],source=c['source']))
    suffix='' if args.tag=='historical' else '_'+args.tag
    out=Path('reports/paper_revision/tables/phase2_cliff_integrity'+suffix+'_both.csv')
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary={s:dict(compounds=sum(r['split']==s for r in records),eligible=sum(r['split']==s and r['eligible'] for r in records),
        eligibility_disagreements=[r['compound'] for r in records if r['split']==s and not r['eligibility_agrees']],
        external_compounds=[r['compound'] for r in records if r['split']==s and r['existing_fold']=='external_to_pool'])
        for s in ['tan70_v1_1','tan60_v1_1']}
    summary['method']='Morgan radius2 2048 bit; exact BulkTanimoto against every current val/test compound; exclude similarity >= cutoff; 128 cache fingerprints verified from released SMILES.'
    summary['source_manifest']=str(path)
    Path('reports/paper_revision/notes/phase2_cliff_integrity'+suffix+'.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
