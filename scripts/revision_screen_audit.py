"""Retain measurement-level hERGCentral identity and source overlap evidence."""
import csv
import hashlib
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path


def init():
    global Chem,salt,largest,uncharger
    from rdkit import Chem,RDLogger
    from rdkit.Chem.SaltRemover import SaltRemover
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*');salt=SaltRemover();largest=rdMolStandardize.LargestFragmentChooser();uncharger=rdMolStandardize.Uncharger()


def standardize(row):
    try:
        mol=Chem.MolFromSmiles(row['X'].strip('"'));a=salt.StripMol(mol)
        if a.GetNumHeavyAtoms()==0:a=max(Chem.GetMolFrags(mol,asMols=True),key=lambda m:m.GetNumHeavyAtoms())
        a=uncharger.uncharge(a);Chem.SanitizeMol(a)
        b=uncharger.uncharge(largest.choose(mol));Chem.SanitizeMol(b)
        return dict(source_id=row['ID'],source_smiles=row['X'].strip('"'),
            original_pipeline_inchikey=Chem.MolToInchiKey(a),v2_pipeline_inchikey=Chem.MolToInchiKey(b),
            percent_change_1um=float(row['hERG_at_1uM']),percent_change_10um=float(row['hERG_at_10uM']),
            source_blocker=int(row['hERG_inhib']))
    except Exception:return dict(source_id=row['ID'],source_smiles=row['X'].strip('"'),
            original_pipeline_inchikey='',v2_pipeline_inchikey='',percent_change_1um='',percent_change_10um='',source_blocker='')


def main():
    path=Path(__file__).resolve().parents[2]/'asi/data/herg-central-cache/herg_central.tab'
    with open('data/compounds/compounds.csv') as f:released={r['inchikey']:int(r['row_idx']) for r in csv.DictReader(f)}
    with path.open() as f:raw=list(csv.DictReader(f,delimiter='\t'))
    with Pool(4,initializer=init) as pool:rows=list(pool.imap(standardize,raw,chunksize=100))
    for r in rows:
        r['original_released_row_idx']=released.get(r['original_pipeline_inchikey'],'')
        r['v2_released_row_idx']=released.get(r['v2_pipeline_inchikey'],'')
    out=Path('data/curation/revision/hergcentral_records.csv');out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(source=str(path),raw_rows=len(rows),raw_unique_ids=len(set(r['source_id'] for r in rows)),
        standardization_failed=sum(not r['original_pipeline_inchikey'] for r in rows),
        standardization_disagreement_rows=sum(r['original_pipeline_inchikey']!=r['v2_pipeline_inchikey'] for r in rows),
        source_sign_disagreements=sum((r['percent_change_10um'] < -50)!=bool(r['source_blocker']) for r in rows if r['source_blocker']!=''))
    for policy in ['original','v2']:
        keycol=policy+'_pipeline_inchikey';counts=Counter(r[keycol] for r in rows if r[keycol])
        report[policy]=dict(unique_standardized=len(counts),duplicate_measurement_rows=sum(n-1 for n in counts.values()),
            unique_released_compounds=len(set(r[policy+'_released_row_idx'] for r in rows if r[policy+'_released_row_idx']!='')))
    with path.open('rb') as f:report['source_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    with out.open('rb') as f:report['export_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    Path('reports/paper_revision/notes/phase2_screen_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    # Isolate identity handling while holding v2's evidence-voting rule fixed.
    # These are candidate corrections, not silently edited production labels.
    from scripts.rebuild_herg_labels import blocker_10um
    with open('data/labels/labels_v2.csv') as f:labels={r['inchikey']:r for r in csv.DictReader(f)}
    screen={}
    for r in rows:
        k=r['original_pipeline_inchikey']
        if not k:continue
        if k not in screen or r['percent_change_10um']<screen[k]['percent_change_10um']:screen[k]=r
    changes=[]
    for k,r in screen.items():
        old=labels.get(k)
        if old is None:continue
        a,b=r['percent_change_1um'],r['percent_change_10um']
        potency=float(old['herg_pchembl']) if old['herg_pchembl'] else None
        for col,new in [('herg_blocker_10um',blocker_10um(a,b,potency)),('herg_blocker_1um',int(a < -50))]:
            if old[col]=='' or int(float(old[col]))!=new:
                changes.append(dict(row_idx=old['row_idx'],inchikey=k,head=col,old=old[col],candidate_new=new,
                    source_id=r['source_id'],percent_change_1um=a,percent_change_10um=b))
    with Path('reports/paper_revision/tables/phase2_screen_standardization_label_changes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['row_idx','inchikey','head','old','candidate_new','source_id','percent_change_1um','percent_change_10um']);w.writeheader();w.writerows(changes)
    result=dict(affected_compounds=len(set(r['row_idx'] for r in changes)),changed_cells=len(changes),
        changes=dict(Counter(r['head']+':'+r['old']+'->'+str(r['candidate_new']) for r in changes)),
        interpretation='Candidate identity-only fix under v2 voting; full source evidence reconciliation pending. labels_v2.csv unchanged.')
    Path('reports/paper_revision/notes/phase2_screen_label_candidates.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
