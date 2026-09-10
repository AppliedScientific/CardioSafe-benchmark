"""Versioned, narrowly scoped source corrections; retain v2 and fixed row order."""
import csv
import json
import hashlib
from collections import Counter
from pathlib import Path
from scripts.revision_curation import reconstruct,read,COLS,CHANNELS
from scripts.revision_assay_decisions import EXCLUDED
from scripts.rebuild_herg_labels import blocker_10um


def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,rows,fields):
    if path.exists():raise FileExistsError(f'Refusing to overwrite {path}')
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    source=Path('data/labels/labels_v2.csv')
    with source.open() as f:reader=csv.DictReader(f);fields=reader.fieldnames;original=list(reader)
    new=[r.copy() for r in original];row_of={r['inchikey']:i for i,r in enumerate(new)}
    evidence=read('chembl36_eligible_evidence.csv');screen=read('hergcentral_records.csv')
    affected={(r['standardized_inchikey'],r['channel']) for r in evidence if int(r['assay_id']) in EXCLUDED}
    filtered,counts=reconstruct(excluded_assays=EXCLUDED,screen_policy='corrected_identity',quarantine_fallback=True)
    reasons={}
    for key,channel in affected:
        if key not in row_of:continue
        i=row_of[key];values=filtered.get(key,[None]*8)
        for col in CHANNELS[channel]:
            name=COLS[col];v=values[col]
            new[i][name]='' if v is None else f'{v:.4f}' if 'pchembl' in name else str(int(v))
            reasons[i,name]='Remove non-blockade/unresolved EC50 evidence; prohibit untraceable curated fallback for affected compound-channel'
    chosen={}
    for r in screen:
        key=r['original_pipeline_inchikey']
        if key not in chosen or float(r['percent_change_10um'])<float(chosen[key]['percent_change_10um']):chosen[key]=r
    for key,r in chosen.items():
        if key not in row_of:continue
        i=row_of[key];p=float(new[i]['herg_pchembl']) if new[i]['herg_pchembl'] else None
        a,b=float(r['percent_change_1um']),float(r['percent_change_10um'])
        new[i]['herg_blocker_10um']=str(blocker_10um(a,b,p));new[i]['herg_blocker_1um']=str(int(a < -50))
        for name in ['herg_blocker_10um','herg_blocker_1um']:
            reasons[i,name]='Original ingestion identity mapping + corrected screen sign; v2 voting rule with corrected retained potency'
    changes=[]
    for i,(old,row) in enumerate(zip(original,new)):
        for name in COLS:
            before=float(old[name]) if old[name] else None;after=float(row[name]) if row[name] else None
            if before!=after:changes.append(dict(row_idx=i,inchikey=row['inchikey'],head=name,old=old[name],new=row[name],reason=reasons[i,name]))
    primary=Path('data/labels/labels_v3.csv');write(primary,new,fields)
    write(Path('reports/paper_revision/tables/phase2_labels_v3_changes.csv'),changes,['row_idx','inchikey','head','old','new','reason'])
    # Assay-pure IC50 endpoint matrix, preserving all released compound rows.
    # Curated CToxPred summaries lack assay relations and are not silently
    # recategorized as individually verified exact IC50 measurements.
    strict,_=reconstruct(excluded_assays=EXCLUDED,ic50_only=True,retain_screen=False,ctoxpred=False)
    strict_rows=[]
    for old in original:
        row={k:old[k] for k in ['row_idx','inchikey']};values=strict.get(old['inchikey'],[None]*8)
        for col,name in enumerate(COLS):
            v=values[col];row[name]='' if v is None else f'{v:.4f}' if 'pchembl' in name else str(int(v))
        strict_rows.append(row)
    ic50=Path('data/labels/labels_revision_ic50.csv');write(ic50,strict_rows,fields)
    manifest=dict(schema_version=3,primary=str(primary),source_labels=str(source),source_labels_sha256=sha(source),
        primary_sha256=sha(primary),strict_ic50_sha256=sha(ic50),compounds=len(new),
        changed_cells=len(changes),affected_compounds=len(set(r['row_idx'] for r in changes)),
        changes_by_head=dict(Counter(r['head'] for r in changes)),
        primary_label_counts={c:sum(bool(r[c]) for r in new) for c in COLS},
        primary_positive_counts={c:sum(r[c]=='1' for r in new) for c in COLS if 'blocker' in c},
        strict_ic50_label_counts={c:sum(bool(r[c]) for r in strict_rows) for c in COLS},
        excluded_assays=sorted(EXCLUDED),source_correction_counts=counts,
        source_evidence_sha256=sha('data/curation/revision/chembl36_eligible_evidence.csv'),
        screen_evidence_sha256=sha('data/curation/revision/hergcentral_records.csv'),
        curated_evidence_sha256=sha('data/curation/revision/ctoxpred_records.csv'),
        policy='Targeted v2 corrections only: EC50 direction/protection/kinetics and source identity mismatch. Other v2 label values retained, including unresolved historical replay differences documented separately. Missing labels stay missing.',
        strict_ic50_policy='ChEMBL36 IC50 exact/censored records only; no screen, Ki/Kd/EC50/AC50/XC50 or curated compound summaries without assay metadata.',
        splits='Unchanged tan70_v1_1 and tan60_v1_1; rows that lose all labels are not reassigned.',
        scope='No cliff fine-tuning labels are included in either main label matrix.')
    Path('data/labels/MANIFEST_v3.json').write_text(json.dumps(manifest,indent=2))
    Path('reports/paper_revision/notes/phase2_label_freeze.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
