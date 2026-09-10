"""Export individual ChEMBL evidence with source IDs before any label revision.

Reproduce the located ASI extraction eligibility exactly, including its permissive
pChEMBL clause. Preserve invalid/unused records in a separate raw-target export.
No labels or splits are changed by this script.
"""
import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

TARGETS = {'CHEMBL240':'herg','CHEMBL1980':'nav15','CHEMBL1940':'cav12','CHEMBL2221347':'iks'}
ACTIVATION = re.compile(r'\b(?:activat\w*|agonis\w*|opener\w*|potentiat\w*|enhanc\w*)\b',re.I)


def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_csv(path, rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--db',type=Path,default=Path('/tmp/cardiosafe_revision_chembl36.db'))
    a=p.parse_args();out=Path('reports/paper_revision');data=Path('data/curation/revision');data.mkdir(parents=True,exist_ok=True)
    RDLogger.DisableLog('rdApp.*');salt=SaltRemover();uncharger=rdMolStandardize.Uncharger();largest=rdMolStandardize.LargestFragmentChooser()
    cache={}
    def standardize(smi):
        if smi in cache:return cache[smi]
        try:
            mol=Chem.MolFromSmiles(smi);strip=salt.StripMol(mol)
            if strip.GetNumHeavyAtoms()==0:
                strip=max(Chem.GetMolFrags(mol,asMols=True),key=lambda m:m.GetNumHeavyAtoms())
            strip=uncharger.uncharge(strip);Chem.SanitizeMol(strip)
            key=Chem.MolToInchiKey(strip)
            alt=uncharger.uncharge(largest.choose(mol));Chem.SanitizeMol(alt)
            value=(key,Chem.MolToSmiles(strip),Chem.MolToInchiKey(alt))
        except Exception:value=('','','')
        cache[smi]=value;return value
    with open('data/compounds/compounds.csv') as f:released={r['inchikey']:int(r['row_idx']) for r in csv.DictReader(f)}
    con=sqlite3.connect(f'file:{a.db.resolve()}?mode=ro',uri=True);con.row_factory=sqlite3.Row
    versions=[dict(r) for r in con.execute('SELECT * FROM version')]
    query='''SELECT act.activity_id, act.assay_id, a.chembl_id AS assay_chembl_id,
      md.chembl_id AS molecule_chembl_id, td.chembl_id AS target_chembl_id,
      act.standard_type,act.standard_relation,act.standard_value,act.standard_units,
      act.pchembl_value,act.data_validity_comment,act.potential_duplicate,
      a.description AS assay_description,a.assay_type,a.confidence_score,
      act.doc_id,cs.canonical_smiles
      FROM target_dictionary td JOIN assays a ON a.tid=td.tid
      JOIN activities act ON act.assay_id=a.assay_id
      JOIN molecule_dictionary md ON md.molregno=act.molregno
      JOIN compound_structures cs ON cs.molregno=act.molregno
      WHERE td.chembl_id IN (?,?,?,?) ORDER BY act.activity_id'''
    rows=[]
    for r in con.execute(query,tuple(TARGETS)):
        r=dict(r);smi=r['canonical_smiles'];key,canonical,alt=standardize(smi) if smi else ('','','')
        valid=r['data_validity_comment'] is None and bool(smi)
        potency=valid and (r['pchembl_value'] is not None or (
            r['standard_relation'] in ('>','>=','<','<=') and r['standard_value'] is not None
            and r['standard_units']=='nM' and r['standard_type'] in ('IC50','Ki','EC50','Kd')))
        inhib=valid and r['standard_type']=='Inhibition' and r['standard_value'] is not None and r['standard_units']=='%'
        match=re.search(r'at (\d+(?:\.\d+)?)\s*[uµ]M',r['assay_description'] or '',re.I)
        r.update(channel=TARGETS[r['target_chembl_id']],source_eligible=int(potency or inhib),
            evidence_type='potency' if potency else 'inhibition' if inhib else '',
            parsed_concentration_um=float(match.group(1)) if inhib and match else '',
            standardized_inchikey=key,standardized_smiles=canonical,largest_fragment_inchikey=alt,
            released_row_idx=released.get(key,''),largest_fragment_released_row_idx=released.get(alt,''),
            activation_language_flag=int(bool(ACTIVATION.search(r['assay_description'] or ''))))
        rows.append(r)
    con.close();write_csv(data/'chembl36_target_records.csv',rows)
    selected=[r for r in rows if r['source_eligible']];write_csv(data/'chembl36_eligible_evidence.csv',selected)
    census=[]
    for population,subset in [('all_target_records',rows),('source_eligible',selected),('eligible_released_compounds',[r for r in selected if r['released_row_idx']!=''])]:
        totals=Counter(r['channel'] for r in subset);counts=Counter((r['channel'],r['standard_type'],r['standard_relation']) for r in subset)
        for (channel,typ,rel),n in sorted(counts.items(),key=lambda x:str(x[0])):
            census.append(dict(population=population,channel=channel,standard_type=typ,standard_relation=rel,
                n_measurements=n,channel_measurements=totals[channel],percent=100*n/totals[channel]))
    write_csv(out/'tables/phase2_endpoint_census_both.csv',census)
    ec50=[r for r in selected if r['standard_type']=='EC50'];write_csv(out/'tables/phase2_ec50_all_source_records.csv',ec50)
    flags=[r for r in selected if r['activation_language_flag']];write_csv(out/'tables/phase2_activation_language_review.csv',flags)
    report=dict(source_db=str(a.db),database_version=versions,all_target_rows=len(rows),eligible_rows=len(selected),
        eligible_ec50_by_channel={c:sum(r['channel']==c for r in ec50) for c in TARGETS.values()},
        activation_language_by_channel={c:sum(r['channel']==c for r in flags) for c in TARGETS.values()},
        standardization_disagreement_unique_smiles=sum(k!=alt for k,_,alt in cache.values()),
        standardization_failures_unique_smiles=sum(not k for k,_,alt in cache.values()),
        evidence_sha256=sha(data/'chembl36_eligible_evidence.csv'),
        note='Language flags require contextual review; not automatic exclusions. Census counts measurements, not compounds. No labels changed.')
    (out/'notes/phase2_full_assay_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
