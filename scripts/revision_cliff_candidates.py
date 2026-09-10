"""Propose identity corrections while preserving unverified historical potency data."""
import csv
import json
from pathlib import Path
from rdkit import Chem,RDLogger


def main():
    RDLogger.DisableLog('rdApp.*')
    with open('data/supplementary/note_s2_per_split_manifest.csv') as f:rows=list(csv.DictReader(f))
    with open('reports/paper_revision/tables/phase2_cliff_drug_identity.csv') as f:identities={r['compound']:r for r in csv.DictReader(f)}
    conflicts=[]
    for r in rows:
        identity=identities[r['compound']]
        r['historical_smiles']=r['smiles'];r['identity_source']=identity['source_url']
        if identity['connectivity_matches']=='False':
            r['smiles']=identity['pubchem_smiles']
            assert r['smiles'] and Chem.MolToInchiKey(Chem.MolFromSmiles(r['smiles']))==identity['pubchem_inchikey']
        r['structure_changed']=identity['connectivity_matches']=='False'
        r['potency_source_verified']=False
        r['training_release_status']='QUARANTINED_PENDING_POTENCY_SOURCE_REVIEW'
        value=float(r['ic50_um'])
        for column,threshold in [('herg_1um_block',1.),('herg_10um_block',10.)]:
            expected=int(value<threshold)
            if int(r[column])!=expected:
                conflicts.append(dict(compound=r['compound'],column=column,stored_binary=int(r[column]),
                    stated_ic50_um=value,binary_implied_by_stated_ic50=expected,source=r['source'],
                    note='May reflect distinct assays or clinical-role labelling; source review required, not automatically overwritten.'))
    out=Path('data/curation/revision/cliff_identity_corrected_candidates.csv')
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out=Path('reports/paper_revision/tables/phase2_cliff_potency_binary_disagreements.csv')
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(conflicts[0]));w.writeheader();w.writerows(conflicts)
    print(json.dumps(dict(structures_corrected=sum(r['structure_changed'] for r in rows),
        potency_binary_disagreements=len(conflicts),affected_compounds=sorted(set(r['compound'] for r in conflicts))),indent=2))


if __name__=='__main__':main()
