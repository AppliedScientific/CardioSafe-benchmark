"""Recover the additional curated source used by the original ingestion pipeline."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def main():
    asi=Path(__file__).resolve().parents[2]/'asi'
    sys.path.insert(0,str(asi/'services/bio-intelligence/src'))
    from service_bio_intelligence.ingestion.preprocessing import standardize_molecule,compute_inchi_key
    targets={'hERG':'herg','Nav1.5':'nav15','Cav1.2':'cav12'}
    with open('data/compounds/compounds.csv') as f:released={r['inchikey']:int(r['row_idx']) for r in csv.DictReader(f)}
    records=[];cache={};hashes={}
    for directory,channel in targets.items():
        for path in sorted((asi/'data/ctoxpred-cache'/directory).glob('*.csv')):
            with path.open() as f:rows=list(csv.DictReader(f))
            if not rows or 'pIC50' not in rows[0]:continue
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            for line,row in enumerate(rows,2):
                smi=row['SMILES'].strip()
                if smi not in cache:
                    result=standardize_molecule(smi);cache[smi]=compute_inchi_key(result[0]) if result else ''
                try:value=float(row['pIC50'])
                except (ValueError,KeyError):continue
                key=cache[smi]
                records.append(dict(channel=channel,source_file=str(path),source_line=line,source_smiles=smi,
                    standardized_inchikey=key,released_row_idx=released.get(key,''),pIC50=value,
                    upstream_source=row.get('Source',''),chembl_id=row.get('ChEMBL ID',''),pubchem_cid=row.get('PubChem CID',''),
                    source_used_as=row.get('USED_AS',''),assay_id='',standard_relation='not_available_in_curated_export'))
    out=Path('data/curation/revision/ctoxpred_records.csv')
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    signature=Counter((r['channel'],r['standardized_inchikey'],r['pIC50'],r['upstream_source']) for r in records)
    source_counts=Counter((r['channel'],r['upstream_source']) for r in records)
    report=dict(raw_curated_file_rows=len(records),unique_standardized_compounds=len(set(r['standardized_inchikey'] for r in records)),
        matched_released_compounds=len(set(r['released_row_idx'] for r in records if r['released_row_idx']!='')),
        redundant_curated_rows_same_channel_key_potency_source=sum(n-1 for n in signature.values()),
        source_files_sha256=hashes,counts_by_channel_upstream_source={str(k):v for k,v in source_counts.items()},
        limitation='Curated compound-level pIC50 values; individual assay IDs, original relation and measurement multiplicity are unavailable here. Repeated rows across development/evaluation exports are not independent experimental measurements.')
    Path('reports/paper_revision/notes/phase2_ctoxpred_source_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
