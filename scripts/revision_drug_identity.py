"""Audit historical named cliff structures against PubChem identifiers.

Name lookup is a cross-check, not an automatic structure/label replacement.
Retain every response so discrepancies can be reviewed and reproduced.
"""
import csv
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from rdkit import Chem,RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize


def main():
    RDLogger.DisableLog('rdApp.*');salt=SaltRemover();uncharger=rdMolStandardize.Uncharger()
    source=Path('data/supplementary/note_s2_per_split_manifest.csv')
    with source.open() as f:rows=list(csv.DictReader(f))
    cache=Path('data/curation/revision/pubchem_drug_identity');cache.mkdir(parents=True,exist_ok=True)
    records=[]
    for r in rows:
        name=r['compound'];path=cache/(name.lower().replace(' ','_')+'.json')
        url='https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/'+urllib.parse.quote(name,safe='')+'/property/InChIKey,CanonicalSMILES,IsomericSMILES,MolecularFormula/JSON'
        if not path.exists() or 'SMILES' not in json.loads(path.read_text())['response']['PropertyTable']['Properties'][0]:
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url,timeout=45) as response:body=json.load(response)
                    path.write_text(json.dumps(dict(url=url,response=body),indent=2));break
                except Exception as error:
                    if attempt==3:raise
                    time.sleep(2**attempt)
            time.sleep(.3)
        data=json.loads(path.read_text())['response']['PropertyTable']['Properties']
        assert len(data)==1,(name,'ambiguous name',data)
        pub=data[0];m=Chem.MolFromSmiles(r['smiles']);m=uncharger.uncharge(salt.StripMol(m));Chem.SanitizeMol(m)
        key=Chem.MolToInchiKey(m);reference=pub['InChIKey']
        records.append(dict(compound=name,historical_smiles=r['smiles'],historical_inchikey=key,
            historical_formula=rdMolDescriptors.CalcMolFormula(m),pubchem_cid=pub['CID'],pubchem_inchikey=reference,
            pubchem_formula=pub['MolecularFormula'],pubchem_smiles=pub.get('SMILES',pub.get('IsomericSMILES','')),
            connectivity_matches=key[:14]==reference[:14],full_key_matches=key==reference,
            source_url='https://pubchem.ncbi.nlm.nih.gov/compound/'+str(pub['CID']),
            disposition='identity_supported' if key[:14]==reference[:14] else 'REVIEW_STRUCTURE_MISMATCH'))
        print(name,records[-1]['disposition'],flush=True)
    out=Path('reports/paper_revision/tables/phase2_cliff_drug_identity.csv')
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    report=dict(compounds=len(records),connectivity_mismatches=[r['compound'] for r in records if not r['connectivity_matches']],
        caution='PubChem name lookup cross-check. Review salts, stereochemistry and structures before changing any labels or features. Historical cliff integrity was tested on the historical structures only.')
    Path('reports/paper_revision/notes/phase2_cliff_drug_identity.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
