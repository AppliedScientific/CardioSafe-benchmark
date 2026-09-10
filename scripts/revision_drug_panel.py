"""Resolve the requested named-drug panel to released identities and actual folds."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
from train.train_revision import read_labels,LABEL_TO_HEAD
from train.readout_contract import sha

NAMES=['Terfenadine','Fexofenadine','Cisapride','Astemizole','Sertindole','Grepafloxacin',
       'Verapamil','Amiodarone','Ranolazine','Ondansetron','Domperidone']


def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',action='store_true');a=p.parse_args()
    source=Path('reports/paper_revision/inputs/source_snapshots/ingestion/preprocessing.py')
    spec=importlib.util.spec_from_file_location('panel_standardizer',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    compounds=pd.read_csv('data/compounds/compounds.csv').set_index('inchikey',verify_integrity=True)
    labels=read_labels(Path('data/labels/labels_v3.csv'),len(compounds));records=[];sources={}
    folds={s:pd.read_csv(f'data/splits/{s}.csv').set_index('row_idx')['fold'] for s in ['tan70_v1_1','tan60_v1_1']}
    for name in NAMES:
        cache=Path('data/curation/revision/pubchem_drug_identity')/(name.lower()+'.json')
        if not cache.exists():
            url='https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/'+urllib.parse.quote(name)+'/property/InChIKey,CanonicalSMILES,IsomericSMILES,MolecularFormula/JSON'
            with urllib.request.urlopen(url,timeout=45) as response:body=json.load(response)
            cache.write_text(json.dumps(dict(url=url,response=body),indent=2))
        sources[str(cache)]=sha(cache)
        values=json.loads(cache.read_text())['response']['PropertyTable']['Properties'];assert len(values)==1
        value=values[0];smi=value.get('SMILES',value.get('IsomericSMILES'));mol=mod.standardize_molecule(smi)
        assert mol is not None;key=mod.compute_inchi_key(mol[0]);assert key in compounds.index,(name,key)
        c=compounds.loc[key];row=int(c.row_idx)
        for split in folds:
            fold=folds[split].get(row,'excluded')
            r=dict(compound=name,row_idx=row,inchikey=key,smiles=c.smiles,split=split,fold=fold,
                identity_source=f'https://pubchem.ncbi.nlm.nih.gov/compound/{value["CID"]}',
                evaluation_role='Held-out validation example used in model selection' if fold=='val' else
                    ('Training-set illustration; not out-of-sample evidence' if fold=='train' else 'Test-set illustration'),
                clinical_safety_label='Not assigned; channel predictions are not clinical risk classifications')
            for col,head in LABEL_TO_HEAD.items():r[head+'_observed']=float(labels[row,col]) if np.isfinite(labels[row,col]) else None
            records.append(r)
    out=Path('reports/paper_revision/tables/phase4_drug_panel_identity.csv')
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    out.with_suffix('.json').write_text(json.dumps(dict(compounds=len(NAMES),source_sha256=sources,
        standardizer_sha256=sha(source),labels_sha256=sha('data/labels/labels_v3.csv'),
        policy='Exact original-standardized InChIKey only; curated v3 observations; no historical cliff potency/role labels reused'),indent=2))
    print(pd.DataFrame(records)[['compound','split','fold','row_idx']].to_string(index=False))


if __name__=='__main__':main()
