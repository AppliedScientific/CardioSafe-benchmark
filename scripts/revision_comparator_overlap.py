"""Audit each comparator's actual released training pool on fixed v3 test endpoints."""
import base64
import csv
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import time
import h5py
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from train.train_revision import read_split,read_labels
from train.readout_contract import sha

POOLS=QUERIES=None


def nearest(i):
    fp=QUERIES[i]
    return i,{name:float(max(DataStructs.BulkTanimotoSimilarity(fp,pool))) for name,pool in POOLS.items()}


def main():
    global POOLS,QUERIES
    out=Path('data/comparators/revision_overlap');out.mkdir(exist_ok=False)
    source=Path('reports/paper_revision/inputs/source_snapshots/ingestion/preprocessing.py')
    spec=importlib.util.spec_from_file_location('revision_source_standardizer',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    compounds=pd.read_csv('data/compounds/compounds.csv').set_index('row_idx');labels=read_labels(Path('data/labels/labels_v3.csv'),len(compounds))
    splits={s:read_split(Path('data/splits')/(s+'.csv'))['test'] for s in ['tan70_v1_1','tan60_v1_1']}
    union=np.unique(np.concatenate(list(splits.values())));ct=pd.read_csv('data/curation/revision/ctoxpred_records.csv')
    summaries=[];records=[];pool_rows=[];started=time.monotonic()
    for channel,name,column in [('herg','hERG',1),('nav15','Nav',4),('cav12','Cav',6)]:
        refs={'ctoxpred2':ct.loc[(ct.channel==channel)&ct.source_used_as.eq('Train'),'source_smiles'].tolist()}
        path=Path('assets/revision_comparators/CardioGenAI')/('train_'+name+'.h5')
        with h5py.File(path) as f:refs['cardiogenai']=[base64.urlsafe_b64decode(k).decode() for k in f.keys()]
        POOLS={};keys={};rawkeys={}
        for tag,smiles in refs.items():
            fps=[];identities=set();raw=set()
            for smi in smiles:
                mol=Chem.MolFromSmiles(smi);assert mol is not None
                fps.append(gen.GetFingerprint(mol));rawkey=Chem.MolToInchiKey(mol);raw.add(rawkey)
                standardized=module.standardize_molecule(smi);assert standardized is not None
                key=module.compute_inchi_key(standardized[0]);assert key;identities.add(key)
                pool_rows.append(dict(comparator=tag,channel=channel,source_smiles=smi,raw_inchikey=rawkey,standardized_inchikey=key))
            POOLS[tag]=fps;keys[tag]=identities;rawkeys[tag]=raw
        selected=union[np.isfinite(labels[union,column])]
        QUERIES=[gen.GetFingerprint(Chem.MolFromSmiles(compounds.loc[i,'smiles'])) for i in selected]
        maxima={tag:np.full(len(selected),np.nan) for tag in refs}
        with mp.get_context('fork').Pool(4) as pool:
            for done,(i,values) in enumerate(pool.imap_unordered(nearest,range(len(selected)),chunksize=32),1):
                for tag,v in values.items():maxima[tag][i]=v
                if done%10000==0:print(channel,done,len(selected),round(time.monotonic()-started),flush=True)
        assert all(np.isfinite(a).all() for a in maxima.values())
        for i,row in enumerate(selected):
            key=compounds.loc[row,'inchikey'];record=dict(row_idx=int(row),head=channel,inchikey=key)
            for tag in refs:
                record[tag+'_max_train_tanimoto']=maxima[tag][i]
                record[tag+'_fp99']=bool(maxima[tag][i]>=.99)
                record[tag+'_standardized_key_overlap']=key in keys[tag]
            records.append(record)
        summaries.append(dict(channel=channel,training_rows={t:len(v) for t,v in refs.items()},
            unique_raw_keys={t:len(v) for t,v in rawkeys.items()},unique_standardized_keys={t:len(v) for t,v in keys.items()},
            raw_key_intersection=len(rawkeys['ctoxpred2']&rawkeys['cardiogenai']),
            standardized_key_intersection=len(keys['ctoxpred2']&keys['cardiogenai']),cga_training_file_sha256=sha(path)))
    for name,rows in [('overlap.csv',records),('training_reference_pools.csv',pool_rows)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest=dict(status='complete',pools=summaries,source_sha256=sha(__file__),standardizer_sha256=sha(source),
        compound_sha256=sha('data/compounds/compounds.csv'),labels_sha256=sha('data/labels/labels_v3.csv'),
        ctox_evidence_sha256=sha('data/curation/revision/ctoxpred_records.csv'),
        output_sha256={name:sha(out/name) for name in ['overlap.csv','training_reference_pools.csv']},
        fingerprint='Morgan radius2 2048 bits, chiralityFalse, raw supplied comparator training SMILES',
        masks='Original FP>=0.99 and separate standardized-identity overlap; report union as identity-aware sensitivity',
        limitation='Publicly released training reference files; checkpoint files do not independently attest their complete training histories',
        elapsed_seconds=time.monotonic()-started)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('complete',len(records),flush=True)


if __name__=='__main__':main()
