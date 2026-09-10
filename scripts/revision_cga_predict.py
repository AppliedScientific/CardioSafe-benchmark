"""Pinned CardioGenAI predictions with explicit compound identities and strict weights."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from model.revision_comparator_adapters import load_cga,cga_token_batch,ASSETS
from train.train_revision import read_split
from train.readout_contract import sha


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','mps'],default='cpu');p.add_argument('--limit',type=int)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);RDLogger.DisableLog('rdApp.warning')
    a.out.mkdir(parents=True,exist_ok=False)
    extractor,models,audit=load_cga('cpu');extractor.model.to(a.device);extractor.device=torch.device(a.device)
    from src.utils import get_graph_features,get_fingerprint_features
    from openbabel import pybel,openbabel
    from torch_geometric.data import Data,Batch
    compounds=pd.read_csv('data/compounds/compounds.csv').set_index('row_idx')
    ids=np.unique(np.concatenate([read_split(Path('data/splits')/(s+'.csv'))['test'] for s in ['tan70_v1_1','tan60_v1_1']]))
    if a.limit:ids=ids[:a.limit]
    records=[];started=time.monotonic()
    for start in range(0,len(ids),64):
        batch_ids=ids[start:start+64];graphs=[];strings=[];valid=[]
        for row in batch_ids:
            r=compounds.loc[row];smi=r.smiles;record=dict(row_idx=int(row),inchikey=r.inchikey,valid=False,error='')
            for h in models:record[h+'_prob']=None
            records.append(record)
            try:
                # Validate author tokenization/length before the GPU batch.
                ds=extractor.dataset;tokens=ds.smiles_regex.findall('[CLS]'+smi.strip()+'[EOS]')
                if len(tokens)>ds.block_size:raise ValueError('author_token_limit')
                if any(t not in ds.stoi for t in tokens):raise ValueError('author_vocabulary_unknown_token')
                graph=get_graph_features(smi);fp=get_fingerprint_features(smi);mol=pybel.readstring('smi',smi)
                edges=[];attrs=[]
                for bond in openbabel.OBMolBondIter(mol.OBMol):
                    i,j=bond.GetBeginAtomIdx()-1,bond.GetEndAtomIdx()-1
                    edges.extend([(i,j),(j,i)]);attrs.extend([bond.GetBondOrder()]*2)
                if not edges:raise ValueError('author_graph_has_no_bonds')
                if not np.isfinite(graph).all():raise ValueError('nonfinite_graph_features')
                graphs.append(Data(x=torch.tensor(graph,dtype=torch.float32),
                    edge_index=torch.tensor(edges,dtype=torch.long).T.contiguous(),edge_attr=torch.tensor(attrs,dtype=torch.float32)[:,None],
                    fingerprint=torch.tensor(fp,dtype=torch.float32)[None]))
                strings.append(smi);valid.append(record)
            except (ValueError,KeyError,RuntimeError,TypeError,IndexError) as error:record['error']=str(error)
        if graphs:
            tfv=extractor.model(cga_token_batch(extractor,strings))[:,0,:].cpu()
            graph_batch=Batch.from_data_list(graphs)
            for h,model in models.items():
                values=model(graph_batch.clone(),graph_batch.fingerprint,tfv).squeeze(-1).sigmoid().numpy()
                assert np.isfinite(values).all()
                for record,value in zip(valid,values):record[h+'_prob']=float(value);record['valid']=True
        if start%4096==0:print(start,'/',len(ids),'elapsed_s',round(time.monotonic()-started),flush=True)
    dest=a.out/'predictions.csv'
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    model_files=list((ASSETS/'CardioGenAI/model_parameters').rglob('*.pt'))
    manifest=dict(status='complete',rows=len(ids),valid=sum(r['valid'] for r in records),limit=a.limit,device=a.device,
        source_sha256=sha(__file__),compound_sha256=sha('data/compounds/compounds.csv'),
        split_sha256={s:sha('data/splits/'+s+'.csv') for s in ['tan70_v1_1','tan60_v1_1']},
        weights_sha256={str(p):sha(p) for p in model_files},model_loading=audit,
        vocabulary_sha256=sha(ASSETS/'CardioGenAI/vocabulary.json'),prediction_sha256=sha(dest),
        protocol='Authors graph+Morgan1024+final first-token transformer readout; deterministic evaluation; no labels read',
        upstream_commit='2f403b84bb6fb4a44762ef8d118377a2b1529416',elapsed_seconds=time.monotonic()-started)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('complete',manifest['rows'],manifest['valid'],flush=True)


if __name__=='__main__':main()
