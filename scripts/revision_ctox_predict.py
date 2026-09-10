"""Regenerate pinned CToxPred2 DNN/MC-dropout predictions on frozen test IDs."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
from model.revision_comparator_adapters import ASSETS,fixed_descriptor_transform
from train.train_revision import read_split,read_labels
from train.readout_contract import sha

sys.path.insert(0,os.environ.get('CARDIOSAFE_COMPARATOR_DEPS','/tmp/cardiosafe_revision_dependencies'))
sys.path.insert(0,str(ASSETS/'CToxPred2'))
from CToxPred2.pairwise_correlation import CorrelationThreshold  # pickle's original __main__ class


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(20260428)
    from CToxPred2.utils import compute_fingerprint_features
    from CToxPred2.hERG_model import hERGClassifier
    from CToxPred2.nav15_model import Nav15Classifier
    from CToxPred2.cav12_model import Cav12Classifier
    from mordred import Calculator,descriptors
    from rdkit import Chem,RDLogger
    import joblib
    RDLogger.DisableLog('rdApp.warning');warnings.filterwarnings('ignore',message='Trying to unpickle estimator')
    root=ASSETS/'CToxPred2/CToxPred2/models';models={};pipelines={};artifact_hashes={}
    for h,directory,stem,width,cls in [('herg','hERG','herg',1905,hERGClassifier),('nav15','Nav1.5','nav15',2453,Nav15Classifier),('cav12','Cav1.2','cav12',2586,Cav12Classifier)]:
        model=cls(width,2,.2);path=root/'model_weights'/directory/('_'+stem+'_checkpoint.model')
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);model.train();models[h]=model;artifact_hashes[str(path)]=sha(path)
        if h!='herg':
            path=root/'decriptors_preprocessing'/directory/('nav' if h=='nav15' else 'cav')
            path=path.with_name(path.name+'_descriptors_preprocessing_pipeline.sav')
            pipelines[h]=joblib.load(path);artifact_hashes[str(path)]=sha(path)
    compounds=pd.read_csv('data/compounds/compounds.csv').set_index('row_idx')
    ids=np.unique(np.concatenate([read_split(Path('data/splits')/(s+'.csv'))['test'] for s in ['tan70_v1_1','tan60_v1_1']]))
    if a.limit:ids=ids[:a.limit]
    labels=read_labels(Path('data/labels/labels_v3.csv'),len(compounds))
    records=[];started=time.monotonic();calc=Calculator(descriptors,ignore_3D=True)
    for start in range(0,len(ids),256):
        batch_ids=ids[start:start+256];strings=compounds.loc[batch_ids,'smiles'].tolist();fps=[];good=[];chunk=[]
        for i,(row,smi) in enumerate(zip(batch_ids,strings)):
            record=dict(row_idx=int(row),inchikey=compounds.loc[row,'inchikey'])
            for h in models:record[h+'_prob']=None;record[h+'_valid']=False;record[h+'_error']=''
            chunk.append(record)
            try:fp=compute_fingerprint_features([smi])[0];assert fp.shape==(1905,);fps.append(fp);good.append(i)
            except Exception as e:
                for h in models:record[h+'_error']='fingerprint:'+str(e)
        if good:
            fp=np.asarray(fps,dtype=np.float32);x=torch.from_numpy(fp);samples=[]
            for _ in range(100):samples.append(models['herg'](x)[:,1].numpy())
            means=np.mean(samples,0)
            for i,value in zip(good,means):chunk[i]['herg_prob']=float(value);chunk[i]['herg_valid']=True
            # Expensive descriptors are needed only on labelled small-channel
            # endpoints. Label values do not enter features or predictions.
            selected=[j for j,i in enumerate(good) if a.limit or np.isfinite(labels[batch_ids[i],[4,6]]).any()]
            if selected:
                selected_good=[good[j] for j in selected]
                frame=calc.pandas([Chem.MolFromSmiles(strings[i]) for i in selected_good],nproc=1,quiet=True).fill_missing(np.nan)
                for h,column in [('nav15',4),('cav12',6)]:
                    for jj,i in enumerate(selected_good):
                        if not a.limit and not np.isfinite(labels[batch_ids[i],column]):continue
                        try:
                            features=fixed_descriptor_transform(pipelines[h],frame.iloc[[jj]])
                            inp=torch.from_numpy(np.concatenate([fp[selected[jj]][None],features],1).astype(np.float32))
                            values=torch.stack([models[h](inp)[:,1] for _ in range(100)]).mean(0)
                            chunk[i][h+'_prob']=float(values[0]);chunk[i][h+'_valid']=True
                        except (ValueError,TypeError,RuntimeError) as error:chunk[i][h+'_error']='descriptor:'+str(error)
        records.extend(chunk)
        if start%4096==0:print(start,'/',len(ids),'elapsed_s',round(time.monotonic()-started),flush=True)
    dest=a.out/'predictions.csv'
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    manifest=dict(status='complete',rows=len(ids),limit=a.limit,upstream_commit='2a31aa119e27b6b69a5588d18a01f2a27fef4524',
        protocol='Pretrained DNN, dropout0.2 active, 100 MC passes; class1 mean probability',inference_seed=20260428,
        labels_used='v3 missingness only to select NaV/CaV inference rows; never values, feature fitting, or model fitting',
        labels_sha256=sha('data/labels/labels_v3.csv'),compound_sha256=sha('data/compounds/compounds.csv'),
        source_sha256=sha(__file__),artifact_sha256=artifact_hashes,prediction_sha256=sha(dest),
        valid={h:sum(r[h+'_valid'] for r in records) for h in models},elapsed_seconds=time.monotonic()-started)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('complete',manifest['valid'],flush=True)


if __name__=='__main__':main()
